'''
    Numpy/Torch Film Grain Engine for comfyui-jbnodes
    -------------------------------------------------
    Copyright (C) 2026  Albert J. Burton

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

    Replaces the moderngl/GLSL film grain pipeline with a pure
    numpy/torch implementation. No system GL dependencies required.
'''
import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from comfy import model_management

# ---------------------------------------------------------------------------
# Film format → physical width in mm
# ---------------------------------------------------------------------------
# The GLSL uses formatMultiplier = 36.0 / film_width, where film_width is
# the physical frame width of the selected format.  36 mm = 35 mm still film.
FILM_WIDTH_MM = {
    "135":  36.0,
    "120":  56.0,     # 6×6 medium format
    "4x5": 100.0,     # 4×5 sheet film
    "8x10": 200.0,    # 8×10 sheet film
}

# ---------------------------------------------------------------------------
# ① Emulsion Softness (Irradiation)
# ---------------------------------------------------------------------------
def apply_emulsion_softness(img_np, softness, format_multiplier):
    """
    5×5-style Gaussian blur approximating the GLSL's micro-Gaussian.

    Parameters
    ----------
    img_np : np.ndarray  (H, W, 3) float32, 0–1
    softness : float  0.0–2.0  (GLSL uniform emulsion_softness)
    format_multiplier : float  (36.0 / film_width_mm)

    Returns a blurred copy in the same space as the input (negative / raw).
    """
    if softness <= 0.0:
        return img_np.copy()

    h, w = img_np.shape[:2]

    # Resolution-independent blur radius.
    # GLSL: blurRadius = softness * (texSize.x * 0.001)
    # Here we scale by the image diagonal / 1024 so the effect looks the same
    # regardless of image dimensions.
    ref = max(h, w)
    radius_px = softness * (ref * 0.001) * format_multiplier

    if radius_px < 0.5:
        return img_np.copy()

    sigma = max(radius_px / 3.0, 0.5)

    result = np.empty_like(img_np)
    for c in range(3):
        result[..., c] = gaussian_filter(img_np[..., c], sigma=sigma, mode='reflect')
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# ② Halation (Light Piping)  —  torch grid_sample spiral blur
# ---------------------------------------------------------------------------
def _build_halation_kernel(height, width, radius_px, device):
    """
    Pre-compute the 32-tap golden-angle spiral sampling grid for halation.

    Returns a tensor of shape (1, 32, H, W, 2) suitable for
    F.grid_sample, containing normalised coordinates in [-1, 1].
    """
    golden_angle = 2.39996323  # radians

    # Build normalised pixel coordinates  (H, W) in [-1, 1]
    y = torch.linspace(-1.0, 1.0, height, device=device)
    x = torch.linspace(-1.0, 1.0, width, device=device)
    gy, gx = torch.meshgrid(y, x, indexing='ij')  # (H, W)

    # (H, W, 2)  —  base position of every pixel
    base = torch.stack([gx, gy], dim=-1)  # (H, W, 2)

    # Build the 32 spiral offsets in normalised units
    # GLSL: r = sqrt(i+0.5) * (radius / 5.65685),  theta = i * golden_angle
    # radius_px is the halation radius in pixels; convert to normalised.
    norm_r = radius_px / max(width, height) * 2.0  # pixel → [-1,1] units

    taps = []
    for i in range(32):
        r = math.sqrt(i + 0.5) * (norm_r / 5.65685)
        theta = i * golden_angle
        dx = math.cos(theta) * r
        dy = math.sin(theta) * r
        taps.append((dx, dy))

    # Each tap is an offset from the base pixel position.
    # Build (32, H, W, 2)
    tap_offsets = torch.tensor(taps, device=device, dtype=torch.float32)  # (32, 2)
    # Reshape to broadcast: (1, 32, H, W, 2) ← base (1, 1, H, W, 2) + offsets (1, 32, 1, 1, 2)
    base_5d = base.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, 2)
    offsets_5d = tap_offsets.view(1, 32, 1, 1, 2)
    grid = base_5d + offsets_5d
    # Clamp to valid range to avoid edge wrap artefacts
    grid = torch.clamp(grid, -1.0, 1.0)
    return grid


def apply_halation(img_tensor, strength, format_multiplier, _cache={}):
    """
    32-tap golden-angle spiral halation using F.grid_sample.

    Parameters
    ----------
    img_tensor : torch.Tensor  (1, H, W, 3) or (B, H, W, 3), 0–1
    strength : float  0.0–1.0
    format_multiplier : float  (36.0 / film_width_mm)

    Returns the halation glow layer as (1, H, W, 3) in the same colour space
    as the input (negative / raw).
    """
    if strength <= 0.0:
        return torch.zeros_like(img_tensor)

    B, H, W, C = img_tensor.shape
    device = img_tensor.device

    # Resolution-independent radius (same logic as GLSL)
    # GLSL: radius = texSize.x * 0.02 * formatMultiplier
    radius_px = W * 0.02 * format_multiplier
    if radius_px < 2.0:
        return torch.zeros_like(img_tensor)

    # Cache the grid across calls for the same (H, W, device)
    cache_key = (H, W, device)
    if cache_key not in _cache:
        _cache[cache_key] = _build_halation_kernel(H, W, radius_px, device)
    grid = _cache[cache_key]

    # grid_sample expects (N, H, W, 2) but we have (1, 32, H, W, 2).
    # We'll reshape to combine tap and batch dims.
    # img_tensor: (B, H, W, C) → (B, C, H, W) for grid_sample
    img_t = img_tensor.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

    # Reshape grid: (1, 32, H, W, 2) → (32, H, W, 2)
    grid_2d = grid.squeeze(0)  # (32, H, W, 2)

    # Sample the image at each tap location.
    # grid_sample expects grid (N, H, W, 2); we have 32 separate grids.
    # Process each tap then compose.
    # We can batch all 32 taps by expanding the image batch dimension:
    # img_t: (B, C, H, W) → tap all 32 locations
    samples = []
    for i in range(32):
        g = grid_2d[i].unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
        s = F.grid_sample(
            img_t, g, mode='bilinear', padding_mode='zeros', align_corners=False
        )  # (B, C, H, W)
        samples.append(s)

    # Stack and average with exponential falloff matching GLSL
    # GLSL: w = exp(-i / 10.0)
    stacked = torch.stack(samples, dim=1)  # (B, 32, C, H, W)

    weights = torch.exp(-torch.arange(32, device=device, dtype=torch.float32) / 10.0)
    weights = weights.view(1, 32, 1, 1, 1)

    # Weighted sum
    weighted = (stacked * weights).sum(dim=1)  # (B, C, H, W)
    weight_sum = weights.sum()

    # Convert to positive for highlight extraction
    # GLSL: s_pos = 1.0 - s_neg  (the image is stored as a negative)
    positive = 1.0 - (weighted / weight_sum)  # (B, C, H, W)

    # Extract luminance and isolate highlights (smoothstep 0.5–1.0)
    # ITU-R BT.601 luma
    luma = (
        0.299 * positive[:, 0:1, :, :]
        + 0.587 * positive[:, 1:2, :, :]
        + 0.114 * positive[:, 2:3, :, :]
    )
    bright = torch.clamp((luma - 0.5) / 0.5, 0.0, 1.0)  # smoothstep(0.5, 1.0, luma)
    # Smoother: smoothstep ≈ 3t² - 2t³
    bright = bright * bright * (3.0 - 2.0 * bright)

    # GLSL: halo = s_pos * bright * w  (but already applied above)
    # The glow is the highlight-weighted average
    glow = positive * bright * strength

    # Clamp the glow
    glow = torch.clamp(glow, 0.0, 1.0)

    return glow.permute(0, 2, 3, 1)  # (B, H, W, C)


# ---------------------------------------------------------------------------
# ③ Luminance Mask
# ---------------------------------------------------------------------------
def luminance_mask(luminance, peak_bias):
    """
    Parabolic attenuation peaking at luminance_peak_bias, multiplied by a
    smoothstep highlight rolloff.

    Parameters
    ----------
    luminance : torch.Tensor  (H, W) or (B, H, W) float32, 0–1
    peak_bias : float  0.0–1.0

    Returns mask tensor of same shape, 0–1 range.
    """
    denom = max(max(peak_bias, 1.0 - peak_bias), 0.0001)
    # M = 1 - ((L - peak_bias) / denom)²
    M = 1.0 - ((luminance - peak_bias) / denom) ** 2
    M = torch.clamp(M, 0.0, 1.0)
    # Highlight rolloff: smoothstep(1.0, 0.8, L)
    rolloff = torch.clamp((luminance - 0.8) / 0.2, 0.0, 1.0)
    rolloff = 1.0 - (rolloff * rolloff * (3.0 - 2.0 * rolloff))
    M = M * rolloff
    return M


# ---------------------------------------------------------------------------
# ④a  Cubic Grain  —  Multi-octave filtered noise  (numpy / scipy)
# ---------------------------------------------------------------------------
def generate_cubic_grain(height, width, seed, octaves, morph_variance, film_grit):
    """
    Multi-octave Gaussian-filtered noise for cubic-emulsion grain.

    Each octave doubles frequency.  Amplitude is multiplied by
    morph_variance each octave (morph_variance < 1 = decay).
    The raw noise is sharpened via smoothstep controlled by film_grit.

    Parameters
    ----------
    height, width : int  image dimensions in pixels
    seed : int  RNG seed  (per-image for batch variation)
    octaves : int  1–10  number of noise octaves
    morph_variance : float  0.0–2.0  amplitude multiplier per octave
    film_grit : float  0.0–1.0  grain-edge sharpness

    Returns np.ndarray (H, W) float32, roughly 0–1 range.
    """
    rng = np.random.RandomState(seed)

    # Reference resolution for consistent grain sizing
    ref = max(height, width)
    size_scale = ref / 1024.0

    # Base sigma calibrated to produce grain similar to the GLSL output.
    # GLSL: baseGrainUV = 0.002, then scaleFactorUV = (rms/8) * baseGrainUV * formatMultiplier
    #       uv_noise = uv_aspect / scaleFactorUV
    # The key insight: the noise frequency is (1 / scaleFactorUV) in UV space.
    # For 1024 px wide image at rms 8, 35 mm format:
    #   scaleFactorUV = (8/8) * 0.002 * 1.0 = 0.002
    #   This means 1 cycle per 0.002 UV = 1 cycle per ~2 px
    # The Gaussian sigma for equivalent visual effect is ~1.5 px at base.
    base_sigma = 1.5 * size_scale

    grain = np.zeros((height, width), dtype=np.float32)
    total_amp = 0.0
    amp = 1.0

    for i in range(octaves):
        octave_freq = 2.0 ** i
        sigma = base_sigma / octave_freq if octave_freq > 1 else base_sigma

        noise = rng.randn(height, width).astype(np.float32)

        # Only blur if sigma is meaningful; below ~0.3 px the Gaussian
        # kernel is essentially a delta function.
        if sigma >= 0.4:
            noise = gaussian_filter(noise, sigma=sigma, mode='reflect')

        grain += noise * amp
        total_amp += amp
        amp *= morph_variance

    # Normalise to roughly [-1, 1] then map to [0, 1]
    if total_amp > 0.0:
        grain = grain / total_amp

    # Map to [0, 1] —  the GLSL does (n_raw / max_amp) * 0.5 + 0.5
    grain = grain * 0.5 + 0.5
    grain = np.clip(grain, 0.0, 1.0)

    # Film grit: smoothstep thresholding sharpens the grain edges.
    # GLSL: spread = mix(0.5, 0.01, film_grit)
    #        n_raw = smoothstep(0.5 - spread, 0.5 + spread, n_raw)
    spread = np.interp(film_grit, [0.0, 1.0], [0.5, 0.01])
    lower = 0.5 - spread
    upper = 0.5 + spread
    if upper > lower:
        grain = np.clip((grain - lower) / (upper - lower), 0.0, 1.0)

    return grain


# ---------------------------------------------------------------------------
# ④b  Tabular Grain  —  Torch Voronoi (cellular) noise
# ---------------------------------------------------------------------------
def _hash33_torch(p3):
    """
    GLSL hash33 — exact match of the shader's pseudo-random vec3→vec3.

    GLSL reference:
      vec3 hash33(vec3 p3) {
          p3 = fract(p3 * vec3(.1031, .1030, .0973));
          p3 += dot(p3, p3.yxz+33.33);
          return fract((p3.xxy + p3.yxx)*p3.zyx);
      }
    """
    scale = torch.tensor([0.1031, 0.1030, 0.0973], device=p3.device, dtype=torch.float32)
    p3 = p3 * scale
    p3 = p3 - torch.floor(p3)  # fract

    # dot(p3, p3.yxz + 33.33)
    yxz = torch.stack([p3[..., 1], p3[..., 0], p3[..., 2]], dim=-1)
    dot_val = (p3 * (yxz + 33.33)).sum(dim=-1, keepdim=True)
    p3 = p3 + dot_val

    # fract((p3.xxy + p3.yxx) * p3.zyx)
    xxy = torch.stack([p3[..., 0], p3[..., 0], p3[..., 1]], dim=-1)
    yxx = torch.stack([p3[..., 1], p3[..., 0], p3[..., 0]], dim=-1)
    zyx = torch.stack([p3[..., 2], p3[..., 1], p3[..., 0]], dim=-1)
    result = (xxy + yxx) * zyx
    result = result - torch.floor(result)  # fract
    return result


def generate_tabular_grain_torch(height, width, seed, octaves, morph_variance,
                                 temporal_entropy, device):
    """
    3D Voronoi / cellular noise for tabular-emulsion (T-grain) grain,
    implemented natively in torch with vectorized 27-neighbor search.

    Memory-efficient: processes the 27 neighbors in 3 batches of 9
    (one Z-layer at a time) to avoid a 27×H×W×3 intermediate tensor.
    """
    ref = max(height, width)
    size_scale = ref / 1024.0
    base_scale = 0.002
    scale = base_scale * size_scale

    seed_offset = (seed % 10000) * 7.193

    y = torch.linspace(0.0, height - 1, height, device=device) * scale + seed_offset
    x = torch.linspace(0.0, width - 1, width, device=device) * scale + seed_offset
    gy, gx = torch.meshgrid(y, x, indexing='ij')

    aspect_val = height / width
    noise_x = gx
    noise_y = gy * aspect_val
    noise_z = torch.full_like(noise_x, temporal_entropy + seed_offset * 0.01)

    raw = torch.zeros((1, 1, height, width), device=device, dtype=torch.float32)
    max_amp = 0.0
    amp = 1.0

    cos_a, sin_a = 0.8, 0.6
    px, py, pz = noise_x.clone(), noise_y.clone(), noise_z.clone()

    # XY offsets for the 3×3 XY grid
    xy_offsets = []
    for j in (-1, 0, 1):
        for i in (-1, 0, 1):
            xy_offsets.append((float(i), float(j)))
    xy_offs = torch.tensor(xy_offsets, device=device, dtype=torch.float32)  # (9, 2)

    for oct in range(octaves):
        freq = 2.0 ** oct
        cx = px * freq
        cy = py * freq
        cz = pz * freq

        cell_x = torch.floor(cx)
        cell_y = torch.floor(cy)
        cell_z = torch.floor(cz)
        fx = cx - cell_x
        fy = cy - cell_y
        fz = cz - cell_z

        # Initialise F1, F2 to large values
        F1 = torch.full((height, width), float('inf'), device=device, dtype=torch.float32)
        F2 = torch.full((height, width), float('inf'), device=device, dtype=torch.float32)

        # Process the 27 neighbours in 3 Z-layers to keep memory low
        for k in (-1, 0, 1):
            n_cell_z_k = cell_z + k  # (H, W)

            # (9, H, W) — broadcast cell_(x|y) with xy offsets
            n_cell_x = cell_x.unsqueeze(0) + xy_offs[:, 0:1].view(9, 1, 1)
            n_cell_y = cell_y.unsqueeze(0) + xy_offs[:, 1:2].view(9, 1, 1)

            # Build cell key for all 9 neighbours: (9, H, W, 3)
            cell_key = torch.stack([
                n_cell_x,
                n_cell_y,
                n_cell_z_k.unsqueeze(0).expand(9, -1, -1),
            ], dim=-1)  # (9, H, W, 3)

            # Compute hash for all 9 neighbours at once
            h = _hash33_torch(cell_key)  # (9, H, W, 3)

            # Displacement from pixel to feature point:
            #   r = g - f + o   where g = offset, f = fract, o = hash
            dx = xy_offs[:, 0:1].view(9, 1, 1) - fx.unsqueeze(0) + h[..., 0]
            dy = xy_offs[:, 1:2].view(9, 1, 1) - fy.unsqueeze(0) + h[..., 1]
            dz = k - fz.unsqueeze(0) + h[..., 2]

            d2 = dx * dx + dy * dy + dz * dz  # (9, H, W)

            # Update F1 and F2 across all 9 neighbours
            # For each y in (9, H, W), find best two along dim=0
            for n in range(9):
                d2_n = d2[n]
                closer = d2_n < F1
                further = (~closer) & (d2_n < F2)
                F2 = torch.where(closer, F1, F2)
                F1 = torch.where(closer, d2_n, F1)
                F2 = torch.where(further, d2_n, F2)

        # Voronoi value = sqrt(F2) - sqrt(F1)
        v_val = torch.sqrt(F2) - torch.sqrt(F1)

        raw += v_val.unsqueeze(0).unsqueeze(0) * amp
        max_amp += amp
        amp *= morph_variance

        # Rotate and scale coordinates for next octave
        rx = cos_a * px - sin_a * py
        ry = sin_a * px + cos_a * py
        px, py = rx, ry
        pz = pz * 2.0

    if max_amp > 0.0:
        raw = raw / max_amp

    # GLSL centering: (n_raw - 0.25) * 2.0 + 0.5
    raw = (raw - 0.25) * 2.0 + 0.5
    raw = torch.clamp(raw, 0.0, 1.0)

    return raw


# ---------------------------------------------------------------------------
# ⑤  Blend Modes
# ---------------------------------------------------------------------------
def _blend_soft_light(base, blend):
    """Soft Light blend mode — the default."""
    # GLSL: (blend < 0.5) ? base - (1-2*blend)*base*(1-base)
    #                   : base + (2*blend-1)*(d(base) - base)
    # where d(x) = ((x <= 0.25) ? ((16x-12)x+4)x : sqrt(x))
    mask = blend < 0.5
    low = base - (1.0 - 2.0 * blend) * base * (1.0 - base)
    # 'd' function for the highlight branch
    d_vals = torch.where(
        base <= 0.25,
        ((16.0 * base - 12.0) * base + 4.0) * base,
        torch.sqrt(base)
    )
    high = base + (2.0 * blend - 1.0) * (d_vals - base)
    return torch.where(mask, low, high)


def _blend_overlay(base, blend):
    """Overlay blend mode."""
    mask = base < 0.5
    low = 2.0 * base * blend
    high = 1.0 - 2.0 * (1.0 - base) * (1.0 - blend)
    return torch.where(mask, low, high)


def _blend_linear_light(base, blend):
    """Linear Light blend mode."""
    return torch.clamp(base + 2.0 * blend - 1.0, 0.0, 1.0)


def apply_blend(base, grain, mode):
    """
    Apply one of three blend modes between the base image and the grain layer.

    Parameters
    ----------
    base : torch.Tensor  (B, H, W, C) — image to grain
    grain : torch.Tensor  (B, H, W, C) — grain layer (grain value broadcast
                           to all channels)
    mode : int  0 = Soft Light, 1 = Overlay, 2 = Linear Light

    Returns (B, H, W, C)
    """
    if mode == 1:
        return _blend_overlay(base, grain)
    elif mode == 2:
        return _blend_linear_light(base, grain)
    else:
        return _blend_soft_light(base, grain)


# ---------------------------------------------------------------------------
# ⑥  Shadow Dither
# ---------------------------------------------------------------------------
def apply_shadow_dither(image, raw_noise_np, luminance, shadow_dither_amount,
                        intensity):
    """
    Add unstructured noise to deep shadow regions to break posterisation.

    Parameters
    ----------
    image : torch.Tensor  (B, H, W, C) — current image after blending
    raw_noise_np : np.ndarray  (H, W) — raw noise before thresholding
    luminance : torch.Tensor  (B, H, W) — luminance of the base image
    shadow_dither_amount : float  0.0–2.0
    intensity : float  (rms_granularity * 0.02) per the GLSL

    Returns (B, H, W, C)
    """
    if shadow_dither_amount <= 0.0:
        return image

    B = image.shape[0]
    device = image.device

    # Shadow mask: smoothstep(0.3, 0.0, L) — strongest in the deepest shadows
    shadow_mask = torch.clamp((0.3 - luminance) / 0.3, 0.0, 1.0)

    # GLSL: finalColor += (n_raw - 0.5) * shadow_mask * shadow_dither * intensity
    noise_t = torch.from_numpy(raw_noise_np).float().to(device)
    # noise_t (H,W) × shadow_mask (B,H,W) → (B,H,W) via broadcasting
    dither = (noise_t - 0.5) * shadow_mask * shadow_dither_amount * intensity

    # Expand to 4-channel: (B, H, W) → (B, H, W, 1) → (B, H, W, 3)
    dither = dither.unsqueeze(-1).expand(-1, -1, -1, 3)
    result = image + dither
    return torch.clamp(result, 0.0, 1.0)


# ---------------------------------------------------------------------------
# RGB → Luminance  (Rec.709 / BT.601 compatible)
# ---------------------------------------------------------------------------
def luminance_torch(img):
    """BT.601 luma from a (B, H, W, C) tensor.  Returns (B, H, W)."""
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])


# ---------------------------------------------------------------------------
# Film format → format multiplier  (36.0 / film_width_mm)
# ---------------------------------------------------------------------------
def get_format_multiplier(film_size):
    """Return the format multiplier given a film_size string key."""
    fw = FILM_WIDTH_MM.get(film_size, 36.0)
    return 36.0 / fw


# ---------------------------------------------------------------------------
# Master orchestrator
# ---------------------------------------------------------------------------
def apply_film_grain(
    image: torch.Tensor,
    rms_granularity: float = 8.0,
    film_size: str = "135",
    emulsion_type: str = "Cubic",
    film_grit: float = 0.2,
    halation: float = 0.1,
    emulsion_softness: float = 0.75,
    blend_mode: str = "Soft Light",
    luminance_peak_bias: float = 0.50,
    algorithmic_octaves: int = 2,
    morphological_variance: float = 1.0,
    temporal_entropy: float = 1.0,
    shadow_dither: float = 0.0,
    seed: int = 0,
    image_index: int = 0,
):
    """
    Full film-grain pipeline, replacing the moderngl/GLSL implementation.

    Accepts a ComfyUI IMAGE tensor  (B, H, W, C)  in sRGB 0–1 range.

    Returns the grained image as a (B, H, W, C) tensor.
    """
    B, H, W, C = image.shape
    device = image.device

    if rms_granularity <= 0.0:
        return image

    fmt_mult = get_format_multiplier(film_size)
    blend_map = {"Soft Light": 0, "Overlay": 1, "Linear Light": 2}
    blend_idx = blend_map.get(blend_mode, 0)
    intensity = rms_granularity * 0.02  # GLSL line: float intensity = safe_rms * 0.02

    # Per-image seed offset
    img_seed = seed + image_index * 7919

    results = []
    for b in range(B):
        # Extract single image
        img_srgb = image[b]  # (H, W, C)

        # ---------------------------------------------------------------
        # Stage ①: Emulsion Softness  (numpy)
        # ---------------------------------------------------------------
        img_np = img_srgb.cpu().numpy().astype(np.float32)
        blurred_np = apply_emulsion_softness(img_np, emulsion_softness, fmt_mult)

        # Convert back to torch
        blurred = torch.from_numpy(blurred_np).to(device)

        # ---------------------------------------------------------------
        # Stage ②: Halation  (torch grid_sample)
        # ---------------------------------------------------------------
        # Halation samples from the ORIGINAL image, not the blurred one.
        # The result is Screen-blended onto the positive version of the image.
        if halation > 0.0:
            glow = apply_halation(
                image[b:b+1], halation, fmt_mult
            )  # (1, H, W, C)

            # GLSL: positiveBase = 1.0 - blurredRGB
            #       positiveFinal = 1.0 - (1.0 - positiveBase) * (1.0 - positiveGlow)
            #       blurredRGB = 1.0 - positiveFinal
            positive_base = 1.0 - blurred
            positive_glow = glow[b]  # (H, W, C)
            positive_final = 1.0 - (1.0 - positive_base) * (1.0 - positive_glow)
            blurred = 1.0 - positive_final

        # ---------------------------------------------------------------
        # Stage ③: Luminance extraction
        # ---------------------------------------------------------------
        L = luminance_torch(blurred.unsqueeze(0)).squeeze(0)  # (H, W)

        # ---------------------------------------------------------------
        # Stage ④: Grain generation
        # ---------------------------------------------------------------
        if emulsion_type == "Tabular":
            # Torch Voronoi noise (on GPU)
            grain_t = generate_tabular_grain_torch(
                H, W, img_seed, algorithmic_octaves, morphological_variance,
                temporal_entropy, device
            )  # (1, 1, H, W)
            grain_np = grain_t.squeeze().cpu().numpy()  # (H, W)
            raw_np = grain_np.copy()  # save for shadow dither
        else:
            # Numpy filtered noise (Cubic)
            grain_np = generate_cubic_grain(
                H, W, img_seed, algorithmic_octaves, morphological_variance, film_grit
            )  # (H, W)
            raw_np = grain_np.copy()  # save raw before thresholding

            # Re-threshold the grain for consistency with the tabular path.
            # The cubic path already applies grit thresholding inside the
            # generator; we keep the post-threshold version for blending.
            # We save raw_np BEFORE the smoothstep for shadow dither use.
            # Actually, generate_cubic_grain returns the post-smoothstep result.
            # For shadow dither, the GLSL uses (n_raw - 0.5) which is PRE-grit.
            # We need to regenerate or save pre-grit noise.
            # Let's regenerate the pre-grit for shadow dither.
            # (We'll handle this in the master function.)

        # ---------------------------------------------------------------
        # Stage ⑤: Luminance mask
        # ---------------------------------------------------------------
        # GLSL: n_final = (n_raw - 0.5) * M * intensity
        # But after the smoothstep/grit threshold, n_raw is already in [0,1].
        # We need to map grain back to centred-around-0.5 for the blend.
        M = luminance_mask(L, luminance_peak_bias)  # (H, W)

        # Convert grain to centred value
        grain_centred = torch.from_numpy(grain_np).float().to(device) - 0.5  # (H, W)
        n_final = grain_centred * M * intensity

        # Build grain layer for blending
        grain_layer = (n_final + 0.5).clamp(0.0, 1.0).unsqueeze(-1)  # (H, W, 1)

        # Broadcast to 3 channels
        grain_rgb = grain_layer.expand(-1, -1, 3)  # (H, W, 3)

        # ---------------------------------------------------------------
        # Stage ⑥: Apply blend
        # ---------------------------------------------------------------
        result = apply_blend(blurred, grain_rgb, blend_idx)

        # ---------------------------------------------------------------
        # Stage ⑦: Shadow dither
        # ---------------------------------------------------------------
        if shadow_dither > 0.0:
            # We need raw noise (pre-grit).  For cubic, regenerate without
            # grit.  For tabular, raw_np was saved pre-normalisation.
            # Simpler approach: generate raw unshaped noise for dither only.
            if emulsion_type == "Tabular":
                dither_noise = raw_np
            else:
                # Generate pre-grit noise for shadow dither
                rng = np.random.RandomState(img_seed + 31337)
                dither_noise = rng.randn(H, W).astype(np.float32) * 0.5 + 0.5
                dither_noise = np.clip(dither_noise, 0.0, 1.0)

            result = apply_shadow_dither(
                result.unsqueeze(0), dither_noise, L.unsqueeze(0),
                shadow_dither, intensity
            ).squeeze(0)

        # Preserve the original alpha channel if present
        if C == 4:
            result = torch.cat([result, image[b, :, :, 3:4]], dim=-1)

        results.append(result)

    return torch.stack(results, dim=0)
