import torch

from comfy import model_management

def pchip_interpolate_torch(x, y, x_new):
    """
    Batched Monotone Cubic Spline (Fritsch-Carlson) natively in PyTorch.
    Ensures that interpolated points strictly follow the trajectory of the 
    data without overshooting (which preserves the true toe/shoulder roll-off).
    """
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    
    # Secant slopes
    S = dy / dx
    
    m = torch.zeros_like(x)
    
    # Interior points: harmonic mean of adjacent secants
    mask = S[:-1] * S[1:] > 0
    w1 = 2 * dx[1:] + dx[:-1]
    w2 = dx[1:] + 2 * dx[:-1]
    
    m[1:-1] = torch.where(
        mask,
        (w1 + w2) / (w1 / S[:-1] + w2 / S[1:]),
        torch.zeros_like(S[:-1])
    )
    
    # Endpoints (standard finite difference)
    m[0] = S[0]
    m[-1] = S[-1]
    
    # Evaluate spline at x_new
    idx = torch.searchsorted(x, x_new) - 1
    idx = torch.clamp(idx, 0, len(x) - 2)
    
    x_k = x[idx]
    y_k = y[idx]
    m_k = m[idx]
    dx_k = dx[idx]
    
    m_k1 = m[idx + 1]
    y_k1 = y[idx + 1]
    
    t = (x_new - x_k) / dx_k
    t2 = t * t
    t3 = t2 * t
    
    # Hermite basis functions
    h00 = 2*t3 - 3*t2 + 1
    h10 = t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 = t3 - t2
    
    y_new = h00 * y_k + h10 * dx_k * m_k + h01 * y_k1 + h11 * dx_k * m_k1
    return y_new

def apply_1d_lut(tensor, lut):
    """
    Applies a 1D PyTorch tensor LUT to an image tensor using linear interpolation.
    """
    N = lut.shape[0]
    x_scaled = tensor * (N - 1)
    x_floor = x_scaled.floor().long()
    x_ceil = torch.clamp(x_floor + 1, max=N - 1)
    weight = x_scaled - x_floor.float()
    return torch.lerp(lut[x_floor], lut[x_ceil], weight)

def get_print_image(film_negative, contrast_factor=None, exposure_secs=10.0, hd_curve_points=None, d_max=2.1, d_min=0.04, precision=4096):
    """
    Simulates darkroom printing natively in PyTorch.
    Maps normalized exposure to absolute paper reflectance based on D_max and D_min.
    """
    
    device = model_management.get_torch_device()

    if contrast_factor is None and hd_curve_points is None:
        raise ValueError("Either contrast_factor or hd_curve_points must be provided.")

    # Move incoming image to the active ComfyUI computing device (GPU/CPU)
    film_negative = film_negative.to(device)

    has_alpha = film_negative.shape[-1] == 4
    if has_alpha:
        rgb = film_negative[..., :3]
        alpha = film_negative[..., 3:]
    else:
        rgb = film_negative

    # 1. Invert negative
    inverted = 1.0 - rgb
    
    # 2. Sensitometric Exposure Calculation
    safe_exposure = max(exposure_secs, 0.001) 
    stops_shift = torch.log2(torch.tensor(safe_exposure / 10.0, device=device))
    exposure_offset = -stops_shift * 0.18
    exposed = torch.clamp(inverted + exposure_offset, 0.0, 1.0)
    
    # 3. Apply Contrast Curve [Normalized 0.0 to 1.0]
    if hd_curve_points is not None:
        # Move raw coordinate points to the GPU
        hd_curve_points = hd_curve_points.to(device)
        log_e = hd_curve_points[:, 0]
        density = hd_curve_points[:, 1]
        
        # Build the high-precision target axis
        log_e_min, log_e_max = log_e.min(), log_e.max()
        lut_axis = torch.linspace(log_e_min, log_e_max, precision, device=device)
        
        # Run Monotone Cubic Spline to generate high-precision density map
        dense_density = pchip_interpolate_torch(log_e, density, lut_axis)
        
        # Normalize Density map to [0.0, 1.0] so it aligns with 'exposed' tensor
        # norm_tensor=1.0 represents White (D_min)
        # norm_tensor=0.0 represents Black (D_max)
        norm_density_lut = (d_max - dense_density) / (d_max - d_min)
        norm_density_lut = torch.clamp(norm_density_lut, 0.0, 1.0)
        
        # Apply the high-precision LUT to the image pixels
        norm_tensor = 1.0 - apply_1d_lut(exposed, norm_density_lut)
        
        # 4. Map to Physical Paper Density
        density = d_max - norm_tensor * (d_max - d_min)
        
        # 5. Convert Density to Linear Reflectance
        linear_reflectance = 10.0 ** (-density)

        # 6. Encode to sRGB for proper display in ComfyUI
        safe_reflectance = torch.clamp(linear_reflectance, 0.0, 1.0)
        mask = safe_reflectance <= 0.0031308
        srgb_tensor = torch.where(
            mask, 
            safe_reflectance * 12.92, 
            1.055 * torch.pow(safe_reflectance + 1e-8, 1.0 / 2.4) - 0.055
        )
        
        result_tensor = torch.clamp(srgb_tensor, 0.0, 1.0)

    else:
        k = contrast_factor
        curve = 1.0 / (1.0 + torch.exp(-k * (exposed - 0.5)))
        
        ref_k = 3.0
        min_val = 1.0 / (1.0 + torch.exp(torch.tensor(ref_k / 2.0, device=device)))
        max_val = 1.0 / (1.0 + torch.exp(torch.tensor(-ref_k / 2.0, device=device)))
        
        norm_tensor = (curve - min_val) / (max_val - min_val)
        result_tensor = torch.clamp(norm_tensor, 0.0, 1.0)

    if has_alpha:
        result_tensor = torch.cat([result_tensor, alpha], dim=-1)

    return (result_tensor.cpu(),)