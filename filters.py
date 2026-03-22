import torch
import numpy as np
import colour

from colour.recovery import RGB_to_sd_Smits1999
from colour.models import XYZ_to_sRGB

from .util import srgb_to_linear_torch, linear_to_srgb_torch

from comfy import model_management

def get_filter_image(image, transmission, filter_factor, auto_exposure, auto_factor, illuminant_name):

    # Retrieve the active ComfyUI PyTorch device (GPU or CPU) for efficient tensor processing.
    device = model_management.get_torch_device()
    
    wl_min = 380
    wl_max = 720
    wl_step = 2

    shape = colour.SpectralShape(wl_min, wl_max, wl_step)
    sd_filter = colour.SpectralDistribution(transmission, shape, name="Wratten Filter")
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(shape)
    illuminant = colour.SDS_ILLUMINANTS[illuminant_name].copy().align(shape)

    # ---------------------------------------------------------
    # PRECOMPUTE BASES ON CPU (Executes exactly once)
    # ---------------------------------------------------------
    # Smits 1999 uses 7 primary bases to reconstruct any spectrum.
    # Since Spectral Multiplication -> XYZ Integration -> Linear sRGB is linear,
    # we can just precalculate the filtered output for the 7 bases.
    bases_lin = np.array([
        [1.0, 1.0, 1.0], # 0: White
        [0.0, 1.0, 1.0], # 1: Cyan
        [1.0, 0.0, 1.0], # 2: Magenta
        [1.0, 1.0, 0.0], # 3: Yellow
        [1.0, 0.0, 0.0], # 4: Red
        [0.0, 1.0, 0.0], # 5: Green
        [0.0, 0.0, 1.0]  # 6: Blue
    ], dtype=np.float32)

    projected_bases = []
    for b in bases_lin:
        # Recover approximate spectrum from primary
        sd = RGB_to_sd_Smits1999(b).align(shape)
        
        # Apply filter transmission
        sd_filtered = colour.SpectralDistribution(sd.values * sd_filter.values, shape)
        
        # Spectrum -> XYZ -> linear sRGB
        XYZ = colour.sd_to_XYZ(sd_filtered, cmfs=cmfs, illuminant=illuminant, method="Integration")
        rgb_lin_out = XYZ_to_sRGB(XYZ / 100.0, apply_cctf_encoding=False)
        projected_bases.append(rgb_lin_out)

    # Move our 7 projected basis vectors to the GPU
    projected_bases = np.array(projected_bases, dtype=np.float32)
    bases_tensor = torch.from_numpy(projected_bases).float().to(device)

    P_white   = bases_tensor[0]
    P_cyan    = bases_tensor[1]
    P_magenta = bases_tensor[2]
    P_yellow  = bases_tensor[3]
    P_red     = bases_tensor[4]
    P_green   = bases_tensor[5]
    P_blue    = bases_tensor[6]

    # ---------------------------------------------------------
    # BATCH GPU TENSOR PROCESSING
    # ---------------------------------------------------------
    # Move the entire image batch tensor to the GPU natively
    img_tensor = image.to(device)

    # Handle Grayscale inputs mapped as (B, H, W, 1)
    if img_tensor.shape[-1] == 1:
        img_tensor = img_tensor.repeat(1, 1, 1, 3)

    has_alpha = img_tensor.shape[-1] == 4
    if has_alpha:
        alpha = img_tensor[..., 3:4]
        rgb = img_tensor[..., :3]
    else:
        alpha = None
        rgb = img_tensor

    rgb = torch.clamp(rgb, 0.0, 1.0)
    
    # Fast GPU vector math: sRGB -> Linear
    rgb_lin = srgb_to_linear_torch(rgb)

    R = rgb_lin[..., 0]
    G = rgb_lin[..., 1]
    B = rgb_lin[..., 2]

    # Extract Smits Basis Weights for every pixel simultaneously
    w_white = torch.amin(rgb_lin, dim=-1)
    w_cyan = torch.clamp(torch.minimum(G, B) - R, min=0.0)
    w_magenta = torch.clamp(torch.minimum(R, B) - G, min=0.0)
    w_yellow = torch.clamp(torch.minimum(R, G) - B, min=0.0)
    w_red = torch.clamp(R - torch.maximum(G, B), min=0.0)
    w_green = torch.clamp(G - torch.maximum(R, B), min=0.0)
    w_blue = torch.clamp(B - torch.maximum(R, G), min=0.0)

    # Composite the final image linearly using the pre-filtered bases
    out_lin = (
        w_white.unsqueeze(-1) * P_white +
        w_cyan.unsqueeze(-1) * P_cyan +
        w_magenta.unsqueeze(-1) * P_magenta +
        w_yellow.unsqueeze(-1) * P_yellow +
        w_red.unsqueeze(-1) * P_red +
        w_green.unsqueeze(-1) * P_green +
        w_blue.unsqueeze(-1) * P_blue
    )

    # Apply exposure compensation
    if not auto_exposure:
        out_lin = out_lin * filter_factor
    else:
        out_lin = out_lin * auto_factor

        # Find the max pixel value per-image across Height, Width, and Channels
        max_vals = torch.amax(out_lin, dim=(1, 2, 3), keepdim=True)
        
        # Prevent blowout: scale down only if max_vals > 1.0
        # If max_vals < 1.0, scale_factor becomes 1.0, leaving the image unchanged.
        scale_factors = torch.clamp(max_vals, min=1.0)
        out_lin = out_lin / scale_factors

    # Fast GPU vector math: Linear -> sRGB
    out_srgb = linear_to_srgb_torch(torch.clamp(out_lin, 0.0, 1.0))
    out_srgb = torch.clamp(out_srgb, 0.0, 1.0)

    # Reattach Alpha if necessary
    if has_alpha:
        out_img = torch.cat([out_srgb, alpha], dim=-1)
    else:
        out_img = out_srgb

    # Return tensor to CPU memory for ComfyUI inter-node routing
    return (out_img.cpu(),)