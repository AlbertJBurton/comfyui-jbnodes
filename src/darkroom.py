import torch

from comfy import model_management

from .lut import apply_1d_lut
from .interpolate import pchip_interpolate_torch

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

    # Invert negative
    inverted = 1.0 - rgb
    
    # Sensitometric Exposure Calculation
    safe_exposure = max(exposure_secs, 0.001) 
    stops_shift = torch.log2(torch.tensor(safe_exposure / 10.0, device=device))
    exposure_offset = -stops_shift * 0.18
    exposed = torch.clamp(inverted + exposure_offset, 0.0, 1.0)
    
    # Apply Contrast Curve [Normalized 0.0 to 1.0]
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