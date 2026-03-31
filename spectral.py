import torch
import numpy as np
import colour

from colour.recovery import RGB_to_sd_Smits1999

from .util import linear_to_srgb_torch, srgb_to_linear_torch, pchip_interpolate_numpy
from .classes import FilmFormat, HDCurve, FilmStock

from comfy import model_management

# Processes a batch of ComfyUI image tensors natively in PyTorch.
# Utilizes Smits 1999 spectral integration when raw coordinate data is available.
#
# Args:
#     image (torch.Tensor): Shape (B, H, W, 3/4) tensor.
#     weights (list/tuple): Fallback [rW, gW, bW] floats.
#     spectral_points (list): Nested list of [Wavelength, Sensitivity] pairs.
#     illuminant_name (str): Key of the illuminating light source.
#     char_lut (np.ndarray): 1D uint8 array representing the characteristic curve.

def get_spectral_image(image, weights, spectral_points, illuminant_name, char_lut):

    device = model_management.get_torch_device()
    img_tensor = image.to(device)

    # Handle Alpha channel mapping
    has_alpha = img_tensor.shape[-1] == 4
    if has_alpha:
        alpha = img_tensor[..., 3:]
        rgb = img_tensor[..., :3]
    else:
        alpha = None
        rgb = img_tensor

    rgb = torch.clamp(rgb, 0.0, 1.0)
    rgb_lin = srgb_to_linear_torch(rgb)

    if spectral_points is not None:
        # ---------------------------------------------------------
        # PRECOMPUTE SPD INTEGRATION ON CPU
        # ---------------------------------------------------------
        wl_min = 380
        wl_max = 720
        wl_step = 2
        shape = colour.SpectralShape(wl_min, wl_max, wl_step)
        
        # Interpolate raw empirical points into a 171-bin continuous spectrum
        pts = np.array(spectral_points)
        wl_data = pts[:, 0]
        sens_data = pts[:, 1]
        
        wl_axis = np.arange(wl_min, wl_max + wl_step, wl_step)
        dense_spd = pchip_interpolate_numpy(wl_data, sens_data, wl_axis)

        sd_film = colour.SpectralDistribution(dense_spd, shape)
        sd_illuminant = colour.SDS_ILLUMINANTS[illuminant_name].copy().align(shape)
        
        bases_lin = np.array([
            [1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        responses = []
        for b in bases_lin:
            sd_basis = RGB_to_sd_Smits1999(b).align(shape)
            # Physical Exposure = Basis * Illuminant * Film Sensitivity
            response = np.sum(sd_basis.values * sd_film.values * sd_illuminant.values)
            responses.append(response)

        responses = np.array(responses, dtype=np.float32)
        
        # Normalize to the White Basis so pure white strictly = 1.0 latent exposure
        responses = responses / responses[0]

        weights_tensor = torch.from_numpy(responses).float().to(device)

        W   = weights_tensor[0]
        C   = weights_tensor[1]
        M   = weights_tensor[2]
        Y   = weights_tensor[3]
        R_w = weights_tensor[4]
        G_w = weights_tensor[5]
        B_w = weights_tensor[6]

        # Extract Smits Basis Weights for every pixel
        R = rgb_lin[..., 0]
        G = rgb_lin[..., 1]
        B = rgb_lin[..., 2]

        w_white   = torch.amin(rgb_lin, dim=-1)
        w_cyan    = torch.clamp(torch.minimum(G, B) - R, min=0.0)
        w_magenta = torch.clamp(torch.minimum(R, B) - G, min=0.0)
        w_yellow  = torch.clamp(torch.minimum(R, G) - B, min=0.0)
        w_red     = torch.clamp(R - torch.maximum(G, B), min=0.0)
        w_green   = torch.clamp(G - torch.maximum(R, B), min=0.0)
        w_blue    = torch.clamp(B - torch.maximum(R, G), min=0.0)

        # Composite the final latent linear exposure
        latent_lin = (
            w_white * W +
            w_cyan * C +
            w_magenta * M +
            w_yellow * Y +
            w_red * R_w +
            w_green * G_w +
            w_blue * B_w
        )
    else:
        # Fallback to simple RGB dot product for legacy stocks missing SPD
        weights_tensor = torch.tensor(weights, device=device, dtype=torch.float32)
        latent_lin = (
            rgb_lin[..., 0] * weights_tensor[0] +
            rgb_lin[..., 1] * weights_tensor[1] +
            rgb_lin[..., 2] * weights_tensor[2]
        )

    # Rescale edge cases
    max_val = torch.max(latent_lin)
    if max_val > 1.0:
        latent_lin = latent_lin / max_val
        
    # Apply Characteristic Curve (Linear -> Display sRGB Negative)
    precision = char_lut.shape[0]
    
    # Scale 0.0-1.0 latent linear values to LUT indices
    indices = (latent_lin * (precision - 1)).long()
    indices = torch.clamp(indices, 0, precision - 1)
    
    # Move LUT to GPU and normalize to 0.0-1.0 float
    char_lut_tensor = torch.from_numpy(char_lut).float().to(device) / 255.0
    
    # Fast GPU indexing
    final_pixel = char_lut_tensor[indices]

    # Reconstruct Output (Broadcast 1D to 3 channels)
    out_rgb = final_pixel.unsqueeze(-1).expand(*final_pixel.shape, 3)
    out_rgb = linear_to_srgb_torch(out_rgb)

    if has_alpha:
        stacked_output = torch.cat([out_rgb, alpha], dim=-1)
    else:
        stacked_output = out_rgb

    # Invert the tensor to create the negative, preserving alpha
    stacked_inverted = stacked_output.clone()
    stacked_inverted[..., :3] = 1.0 - stacked_inverted[..., :3]

    return (stacked_output.cpu(), stacked_inverted.cpu())


# Applies the spectral power distribution mapping to generate a linear latent image,
# bypassing the characteristic curve application and returning only the single processed image.

def get_camera_image(image, camera):
    device = model_management.get_torch_device()
    img_tensor = image.to(device)

    # Handle Alpha channel mapping
    has_alpha = img_tensor.shape[-1] == 4
    if has_alpha:
        alpha = img_tensor[..., 3:]
        rgb = img_tensor[..., :3]
    else:
        alpha = None
        rgb = img_tensor

    rgb = torch.clamp(rgb, 0.0, 1.0)
    rgb_lin = srgb_to_linear_torch(rgb)

    if camera.film_stock.spectral_points is not None:
        # ---------------------------------------------------------
        # PRECOMPUTE SPD INTEGRATION ON CPU
        # ---------------------------------------------------------
        wl_min = 380
        wl_max = 720
        wl_step = 2
        shape = colour.SpectralShape(wl_min, wl_max, wl_step)
        
        # Interpolate raw empirical points into a 171-bin continuous spectrum
        pts = np.array(camera.film_stock.spectral_points)
        wl_data = pts[:, 0]
        sens_data = pts[:, 1]
        
        wl_axis = np.arange(wl_min, wl_max + wl_step, wl_step)
        dense_spd = pchip_interpolate_numpy(wl_data, sens_data, wl_axis)

        sd_film = colour.SpectralDistribution(dense_spd, shape)
        sd_illuminant = colour.SDS_ILLUMINANTS[camera.illuminant_key].copy().align(shape)
        
        bases_lin = np.array([
            [1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        responses = []
        for b in bases_lin:
            sd_basis = RGB_to_sd_Smits1999(b).align(shape)
            # Physical Exposure = Basis * Illuminant * Film Sensitivity
            response = np.sum(sd_basis.values * sd_film.values * sd_illuminant.values)
            responses.append(response)

        responses = np.array(responses, dtype=np.float32)
        
        # Normalize to the White Basis so pure white strictly = 1.0 latent exposure
        responses = responses / responses[0]

        weights_tensor = torch.from_numpy(responses).float().to(device)

        W   = weights_tensor[0]
        C   = weights_tensor[1]
        M   = weights_tensor[2]
        Y   = weights_tensor[3]
        R_w = weights_tensor[4]
        G_w = weights_tensor[5]
        B_w = weights_tensor[6]

        # Extract Smits Basis Weights for every pixel
        R = rgb_lin[..., 0]
        G = rgb_lin[..., 1]
        B = rgb_lin[..., 2]

        w_white   = torch.amin(rgb_lin, dim=-1)
        w_cyan    = torch.clamp(torch.minimum(G, B) - R, min=0.0)
        w_magenta = torch.clamp(torch.minimum(R, B) - G, min=0.0)
        w_yellow  = torch.clamp(torch.minimum(R, G) - B, min=0.0)
        w_red     = torch.clamp(R - torch.maximum(G, B), min=0.0)
        w_green   = torch.clamp(G - torch.maximum(R, B), min=0.0)
        w_blue    = torch.clamp(B - torch.maximum(R, G), min=0.0)

        # Composite the final latent linear exposure
        latent_lin = (
            w_white * W +
            w_cyan * C +
            w_magenta * M +
            w_yellow * Y +
            w_red * R_w +
            w_green * G_w +
            w_blue * B_w
        )
    else:
        # Fallback to simple RGB dot product for legacy stocks missing SPD
        weights_tensor = torch.tensor(camera.film_stock.weights, device=device, dtype=torch.float32)
        latent_lin = (
            rgb_lin[..., 0] * weights_tensor[0] +
            rgb_lin[..., 1] * weights_tensor[1] +
            rgb_lin[..., 2] * weights_tensor[2]
        )

    # Rescale edge cases
    max_val = torch.max(latent_lin)
    if max_val > 1.0:
        latent_lin = latent_lin / max_val
            
    # Reconstruct Output (Broadcast 1D back to 3 channels)
    out_rgb = latent_lin.unsqueeze(-1).expand(*latent_lin.shape, 3)
    out_rgb = linear_to_srgb_torch(out_rgb)

    if has_alpha:
        stacked_output = torch.cat([out_rgb, alpha], dim=-1)
    else:
        stacked_output = out_rgb

    return stacked_output.cpu()
