import torch
import numpy as np
import colour

from .util import get_srgb_lut

from colour.recovery import RGB_to_sd_Smits1999
from colour.models import XYZ_to_sRGB

from comfy import model_management

# Processes an image by linearizing RGB values, applying a spectral weight 
# dot product, and applying a characteristic curve via a high-precision LUT.
# 
# Args:
#     image_data (np.ndarray): Shape (H, W, 3/4) uint8 array.
#     weights (list/tuple): [rW, gW, bW] floats.
#     linear_lut (np.ndarray): 256-entry float32 sRGB LUT (Input).
#     characteristic_lut (np.ndarray): High-precision uint8 LUT (Output).

def process_image(image_data, weights, linear_lut, characteristic_lut):

    h, w, c = image_data.shape
    # Dynamically handle RGB or RGBA
    pixels = image_data.reshape(-1, c)
    
    # Extract channels and linearize (Input sRGB -> Linear)
    r_lin = linear_lut[pixels[:, 0]]
    g_lin = linear_lut[pixels[:, 1]]
    b_lin = linear_lut[pixels[:, 2]]

    # Apply Spectral Weights (Dot Product)
    latent_lin = (r_lin * weights[0]) + (g_lin * weights[1]) + (b_lin * weights[2])
    
    # Weights should combine to be [0..1], but we rescale here for edge cases.
    max_val = np.max(latent_lin)
    if (max_val > 1.0):
        latent_lin = latent_lin / max_val if max_val != 0 else latent_lin.astype(np.float32)

    # Apply Characteristic Curve (Linear -> Output sRGB)
    # Map 0.0-1.0 to the indices of the characteristic_lut
    precision = len(characteristic_lut)
    indices = (latent_lin * (precision - 1)).astype(np.int32)
    final_pixel = characteristic_lut[indices]

    # Reconstruct Output
    output = np.empty_like(pixels)
    output[:, 0] = final_pixel  # R
    output[:, 1] = final_pixel  # G
    output[:, 2] = final_pixel  # B
    
    # Preserve Alpha if present
    if c == 4:
        output[:, 3] = pixels[:, 3] 

    return output.reshape(h, w, c)

# Processes a batch of ComfyUI image tensors.
# 
# Args:
#     image (np.ndarray): Shape (H, W, 3/4) uint8 array.
#     weights (list/tuple): [rW, gW, bW] floats.
#     linear_lut (np.ndarray): 256-entry float32 sRGB LUT (Input).
#     characteristic_lut (np.ndarray): High-precision uint8 LUT (Output).

def get_spectral_image(image, weights, linear_lut, characteristic_lut):

    # ComfyUI tensors are (B, H, W, C)
    batch_size = image.shape[0]
    output_tensors = []

    for i in range(batch_size):
        # Convert to NumPy (on CPU)
        img_np = image[i].detach().cpu().numpy()
        
        # Scale float [0,1] -> uint8 [0,255] for LUT processing
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        
        # Process
        result_np = process_image(img_np, weights, linear_lut, characteristic_lut)
        
        # Convert back to torch float32 [0, 1]
        result_tensor = torch.from_numpy(result_np).float() / 255.0
        output_tensors.append(result_tensor)

    stacked_output = torch.stack(output_tensors)

    # Invert the tensor to create the negative, preserving alpha channel if it exists
    if stacked_output.shape[-1] == 4:
        stacked_inverted = stacked_output.clone()
        stacked_inverted[..., :3] = 1.0 - stacked_inverted[..., :3]
    else:
        stacked_inverted = 1.0 - stacked_output

    return (stacked_output, stacked_inverted)