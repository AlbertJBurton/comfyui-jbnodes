import torch
import numpy as np
from .util import get_srgb_lut

# Processes an image by linearizing RGB values, applying a spectral weight 
# dot product, and re-applying a characteristic curve via a high-precision LUT.
# 
# Args:
#     image_data (np.ndarray): Shape (H, W, 3/4) uint8 array.
#     weights (list/tuple): [rW, gW, bW] floats.
#     linear_lut (np.ndarray): 256-entry float32 sRGB LUT (Input).
#     encoding_lut (np.ndarray): High-precision uint8 LUT (Output).

def process_image(image_data, weights, linear_lut, encoding_lut):

    h, w, c = image_data.shape
    # Dynamically handle RGB or RGBA
    pixels = image_data.reshape(-1, c)
    
    # Extract channels
    r_raw = pixels[:, 0]
    g_raw = pixels[:, 1]
    b_raw = pixels[:, 2]

    # 1. Linearize (Input sRGB -> Linear)
    r_lin = linear_lut[r_raw]
    g_lin = linear_lut[g_raw]
    b_lin = linear_lut[b_raw]

    # 2. Apply Spectral Weights (Dot Product)
    latent_lin = (r_lin * weights[0]) + (g_lin * weights[1]) + (b_lin * weights[2])
    
    # Clip to legal linear range [0, 1]
    latent_lin = np.clip(latent_lin, 0, 1)

    # 3. Apply Characteristic Curve (Linear -> Output sRGB)
    # Map 0.0-1.0 to the indices of the encoding_lut
    precision = len(encoding_lut)
    indices = (latent_lin * (precision - 1)).astype(np.int32)
    final_pixel = encoding_lut[indices]

    # 4. Reconstruct Output
    output = np.empty_like(pixels)
    output[:, 0] = final_pixel  # R
    output[:, 1] = final_pixel  # G
    output[:, 2] = final_pixel  # B
    
    # Preserve Alpha if present
    if c == 4:
        output[:, 3] = pixels[:, 3] 

    return output.reshape(h, w, c)

def get_spectral_image(image, weights, linear_lut, encoding_lut):

    # ComfyUI tensors are (B, H, W, C)
    batch_size = image.shape[0]
    output_tensors = []

    for i in range(batch_size):
        # Convert to NumPy (on CPU)
        img_np = image[i].detach().cpu().numpy()
        
        # Scale float [0,1] -> uint8 [0,255] for LUT processing
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        
        # Process
        result_np = process_image(img_np, weights, linear_lut, encoding_lut)
        
        # Convert back to torch float32 [0, 1]
        result_tensor = torch.from_numpy(result_np).float() / 255.0
        output_tensors.append(result_tensor)

    return (torch.stack(output_tensors),)