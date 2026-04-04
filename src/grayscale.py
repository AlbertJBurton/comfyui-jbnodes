'''
    Grayscale Conversion Library Functions
    --------------------------------------
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
'''

import torch

from comfy import model_management

from .srgb import srgb_to_linear_torch, linear_to_srgb_torch

def get_grayscale_image(image, weights):
    '''
    Processes a batch of ComfyUI image tensors directly on the GPU.
    Args:
        image (torch.Tensor): Shape (B, H, W, 3/4) float tensor.
        weights (list/tuple): [rW, gW, bW] floats.
    '''

    # Retrieve the active ComfyUI computing device
    device = model_management.get_torch_device()
    
    # Move tensor to the target device
    img_tensor = image.to(device)

    # Separate Alpha channel if present
    has_alpha = img_tensor.shape[-1] == 4
    if has_alpha:
        rgb = img_tensor[..., :3]
        alpha = img_tensor[..., 3:]
    else:
        rgb = img_tensor
        alpha = None

    # Linearize input directly on GPU
    rgb_lin = srgb_to_linear_torch(rgb)
    
    # Extract channels
    r_lin = rgb_lin[..., 0]
    g_lin = rgb_lin[..., 1]
    b_lin = rgb_lin[..., 2]

    # Apply Spectral Weights (Batched Dot Product)
    latent_lin = (r_lin * weights[0]) + (g_lin * weights[1]) + (b_lin * weights[2])
    
    # Weights should combine to be [0..1], but we rescale here for edge cases
    max_val = torch.max(latent_lin)
    if max_val > 1.0:
        latent_lin = latent_lin / max_val
        
    # Convert Linear grayscale back to sRGB display curve
    latent_srgb = linear_to_srgb_torch(latent_lin)
    latent_srgb = torch.clamp(latent_srgb, 0.0, 1.0)
    
    # Reconstruct Output (Broadcast 1D grayscale to 3D RGB channels)
    out_rgb = latent_srgb.unsqueeze(-1).expand(*latent_srgb.shape, 3)
    
    # Reattach Alpha if necessary
    if has_alpha:
        out_img = torch.cat([out_rgb, alpha], dim=-1)
    else:
        out_img = out_rgb

    # Return tensor to CPU memory for ComfyUI inter-node routing
    return (out_img.cpu(),)