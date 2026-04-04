'''
    PyTorch sRGB Library Functions
    ------------------------------
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
import numpy as np

def srgb_to_linear_torch(tensor):
    ''' Vectorized PyTorch sRGB to Linear decoding. '''

    mask = tensor <= 0.04045
    safe_tensor = torch.clamp(tensor, min=0.0)
    
    return torch.where(mask, tensor / 12.92, torch.pow((safe_tensor + 0.055) / 1.055, 2.4))

def linear_to_srgb_torch(tensor):
    ''' Vectorized PyTorch Linear to sRGB encoding. '''
    
    mask = tensor <= 0.0031308
    safe_tensor = torch.clamp(tensor, min=1e-8)
    
    return torch.where(mask, tensor * 12.92, 1.055 * torch.pow(safe_tensor, 1.0 / 2.4) - 0.055)

def get_srgb_lut():
    '''
    Pre-calculate sRGB linearization lookup table for performance
    Maps 0-255 uint8 input to 0.0-1.0 linear float output
    Input is always 8-bit, so this table size is fixed at 256.
    '''

    # Create an array of 256 values from 0 to 255
    i = np.arange(256)
    normalized = i / 255.0
    
    # Apply the sRGB transfer function logic
    # Using np.where acts as a vectorized if/else
    linear_lut = np.where(
        normalized <= 0.04045,
        normalized / 12.92,
        np.power((normalized + 0.055) / 1.055, 2.4)
    )
    
    return linear_lut.astype(np.float32)

