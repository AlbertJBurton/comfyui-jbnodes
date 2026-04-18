'''
    Lookup Table (LUT) Library Functions
    ------------------------------------
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
import logging

from ..models.hdcurve import HDCurve

def apply_1d_lut(tensor, lut):
    ''' Apply a 1D PyTorch tensor LUT to an image tensor using linear interpolation. '''

    N = lut.shape[0]
    x_scaled = tensor * (N - 1)
    x_floor = x_scaled.floor().long()
    x_ceil = torch.clamp(x_floor + 1, max = N - 1)
    weight = x_scaled - x_floor.float()

    return torch.lerp(lut[x_floor], lut[x_ceil], weight)

def get_hd_curve_lut(hd_curve: HDCurve, precision: int = 4096, ei: float = 0.1, dev_offset: int = 0, dynamic_range: float = 10):
    '''
    Generates a Characteristic Curve LUT directly from an empirical HDCurve object.
    Uses the Zone System to map the digital 0.0-1.0 space to a dynamic range 
    defined by N development (9 stops default). Zone 0 (black point) is mathematically 
    anchored at the EI density units above base+fog, allowing accurate contrast 
    index shifts across developments.
    '''
    
    if not hd_curve or not hd_curve.curve_points:
        return np.linspace(0, 255, precision, dtype=np.uint8)
        
    points = np.array(hd_curve.curve_points)
    
    # Sort points by x-axis (Log Exposure) to ensure deterministic evaluation
    sorted_indices = np.argsort(points[:, 0])
    xp = points[sorted_indices, 0]
    yp = points[sorted_indices, 1]
    
    yp_min, yp_max = yp.min(), yp.max()
    density_range = yp_max - yp_min
    
    if density_range <= 0.0:
        return np.linspace(0, 255, precision, dtype=np.float32)
    
    # Standard densitometry: Exposure UP -> Density UP.
    # Normalize Density (Y)
    yp_norm = (yp - yp_min) / density_range

    # Define Zone 0 (Base+Fog is yp_min)
    zone_0_target = yp_min + ei
    idx = np.where(yp >= zone_0_target)[0]
        
    # Safely find the LogE (xp) corresponding to Zone 0 density
    xp_start = xp.min() # fallback
    if len(idx) > 0:
        first_idx = idx[0]
        if first_idx > 0:
            x0, x1 = xp[first_idx - 1], xp[first_idx]
            y0, y1 = yp[first_idx - 1], yp[first_idx]
            # Linearly interpolate the exact LogE coordinate for a flawless Zone 0
            if y1 != y0:
                xp_start = x0 + (zone_0_target - y0) * (x1 - x0) / (y1 - y0)
            else:
                xp_start = x1
        else:
            xp_start = xp[first_idx]

    # We assume here that the Zone System defines N development as 7 stops of dynamic 
    # range above Zone 0 and adjust based on the dev_offset (e.g. -2 for N-2, +2 for N+2).
    # 1 stop = log10(2) ≈ 0.30103 LogE units.
    log_e_steps = dynamic_range * np.log10(2.0)
    xp_end = xp_start + log_e_steps

    logging.info(f"esteps: {log_e_steps}, dr: {dynamic_range}")
    
    # CRITICAL FIX: Convert linear pixel values to Log Exposure.
    # The H&D curve x-axis is Log10(Exposure). Our input x_eval is Linear Exposure.
    # We map the maximum linear value (1.0) to xp_end.
    # We use a tiny epsilon to avoid log10(0) for pure black pixels.
    x_eval = np.linspace(0.0, 1.0, precision)
    x_safe = np.clip(x_eval, 1e-10, 1.0)
    xp_eval = np.log10(x_safe) + xp_end

    # Interpolate the empirical density at these exact LogE points.
    # np.interp gracefully handles any xp_eval values extending past the 
    # original densitometer curve by clamping them to the nearest valid density.
    y_eval = np.interp(xp_eval, xp, yp)

    # Apply True N-Development (Contrast Scaling)
    # N+1 increases contrast (slope). N-1 decreases contrast.
    # We scale the density pivoting around Base+Fog (yp_min).
    # A typical N+1 push increases the Contrast Index by roughly 15-20%.
    if dev_offset != 0:
        contrast_factor = 1.0 + (dev_offset * 0.15)
        y_eval = yp_min + ((y_eval - yp_min) * contrast_factor)
    
    # Convert Optical Density to Transmittance (T = 10^-D)
    t_eval = 10.0 ** (-y_eval)
    
    # Scale Transmittance for the LUT output 
    lut = np.clip(t_eval * 255.0, 0, 255).astype(np.float32)    
    
    return lut.astype(np.uint8)

def get_generalized_sigmoid_lut(slope, toe, shoulder, precision):
    '''
    Generates a Characteristic Curve LUT using a shaped sigmoid.
    Args:
      slope: Controls contrast (Gamma). Higher = steeper.
      toe: Controls shadow compression length. Higher = longer toe (crushed blacks).
      shoulder: Controls highlight rolloff. Lower = earlier rolloff.
      precision: The resolution of the output curve (default 1024).
    '''

    x = np.linspace(0.0, 1.0, precision)
    
    # Slope (Gamma/Contrast) applied to midtones
    # Inverted because this is an Encoding LUT (Linear -> Display)
    # A standard gamma is x^(1/2.2). Here 1/slope acts as that exponent.
    y = np.power(x, 1.0 / slope) 

    # Apply Toe Compression (Shadow suppression)
    # If x is in the toe region, we suppress the values to simulate density buildup
    if toe > 0:
        toe_region = np.exp(-10 * (x / toe)) 
        y = y * (1.0 - toe_region * 0.2) 

    # Apply Shoulder Rolloff (Highlight compression)
    # A simple smoothstep-like approximation to roll off highlights
    if shoulder < 1.0:
        shoulder_mask = x > shoulder
        # Compress the range above the shoulder point
        # This is a simplified mathematical model for visual characteristic simulation
        y[shoulder_mask] = y[shoulder_mask] * (1.0 - (x[shoulder_mask] - shoulder) * 0.5)

    # Map the 0.0-1.0 normalized curve to a generic density range (e.g., D_min=0.2, D_max=2.4)
    density = 0.2 + y * (2.4 - 0.2)
    
    # Convert Optical Density to Transmittance (T = 10^-D)
    t_eval = 10.0 ** (-density)

    # Scale back to 8-bit space
    lut = np.clip(t_eval * 255.0, 0, 255).astype(np.float32) 
    
    return lut.astype(np.uint8)
