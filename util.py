import json
import os
import numpy as np
import torch
import dataclasses
import logging

from dataclasses import dataclass, field
from typing import List, Dict, Optional

from .classes import FilmFormat, HDCurve, FilmStock

def srgb_to_linear_torch(tensor):
    """Vectorized PyTorch sRGB to Linear decoding."""
    mask = tensor <= 0.04045
    safe_tensor = torch.clamp(tensor, min=0.0)
    return torch.where(mask, tensor / 12.92, torch.pow((safe_tensor + 0.055) / 1.055, 2.4))

def linear_to_srgb_torch(tensor):
    """Vectorized PyTorch Linear to sRGB encoding."""
    mask = tensor <= 0.0031308
    safe_tensor = torch.clamp(tensor, min=1e-8)
    return torch.where(mask, tensor * 12.92, 1.055 * torch.pow(safe_tensor, 1.0 / 2.4) - 0.055)

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
    # Explicitly call .contiguous() on the boundary tensor to prevent 
    # PyTorch memory reallocation warnings and performance hits.
    idx = torch.searchsorted(x.contiguous(), x_new) - 1
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

def pchip_interpolate_numpy(x, y, x_new):
    """
    Monotone Cubic Spline (Fritsch-Carlson) in NumPy.
    Runs on the CPU to generate a high-precision, non-ringing 
    spectral power distribution from empirical coordinates.
    """
    dx = np.diff(x)
    dy = np.diff(y)
    S = dy / dx

    m = np.zeros_like(x)
    mask = S[:-1] * S[1:] > 0
    w1 = 2 * dx[1:] + dx[:-1]
    w2 = dx[1:] + 2 * dx[:-1]

    m[1:-1][mask] = (w1[mask] + w2[mask]) / (w1[mask] / S[:-1][mask] + w2[mask] / S[1:][mask])
    m[0] = S[0]
    m[-1] = S[-1]

    idx = np.searchsorted(x, x_new) - 1
    idx = np.clip(idx, 0, len(x) - 2)

    x_k = x[idx]
    y_k = y[idx]
    m_k = m[idx]
    dx_k = dx[idx]

    m_k1 = m[idx + 1]
    y_k1 = y[idx + 1]

    t = (x_new - x_k) / dx_k
    t2 = t * t
    t3 = t2 * t

    h00 = 2*t3 - 3*t2 + 1
    h10 = t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 = t3 - t2

    y_new = h00 * y_k + h10 * dx_k * m_k + h01 * y_k1 + h11 * dx_k * m_k1
    
    # Zero out any values outside the original empirical range 
    # to prevent artificial sensitivity tails in the UV/IR bands.
    y_new[x_new < x[0]] = 0.0
    y_new[x_new > x[-1]] = 0.0
    
    # Physical sensitivity cannot be negative
    return np.maximum(y_new, 0.0)

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

# Pre-calculate sRGB linearization lookup table for performance
# Maps 0-255 uint8 input to 0.0-1.0 linear float output
# Input is always 8-bit, so this table size is fixed at 256.

def get_srgb_lut():

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

# Generates a Characteristic Curve LUT using a shaped sigmoid.
#
# Args:
#   slope: Controls contrast (Gamma). Higher = steeper.
#   toe: Controls shadow compression length. Higher = longer toe (crushed blacks).
#   shoulder: Controls highlight rolloff. Lower = earlier rolloff.
#   precision: The resolution of the output curve (default 1024).

def get_generalized_sigmoid_lut(slope, toe, shoulder, precision):

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

    # Normalize to 0-255
    lut = np.clip(y * 255.0, 0, 255).astype(np.float32) 
    
    return lut.astype(np.uint8)


# Generates a Characteristic Curve LUT directly from an empirical HDCurve object.
#   
# Uses the Zone System to map the digital 0.0-1.0 space to a dynamic range 
# defined by N development (7 stops default). Zone 0 (black point) is mathematically 
# anchored at the EI density units above base+fog, allowing accurate contrast 
# index shifts across developments.

def get_hd_curve_lut(hd_curve: HDCurve, precision: int = 4096, ei: float = 0.1, dev_offset: int = 0):
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
        return np.linspace(0, 255, precision, dtype=np.uint8)
    
    # Check empirical curve direction. 
    # Standard densitometry: Exposure UP -> Density UP.
    # Inverted/Transmission: Exposure UP -> Value DOWN.
    is_ascending = yp[0] < yp[-1]
    
    if is_ascending:
        # 1. Normalize Density (Y)
        yp_norm = (yp - yp_min) / density_range
        # 2. Define Zone 0 (Base+Fog is yp_min)
        zone_0_target = yp_min + ei
        idx = np.where(yp >= zone_0_target)[0]
    else:
        # 1. Normalize and FLIP Density (Y) so the LUT is always a positive 0.0-1.0 map
        yp_norm = (yp_max - yp) / density_range
        # 2. Define Zone 0 (Base+Fog is yp_max in inverted data)
        zone_0_target = yp_max - ei
        idx = np.where(yp <= zone_0_target)[0]
        
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

    # The Zone System defines N development as 7 stops of dynamic range above Zone 0.
    # We adjust this based on the dev_offset (e.g. -2 for N-2, +2 for N+2).
    # 1 stop = log10(2) ≈ 0.30103 LogE units.
    dynamic_range_stops = 7.0 + dev_offset
    log_e_stops = dynamic_range_stops * np.log10(2.0)
    
    xp_end = xp_start + log_e_stops
    
    # 3. Generate LUT over the fixed window
    x_eval = np.linspace(0.0, 1.0, precision)
    
    # Map digital 0.0-1.0 exactly to the [xp_start, xp_end] LogE range
    xp_eval = xp_start + (x_eval * (xp_end - xp_start))
    
    # Interpolate the normalized empirical density at these exact LogE points.
    # np.interp gracefully handles any xp_eval values extending past the 
    # original densitometer curve by clamping them to the nearest valid density.
    y_eval = np.interp(xp_eval, xp, yp_norm)
    
    # Scale back to 8-bit space
    lut = np.clip(y_eval * 255.0, 0, 255).astype(np.float32)
    
    return lut.astype(np.uint8)

#
# Generates a luminosity Look-Up Table (LUT).
#
# Args:
#    lum_mask: Array-like containing [a, b, c, f] parameters.
#    precision: The size of the LUT array (default 256 for 8-bit).

def get_luminosity_lut(lum_mask, precision):
    
    x = np.linspace(0.0, 1.0, precision)

    a = lum_mask[0]
    b = lum_mask[1]
    c = lum_mask[2]
    f = lum_mask[3]

    y = (1.0 - f) * ( c * x**a * (1.0 - x)**b ) + f
    
    lut = (y).astype(np.uint8) 
    
    return lut

def sigmoid(x):
    """
    Vectorized sigmoid function. 
    Relies on numpy's native C-level broadcasting for optimal speed.
    """
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    mask = [1.5, 2.0, 1.2, 0.1]
    precision=1024
    # Generating a standard 256-value 8-bit LUT
    my_lut = get_luminosity_lut(mask, precision)
    print(f"LUT shape: {my_lut.shape}, dtype: {my_lut.dtype}")
    my_lut = get_generalized_sigmoid_lut(2.2, 0.0, 1.0, precision)
    print(f"LUT shape: {my_lut.shape}, dtype: {my_lut.dtype}")