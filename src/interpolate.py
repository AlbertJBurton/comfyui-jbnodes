'''
    Interpolation Library Functions
    -------------------------------
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

import numpy as np
import torch

def pchip_interpolate_torch(x, y, x_new):
    '''
    Batched Monotone Cubic Spline (Fritsch-Carlson) natively in PyTorch.
    Ensures that interpolated points strictly follow the trajectory of the 
    data without overshooting (which preserves the true toe/shoulder roll-off).
    '''

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
    '''
    Monotone Cubic Spline (Fritsch-Carlson) in NumPy.
    Runs on the CPU to generate a high-precision, non-ringing 
    spectral power distribution from empirical coordinates.
    '''

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
