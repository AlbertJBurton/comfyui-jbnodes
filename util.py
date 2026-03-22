import torch
import numpy as np

from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class FilmStock:
    id: str
    name: str
    description: str
    # Provide safe defaults based on the fallbacks in your nodes.py
    weights: List[float] = field(default_factory=lambda: [0.33, 0.33, 0.33])
    luminosity_mask: List[float] = field(default_factory=lambda: [2.8, 1.1, 10.18, 0.0])
    params: Dict[str, float] = field(default_factory=lambda: {"slope": 1.8, "toe": 0.2, "shoulder": 0.8})
    # spd is optional since most stocks in your JSON don't have it defined
    spd: Optional[List[float]] = None 

    @classmethod
    def from_dict(cls, data: dict):
        """Creates a FilmStock object from a parsed JSON dictionary."""
        return cls(
            id=data.get("id", "unknown"),
            name=data.get("name", "Unknown Stock"),
            description=data.get("description", ""),
            weights=data.get("weights", [0.33, 0.33, 0.33]),
            luminosity_mask=data.get("luminosity_mask", [2.8, 1.1, 10.18, 0.0]),
            params=data.get("params", {"slope": 1.8, "toe": 0.2, "shoulder": 0.8}),
            spd=data.get("spd")
        )

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
	output=np.empty_like(x)
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			output[i,j] = 1 / (1 + np.exp(-x[i,j]))
	return output

if __name__ == "__main__":
    mask = [1.5, 2.0, 1.2, 0.1]
    precision=1024
    # Generating a standard 256-value 8-bit LUT
    my_lut = get_luminosity_lut(mask, precision)
    print(f"LUT shape: {my_lut.shape}, dtype: {my_lut.dtype}")
    my_lut = get_generalized_sigmoid_lut(2.2, 0.0, 1.0, precision)
    print(f"LUT shape: {my_lut.shape}, dtype: {my_lut.dtype}")