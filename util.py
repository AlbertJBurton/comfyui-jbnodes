import numpy as np

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
# slope: Controls contrast (Gamma). Higher = steeper.
# toe: Controls shadow compression length. Higher = longer toe (crushed blacks).
# shoulder: Controls highlight rolloff. Lower = earlier rolloff.
# precision: The resolution of the output curve (default 1024).

def get_generalized_sigmoid_lut(slope, toe, shoulder, precision=1024):

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

def sigmoid(x):
	output=np.empty_like(x)
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			output[i,j] = 1 / (1 + np.exp(-x[i,j]))
	return output