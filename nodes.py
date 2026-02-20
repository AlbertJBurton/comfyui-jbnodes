import json
import os
import numpy as np
from .spectral import get_spectral_image
from .util import get_srgb_lut, get_generalized_sigmoid_lut

current_dir = os.path.dirname(os.path.realpath(__file__))
json_path = os.path.join(current_dir, "film_stocks.json")

with open(json_path, 'r') as f:
    STOCK_DATA = json.load(f)

# Create mappings
STOCK_MAP = {}
STOCK_NAMES = []
for group in STOCK_DATA["film_stock_groups"]:
    for stock in group["stocks"]:
        STOCK_MAP[stock["name"]] = stock
        STOCK_NAMES.append(stock["name"])

DEVELOPERS = [
    "Standard (D-76)", 
    "Contrast (HC-110)", 
    "Acutance (Rodinal)", 
    "Fine Grain (Xtol)"
]

class SpectralLab:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "image": ("IMAGE",),
                "film_stock": (STOCK_NAMES, {"default": "Kodak / Tri-X 400"}),
                "developer": (DEVELOPERS, {"default": "Standard (D-76)"}),
                "precision": ("INT", {"default": 1024, "min": 256, "max": 65536, "step": 256}),
            },
            "optional": {
                "contrast_index_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "push_pull_stops": ("FLOAT", {"default": 0.0, "min": -3.0, "max": 3.0, "step": 0.1}),
            }        
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image",)
    FUNCTION = "build_spectral_image"
    CATEGORY = "JBNodes"
    DESCRIPTION = """Simulate black and white film stocks with customizable development processes."""

    def build_spectral_image(self, image, film_stock, developer, push_pull_stops, contrast_index_offset, precision):

        # Retrieve Stock Data
        stock_data = STOCK_MAP.get(film_stock)
        # Fallback defaults if json is malformed
        weights = stock_data.get("weights", [0.33, 0.33, 0.33])
        params = stock_data.get("params", {"slope": 1.8, "toe": 0.2, "shoulder": 0.8})

        slope = params["slope"]
        toe = params["toe"]
        shoulder = params["shoulder"]

        # Apply Target CI offset
        # CI (Contrast Index) is roughly 1.0 / slope in the sigmoid model.
        # Standard Tri-X (CI 0.56) has slope ~1.8. 
        # A higher CI (0.70) -> Lower Slope (1.4).
        if contrast_index_offset != 0.0:
            ci = 1 / slope
            slope = 1 / (ci + contrast_index_offset)
            
        # Apply Developer Modifiers
        if "Contrast" in developer:
            slope += 0.2
            toe = max(0.0, toe - 0.05)
        elif "Acutance" in developer:
            slope = max(0.1, slope - 0.05)
            shoulder = max(0.0, shoulder - 0.10)
        elif "Fine Grain" in developer:
            slope = max(0.1, slope - 0.10)
            toe += 0.05

        # Apply Push/Pull (Simulates development time)
        # Pushing increases contrast (slope)
        slope += (push_pull_stops * 0.1)

        # Generate LUTs
        lin_lut = get_srgb_lut() # Fixed 256 size for input linearization
        enc_lut = get_generalized_sigmoid_lut(slope, toe, shoulder, precision)

        return get_spectral_image(image, weights, lin_lut, enc_lut)


class SpectralLabCustom:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "image": ("IMAGE",)            
            },
            "optional": {
                "red": ("FLOAT", {"default": 0.2126, "min": 0.0, "max": 1.0, "step": 0.001}),
                "green": ("FLOAT", {"default": 0.7152, "min": 0.0, "max": 1.0, "step": 0.001}),
                "blue": ("FLOAT", {"default": 0.0722, "min": 0.0, "max": 1.0, "step": 0.001}),
                "slope": ("FLOAT", {"default": 2.2, "min": 0.1, "max": 5.0, "step": 0.05}),
                "toe": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "shoulder": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "precision": ("INT", {"default": 1024, "min": 256, "max": 65536, "step": 256}),
            }, 
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image",)
    FUNCTION = "build_custom_spectral_image"
    CATEGORY = "JBNodes"
    DESCRIPTION = """Create a custom spectral response with manual characteristic curve control."""

    def build_custom_spectral_image(self, image, red, green, blue, slope, toe, shoulder, precision):
        
        weights = [red, green, blue]
        
        lin_lut = get_srgb_lut()
        enc_lut = get_generalized_sigmoid_lut(slope, toe, shoulder, precision)
        
        return get_spectral_image(image, weights, lin_lut, enc_lut)