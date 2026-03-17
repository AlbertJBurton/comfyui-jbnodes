import json
import os
import numpy as np
from .filters import get_filter_image
from .spectral import get_spectral_image
from .util import get_srgb_lut, get_generalized_sigmoid_lut, get_luminosity_lut

current_dir = os.path.dirname(os.path.realpath(__file__))
stock_path = os.path.join(current_dir, "film_stocks.json")
filter_path = os.path.join(current_dir, "filters.json")
source_path = os.path.join(current_dir, "illuminants.json")

with open(stock_path, 'r') as film:
    STOCK_DATA = json.load(film)

with open(filter_path, 'r') as filter:
    FILTER_DATA = json.load(filter)

with open(source_path, 'r') as source:
    SOURCE_DATA = json.load(source)

# Create mappings
STOCK_MAP = {}
STOCK_NAMES = []
for group in STOCK_DATA["film_stock_groups"]:
    for stock in group["stocks"]:
        STOCK_MAP[stock["name"]] = stock
        STOCK_NAMES.append(stock["name"])

FILTER_MAP = {}
FILTER_NAMES = []
for filter in FILTER_DATA["filters"]:
    FILTER_MAP[filter["name"]] = filter
    FILTER_NAMES.append(filter["name"])

SOURCE_MAP = {}
SOURCE_NAMES = []
for source in SOURCE_DATA["sources"]:
    SOURCE_MAP[source["label"]] = source
    SOURCE_NAMES.append(source["label"])

DEVELOPERS = [
    "None", 
    "Standard (D-76)", 
    "Contrast (HC-110)", 
    "Acutance (Rodinal)", 
    "Fine Grain (Xtol)"
]

class FilterLab:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "filter": (FILTER_NAMES, {"default": "No. 0 - Colorless"}),
            },
            "optional": {
                "filter_factor": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 10.00, "step": 0.01}),
                "auto_filter_factor": ("BOOLEAN", {"default": False}),
                "light_source": (SOURCE_NAMES, {"default": "Noon Daylight (6500 K)"}),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_filter"
    CATEGORY = "JBNodes"
    DESCRIPTION = """Apply a Wratten filter to an image."""

    def apply_filter(self, image, filter, filter_factor, auto_filter_factor, light_source):
        
        # Retrieve Filter Data
        filter_data = FILTER_MAP.get(filter)
        transmission = filter_data.get("transmission")
        auto_factor = 1.0 / filter_data.get("visual_transmission")
        
        source_data = SOURCE_MAP.get(light_source)
        illuminant_name = source_data["key"]

        return get_filter_image(image, transmission, filter_factor, auto_filter_factor, auto_factor, illuminant_name)


class SpectralLab:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "image": ("IMAGE",),
                "film_stock": (STOCK_NAMES, {"default": "Kodak / Tri-X 400"}),
                "developer": (DEVELOPERS, {"default": "Standard (D-76)"}),
            },
            "optional": {
                "precision": ("INT", {"default": 1024, "min": 256, "max": 65536, "step": 256}),
                "contrast_index_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "push_pull_stops": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.1}),
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
        lum_mask = stock_data.get("luminosity_mask", [2.8, 1.1, 10.18, 0.0])

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
        elif "None" in developer:
            slope = 2.2
            toe = 0

        # Apply Push/Pull (Simulates development time)
        # Pushing increases contrast (slope)
        slope += (push_pull_stops * 0.1)

        # Generate LUTs
        lin_lut = get_srgb_lut() # Fixed 256 size for input linearization
        char_lut = get_generalized_sigmoid_lut(slope, toe, shoulder, precision)
        mask_lut = get_luminosity_lut(lum_mask, precision)

        return get_spectral_image(image, weights, lin_lut, char_lut)


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
        char_lut = get_generalized_sigmoid_lut(slope, toe, shoulder, precision)
        
        return get_spectral_image(image, weights, lin_lut, char_lut)