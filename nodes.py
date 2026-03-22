import json
import os
import numpy as np
import torch
import dataclasses

from comfy import model_management

from PIL import ImageEnhance, Image

from .filters import get_filter_image
from .spectral import get_spectral_image
from .print import get_print_image
from .grayscale import get_grayscale_image
from .util import get_srgb_lut, get_generalized_sigmoid_lut, get_luminosity_lut, FilmStock

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
FILM_STOCK_JSON_PATH = os.path.join(CURRENT_DIR, "film_stocks.json")
FILTER_JSON_PATH = os.path.join(CURRENT_DIR, "wratten_filters.json")
ILLUMINANT_JSON_PATH = os.path.join(CURRENT_DIR, "illuminants.json")
CONTRAST_FILTER_JSON_PATH = os.path.join(CURRENT_DIR, "contrast_filters.json")
PAPER_JSON_PATH = os.path.join(CURRENT_DIR, "papers.json")
GRAYSCALE_JSON_PATH = os.path.join(CURRENT_DIR, "grayscale.json")

with open(FILM_STOCK_JSON_PATH, 'r') as film:
    STOCK_DATA = json.load(film)

with open(FILTER_JSON_PATH, 'r') as filter:
    FILTER_DATA = json.load(filter)

with open(ILLUMINANT_JSON_PATH, 'r') as source:
    SOURCE_DATA = json.load(source)

with open(CONTRAST_FILTER_JSON_PATH, 'r') as contrast:
    CONTRAST_FILTER_DATA = json.load(contrast)

with open(PAPER_JSON_PATH, 'r') as paper:
    PAPER_DATA = json.load(paper)

with open(GRAYSCALE_JSON_PATH, 'r') as grayscale:
    GRAYSCALE_DATA = json.load(grayscale)

# Create mappings
STOCK_MAP = {}
STOCK_NAMES = []
for group in STOCK_DATA["film_stock_groups"]:
    for stock in group["stocks"]:
        STOCK_MAP[stock["name"]] = stock
        STOCK_NAMES.append(stock["name"])

GRAYSCALE_MAP = {}
GRAYSCALE_NAMES = []
for grayscale in GRAYSCALE_DATA["grayscale"]:
    GRAYSCALE_MAP[grayscale["name"]] = grayscale
    GRAYSCALE_NAMES.append(grayscale["name"])

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

CONTRAST_MAP = {}
CONTRAST_NAMES = []
for filter in CONTRAST_FILTER_DATA["filters"]:
    CONTRAST_MAP[filter["label"]] = filter
    CONTRAST_NAMES.append(filter["label"])

GRADED_PAPER_MAP = {}
GRADED_PAPER_NAMES = []
for paper in PAPER_DATA["graded_papers"]:
    GRADED_PAPER_MAP[paper["name"]] = paper
    GRADED_PAPER_NAMES.append(paper["name"])

DEVELOPERS = [
    "None", 
    "Standard (D-76)", 
    "Contrast (HC-110)", 
    "Acutance (Rodinal)", 
    "Fine Grain (Xtol)"
]

class FilmLab:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "film_stock": (STOCK_NAMES, {"default": "Kodak / Tri-X 400"}),
            }, 
        }

    RETURN_TYPES = ("FILM_STOCK",)
    RETURN_NAMES = ("film_stock",)
    FUNCTION = "get_film_stock"
    CATEGORY = "JBNodes"
    DESCRIPTION = """Simulated black and white film stocks."""

    def get_film_stock(self, film_stock):
        stock_data = STOCK_MAP.get(film_stock)
        return (FilmStock.from_dict(stock_data),)

class PrintLabMultigrade:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "film_negative": ("IMAGE",),
            },
            "optional": {
                "contrast_filter": (CONTRAST_NAMES, {"default": "2"}),
                "exposure_secs": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "print_image"
    CATEGORY = "JBNodes"
    DESCRIPTION = """Simulate printing a negative to photographic paper."""

    def print_image(self, film_negative, contrast_filter, exposure_secs):
        filter_data = CONTRAST_MAP.get(contrast_filter)
        contrast_factor = filter_data.get("factor")
        return get_print_image(film_negative, contrast_factor=contrast_factor, exposure_secs=exposure_secs)

class PrintLabGraded:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "film_negative": ("IMAGE",),
            },
            "optional": {
                "graded_paper": (GRADED_PAPER_NAMES, {"default": "Ilford / Ilfobrom Galerie FB / Grade 2"}),
                "exposure_secs": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "precision": ("INT", {"default": 4096, "min": 256, "max": 65536, "step": 256}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "print_image"
    CATEGORY = "JBNodes"
    DESCRIPTION = """Simulate printing a negative to graded photographic paper."""

    def print_image(self, film_negative, graded_paper, exposure_secs, precision=4096):
        paper_data = GRADED_PAPER_MAP.get(graded_paper)
        grade = paper_data.get("grade")
        
        # Read the raw empirical points
        hd_curve_raw = paper_data.get("hd_curve_points", None)
        d_max = paper_data.get("d_max", 2.1)
        d_min = paper_data.get("d_min", 0.04)
        
        hd_curve_points = None
        contrast_factor = None
        
        if hd_curve_raw:
            # Cast the JSON array of coordinates into a PyTorch tensor
            device = model_management.get_torch_device()
            hd_curve_points = torch.tensor(hd_curve_raw, dtype=torch.float32, device=device)
        else:
            # Fallback to the algorithmic method if no empirical data exists
            filter_data = CONTRAST_MAP.get(grade)
            contrast_factor = filter_data.get("factor")

        return get_print_image(film_negative, contrast_factor=contrast_factor, exposure_secs=exposure_secs, hd_curve_points=hd_curve_points, d_max=d_max, d_min=d_min, precision=precision)

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

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_filter"
    CATEGORY = "JBNodes"
    DESCRIPTION = """Apply a Wratten filter to an image."""

    def apply_filter(self, image, filter, filter_factor, auto_filter_factor, light_source):
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
                "film_stock": ("FILM_STOCK",), 
                "developer": (DEVELOPERS, {"default": "Standard (D-76)"}),
            },
            "optional": {
                "precision": ("INT", {"default": 1024, "min": 256, "max": 65536, "step": 256}),
                "contrast_index_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "push_pull_stops": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.1}),
            }        
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("preview", "film_negative")
    FUNCTION = "build_spectral_image"
    CATEGORY = "JBNodes"
    DESCRIPTION = """Simulate black and white film stocks with customizable development processes."""

    def build_spectral_image(self, image, film_stock, developer, push_pull_stops, contrast_index_offset, precision):
        stock_data = dataclasses.asdict(film_stock) if isinstance(film_stock, FilmStock) else STOCK_MAP.get(film_stock)

        weights = stock_data.get("weights", [0.33, 0.33, 0.33])
        params = stock_data.get("params", {"slope": 1.8, "toe": 0.2, "shoulder": 0.8})
        lum_mask = stock_data.get("luminosity_mask", [2.8, 1.1, 10.18, 0.0])

        slope = params["slope"]
        toe = params["toe"]
        shoulder = params["shoulder"]

        if contrast_index_offset != 0.0:
            ci = 1 / slope
            slope = 1 / (ci + contrast_index_offset)
            
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

        slope += (push_pull_stops * 0.1)

        lin_lut = get_srgb_lut()
        char_lut = get_generalized_sigmoid_lut(slope, toe, shoulder, precision)

        return get_spectral_image(image, weights, lin_lut, char_lut)

class GrayscaleLab:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "image": ("IMAGE",),            
                "method": (GRAYSCALE_NAMES, {"default": "ITU-R BT.709 Luminance"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "build_grayscale_image"
    CATEGORY = "JBNodes"
    DESCRIPTION = """Create a grayscale image with a specific sprectral curve."""

    def build_grayscale_image(self, image, method):
        grayscale_data = GRAYSCALE_MAP.get(method)
        weights = grayscale_data.get("weights", [0.33, 0.33, 0.33])       
        return get_grayscale_image(image, weights)
    