from .nodes import PrintLabGraded, PrintLabMultigrade, SpectralLab, FilterLab, FilmLab, GrayscaleLab, CameraLab, ShaderLab

NODE_CLASS_MAPPINGS = {
    "SpectralLab": SpectralLab,
    "FilterLab": FilterLab,
    "PrintLabMultigrade": PrintLabMultigrade,
    "PrintLabGraded": PrintLabGraded,
    "FilmLab": FilmLab,
    "GrayscaleLab": GrayscaleLab,
    "CameraLab": CameraLab,
    "ShaderLab": ShaderLab,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpectralLab": "JB Film Development",
    "FilterLab": "JB Wratten Filter",
    "PrintLabMultigrade": "JB Darkroom Enlarger (Multigrade)",
    "PrintLabGraded": "JB Darkroom Enlarger",
    "FilmLab": "JB Film Stock",
    "GrayscaleLab": "JB Grayscale Image",
    "CameraLab": "JB Camera",
    "ShaderLab": "JB Film Grain",
}

WEB_DIRECTORY = "./web"

__version__ = "0.1.1"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

