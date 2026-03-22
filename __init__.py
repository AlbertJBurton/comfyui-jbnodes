from .nodes import PrintLabGraded, PrintLabMultigrade, SpectralLab, FilterLab, FilmLab, GrayscaleLab

NODE_CLASS_MAPPINGS = {
    "SpectralLab": SpectralLab,
    "FilterLab": FilterLab,
    "PrintLabMultigrade": PrintLabMultigrade,
    "PrintLabGraded": PrintLabGraded,
    "FilmLab": FilmLab,
    "GrayscaleLab": GrayscaleLab,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpectralLabCustom": "JB Spectral Lab (Custom)",
    "SpectralLab": "JB Film Development",
    "FilterLab": "JB Wratten Filter",
    "PrintLabMultigrade": "JB Darkroom Enlarger (Multigrade)",
    "PrintLabGraded": "JB Darkroom Enlarger (Graded)",
    "FilmLab": "JB BW Film Stocks",
    "GrayscaleLab": "JB Image to Grayscale"
}

__version__ = "0.1.1"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
