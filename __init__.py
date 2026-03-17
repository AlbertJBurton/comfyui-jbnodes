from .nodes import SpectralLab, SpectralLabCustom, FilterLab

NODE_CLASS_MAPPINGS = {
    "SpectralLabCustom": SpectralLabCustom,
    "SpectralLab": SpectralLab,
    "FilterLab": FilterLab
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpectralLabCustom": "Spectral Lab (Custom)",
    "SpectralLab": "Spectral Lab (Film Stock)",
    "FilterLab": "Wratten Filter"
}

__version__ = "0.1.1"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
