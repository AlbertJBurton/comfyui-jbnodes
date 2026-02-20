from .nodes import SpectralLab, SpectralLabCustom

NODE_CLASS_MAPPINGS = {
    "SpectralLabCustom": SpectralLabCustom,
    "SpectralLab": SpectralLab
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpectralLabCustom": "Spectral Lab (Custom)",
    "SpectralLab": "Spectral Lab (Film Stock)"
}

__version__ = "0.1.0"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
