from .nodes import PrintLabGraded, PrintLabMultigrade, DeveloperLab, FilterLab, GrayscaleLab, CameraLab, FilmGrainLab, CropFilmAspectRatio, FilmAspectRatio

NODE_CLASS_MAPPINGS = {
    "DeveloperLab": DeveloperLab,
    "FilterLab": FilterLab,
    "PrintLabMultigrade": PrintLabMultigrade,
    "PrintLabGraded": PrintLabGraded,
    "GrayscaleLab": GrayscaleLab,
    "CameraLab": CameraLab,
    "FilmGrainLab": FilmGrainLab,
    "CropFilmAspectRatio": CropFilmAspectRatio,
    "FilmAspectRatio": FilmAspectRatio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeveloperLab": "Film Development",
    "FilterLab": "Wratten Filter",
    "PrintLabMultigrade": "Darkroom Enlarger (Multigrade)",
    "PrintLabGraded": "Darkroom Enlarger",
    "GrayscaleLab": "Grayscale Image",
    "CameraLab": "B&W Film Camera",
    "FilmGrainLab": "Film Grain",
    "CropFilmAspectRatio": "Crop Film Aspect Ratio",
    "FilmAspectRatio": "Film Aspect Ratio",
}

WEB_DIRECTORY = "./web"

__version__ = "0.1.1"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

