'''
    ComfyUI Nodes for Film Simulation and Photographic Principles
    -------------------------------------------------------------
    Copyright (C) 2026  Albert J. Burton

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

__version__ = "0.4.0"

failed_status = False

print("\n\033[32m[comfyui-jbnodes]\033[0m Loading ComfyUI Nodes for B&W Film Emulation (version {})".format(__version__))

''' --- UTILITY NODES --- '''
try:
    from .nodes.film_aspect_ratio import FilmAspectRatio
    from .nodes.crop_film_aspect_ratio import CropFilmAspectRatio
    from .nodes.grayscale_lab import GrayscaleLab
    from .nodes.color_chart_Image import ColorChartImageLoader
    from .nodes.merge_rgb_channels import MergeRGBImageChannel
    from .nodes.prompt_lab import PromptLab
    from .nodes.camera_image_pipe import CameraImagePipeLoader, CameraImagePipeDecomposer, CameraImagePipeComposer
except ImportError as e:
    failed_status = True
    print("\033[31m[comfyui-jbnodes]\033[0m Warning: Failed to import utility nodes. Some functionality may be limited. Error details: {}".format(e))
finally:
    if not failed_status:
        print("\033[32m[comfyui-jbnodes]\033[0m Utility nodes loaded successfully.")

''' --- PHOTOGRAPHY NODES --- '''
try:
    from .nodes.camera_lab import CameraLab
    from .nodes.filter_lab import FilterLab
    from .nodes.film_grain_lab import FilmGrainLab
    from .nodes.developer_lab import DeveloperLab
except ImportError as e:
    failed_status = True
    print("\033[31m[comfyui-jbnodes]\033[0m Warning: Failed to import photography nodes. Some functionality may be limited. Error details: {}".format(e))
finally:
    if not failed_status:
        print("\033[32m[comfyui-jbnodes]\033[0m Photography nodes loaded successfully.")

''' --- DARKROOM NODES --- '''
try:
    from .nodes.print_lab_multigrade import PrintLabMultigrade
    from .nodes.print_lab_graded import PrintLabGraded
    from .nodes.print_lab_splitgrade import PrintLabSplitGrade
except ImportError as e:
    failed_status = True
    print("\033[31m[comfyui-jbnodes]\033[0m Warning: Failed to import darkroom nodes. Some functionality may be limited. Error details: {}".format(e))
finally:
    if not failed_status:
        print("\033[32m[comfyui-jbnodes]\033[0m Darkroom nodes loaded successfully.")

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
    "ColorChartImageLoader": ColorChartImageLoader,
    "MergeRGBImageChannel": MergeRGBImageChannel,
    "PrintLabSplitGrade": PrintLabSplitGrade,
    "PromptLab": PromptLab,
    "CameraImagePipeLoader": CameraImagePipeLoader,
    "CameraImagePipeDecomposer": CameraImagePipeDecomposer,
    "CameraImagePipeComposer": CameraImagePipeComposer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeveloperLab": "Film Development",
    "FilterLab": "Wratten Filter",
    "GrayscaleLab": "Grayscale Image",
    "CameraLab": "B&W Film Camera",
    "FilmGrainLab": "Film Grain",
    "CropFilmAspectRatio": "Crop Film Aspect Ratio",
    "FilmAspectRatio": "Film Aspect Ratio",
    "ColorChartImageLoader": "Color Test Chart Image",
    "MergeRGBImageChannel": "Merge RGB Channels",
    "PrintLabSplitGrade": "Darkroom Enlarger (Split Grade)",
    "PrintLabMultigrade": "Darkroom Enlarger (Multigrade)",
    "PrintLabGraded": "Darkroom Enlarger",
    "PromptLab": "Film Stock Prompt Manager",
    "CameraImagePipeLoader": "Camera Image Pipe Loader",
    "CameraImagePipeDecomposer": "Camera Image Pipe Decomposer",
    "CameraImagePipeComposer": "Camera Image Pipe Composer",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY", __version__]

