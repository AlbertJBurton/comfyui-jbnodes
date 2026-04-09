'''
    Film Aspect Ratio Node for ComfyUI Custom Nodes
    -----------------------------------------------
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

import torch

from ..node_config import FILM_FORMAT_NAMES, FILM_FORMAT_MAP
from ..models.filmformat import FilmFormat
from ..models.latentsize import LatentSize

class FilmAspectRatio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "film_format": (FILM_FORMAT_NAMES, {}),
                "resolution": ([None], {}),
                "swap_dimensions": ("BOOLEAN", {"default": False}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
            },
        }
    
    # Bypass ComfyUI's strict dropdown validation for dynamic widgets
    @classmethod
    def VALIDATE_INPUTS(s, resolution):
        return True

    RETURN_TYPES = ("FLOAT", "FLOAT", "LATENT", "FILMFORMAT")
    RETURN_NAMES = ("width", "height", "empty_latent", "film_format")
    FUNCTION = "get_aspect_ratio"
    CATEGORY = "JBNodes/Utility"
    DESCRIPTION = """Get a latent image with the aspect ratio of a specific film format."""

    def get_aspect_ratio(self, film_format, resolution, swap_dimensions, batch_size):

        # Safe defaults in case something goes wrong with the format lookup or resolution parsing
        width = 1024
        height = 1024

        film_format_obj = FilmFormat.from_dict(FILM_FORMAT_MAP.get(film_format))

        if not film_format_obj or not isinstance(film_format_obj, FilmFormat):
            print(f"[comfyui-jbnodes] Warning: Film format '{film_format}' not found. Using default aspect ratio.")
            return (width, height, {"samples":torch.zeros((batch_size, 4, width // 8, height // 8), dtype=torch.float32)}, None)
        
        # Re-associate the string from the dropdown with the actual LatentSize (resolution) object
        size = None
        if film_format_obj.latent_sizes and resolution != "None":
            for lsize in film_format_obj.latent_sizes:
                # This must perfectly match the string format generated in the API route
                display_name = f"{lsize.name}"
                if display_name == resolution:
                    size = lsize
                    break

        if size:
            width = size.width
            height = size.height

        if swap_dimensions:
            width, height = height, width

        empty_latent = torch.zeros((batch_size, 4, width // 8, height // 8), dtype=torch.float32)
        
        return (width, height, {"samples":empty_latent}, film_format_obj)

