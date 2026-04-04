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

from ..node_config import FILM_SIZE_NAMES, FILM_SIZE_MAP, RESOLUTIONS

class FilmAspectRatio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "film_size": (FILM_SIZE_NAMES, {}),
                "resolution": (RESOLUTIONS, {}),
                "swap_dimensions": ("BOOLEAN", {"default": False}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
            },
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "LATENT")
    RETURN_NAMES = ("width", "height", "aspect_ratio", "empty_latent")
    FUNCTION = "get_aspect_ratio"
    CATEGORY = "JBNodes/Utility"
    DESCRIPTION = """Get a latent image with the aspect ratio of a specific film format."""

    def get_aspect_ratio(self, film_size, resolution, swap_dimensions, batch_size):
        size_data = FILM_SIZE_MAP.get(film_size)

        width = 1024
        height = 1024
        target_aspect = width / height

        for lsize in size_data["latent_sizes"]:
            if lsize["name"].startswith(resolution):
                width = lsize.get("width", 1024)
                height = lsize.get("height", 1024)
                target_aspect = width / height
                break

        if swap_dimensions:
            target_aspect = 1 / target_aspect
            width, height = height, width

        empty_latent = torch.zeros((batch_size, 4, width // 8, height // 8), dtype=torch.float32)
        
        return (width, height, target_aspect, {"samples":empty_latent})

