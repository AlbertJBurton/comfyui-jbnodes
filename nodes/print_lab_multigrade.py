'''
    Multigrade Paper Darkroom Enlarger Node for ComfyUI Custom Nodes
    ----------------------------------------------------------------
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

from ..node_config import CONTRAST_FILTER_NAMES, CONTRAST_FILTER_MAP

from ..src.darkroom_lib import get_print_image

class PrintLabMultigrade:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "film_negative": ("IMAGE",),
            },
            "optional": {
                "contrast_filter": (CONTRAST_FILTER_NAMES, {"default": "2"}),
                "exposure_secs": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "print_image"
    CATEGORY = "JBNodes/Darkroom"
    DESCRIPTION = """Simulate printing a negative to photographic paper."""

    def print_image(self, film_negative, contrast_filter, exposure_secs):

        filter_data = CONTRAST_FILTER_MAP.get(contrast_filter)
        contrast_factor = filter_data.get("factor")

        return get_print_image(film_negative, contrast_factor = contrast_factor, exposure_secs = exposure_secs)

