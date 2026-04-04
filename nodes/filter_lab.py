'''
    Wratten Filter Node for ComfyUI Custom Nodes
    --------------------------------------------
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

from ..node_config import FILTER_NAMES, FILTER_MAP

from ..src.filters import get_filter_image

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
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_filter"
    CATEGORY = "JBNodes"
    DESCRIPTION = """Apply a Wratten filter to an image."""

    def apply_filter(self, image, filter, filter_factor, auto_filter_factor):
        filter_data = FILTER_MAP.get(filter)
        if not filter_data:
            return (image,) 
        
        transmission = filter_data.get("transmission")
        auto_factor = 1.0 / filter_data.get("visual_transmission")

        return get_filter_image(image, transmission, filter_factor, auto_filter_factor, auto_factor)

