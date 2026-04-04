'''
    Grayscale Image Node for ComfyUI Custom Nodes
    ---------------------------------------------
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

from ..node_config import GRAYSCALE_NAMES, GRAYSCALE_MAP
from ..src.grayscale import get_grayscale_image

class GrayscaleLab:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "image": ("IMAGE",),            
                "method": (GRAYSCALE_NAMES, {}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "build_grayscale_image"
    CATEGORY = "JBNodes/Utility"
    DESCRIPTION = """Create a grayscale image with a specific sprectral curve."""

    def build_grayscale_image(self, image, method):
        
        grayscale_data = GRAYSCALE_MAP.get(method)
        weights = grayscale_data.get("weights", [0.33, 0.33, 0.33])       
        
        return get_grayscale_image(image, weights)
    
