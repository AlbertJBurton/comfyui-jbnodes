
'''
    Crop Film Aspect Ratio Node for ComfyUI Custom Nodes
    ----------------------------------------------------
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

from ..node_config import FILM_FORMAT_NAMES, FILM_FORMAT_MAP

class CropFilmAspectRatio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "image": ("IMAGE",),
                "film_size": (FILM_FORMAT_NAMES, {}),
                "orientation": (["Auto", "Landscape", "Portrait"], {}),
                "shift": ("FLOAT", {"default": 0.00, "min": -1.00, "max": 1.00, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "enforce_aspect_ratio"
    CATEGORY = "JBNodes/Utility"
    DESCRIPTION = """Crop the image to match the aspect ratio of a specific film format."""

    def enforce_aspect_ratio(self, image, film_size, orientation, shift):

        size_data = FILM_FORMAT_MAP.get(film_size)
        if not size_data:
            return (image,) # Safety fallback
            
        _, height, width, _ = image.shape
        current_aspect = width / height

        if orientation == "Auto":
            if width > height:
                target_aspect = size_data["frame_size"][0] / size_data["frame_size"][1]
            else:            
                target_aspect = size_data["frame_size"][1] / size_data["frame_size"][0]
        else:
            if orientation == "Landscape":
                target_aspect = size_data["frame_size"][0] / size_data["frame_size"][1]
            else:
                target_aspect = size_data["frame_size"][1] / size_data["frame_size"][0]

        # Use a small tolerance for floating point comparisons to prevent microscopic 1-pixel jitters
        if abs(current_aspect - target_aspect) < 0.001:
            return (image,)
        
        # Calculate Target Dimensions
        if current_aspect > target_aspect:
            # Image is too wide, preserve height and crop width
            new_width = int(round(height * target_aspect))
            new_height = height
        else:
            # Image is too tall, preserve width and crop height
            new_width = width
            new_height = int(round(width / target_aspect))

        x_offset = 0 
        y_offset = 0
        shift_offset = 0
        
        x_offset = (width - new_width) // 2
        y_offset = (height - new_height) // 2        
        shift_offset = int(((height - new_height) // 2) * (-shift))

        # Apply user adjustment for shift in cropping position. Default is 1.0 (no change). Max 2.0 (double the shift).
        y_offset += shift_offset

        image = image[:, y_offset:y_offset + new_height, x_offset:x_offset + new_width, :]

        return (image,)
