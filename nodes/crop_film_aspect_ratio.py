
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
                "film_format": (FILM_FORMAT_NAMES, {}),
                "orientation": (["Auto", "Landscape", "Portrait"], {}),
                "shift": ("FLOAT", {"default": 0.00, "min": -1.00, "max": 1.00, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "FILMFORMAT")
    RETURN_NAMES = ("image", "film_format")
    FUNCTION = "enforce_aspect_ratio"
    CATEGORY = "JBNodes/Utility"
    DESCRIPTION = """Crop the image to match the aspect ratio of a specific film format."""

    def enforce_aspect_ratio(self, image, film_format, orientation, shift):

        film_format_obj = FILM_FORMAT_MAP.get(film_format)
        if not film_format_obj:
            return (image, None) # Safety fallback
            
        _, height, width, _ = image.shape
        current_aspect = width / height

        if orientation == "Auto":
            if width > height:
                target_aspect = film_format_obj["frame_size"][0] / film_format_obj["frame_size"][1]
            else:            
                target_aspect = film_format_obj["frame_size"][1] / film_format_obj["frame_size"][0]
        else:
            if orientation == "Landscape":
                target_aspect = film_format_obj["frame_size"][0] / film_format_obj["frame_size"][1]
            else:
                target_aspect = film_format_obj["frame_size"][1] / film_format_obj["frame_size"][0]

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

        # Apply user adjustment for shift in cropping position. Default is 1.0 (no change). 
        # Positive values shift the crop towards the right (for landscape) or down
        # (for portrait), while negative values shift it left/up.
        x_offset = int((width - new_width) // 2)
        y_offset = int((height - new_height) // 2)

        if current_aspect > target_aspect:
            x_offset += int(x_offset * (-shift))
        else:
            y_offset += int(y_offset * (-shift))

        image = image[:, y_offset:y_offset + new_height, x_offset:x_offset + new_width, :]

        return (image, film_format_obj)
