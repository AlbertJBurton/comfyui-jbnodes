'''
    BW Film Camera Node for ComfyUI Custom Nodes
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

from ..node_config import CAMERA_NAMES, CAMERA_MAP, BW_FILTER_NAMES, BW_FILTER_MAP, STOCK_NAMES, STOCK_MAP, FILM_FORMAT_MAP, SOURCE_NAMES, SOURCE_MAP

from ..src.spectral import get_camera_image
from ..src.filters import get_filter_image

from ..models.filmstock import FilmStock
from ..models.filmformat import FilmFormat
from ..models.camera import Camera

class CameraLab:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "image": ("IMAGE",),
                "film_format": ("FILMFORMAT",),
                "camera": ([None],),
                "filter": (BW_FILTER_NAMES, {"default": "None"}),
                "film": ([None],),
            },
            "optional": {
                "light_source": (SOURCE_NAMES, ),
            }
        }

    # Bypass ComfyUI's strict dropdown validation for dynamic widgets
    @classmethod
    def VALIDATE_INPUTS(s, camera, film):
        return True

    RETURN_TYPES = ("CAMERA", "IMAGE")
    RETURN_NAMES = ("camera", "preview")
    FUNCTION = "get_camera"
    CATEGORY = "JBNodes"
    DESCRIPTION = """Classic black-and-white film cameras."""

    def get_camera(self, image, film_format, camera, filter, film, light_source):
        camera_obj = Camera.from_dict(CAMERA_MAP.get(camera))

        film_obj = FilmStock.from_dict(STOCK_MAP.get(film))
        camera_obj.film_stock = film_obj

        camera_obj.film_format = film_format if film_format else FilmFormat.from_dict(FILM_FORMAT_MAP.get("135"))  # Default to 35mm if no format provided

        illuminant = SOURCE_MAP.get(light_source)
        camera_obj.illuminant_key = illuminant["key"] if illuminant else "D65"

        if filter != "None":
            filter_data = BW_FILTER_MAP.get(filter)
            if filter_data:
                transmission = filter_data.get("transmission")
                auto_factor = 1.0 / filter_data.get("visual_transmission")
                image = get_filter_image(image, transmission, 1.0, True, auto_factor)
                if isinstance(image, tuple):
                    image = image[0] if len(image) > 0 else image

        camera_image, preview = get_camera_image(image, camera_obj)
        camera_obj.image = camera_image

        return(camera_obj, preview)
    
