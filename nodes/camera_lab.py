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

from ..node_config import CAMERA_MAP, BW_FILTER_NAMES, BW_FILTER_MAP, FILM_STOCK_MAP, FILM_FORMAT_MAP, FILM_FORMAT_NAME_TO_ID, ILLUMINANT_NAMES, ILLUMINANT_MAP

from ..src.spectral_lib import get_camera_image
from ..src.filters_lib import get_filter_image

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
                "light_source": (ILLUMINANT_NAMES, ),
            }
        }

    # Bypass ComfyUI's strict dropdown validation for dynamic widgets
    @classmethod
    def VALIDATE_INPUTS(s, camera, film):
        return True

    RETURN_TYPES = ("CAMERA", "IMAGE")
    RETURN_NAMES = ("camera_roll", "film_latent_preview")
    FUNCTION = "get_camera"
    CATEGORY = "JBNodes"
    DESCRIPTION = """Classic black-and-white film camera simulation."""

    def get_camera(self, image, film_format, camera, filter, film, light_source):
        camera_obj = Camera.from_dict(CAMERA_MAP.get(camera))
        camera_obj.image = image

        if isinstance(film_format, FilmFormat):
            camera_obj.film_format = film_format
        elif isinstance(film_format, dict):
            camera_obj.film_format = FilmFormat.from_dict(film_format)
        else:
            film_format_id = FILM_FORMAT_NAME_TO_ID.get(film_format, film_format)
            camera_obj.film_format = FilmFormat.from_dict(FILM_FORMAT_MAP.get(film_format_id, {}))

        film_obj = FilmStock.from_dict(FILM_STOCK_MAP.get(film))
        camera_obj.film_stock = film_obj

        # Populate the film grain object with the selected film size if appropriate.
        if camera_obj.film_stock.film_grain:
            camera_obj.film_stock.film_grain.film_size = camera_obj.film_format.id
            
        if camera_obj.film_stock.hd_curves:
            for curve in camera_obj.film_stock.hd_curves:
                if curve.film_grain:
                    curve.film_grain.film_size = camera_obj.film_format.id

        illuminant = ILLUMINANT_MAP.get(light_source)
        camera_obj.illuminant_key = illuminant["key"] if illuminant else "D65"

        if filter != "None":
            filter_data = BW_FILTER_MAP.get(filter)
            if filter_data:
                transmission = filter_data.get("transmission")
                auto_factor = 1.0 / filter_data.get("visual_transmission")
                camera_obj.image = get_filter_image(camera_obj.image, transmission, 1.0, True, auto_factor)
                if isinstance(camera_obj.image, tuple):
                    camera_obj.image = camera_obj.image[0] if len(camera_obj.image) > 0 else camera_obj.image

        camera_obj.image, preview = get_camera_image(camera_obj.image, camera_obj)

        return(camera_obj, preview)
    
