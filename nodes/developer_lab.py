'''
    Film Developer Node for ComfyUI Custom Nodes
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

import logging


from ..node_config import STOCK_MAP

from ..models.camera import Camera
from ..models.filmstock import FilmStock
from ..models.filmgrain import FilmGrain

from ..src.srgb import linear_to_srgb_torch
from ..src.spectral import get_spectral_image
from ..src.lut import get_generalized_sigmoid_lut, get_hd_curve_lut
from ..src.filmgrain import get_film_grain_image

class DeveloperLab:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "camera": ("CAMERA",),
                "developer": (["None"],),
            },
            "optional": {
                "apply_film_grain": ("BOOLEAN", {"default": True}),
                "precision": ("INT", {"default": 4096, "min": 256, "max": 65536, "step": 256}),
                "exposure_index": ("FLOAT", {"default": 0.03, "min": 0.00, "max": 1.00, "step": 0.01}),
                "N_development": ("INT", {"default": 0.0, "min": -3.0, "max": 3.0, "step": 1}),
            }        
        }

    # Bypass ComfyUI's strict dropdown validation for dynamic widgets
    @classmethod
    def VALIDATE_INPUTS(s, developer):
        return True

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("preview", "film_negative")
    FUNCTION = "build_spectral_image"
    CATEGORY = "JBNodes"
    DESCRIPTION = """Simulate black and white film stocks with customizable development processes."""

    def build_spectral_image(self, camera, apply_film_grain, precision, exposure_index, N_development, developer = None):
        
        if isinstance(camera, Camera):
            film_stock = camera.film_stock
            illuminant_key = camera.illuminant_key
            image = linear_to_srgb_torch(camera.image)
        else:
            return (None, None) 
        
        if isinstance(film_stock, FilmStock):
            weights = film_stock.weights
            params = film_stock.params
            stock_name = film_stock.name
        else:
            stock_data = STOCK_MAP.get(film_stock, {})
            weights = stock_data.get("weights", [0.33, 0.33, 0.33])
            params = stock_data.get("params", {"slope": 1.8, "toe": 0.2, "shoulder": 0.8})
            stock_name = str(film_stock)

        # Re-associate the string from the dropdown with the actual HDCurve object
        curve = None
        if film_stock.hd_curves and developer != "None":
            for c in film_stock.hd_curves:
                # This must perfectly match the string format generated in the API route
                display_name = f"{c.name} ({c.time}m at {c.temp}C)"
                if display_name == developer:
                    curve = c
                    break

        # Apply the film grain tot he linear image before applying the characteristic curve
        if film_stock.film_grain and apply_film_grain and developer != "None":
            image = get_film_grain_image(image, **film_stock.film_grain.__dict__)

        slope = params.get("slope", 1.8)
        toe = params.get("toe", 0.2)
        shoulder = params.get("shoulder", 0.8)

        # Get the characteristic curve LUT
        if not curve:
            char_lut = get_generalized_sigmoid_lut(slope, toe, shoulder, precision)
        else:
            char_lut = get_hd_curve_lut(curve, precision, ei=exposure_index, dev_offset=N_development)
            try:
                logging.info(f"[comfyui-jbnodes] applying {stock_name} - {curve.name} characteristic curve with EI: {exposure_index}, N-development: {N_development}")
            except:
                pass

        return get_spectral_image(image, weights, None, illuminant_key, char_lut)

