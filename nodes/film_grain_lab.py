'''
    Film Grain Node for ComfyUI Custom Nodes
    ----------------------------------------
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

from ..node_config import GLSL_DIR
from ..src.filmgrain import get_film_grain_image

class FilmGrainLab:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "rms_granularity": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 50.0, "step": 0.1}),
                "film_size": (["135","120","4x5","8x10"], {}),
                "emulsion_type": (["Cubic", "Tabular"], {}),
                "film_grit": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "halation": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "emulsion_softness": ("FLOAT", {"default": 0.75, "min": 0.00, "max": 1.50, "step": 0.01}),
                "blend_mode": (["Soft Light", "Overlay", "Linear Light"], {}),
                "luminance_peak_bias": ("FLOAT", {"default": 0.50, "min": 0.00, "max": 1.00, "step": 0.01}),
                "algorithmic_octaves": ("INT", {"default": 2, "min": 1, "max": 4, "step": 1}),
                "morphological_variance": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "temporal_entropy": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_shader"
    CATEGORY = "JBNodes"
    DESCRIPTION = "Execute custom GLSL shaders with an extended parameter set using moderngl."

    def apply_shader(self, image, **kwargs):

        result = get_film_grain_image(image, **kwargs)

        return (result,)
           
