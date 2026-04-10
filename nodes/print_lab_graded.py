'''
    Graded Paper Darkroom Enlarger Node for ComfyUI Custom Nodes
    ------------------------------------------------------------
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

from comfy import model_management

from ..node_config import GRADED_PAPER_NAMES, GRADED_PAPER_MAP, CONTRAST_FILTER_MAP

from ..src.darkroom_lib import get_print_image

class PrintLabGraded:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "film_negative": ("IMAGE",),
            },
            "optional": {
                "graded_paper": (GRADED_PAPER_NAMES, {"default": "Ilford / Ilfobrom Galerie FB / Grade 2"}),
                "exposure_secs": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 60.0, "step": 0.1}),
                "precision": ("INT", {"default": 4096, "min": 256, "max": 65536, "step": 256}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "print_image"
    CATEGORY = "JBNodes/Darkroom"
    DESCRIPTION = """Simulate printing a negative to graded photographic paper."""

    def print_image(self, film_negative, graded_paper, exposure_secs, precision=4096):
        paper_data = GRADED_PAPER_MAP.get(graded_paper)
        grade = paper_data.get("grade")
        
        # Read the raw empirical points
        hd_curve_raw = paper_data.get("hd_curve_points", None)
        d_max = paper_data.get("d_max", 2.1)
        d_min = paper_data.get("d_min", 0.04)
        
        hd_curve_points = None
        contrast_factor = None
        
        if hd_curve_raw:
            # Cast the JSON array of coordinates into a PyTorch tensor
            device = model_management.get_torch_device()
            hd_curve_points = torch.tensor(hd_curve_raw, dtype=torch.float32, device=device)
        else:
            # Fallback to the algorithmic method if no empirical data exists
            filter_data = CONTRAST_FILTER_MAP.get(grade)
            contrast_factor = filter_data.get("factor")

        return get_print_image(film_negative, contrast_factor = contrast_factor, exposure_secs = exposure_secs, hd_curve_points = hd_curve_points, d_max = d_max, d_min = d_min, precision = precision)

