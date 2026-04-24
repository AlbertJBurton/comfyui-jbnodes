'''
    Split Grade Darkroom Enlarger Node for ComfyUI Custom Nodes
    -----------------------------------------------------------
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
import numpy as np

from PIL import Image, ImageOps

from ..node_config import CONTRAST_FILTER_NAMES, CONTRAST_FILTER_MAP

from ..src.darkroom_lib import get_print_image

class PrintLabSplitGrade:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "film_negative": ("IMAGE",),
                "contrast_filter_1": (CONTRAST_FILTER_NAMES, {"default": "2"}),
                "exposure_secs_1": ("FLOAT", {"default": 10, "min": 0, "max": 60, "step": 1}),
                "contrast_filter_2": (CONTRAST_FILTER_NAMES, {"default": "2"}),
                "exposure_secs_2": ("FLOAT", {"default": 10, "min": 0, "max": 60, "step": 1}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "preview_1", "preview_2")
    FUNCTION = "print_image"
    CATEGORY = "JBNodes/Darkroom"
    DESCRIPTION = """Simulate printing a negative to photographic paper using a split grade filter process."""

    def print_image(self, film_negative, contrast_filter_1, exposure_secs_1, contrast_filter_2, exposure_secs_2):
    
        filter_data_1 = CONTRAST_FILTER_MAP.get(contrast_filter_1)
        contrast_factor_1 = filter_data_1.get("factor")
    
        filter_data_2 = CONTRAST_FILTER_MAP.get(contrast_filter_2)
        contrast_factor_2 = filter_data_2.get("factor")
    
        image_1 = get_print_image(film_negative, contrast_factor = contrast_factor_1, exposure_secs = exposure_secs_1)
        image_2 = get_print_image(film_negative, contrast_factor = contrast_factor_2, exposure_secs = exposure_secs_2)

        print_image_1 = Image.fromarray(np.clip(255. * image_1[0].cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        print_image_2 = Image.fromarray(np.clip(255. * image_2[0].cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

        blend_percent = 0.5
        blend_mask = Image.new(mode="L", size=(print_image_1.size), color=(round(blend_percent * 255)))
        blend_mask = ImageOps.invert(blend_mask)

        # Blend image
        img_result = Image.composite(print_image_1, print_image_2, blend_mask)

        del print_image_1, print_image_2, blend_mask

        image_out = torch.from_numpy(np.array(img_result).astype(np.float32) / 255.0).unsqueeze(0)
    
        return (image_out, image_1[0], image_2[0])
