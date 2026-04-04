'''
    Merge RGB Channels Node for ComfyUI Custom Nodes
    ------------------------------------------------
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

class MergeRGBImageChannel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "red": ("IMAGE",),
            "green": ("IMAGE",),
            "blue": ("IMAGE",),
    },
            "optional": {
                "normalize": ("BOOLEAN", {"default": False}),
                },
            }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "get_merged_image"
    CATEGORY = "JBNodes/Utility"
    DESCRIPTION = """Merges red, green, and blue channels into an image."""
        
    def get_merged_image(self, red, green, blue, normalize=False):

        image = torch.stack([red[..., 0, None], green[..., 1, None], blue[..., 2, None]], dim=-1)
        image = image.squeeze(-2)
    
        if normalize:
            image = image / torch.max(image)

        return (image,)
