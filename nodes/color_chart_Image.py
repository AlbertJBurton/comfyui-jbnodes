'''
    Color Chart Image Loader Node for ComfyUI Custom Nodes
    ------------------------------------------------------
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

import os
import torch
import numpy as np
from PIL import Image, ImageOps
import folder_paths
import random
import logging

from ..node_config import IMAGES_DIR

class ColorChartImageLoader:
    @classmethod
    def INPUT_TYPES(s):
        
        if not os.path.exists(IMAGES_DIR):
            os.makedirs(IMAGES_DIR)
        image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')) and f.lower().startswith("color_chart")]

        if not image_files:
            image_files = ["none"]
        
        return {
            "required": {
                "image": (sorted(image_files),),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_image"
    CATEGORY = "JBNodes/Utility"
    DESCRIPTION = "Loads a color chart image from the comfyui-jbnodes/images folder."

    def load_image(self, image):

        if image == "none":
            return (torch.zeros((1, 512, 512, 3), dtype=torch.float32),)
        
        image_path = os.path.join(IMAGES_DIR, image)
        
        try:
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            img = i.convert("RGB")
        except Exception as e:
            logging.error(f"[comfyui-jbnodes] Error loading image {image_path}: {e}")
            # Return a blank image if it fails (e.g., empty file)
            img = Image.new("RGB", (512, 512), color=(0, 0, 0))
            i = img
            
        img_tensor = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_tensor)[None,]
        
        # Save a temporary copy for the UI preview
        temp_dir = folder_paths.get_temp_directory()
        temp_filename = f"jbnode_preview_{random.randint(0, 1000000)}.png"
        temp_path = os.path.join(temp_dir, temp_filename)
        i.save(temp_path)
        
        return {"ui": {"images": [{"filename": temp_filename, "subfolder": "", "type": "temp"}]}, "result": (img_tensor,)}
