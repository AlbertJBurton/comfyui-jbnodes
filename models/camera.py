'''
    Camera Class for ComfyUI Custom Nodes
    -------------------------------------
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

from dataclasses import dataclass
from typing import Optional

from .filmstock import FilmStock

@dataclass
class Camera:
    name: str
    illuminant_key: str
    film_stock: Optional[FilmStock] = None
    image: Optional[torch.Tensor] = None

    @classmethod
    def from_dict(cls, data: dict):

        return cls(
            name = data.get("name", "Unknown Camera"),
            illuminant_key = data.get("illuminant_key", "D65"),
            film_stock = None,  # To be set later when we have the FilmStock object
            image = None        # To be set later when we load the image tensor
        )

