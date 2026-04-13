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
from .filmformat import FilmFormat

@dataclass
class Camera:
    id: str
    name: str
    illuminant_key: str
    film_format: Optional[FilmFormat] = None
    film_stock: Optional[FilmStock] = None
    image: Optional[torch.Tensor] = None

    @classmethod
    def from_dict(cls, data: dict):

        film_format_data = data.get("film_format")
        film_stock_data = data.get("film_stock")

        return cls(
            id = data.get("id", "285E63ED-C3AA-4C79-B7C9-06CF58D73357"),
            name = data.get("name", "Generic Camera"),
            illuminant_key = data.get("illuminant_key", "D65"),
            film_format = FilmFormat.from_dict(film_format_data) if isinstance(film_format_data, dict) else None,
            film_stock = FilmStock.from_dict(film_stock_data) if isinstance(film_stock_data, dict) else None,  
            image = data.get("image", None)        
        )

