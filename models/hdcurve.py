'''
    Hunter and Driffield Characteristic Curve Class for ComfyUI Custom Nodes
    ------------------------------------------------------------------------
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

from dataclasses import dataclass
from typing import List, Dict, Optional
from .filmgrain import FilmGrain

@dataclass
class HDCurve:
    id: str
    name: str
    time: float
    temp: float
    curve_points: List[List[float]]
    film_grain: Optional[FilmGrain] = None

    @classmethod
    def from_dict(cls, data: dict, developer_name: str = "Generic Developer", film_grain: Optional[FilmGrain] = None):
        return cls(
            id = data.get("id", "00000000-0000-0000-0000-000000000000"),
            name = developer_name,
            time = float(data.get("time", 0.0)),
            temp = float(data.get("temp", 20.0)),
            curve_points = data.get("curve_points", []),
            film_grain = film_grain
        )

