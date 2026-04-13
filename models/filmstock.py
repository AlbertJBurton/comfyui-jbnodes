'''
    Film Stock Class for ComfyUI Custom Nodes
    -----------------------------------------
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

from ..node_config import FILM_FORMAT_NAMES

from dataclasses import dataclass, field
from typing import List, Dict, Optional

from .hdcurve import HDCurve
from .filmgrain import FilmGrain

@dataclass
class FilmStock:
    id: str
    name: str
    description: str
    iso: int = 100
    weights: Optional[List[float]] = None
    params: Optional[Dict[str, float]] = None
    film_formats: Optional[List[str]] = None
    film_grain: Optional[FilmGrain] = None
    spectral_points: Optional[List[List[float]]] = None 
    hd_curves: Optional[List[HDCurve]] = None

    @classmethod
    def from_dict(cls, data: dict):
        
        # Parse HD curves if they exist in the payload
        raw_hd_curves = data.get("hd_curves")
        parsed_hd_curves = []
        if raw_hd_curves:
            for dev_group in raw_hd_curves:
                dev_name = dev_group.get("developer", "Generic Developer")
                raw_grain = dev_group.get("film_grain")
                dev_grain = FilmGrain.from_dict(raw_grain) if raw_grain else None
                for curve_data in dev_group.get("curves", []):
                    parsed_hd_curves.append(HDCurve.from_dict(curve_data, developer_name=dev_name, film_grain=dev_grain))
                    
        raw_film_grain = data.get("film_grain")
        parsed_film_grain = FilmGrain.from_dict(raw_film_grain) if raw_film_grain else None

        return cls(
            id = data.get("id", "3F3836A6-8070-4F70-8086-520E47BB5143"),
            name = data.get("name", "Generic Film Stock"),
            description = data.get("description", ""),
            iso = int(data.get("iso", 100)),
            film_formats = data.get("film_formats", FILM_FORMAT_NAMES),  # Default to all formats if not specified
            weights = data.get("weights", None),
            params = data.get("params", None),
            spectral_points = data.get("spectral_points", None),
            hd_curves = parsed_hd_curves,
            film_grain = parsed_film_grain
        )

