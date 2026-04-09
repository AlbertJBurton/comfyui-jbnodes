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

from ..node_config import FILM_FORMAT_NAMES, FILM_FORMAT_MAP

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
    weights: List[float] = field(default_factory = lambda: [0.33, 0.33, 0.33])
    params: Dict[str, float] = field(default_factory = lambda: {"slope": 1.8, "toe": 0.2, "shoulder": 0.8})
    film_formats: Optional[List[str]] = None
    film_grain: Optional[Dict[str, float]] = None
    spectral_points: Optional[List[List[float]]] = None 
    hd_curves: Optional[List[HDCurve]] = None

    @classmethod
    def from_dict(cls, data: dict):
        
        # Parse HD curves if they exist in the payload
        raw_hd_curves = data.get("hd_curves")
        parsed_hd_curves = [HDCurve.from_dict(c) for c in raw_hd_curves] if raw_hd_curves else None
        raw_film_grain = data.get("film_grain")
        parsed_film_grain = FilmGrain.from_dict(raw_film_grain) if raw_film_grain else None


        return cls(
            id = data.get("id", "generic"),
            name = data.get("name", "Generic Film Stock"),
            description = data.get("description", ""),
            iso = int(data.get("iso", 100)),
            film_formats = data.get("film_formats", FILM_FORMAT_NAMES),  # Default to all formats if not specified
            weights = data.get("weights", [0.33, 0.33, 0.33]),
            params = data.get("params", {"slope": 1.8, "toe": 0.2, "shoulder": 0.8}),
            spectral_points = data.get("spectral_points", None),
            hd_curves = parsed_hd_curves,
            film_grain = parsed_film_grain
        )

