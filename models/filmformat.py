'''
    Film Format Class for ComfyUI Custom Nodes
    -----------------------------------------------
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

from dataclasses import dataclass, field
from typing import List, Dict, Optional

from .latentsize import LatentSize

@dataclass
class FilmFormat:
    name: str
    description: str
    frame_size: List[float] = field(default_factory = lambda: [1.0, 1.0]) # [width_mm, height_mm]
    latent_sizes: Optional[List[Dict[str, int]]] = None  # List of dicts with 'name', 'width', 'height'

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            name = data.get("name", "Unknown Format"),
            description = data.get("description", ""),
            frame_size = data.get("frame_size", [1.0, 1.0]),
            latent_sizes = [LatentSize.from_dict(size) for size in data.get("latent_sizes", [])] if data.get("latent_sizes") else None
        )
