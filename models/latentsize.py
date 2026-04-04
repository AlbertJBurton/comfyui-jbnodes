'''
    Latent Image Size Class for ComfyUI Custom Nodes
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

from dataclasses import dataclass

@dataclass
class LatentSize:
    name: str
    width: int
    height: int

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            name = data.get("name", "Default Size"),
            width = int(data.get("width", 512)),
            height = int(data.get("height", 512))
        )
