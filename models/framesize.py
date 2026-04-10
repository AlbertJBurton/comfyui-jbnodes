'''
    Frame Size Class for ComfyUI Custom Nodes
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

from dataclasses import dataclass

@dataclass
class FrameSize:
    width: float = 36.0    # default 35mm width
    height: float = 24.0   # default 35mm height
 
    @classmethod
    def from_dict(cls, data: dict):
        if not data:
            return cls()
            
        return cls(
            width = data.get("width", 36.0),
            height = data.get("height", 24.0)
        )
