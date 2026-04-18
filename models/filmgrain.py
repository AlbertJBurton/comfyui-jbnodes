'''
    Film Grain Class for ComfyUI Custom Nodes
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

from dataclasses import dataclass, field

@dataclass
class FilmGrain:
    rms_granularity: float = 8.0
    film_size: str = "135"
    emulsion_type: str = "Cubic"
    film_grit: float = 0.2
    halation: float = 0.1
    emulsion_softness: float = 0.5
    blend_mode: str = "Soft Light"
    luminance_peak_bias: float = 0.5
    algorithmic_octaves: int = 3
    morphological_variance: float = 2.0
    temporal_entropy: float = 1.0
    shadow_dither: float = 0.0

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            rms_granularity = float(data.get("rms_granularity", 8.0)),
            film_size = data.get("film_size", "135"),
            emulsion_type = data.get("emulsion_type", "Cubic"),
            film_grit = float(data.get("film_grit", 0.2)),
            halation = float(data.get("halation", 0.1)),
            emulsion_softness = float(data.get("emulsion_softness", 0.5)),
            blend_mode = data.get("blend_mode", "Soft Light"),
            luminance_peak_bias = float(data.get("luminance_peak_bias", 0.5)),
            algorithmic_octaves = int(data.get("algorithmic_octaves", 3)),
            morphological_variance = float(data.get("morphological_variance", 2.0)),
            temporal_entropy = float(data.get("temporal_entropy", 1.0)),
            shadow_dither = float(data.get("shadow_dither", 0.0))
        )
