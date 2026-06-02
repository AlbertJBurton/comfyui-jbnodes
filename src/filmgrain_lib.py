'''
    Film Grain Emulation Library Functions
    --------------------------------------
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

    Redirect to the numpy/torch grain engine.
    Replaces the prior moderngl/GLSL implementation.
'''
import logging
import warnings

from ..src.grain_engine import apply_film_grain

warnings.warn(
    "[comfyui-jbnodes] filmgrain_lib.py is deprecated. "
    "Use src/grain_engine.py directly.",
    DeprecationWarning,
    stacklevel=2,
)


def get_film_grain_image(image, **kwargs):
    """
    Apply film grain to a batched image tensor.

    Deprecated: calls the new grain_engine internally.
    Kept for backward compatibility with the CAMERA pipeline
    (DeveloperLab etc.).
    """
    logging.info("[comfyui-jbnodes] filmgrain_lib.get_film_grain_image (deprecated path)")

    # The old interface expects kwargs with the same parameter names.
    # Map film_size -> film_size string (passed through, fine)
    # Map blend -> blend_mode (the old lib used 'blend' as an int index)
    # The old lib's 'blend' parameter was an int (0=Soft Light, 1=Overlay, 2=Linear Light).
    if "blend" in kwargs and "blend_mode" not in kwargs:
        blend_map = {0: "Soft Light", 1: "Overlay", 2: "Linear Light"}
        kwargs["blend_mode"] = blend_map.get(kwargs.pop("blend"), "Soft Light")

    # The old lib had 'grain_type' (int) not 'emulsion_type' (str).
    if "grain_type" in kwargs and "emulsion_type" not in kwargs:
        kwargs["emulsion_type"] = "Tabular" if kwargs.pop("grain_type") == 1 else "Cubic"

    result = apply_film_grain(
        image=image,
        **kwargs,
        seed=0,
        image_index=0,
    )
    return result
