'''
    Film Grain Node for ComfyUI Custom Nodes
    ----------------------------------------
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

    Standalone film grain node using the numpy/torch grain engine
    (replaces the moderngl/GLSL implementation — no system GL deps).
'''
import logging

from ..src.grain_engine import apply_film_grain

# ---------------------------------------------------------------------------
# Shared parameter definitions
# ---------------------------------------------------------------------------
FILM_SIZE_OPTIONS = ["135", "120", "4x5", "8x10"]
EMULSION_TYPE_OPTIONS = ["Cubic", "Tabular"]
BLEND_MODE_OPTIONS = ["Soft Light", "Overlay", "Linear Light"]

INPUT_KWARGS = {
    "rms_granularity": ("FLOAT", {
        "default": 8.0, "min": 1.0, "max": 50.0, "step": 0.1,
        "tooltip": "RMS Granularity (8 = T-MAX 100, 17 = Tri-X 400)"
    }),
    "film_size": (FILM_SIZE_OPTIONS, {
        "default": "135",
        "tooltip": "Physical film format. Smaller formats = coarser relative grain"
    }),
    "emulsion_type": (EMULSION_TYPE_OPTIONS, {
        "default": "Cubic",
        "tooltip": "Cubic = traditional grain (Tri-X, HP5+). Tabular = T-grain (T-Max, Delta)"
    }),
    "film_grit": ("FLOAT", {
        "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01,
        "tooltip": "Grain edge sharpness. 0 = soft, 1 = harsh/contrasty"
    }),
    "halation": ("FLOAT", {
        "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01,
        "tooltip": "Light piping in film base. Warm glow around highlights"
    }),
    "emulsion_softness": ("FLOAT", {
        "default": 0.75, "min": 0.0, "max": 1.50, "step": 0.01,
        "tooltip": "Micro-blur simulating light scatter in the emulsion"
    }),
    "blend_mode": (BLEND_MODE_OPTIONS, {
        "default": "Soft Light",
        "tooltip": "How grain is composited onto the image"
    }),
    "luminance_peak_bias": ("FLOAT", {
        "default": 0.50, "min": 0.00, "max": 1.00, "step": 0.01,
        "tooltip": "Brightness zone where grain is most visible (0 = shadows, 1 = highlights)"
    }),
    "algorithmic_octaves": ("INT", {
        "default": 2, "min": 1, "max": 10, "step": 1,
        "tooltip": "Number of noise octaves. More = finer detail in grain structure"
    }),
    "morphological_variance": ("FLOAT", {
        "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1,
        "tooltip": "Amplitude multiplier per octave. <1 = decay (smoother), >1 = growth (more texture)"
    }),
    "temporal_entropy": ("FLOAT", {
        "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
        "tooltip": "3D noise Z-axis offset. Changes the grain seed pattern"
    }),
    "shadow_dither": ("FLOAT", {
        "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05,
        "tooltip": "Additive noise in deep shadows to prevent posterization"
    }),
}


# ---------------------------------------------------------------------------
# Standalone IMAGE → IMAGE node  (primary)
# ---------------------------------------------------------------------------
class FilmGrainNode:
    """
    Standalone film grain node.  Accepts a standard IMAGE tensor directly
    — no CAMERA pipeline required.  All 13 grain parameters are exposed.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                **INPUT_KWARGS,
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFF,
                    "tooltip": "Random seed for reproducible grain"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_grain"
    CATEGORY = "JBNodes"
    DESCRIPTION = "Advanced film grain emulation (numpy/torch engine — no GLSL required)."

    def apply_grain(self, image, **kwargs):
        seed = kwargs.pop("seed", 0)
        logging.info(f"[comfyui-jbnodes] FilmGrainNode: seed={seed}")

        result = apply_film_grain(
            image=image,
            **kwargs,
            seed=seed,
            image_index=0,
        )
        return (result,)


# ---------------------------------------------------------------------------
# CAMERA-pipeline node  (backward compat)
# ---------------------------------------------------------------------------
class FilmGrainLab:
    """
    Film grain in the CAMERA-pipeline workflow.
    Accepts a CAMERA type and applies grain to the image inside it.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "camera_roll": ("CAMERA",),
            },
            "optional": dict(INPUT_KWARGS),
        }

    RETURN_TYPES = ("CAMERA", "IMAGE")
    RETURN_NAMES = ("camera_roll", "preview")
    FUNCTION = "apply_shader"
    CATEGORY = "JBNodes"
    DESCRIPTION = "Film grain emulation (CAMERA pipeline)."

    def apply_shader(self, camera_roll, **kwargs):
        result = apply_film_grain(
            image=camera_roll.image,
            **kwargs,
            seed=0,
            image_index=0,
        )
        camera_roll.image = result
        return (camera_roll, result)


NODE_CLASS_MAPPINGS = {
    "FilmGrainNode": FilmGrainNode,
    "FilmGrainLab": FilmGrainLab,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FilmGrainNode": "Film Grain (Standalone)",
    "FilmGrainLab": "Film Grain",
}
