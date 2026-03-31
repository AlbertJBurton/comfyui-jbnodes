import json
import os
import numpy as np
import torch
import dataclasses
import logging

from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class FilmFormat:
    name: str
    description: str
    frame_size: List[float] = field(default_factory=lambda: [1.0, 1.0]) # [width_mm, height_mm]
    latent_sizes: Optional[List[Dict[str, int]]] = None  # List of dicts with 'name', 'width', 'height'

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            name = data.get("name", "Unknown Format"),
            description = data.get("description", ""),
            frame_size = data.get("frame_size", [1.0, 1.0]),
            latent_sizes = data.get("latent_sizes", [])
        )

@dataclass
class HDCurve:
    name: str
    time: float
    temp: float
    curve_points: List[List[float]]

    @classmethod
    def from_dict(cls, data: dict):

        return cls(
            name=data.get("name", "Unknown Developer"),
            time=float(data.get("time", 0.0)),
            temp=float(data.get("temp", 20.0)),
            curve_points=data.get("curve_points", [])
        )

@dataclass
class FilmStock:
    id: str
    name: str
    description: str
    # Provide safe defaults
    weights: List[float] = field(default_factory=lambda: [0.33, 0.33, 0.33])
    luminosity_mask: List[float] = field(default_factory=lambda: [2.8, 1.1, 10.18, 0.0])
    params: Dict[str, float] = field(default_factory=lambda: {"slope": 1.8, "toe": 0.2, "shoulder": 0.8})
    # spectral_points is optional since some stocks in don't have it defined
    spectral_points: Optional[List[List[float]]] = None 
    # hd_curves contains H&D plot data for specific developers, times, and temps
    hd_curves: Optional[List[HDCurve]] = None

    @classmethod
    def from_dict(cls, data: dict):
        
        # Parse HD curves if they exist in the payload
        raw_hd_curves = data.get("hd_curves")
        parsed_hd_curves = [HDCurve.from_dict(c) for c in raw_hd_curves] if raw_hd_curves else None

        return cls(
            id=data.get("id", "unknown"),
            name=data.get("name", "Unknown Stock"),
            description=data.get("description", ""),
            weights=data.get("weights", [0.33, 0.33, 0.33]),
            luminosity_mask=data.get("luminosity_mask", [2.8, 1.1, 10.18, 0.0]),
            params=data.get("params", {"slope": 1.8, "toe": 0.2, "shoulder": 0.8}),
            spectral_points=data.get("spectral_points"),
            hd_curves=parsed_hd_curves
        )

@dataclass
class Camera:
    name: str
    illuminant_key: str
    film_stock: Optional[FilmStock] = None
    film_width: Optional[float] = None
    image: Optional[torch.Tensor] = None

    @classmethod
    def from_dict(cls, data: dict):

        return cls(
            name = data.get("name", "Unknown Camera"),
            illuminant_key = data.get("illuminant_key", "D65"),
            film_stock = None,  # To be set later when we have the FilmStock object
            film_width = None,  # To be set later based on user input
            image = None       # To be set later when we load the image tensor
        )

