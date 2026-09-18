"""Monocular spatial estimation via the pinhole camera model.

Approximates the Luxonis OAK-D stereo output (X, Y, Z in metres) from a 2D bbox so
proximity hazards can be flagged without a depth sensor. With a real stereo camera
these values come from the disparity map instead.
"""
from __future__ import annotations

from dataclasses import dataclass

from .hazard_detector import Detection


@dataclass
class Spatial3D:
    x: float
    y: float
    z: float
    unit: str = "meters"


class DepthEstimator:
    def __init__(self, image_width: int = 1920, image_height: int = 1080, focal_scale: float = 1800.0, proximity_hazard_m: float = 2.5) -> None:
        self.cx = image_width / 2
        self.cy = image_height / 2
        self.focal_scale = focal_scale
        self.proximity_hazard_m = proximity_hazard_m

    def estimate(self, det: Detection) -> Spatial3D:
        x1, y1, x2, y2 = det.bbox
        bbox_h = max(1, y2 - y1)
        z = max(0.8, round(self.focal_scale / max(50, bbox_h), 2))
        x = round(((x1 + x2) / 2 - self.cx) / 300.0, 2)
        y = round((y2 - self.cy) / 300.0, 2)
        return Spatial3D(x=x, y=y, z=z)

    def is_proximity_hazard(self, det: Detection) -> bool:
        return self.estimate(det).z < self.proximity_hazard_m
