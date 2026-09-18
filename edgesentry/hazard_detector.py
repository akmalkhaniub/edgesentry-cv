"""Polygonal restricted-zone evaluation using real OpenCV ``cv2.pointPolygonTest``."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np


@dataclass
class Detection:
    """A single detected entity in a frame.

    bbox is (x1, y1, x2, y2) in pixel coordinates.
    """

    id: str
    bbox: tuple[int, int, int, int]
    label: str = "person"
    confidence: float = 0.9
    has_hardhat: bool = True
    has_high_vis_vest: bool = True

    @property
    def foot_point(self) -> tuple[int, int]:
        x1, _y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int(y2))  # bottom-center = feet on ground plane


@dataclass
class Zone:
    zone_id: str
    name: str
    polygon: np.ndarray  # int32 array of shape (N, 2)


@dataclass
class Violation:
    entity_id: str
    zone_id: str
    zone_name: str
    foot_point: tuple[int, int]
    is_ppe_compliant: bool
    violation_type: str
    distance_px: float
    timestamp: float = field(default_factory=time.time)


class SpatialHazardDetector:
    """Registers restricted polygonal zones and flags intrusions.

    Point-in-polygon uses ``cv2.pointPolygonTest`` (the OpenCV primitive), so the
    geometry matches what a production OpenCV pipeline would compute.
    """

    def __init__(self) -> None:
        self._zones: dict[str, Zone] = {}

    def register_zone(self, zone_id: str, vertices: Sequence[Sequence[float]], name: str = "Restricted Hazard Perimeter") -> Zone:
        if len(vertices) < 3:
            raise ValueError("A valid polygon requires at least 3 vertices.")
        polygon = np.asarray(vertices, dtype=np.int32).reshape(-1, 2)
        zone = Zone(zone_id=zone_id, name=name, polygon=polygon)
        self._zones[zone_id] = zone
        return zone

    @property
    def zones(self) -> list[Zone]:
        return list(self._zones.values())

    def evaluate(self, detections: Sequence[Detection]) -> list[Violation]:
        violations: list[Violation] = []
        for det in detections:
            fx, fy = det.foot_point
            for zone in self._zones.values():
                # >= 0 means on the edge or inside; measureDist=True returns signed distance.
                dist = cv2.pointPolygonTest(zone.polygon, (float(fx), float(fy)), True)
                if dist >= 0:
                    compliant = det.has_hardhat and det.has_high_vis_vest
                    violations.append(
                        Violation(
                            entity_id=det.id,
                            zone_id=zone.zone_id,
                            zone_name=zone.name,
                            foot_point=(fx, fy),
                            is_ppe_compliant=compliant,
                            violation_type="UNAUTHORIZED_ZONE_INTRUSION" if compliant else "ZONE_INTRUSION_MISSING_PPE",
                            distance_px=float(dist),
                        )
                    )
        return violations
