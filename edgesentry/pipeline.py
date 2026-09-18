"""EdgeSentry pipeline: capture -> detect -> zone eval -> temporal filter -> annotate -> dispatch."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

from .aws_dispatch import AWSServerlessDispatcher, DispatchRecord
from .depth_estimator import DepthEstimator
from .detector import Detector, HOGPeopleDetector
from .hazard_detector import Detection, SpatialHazardDetector, Violation
from .temporal_filter import TemporalEventFilter, VerifiedAlert

_ZONE_COLOR = (0, 0, 255)
_OK_COLOR = (0, 200, 0)
_ALERT_COLOR = (0, 0, 255)


@dataclass
class FrameResult:
    detections: list[Detection]
    violations: list[Violation]
    verified: list[VerifiedAlert]
    dispatched: list[DispatchRecord]
    annotated: np.ndarray


class EdgeSentryPipeline:
    def __init__(
        self,
        detector: Detector | None = None,
        hazard: SpatialHazardDetector | None = None,
        temporal: TemporalEventFilter | None = None,
        depth: DepthEstimator | None = None,
        dispatcher: AWSServerlessDispatcher | None = None,
    ) -> None:
        # Detector is optional: supply an OnnxDetector/HOGPeopleDetector for live
        # frames, or drive process_frame() with explicit detections=.
        self.detector = detector
        self.hazard = hazard or SpatialHazardDetector()
        self.temporal = temporal or TemporalEventFilter()
        self.depth = depth or DepthEstimator()
        self.dispatcher = dispatcher or AWSServerlessDispatcher(force_mock=True)

    def register_zone(self, zone_id: str, vertices: Sequence[Sequence[float]], name: str = "Restricted Zone") -> None:
        self.hazard.register_zone(zone_id, vertices, name)

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        for zone in self.hazard.zones:
            cv2.fillPoly(overlay, [zone.polygon], _ZONE_COLOR)
            cv2.polylines(frame, [zone.polygon], isClosed=True, color=_ZONE_COLOR, thickness=2)
            cv2.putText(frame, zone.name, tuple(zone.polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _ZONE_COLOR, 2)
        return cv2.addWeighted(overlay, 0.20, frame, 0.80, 0)

    def process_frame(self, frame: np.ndarray, now: float | None = None, detections: Sequence[Detection] | None = None) -> FrameResult:
        if detections is not None:
            dets = list(detections)
        elif self.detector is not None:
            dets = self.detector.detect(frame)
        else:
            raise RuntimeError("No detector configured; pass detections= or construct with a detector.")
        violations = self.hazard.evaluate(dets)
        verified = self.temporal.update(violations, now=now)

        annotated = self.draw_zones(frame)
        violating_ids = {v.entity_id for v in violations}
        for det in dets:
            x1, y1, x2, y2 = det.bbox
            color = _ALERT_COLOR if det.id in violating_ids else _OK_COLOR
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            z = self.depth.estimate(det).z
            label = f"{det.label} {det.confidence:.2f} z={z}m{'' if det.has_high_vis_vest else ' NO-PPE'}"
            cv2.putText(annotated, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        dispatched = [self.dispatcher.dispatch(a, annotated) for a in verified]
        return FrameResult(detections=dets, violations=violations, verified=verified, dispatched=dispatched, annotated=annotated)

    def run_stream(self, source: str | int, max_frames: int | None = None, write_path: str | None = None) -> Iterable[FrameResult]:
        """Process a real video source (file path, RTSP URL, or camera index) via cv2.VideoCapture."""
        if self.detector is None:
            raise RuntimeError("run_stream requires a detector (OnnxDetector or HOGPeopleDetector).")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source!r}")
        writer = None
        try:
            count = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                result = self.process_frame(frame)
                if write_path:
                    if writer is None:
                        h, w = result.annotated.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(write_path, fourcc, cap.get(cv2.CAP_PROP_FPS) or 20.0, (w, h))
                    writer.write(result.annotated)
                yield result
                count += 1
                if max_frames is not None and count >= max_frames:
                    break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
