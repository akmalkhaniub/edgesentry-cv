"""People / PPE detection built on OpenCV primitives.

Two backends, both real OpenCV:
- ``HOGPeopleDetector`` uses ``cv2.HOGDescriptor`` with the default people SVM.
- ``OnnxDetector`` runs an ONNX object detector through ``cv2.dnn`` when a model
  path is supplied.

Hi-vis PPE presence is estimated from the upper-body region with an HSV colour mask
(genuine ``cv2.cvtColor`` / ``cv2.inRange`` work), not a hard-coded flag.
"""
from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from .hazard_detector import Detection

# Hi-vis yellow-green / orange ranges in HSV.
_HIVIS_RANGES = [
    (np.array([25, 80, 80]), np.array([45, 255, 255])),   # yellow-green
    (np.array([5, 120, 120]), np.array([20, 255, 255])),  # orange
]


def estimate_hi_vis(frame: np.ndarray, bbox: tuple[int, int, int, int], min_ratio: float = 0.06) -> bool:
    """Return True if the torso region contains enough hi-vis colour."""
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    # Torso band ~ upper-middle of the person box.
    ty1, ty2 = y1 + int(0.15 * h), y1 + int(0.55 * h)
    roi = frame[max(0, ty1):max(1, ty2), max(0, x1):max(1, x2)]
    if roi.size == 0:
        return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in _HIVIS_RANGES:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    ratio = float(cv2.countNonZero(mask)) / float(mask.size)
    return ratio >= min_ratio


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class HOGPeopleDetector:
    """Real OpenCV HOG + linear-SVM pedestrian detector.

    Requires an OpenCV build that includes the ``objdetect`` module (HOGDescriptor).
    The slim ``opencv-python-headless`` OpenCV 5 wheel omits it, so this raises a
    clear error there — use :class:`OnnxDetector` (``cv2.dnn``) or supply detections.
    """

    def __init__(self, confidence_threshold: float = 0.0) -> None:
        if not hasattr(cv2, "HOGDescriptor"):
            raise RuntimeError(
                "cv2.HOGDescriptor is unavailable in this OpenCV build. Install a full "
                "OpenCV (objdetect module), use OnnxDetector, or pass detections explicitly."
            )
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.confidence_threshold = confidence_threshold

    def detect(self, frame: np.ndarray) -> list[Detection]:
        rects, weights = self.hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
        detections: list[Detection] = []
        for i, (x, y, w, h) in enumerate(rects):
            conf = float(weights[i]) if len(weights) > i else 1.0
            if conf < self.confidence_threshold:
                continue
            bbox = (int(x), int(y), int(x + w), int(y + h))
            hi_vis = estimate_hi_vis(frame, bbox)
            detections.append(
                Detection(
                    id=f"hog_{i}",
                    bbox=bbox,
                    label="person",
                    confidence=round(min(1.0, max(0.0, conf)), 3),
                    has_hardhat=hi_vis,          # proxy: crews wearing vests usually wear hardhats
                    has_high_vis_vest=hi_vis,
                )
            )
        return detections


class OnnxDetector:
    """ONNX object detector via ``cv2.dnn`` (e.g. a quantized YOLO person/PPE model)."""

    def __init__(self, model_path: str, input_size: int = 640, conf: float = 0.4, person_class: int = 0) -> None:
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.input_size = input_size
        self.conf = conf
        self.person_class = person_class

    def detect(self, frame: np.ndarray) -> list[Detection]:
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (self.input_size, self.input_size), swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()
        # Support the common YOLO layout (1, N, 85) -> (N, 85).
        preds = out[0] if out.ndim == 3 else out
        h, w = frame.shape[:2]
        sx, sy = w / self.input_size, h / self.input_size
        detections: list[Detection] = []
        for i, row in enumerate(preds):
            obj_conf = float(row[4])
            if obj_conf < self.conf:
                continue
            cx, cy, bw, bh = row[0] * sx, row[1] * sy, row[2] * sx, row[3] * sy
            bbox = (int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2))
            hi_vis = estimate_hi_vis(frame, bbox)
            detections.append(Detection(id=f"onnx_{i}", bbox=bbox, confidence=round(obj_conf, 3), has_hardhat=hi_vis, has_high_vis_vest=hi_vis))
        return detections
