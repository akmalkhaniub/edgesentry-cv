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


#: Hard-hat colours (safety yellow / orange / white) in HSV.
_HARDHAT_RANGES = [
    (np.array([15, 120, 120]), np.array([35, 255, 255])),  # yellow
    (np.array([5, 120, 120]), np.array([15, 255, 255])),   # orange
    (np.array([0, 0, 200]), np.array([180, 40, 255])),     # white
]

#: Coarse PPE ratio bands used by the perception tool (fraction of the torso band).
PPE_MISSING_BELOW = 0.03
PPE_COMPLIANT_ABOVE = 0.12


def _band(frame: np.ndarray, bbox: tuple[int, int, int, int], top: float, bottom: float) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    by1, by2 = y1 + int(top * h), y1 + int(bottom * h)
    return frame[max(0, by1):max(1, by2), max(0, x1):max(1, x2)]


def _ratio(roi: np.ndarray, ranges) -> float:
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    return float(cv2.countNonZero(mask)) / float(mask.size)


def hi_vis_ratio(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    """Fraction of the torso band (0.15h–0.55h) that is lit hi-vis colour (coarse check)."""
    return _ratio(_band(frame, bbox, 0.15, 0.55), _HIVIS_RANGES)


def hardhat_ratio(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    """Fraction of the head band (top 16% of the box) that is hard-hat coloured."""
    return _ratio(_band(frame, bbox, 0.0, 0.16), _HARDHAT_RANGES)


def classify_ppe(ratio: float) -> str:
    """Map a coarse vest ratio to 'compliant' | 'missing' | 'uncertain'."""
    if ratio >= PPE_COMPLIANT_ABOVE:
        return "compliant"
    if ratio < PPE_MISSING_BELOW:
        return "missing"
    return "uncertain"


def estimate_hi_vis(frame: np.ndarray, bbox: tuple[int, int, int, int], min_ratio: float = 0.06) -> bool:
    """Return True if the torso region contains enough hi-vis colour."""
    return hi_vis_ratio(frame, bbox) >= min_ratio


def inspect_ppe(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> dict:
    """Active-perception re-inspection of one worker (the agent's ``inspect_worker`` tool).

    The coarse check needs a *lit* vest (V ≥ 80), so a vest in shadow can read as
    "uncertain". Here we look again, more carefully:

    1. crop the worker with a margin and upscale 2× (``cv2.resize``, cubic);
    2. test chromaticity only — hue + saturation, tolerating low value — so shadow
       does not hide a genuine hi-vis garment;
    3. clean the mask with a morphological open and measure the largest connected
       band (``connectedComponentsWithStats``): a vest spans the torso, a stray
       yellow patch does not.
    """
    x1, y1, x2, y2 = bbox
    h, w = y2 - y1, x2 - x1
    torso = _band(frame, bbox, 0.15, 0.55)
    if torso.size == 0 or w <= 0 or h <= 0:
        return {"vest_present": False, "vest_coverage": 0.0, "vest_span": 0.0, "hardhat_present": False, "method": "empty-roi"}
    up = cv2.resize(torso, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(up, cv2.COLOR_BGR2HSV)
    chroma = cv2.inRange(hsv, np.array([22, 70, 25]), np.array([48, 255, 255]))  # hue+sat, low V allowed
    chroma = cv2.morphologyEx(chroma, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    coverage = float(cv2.countNonZero(chroma)) / float(chroma.size)
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(chroma, connectivity=8)
    span = max((stats[i][cv2.CC_STAT_WIDTH] for i in range(1, n)), default=0) / float(chroma.shape[1])
    vest_present = coverage >= 0.35 and span >= 0.6
    return {
        "vest_present": bool(vest_present),
        "vest_coverage": round(coverage, 3),
        "vest_span": round(float(span), 3),
        "hardhat_present": hardhat_ratio(frame, bbox) >= 0.08,
        "method": "upscale2x + hue/sat chromaticity + morph-open + largest-component span",
    }


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

    def detect(self, frame: np.ndarray) -> list[Detection]:  # pragma: no cover - needs objdetect build
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

    def detect(self, frame: np.ndarray) -> list[Detection]:  # pragma: no cover - needs an ONNX model
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


class ContourDetector:
    """Model-free single-frame detector via foreground thresholding + connected components.

    Real OpenCV (cvtColor / threshold / morphologyEx / connectedComponentsWithStats). Detects
    people/objects that stand out from the background — no trained model or objdetect module
    needed. Tune `bg_value` to the scene's floor/background intensity.
    """

    def __init__(self, min_area: int = 800, min_aspect: float = 1.2, bg_value: int = 60, delta: int = 25) -> None:
        self.min_area = min_area
        self.min_aspect = min_aspect  # height/width — people are taller than wide
        self.bg_value = bg_value
        self.delta = delta

    def detect(self, frame: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Foreground = pixels that differ from the background intensity.
        fg = (np.abs(gray.astype(np.int16) - self.bg_value) > self.delta).astype(np.uint8) * 255
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, _labels, stats, _cent = cv2.connectedComponentsWithStats(fg, connectivity=8)
        detections: list[Detection] = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < self.min_area or w == 0 or h / w < self.min_aspect:
                continue
            bbox = (int(x), int(y), int(x + w), int(y + h))
            hi_vis = estimate_hi_vis(frame, bbox)
            detections.append(Detection(id=f"cc_{i}", bbox=bbox, confidence=1.0, has_hardhat=hi_vis, has_high_vis_vest=hi_vis))
        return detections


class MotionDetector:
    """Model-free streaming detector via MOG2 background subtraction (stateful across frames)."""

    def __init__(self, min_area: int = 800, history: int = 100, var_threshold: float = 40.0) -> None:
        self.min_area = min_area
        self.bg = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold, detectShadows=False)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        mask = self.bg.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []
        for i, c in enumerate(contours):
            if cv2.contourArea(c) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            bbox = (int(x), int(y), int(x + w), int(y + h))
            hi_vis = estimate_hi_vis(frame, bbox)
            detections.append(Detection(id=f"mog_{i}", bbox=bbox, confidence=1.0, has_hardhat=hi_vis, has_high_vis_vest=hi_vis))
        return detections


class PlateDiffDetector:
    """Fixed-camera person detector via clean-plate differencing (no trained model).

    A fixed industrial safety camera sees a static scene, so the empty "clean plate"
    captured at install time is a strong prior. Pipeline (all OpenCV 5):
    ``absdiff`` vs. the plate → grayscale → Gaussian blur → Otsu threshold (with a
    floor so pure sensor noise never passes) → morphological open/close →
    ``connectedComponentsWithStats`` → area + aspect filtering (people are taller
    than wide). Works on real fixed-camera footage given an empty-scene plate.

    Robustness steps before differencing (each found necessary by scripts/evaluate.py):

    * **Registration** — ``phaseCorrelate`` estimates sub-pixel camera drift against the
      plate (vibration, a bumped mount) and ``warpAffine`` undoes it; without this a
      3 px bump turns every edge in the scene into foreground.
    * **Illumination compensation** — the median per-channel offset between frame and
      plate (people cover a small fraction of the view) is removed, so a global lighting
      change does not swamp the difference image.
    * **Misalignment-tolerant differencing** — a pixel only counts as changed if it falls
      outside the [``erode``, ``dilate``] envelope of the plate's 5x5 neighbourhood, so a
      residual sub-pixel shift does not light up high-contrast edges (hazard stripes).
    * **Merged-blob splitting** — two people shoulder to shoulder form one wide blob; it
      is split between heads found on the silhouette's top profile.

    ``health`` reports what was corrected on the last frame (drift, lighting offset,
    foreground fraction) so the system can say when the camera needs attention.
    """

    def __init__(self, background: np.ndarray, min_area: int = 2500, min_aspect: float = 1.6, min_thresh: int = 18,
                 register: bool = True, compensate_light: bool = True, split_merged: bool = True, realign_every: int = 4) -> None:
        self.background = background
        self.min_area = min_area
        self.min_aspect = min_aspect
        self.min_thresh = min_thresh
        self.register, self.compensate_light, self.split_merged = register, compensate_light, split_merged
        self.realign_every = realign_every
        self._warp: np.ndarray | None = None
        self._align_score, self._frames_since_pose = 1.0, 0
        envelope = np.ones((5, 5), np.uint8)
        self._plate_lo, self._plate_hi = cv2.erode(background, envelope), cv2.dilate(background, envelope)
        small = cv2.pyrDown(cv2.cvtColor(background, cv2.COLOR_BGR2GRAY))
        self._plate_small = small.astype(np.float32)
        self._plate_quarter = cv2.pyrDown(small).astype(np.float32)
        self._plate_q_lo = cv2.erode(self._plate_quarter, np.ones((3, 3), np.uint8))
        self._plate_q_hi = cv2.dilate(self._plate_quarter, np.ones((3, 3), np.uint8))
        self._window = cv2.createHanningWindow((small.shape[1], small.shape[0]), cv2.CV_32F)
        self.health: dict = {"shift_px": (0.0, 0.0), "rotation_deg": 0.0, "alignment_score": 1.0,
                             "light_offset": 0.0, "foreground_fraction": 0.0}

    def estimate_camera_pose(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """Euclidean warp (full resolution) mapping plate → frame, plus ECC's correlation score.

        ``phaseCorrelate`` (half resolution) gives a robust translation even for large bumps;
        ``findTransformECC`` (quarter resolution, Euclidean) refines it and adds rotation,
        warm-started from the previous rotation estimate.
        """
        small = cv2.pyrDown(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)).astype(np.float32)
        quarter = cv2.pyrDown(small)
        # People must not drive the camera estimate: mask out large changed blobs (outside the
        # plate's 3x3 envelope, surviving an opening), keep thin misalignment edges, which carry
        # the alignment signal.
        changed = (cv2.max(self._plate_q_lo - quarter, 0) + cv2.max(quarter - self._plate_q_hi, 0)) > 30
        changed = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        background = (cv2.dilate(changed, np.ones((5, 5), np.uint8)) == 0).astype(np.uint8)

        # Copies: in OpenCV 5.0.0 phaseCorrelate applies the window to its inputs *in place*,
        # which would progressively erase the stored plate.
        (dx, dy), _response = cv2.phaseCorrelate(self._plate_small.copy(), small.copy(), self._window)
        identity = np.eye(2, 3, dtype=np.float32)
        init = np.array([[1, 0, dx / 2], [0, 1, dy / 2]], np.float32)  # quarter-resolution units
        if self._warp is not None:
            init[:, :2] = self._warp[:, :2]

        def cc(warp: np.ndarray) -> float:
            moved = cv2.warpAffine(quarter, warp, quarter.shape[::-1], flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            return float(cv2.computeECC(self._plate_quarter, moved, background))

        base = cc(identity)
        start = init if cc(init) > base else identity
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-4)
        try:
            score, warp = cv2.findTransformECC(self._plate_quarter, quarter, start.copy(), cv2.MOTION_EUCLIDEAN, criteria, background, 3)
        except cv2.error:  # ECC did not converge
            score, warp = base, identity
        if score <= base + 1e-3:  # the camera has not measurably moved
            score, warp = base, identity
        full = warp.copy()
        full[:, 2] *= 4.0
        return full, float(score)

    def align(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, dict]:
        """Undo camera motion relative to the plate; the pose is re-estimated every ``realign_every`` frames.

        Returns (aligned frame, full-resolution warp or None if negligible, health fields).
        """
        if self._warp is None or self._frames_since_pose >= self.realign_every:
            self._warp, self._align_score = self.estimate_camera_pose(frame)
            self._frames_since_pose = 0
        self._frames_since_pose += 1
        full = self._warp
        angle = float(np.degrees(np.arctan2(full[1, 0], full[0, 0])))
        fields = {"shift_px": (round(float(full[0, 2]), 2), round(float(full[1, 2]), 2)), "rotation_deg": round(angle, 3),
                  "alignment_score": round(self._align_score, 3)}
        if abs(full[0, 2]) < 1.5 and abs(full[1, 2]) < 1.5 and abs(angle) < 0.05:  # within the 5x5 envelope tolerance
            return frame, None, {**fields, "shift_px": (0.0, 0.0), "rotation_deg": 0.0}
        h, w = frame.shape[:2]
        aligned = cv2.warpAffine(frame, full, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        return aligned, full, fields

    def normalize_light(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """Remove the global per-channel brightness offset between the frame and the plate."""
        sample = (frame.reshape(-1, 3)[::97].astype(np.int16) - self.background.reshape(-1, 3)[::97].astype(np.int16))
        offset = np.median(sample, axis=0)
        if np.all(np.abs(offset) < 4):
            return frame, 0.0
        corrected = np.clip(frame.astype(np.int16) - offset.astype(np.int16), 0, 255).astype(np.uint8)
        return corrected, float(np.mean(offset))

    def foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        warp, align_fields = None, {"shift_px": (0.0, 0.0), "rotation_deg": 0.0, "alignment_score": 1.0}
        if self.register:
            frame, warp, align_fields = self.align(frame)
        light = 0.0
        if self.compensate_light:
            frame, light = self.normalize_light(frame)
        # Distance outside the plate's local [min, max] envelope (0 inside it), per channel.
        diff = cv2.add(cv2.subtract(self._plate_lo, frame), cv2.subtract(frame, self._plate_hi))
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, mask = cv2.threshold(gray, max(float(otsu), float(self.min_thresh)), 255, cv2.THRESH_BINARY)
        if warp is not None:  # the replicated border band after warping is not real foreground
            h, w = mask.shape
            corners = np.float32([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]])
            band = int(np.ceil(np.abs(corners @ warp.T - corners[:, :2]).max())) + 2
            mask[:band, :] = 0
            mask[-band:, :] = 0
            mask[:, :band] = 0
            mask[:, -band:] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        self.health = {**align_fields, "light_offset": round(light, 1),
                       "foreground_fraction": round(float(np.count_nonzero(mask)) / mask.size, 4)}
        return mask

    def _split_heads(self, component: np.ndarray, x: int, y: int) -> list[tuple[int, int, int, int]]:
        """Split a group blob (people shoulder to shoulder) into one box per head.

        Reads the silhouette's top profile (first foreground row per column): heads are runs
        of columns near the top of the blob, separated by lower shoulder gaps; each gap that
        dips at least ``max(6, 0.06h)`` below the head line becomes a cut. Handles any number
        of people in one pass (convex-hull defects miss a middle head that is collinear with
        its neighbours).
        """
        h, w = component.shape
        filled = component.any(axis=0)
        top = np.where(filled, component.argmax(axis=0), h)
        head = top <= 0.12 * h
        edges = np.flatnonzero(np.diff(np.concatenate(([0], head.astype(np.int8), [0]))))
        runs = [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= max(3, int(0.03 * h))]
        cuts = [(a1 + b0) // 2 for (_a0, a1), (b0, _b1) in zip(runs, runs[1:])
                if top[a1:b0].min() >= max(6, int(0.06 * h))]
        bounds = [0, *cuts, w]
        boxes = []
        for c0, c1 in zip(bounds, bounds[1:]):
            ys, xs = np.nonzero(component[:, c0:c1])
            if len(xs):
                boxes.append((x + c0 + int(xs.min()), y + int(ys.min()), x + c0 + int(xs.max()) + 1, y + int(ys.max()) + 1))
        return boxes or [(x, y, x + w, y + h)]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """People (upright, ~3x taller than wide) plus fail-safe "unknown" detections.

        A large changed region that does not look like an upright person (someone kneeling,
        an object left in the cell, an unsplittable group) is still reported, labelled
        ``unknown``: a safety system must never silently drop unexplained presence.
        """
        mask = self.foreground_mask(frame)
        n, labels, stats, _cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
        detections: list[Detection] = []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area < self.min_area or w == 0:
                continue
            boxes = [(x, y, x + w, y + h)]
            if self.split_merged and h / w < 2.4:  # one upright person is ~3x taller than wide
                component = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
                boxes = self._split_heads(component, x, y)
            for k, bbox in enumerate(boxes):
                bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
                upright = bh / max(1, bw) >= self.min_aspect
                ratio = hi_vis_ratio(frame, bbox)
                detections.append(Detection(
                    id=f"pd_{i}_{k}", bbox=bbox, label="person" if upright else "unknown",
                    confidence=1.0 if upright else 0.5,
                    has_hardhat=hardhat_ratio(frame, bbox) >= 0.08,
                    has_high_vis_vest=ratio >= PPE_COMPLIANT_ABOVE,
                ))
        return detections


class CentroidTracker:
    """Assigns stable track IDs across frames by nearest-centroid matching.

    The temporal filter keys hazards on entity ID, so per-frame component indices
    would break persistence. Unmatched detections start new tracks; tracks unseen
    for ``max_missed`` frames are retired.
    """

    def __init__(self, max_distance: float = 90.0, max_missed: int = 4) -> None:
        self.max_distance = max_distance
        self.max_missed = max_missed
        self._next = 1
        self._tracks: dict[str, tuple[float, float]] = {}
        self._missed: dict[str, int] = {}

    @staticmethod
    def _centroid(det: Detection) -> tuple[float, float]:
        x1, y1, x2, y2 = det.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def update(self, detections: list[Detection]) -> list[Detection]:
        unmatched = set(self._tracks)
        assigned: list[Detection] = []
        # Greedy matching on the globally closest pairs first (deterministic).
        pairs = sorted(
            ((np.hypot(c[0] - t[0], c[1] - t[1]), di, tid)
             for di, det in enumerate(detections)
             for tid, t in self._tracks.items()
             for c in [self._centroid(det)]),
            key=lambda p: (p[0], p[1], p[2]),
        )
        det_to_track: dict[int, str] = {}
        for dist, di, tid in pairs:
            if dist > self.max_distance or di in det_to_track or tid not in unmatched:
                continue
            det_to_track[di] = tid
            unmatched.discard(tid)
        for di, det in enumerate(detections):
            tid = det_to_track.get(di)
            if tid is None:
                tid = f"T{self._next}"
                self._next += 1
            self._tracks[tid] = self._centroid(det)
            self._missed[tid] = 0
            det.id = tid
            assigned.append(det)
        for tid in unmatched:
            self._missed[tid] = self._missed.get(tid, 0) + 1
            if self._missed[tid] > self.max_missed:
                self._tracks.pop(tid, None)
                self._missed.pop(tid, None)
        return assigned
