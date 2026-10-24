"""Human-readable frame overlay for the safety clip (what an operator sees).

Drawn causally: each frame shows only what was known at that instant — who is
tracked, how long they have been inside the zone, and whether the intrusion has
been verified by the temporal filter.
"""
from __future__ import annotations

import cv2
import numpy as np

from .hazard_detector import Detection

_GREEN = (90, 190, 60)
_AMBER = (0, 165, 255)
_RED = (50, 50, 225)
_WHITE = (255, 255, 255)
_FONT = cv2.FONT_HERSHEY_DUPLEX


def _chip(img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int], scale: float = 0.62,
          placed: list[tuple[int, int, int, int]] | None = None) -> None:
    """Filled label with its bottom-left near ``org``; nudged upward to avoid chips already ``placed``."""
    (tw, th), base = cv2.getTextSize(text, _FONT, scale, 1)
    x, y = org
    w, h = tw + 12, th + base + 8
    y = max(h, y)
    while placed is not None and any(x < px2 and x + w > px1 and y - h < py2 and y > py1 for px1, py1, px2, py2 in placed) and y - h > h:
        y -= h + 3
    cv2.rectangle(img, (x, y - h), (x + w, y), color, -1, cv2.LINE_AA)
    cv2.putText(img, text, (x + 6, y - base - 4), _FONT, scale, _WHITE, 1, cv2.LINE_AA)
    if placed is not None:
        placed.append((x, y - h, x + w, y))


def annotate_frame(frame: np.ndarray, zone: list[tuple[int, int]], zone_name: str, detections: list[Detection],
                   in_zone: set[str], dwell_s: dict[str, float], verified: set[str]) -> np.ndarray:
    """Return an annotated copy of ``frame`` (the input is never modified)."""
    out = frame.copy()
    poly = np.array(zone, np.int32)
    alarm = bool(in_zone & verified)
    overlay = out.copy()
    cv2.fillPoly(overlay, [poly], _RED if alarm else _AMBER)
    out = cv2.addWeighted(overlay, 0.22 if alarm else 0.10, out, 0.78 if alarm else 0.90, 0)
    cv2.polylines(out, [poly], True, _RED if alarm else _AMBER, 2, cv2.LINE_AA)
    hud = f"{zone_name}: INTRUSION VERIFIED" if alarm else (f"{zone_name}: person in zone" if in_zone else f"{zone_name}: clear")
    _chip(out, hud, (16, 44), _RED if alarm else (_AMBER if in_zone else _GREEN), 0.75)

    placed: list[tuple[int, int, int, int]] = []
    for det in detections:
        color = _RED if det.id in in_zone and det.id in verified else (_AMBER if det.id in in_zone else _GREEN)
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
    for det in sorted(detections, key=lambda d: d.bbox[0]):
        x1, y1, _, _ = det.bbox
        if det.id in in_zone and det.id in verified:
            color, label = _RED, f"{det.id}  verified {dwell_s.get(det.id, 0):.2f}s"
        elif det.id in in_zone:
            color, label = _AMBER, f"{det.id}  in zone {dwell_s.get(det.id, 0):.2f}s"
        else:
            color, label = _GREEN, det.id
        if det.label == "unknown":
            label += "  (shape?)"
        _chip(out, label, (x1 - 2, y1 - 4), color, placed=placed)
    return out
