"""Synthetic frame generator for offline demos and tests (no camera required)."""
from __future__ import annotations

import cv2
import numpy as np


def make_frame(width: int = 960, height: int = 540, people: list[dict] | None = None) -> np.ndarray:
    """Render a simple scene: grey floor + coloured 'people' rectangles.

    Each person dict: {"bbox": (x1,y1,x2,y2), "hi_vis": bool}.
    Hi-vis people are drawn in a yellow-green torso so the HSV PPE check detects it.
    """
    frame = np.full((height, width, 3), 60, dtype=np.uint8)
    cv2.rectangle(frame, (0, int(height * 0.6)), (width, height), (90, 90, 90), -1)  # floor
    for p in people or []:
        x1, y1, x2, y2 = p["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), -1)  # body
        if p.get("hi_vis"):
            th1, th2 = y1 + int(0.15 * (y2 - y1)), y1 + int(0.55 * (y2 - y1))
            cv2.rectangle(frame, (x1, th1), (x2, th2), (60, 230, 230), -1)  # hi-vis torso (BGR yellow-green)
    return frame
