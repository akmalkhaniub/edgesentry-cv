"""Synthetic scenes for offline demos, tests, and evaluation (no camera required).

Two renderers:

* ``make_frame`` — the original minimal scene (kept for the CLI/bench). Colours are
  chosen so the model-free ``ContourDetector`` (background ≈ 60) actually separates
  people from the floor.
* ``render_scene`` / ``render_background`` — a fixed industrial-camera view of a
  robot work cell. ``render_background`` is the empty "clean plate" that
  ``PlateDiffDetector`` differences against, exactly as a real fixed safety camera
  would be calibrated.

``build_scenario`` produces a timed multi-worker sequence with ground truth so the
perception → decision → action loop can be evaluated, not just demoed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Minimal legacy scene
# ---------------------------------------------------------------------------


def make_frame(width: int = 960, height: int = 540, people: list[dict] | None = None) -> np.ndarray:
    """Render a simple scene: background ≈ 60, floor within detector tolerance, dark people.

    Each person dict: {"bbox": (x1,y1,x2,y2), "hi_vis": bool}.
    Hi-vis people get a yellow-green torso so the HSV PPE check detects it.
    """
    frame = np.full((height, width, 3), 60, dtype=np.uint8)
    cv2.rectangle(frame, (0, int(height * 0.6)), (width, height), (72, 72, 72), -1)  # floor: |72-60| < delta
    for p in people or []:
        x1, y1, x2, y2 = p["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (24, 24, 24), -1)  # body: |24-60| > delta
        if p.get("hi_vis"):
            th1, th2 = y1 + int(0.15 * (y2 - y1)), y1 + int(0.55 * (y2 - y1))
            cv2.rectangle(frame, (x1, th1), (x2, th2), (60, 230, 230), -1)  # hi-vis torso (BGR yellow-green)
    return frame


# ---------------------------------------------------------------------------
# Industrial work-cell scene (fixed camera)
# ---------------------------------------------------------------------------

SCENE_W, SCENE_H = 1280, 720
FLOOR_Y = 380
#: Restricted robot work cell painted on the floor (image coordinates).
CELL_ZONE = [(470, 420), (860, 420), (950, 700), (380, 700)]

_HIVIS = (60, 230, 230)       # BGR yellow-green vest
_HARDHAT = (0, 210, 255)      # BGR safety yellow
_BODY = (70, 42, 30)          # dark navy coverall
_SKIN = (120, 150, 190)
_REFLECTIVE = (215, 215, 215)


def render_background(width: int = SCENE_W, height: int = SCENE_H) -> np.ndarray:
    """The empty work cell: wall, concrete floor, painted hazard border, and the robot arm."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Wall: vertical gradient.
    for y in range(FLOOR_Y):
        v = 78 + int(22 * y / FLOOR_Y)
        frame[y, :] = (v + 6, v + 3, v)
    # Concrete floor with expansion joints.
    frame[FLOOR_Y:, :] = (112, 112, 108)
    for x in range(0, width, 160):
        cv2.line(frame, (x, FLOOR_Y), (x - 120, height), (100, 100, 96), 1)
    cv2.line(frame, (0, 540), (width, 540), (100, 100, 96), 1)

    # Hazard-striped border of the robot cell (yellow/black), drawn as a thick polyline.
    zone = np.array(CELL_ZONE, np.int32)
    cv2.polylines(frame, [zone], True, (0, 0, 0), 9)
    cv2.polylines(frame, [zone], True, (0, 200, 235), 5)

    # Industrial robot arm (static) at the back of the cell.
    base = (665, 455)
    cv2.rectangle(frame, (base[0] - 38, base[1] - 10), (base[0] + 38, base[1] + 18), (40, 70, 200), -1)
    cv2.line(frame, base, (610, 300), (40, 90, 230), 22)
    cv2.line(frame, (610, 300), (720, 235), (40, 90, 230), 18)
    cv2.circle(frame, (610, 300), 16, (30, 30, 30), -1)
    cv2.rectangle(frame, (715, 222), (748, 250), (60, 60, 60), -1)
    return frame


@dataclass
class WorkerPose:
    """One worker at one instant. ``vest`` is "full", "shadowed" (mostly in shadow), or "none"."""

    bbox: tuple[int, int, int, int]
    vest: str = "full"
    hardhat: bool = True
    shadow: float = 0.27  # luminance kept by a "shadowed" vest (lower = deeper shadow)


def draw_worker(frame: np.ndarray, pose: WorkerPose) -> None:
    x1, y1, x2, y2 = pose.bbox
    h, w = y2 - y1, x2 - x1
    head_r = max(8, int(0.07 * h))
    cx = (x1 + x2) // 2
    head_cy = y1 + head_r
    # Body / coverall below the head.
    cv2.rectangle(frame, (x1, y1 + 2 * head_r), (x2, y2), _BODY, -1)
    # Head.
    cv2.circle(frame, (cx, head_cy), head_r, _SKIN, -1)
    if pose.hardhat:
        cv2.ellipse(frame, (cx, head_cy - head_r // 3), (head_r + 3, head_r), 0, 180, 360, _HARDHAT, -1)
    # Vest across the torso band (0.15h .. 0.55h), with two reflective strips.
    ty1, ty2 = y1 + int(0.15 * h), y1 + int(0.55 * h)
    if pose.vest == "full":
        cv2.rectangle(frame, (x1, ty1), (x2, ty2), _HIVIS, -1)
    elif pose.vest == "shadowed":
        # Worker stands in the robot's shadow: the vest's luminance collapses, its hue does not.
        dim = tuple(int(c * pose.shadow) for c in _HIVIS)
        cv2.rectangle(frame, (x1, ty1), (x2, ty2), dim, -1)
    if pose.vest in ("full", "shadowed"):
        for frac in (0.32, 0.46):
            sy = y1 + int(frac * h)
            cv2.rectangle(frame, (x1, sy), (x2, sy + 2), _REFLECTIVE, -1)
    # Legs gap (gives the silhouette a person-like shape).
    cv2.rectangle(frame, (cx - 3, y1 + int(0.72 * h)), (cx + 3, y2), (112, 112, 108), -1)


def render_scene(poses: list[WorkerPose], background: np.ndarray | None = None, noise_sigma: float = 2.0, seed: int = 0) -> np.ndarray:
    """Composite workers onto the clean plate and add mild sensor noise."""
    frame = (background if background is not None else render_background()).copy()
    for pose in poses:
        draw_worker(frame, pose)
    if noise_sigma > 0:
        cv2.setRNGSeed(seed)
        noise = np.empty(frame.shape, np.int16)
        cv2.randn(noise, 0, noise_sigma)  # C-speed Gaussian sensor noise
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return frame


@lru_cache(maxsize=1)
def _background() -> np.ndarray:
    bg = render_background()
    bg.setflags(write=False)  # shared clean plate: read-only
    return bg


# ---------------------------------------------------------------------------
# Timed scenario with ground truth
# ---------------------------------------------------------------------------


@dataclass
class GroundTruthWorker:
    worker_id: str
    label: str
    vest: str
    hardhat: bool
    expected_action: str  # trigger_estop_alert | request_supervisor_approval | log_and_monitor
    poses: dict[int, tuple[int, int, int, int]] = field(default_factory=dict)  # frame index -> bbox


@dataclass
class Scenario:
    frames: list[np.ndarray]
    timestamps: list[float]
    background: np.ndarray
    zone: list[tuple[int, int]]
    workers: list[GroundTruthWorker]
    dt: float


@lru_cache(maxsize=8)
def build_scenario(n_frames: int = 24, dt: float = 0.25, seed: int = 7) -> Scenario:
    """A 6-second shift clip in which four people interact with the robot cell.

    * W-12  Operator without PPE stands inside the cell the whole time    → E-stop alert
    * W-07  Technician with full PPE walks in and stays (maintenance?)   → supervisor approval
    * W-03  Passer-by clips the cell corner for half a second            → log & monitor (transient)
    * W-15  Contractor whose vest is in the robot's shadow enters & stays → coarse PPE check is
            uncertain; re-inspection recovers the vest                   → supervisor approval
    """
    W, H = 70, 230
    workers = [
        GroundTruthWorker("W-12", "Operator (no PPE)", "none", False, "trigger_estop_alert"),
        GroundTruthWorker("W-07", "Technician (full PPE)", "full", True, "request_supervisor_approval"),
        GroundTruthWorker("W-03", "Passer-by", "full", True, "log_and_monitor"),
        GroundTruthWorker("W-15", "Contractor (vest in shadow)", "shadowed", True, "request_supervisor_approval"),
    ]
    by_id = {w.worker_id: w for w in workers}
    for f in range(n_frames):
        # W-12: static inside the right of the cell.
        by_id["W-12"].poses[f] = (780, 460, 780 + W, 460 + H)
        # W-07: walks in from the left, then stays.
        x = min(100 + 50 * f, 500)
        by_id["W-07"].poses[f] = (x, 460, x + W, 460 + H)
        # W-03: brushes the lower-left corner on frames 13-14 only.
        if 11 <= f <= 16:
            px = {11: 250, 12: 300, 13: 360, 14: 380, 15: 300, 16: 240}[f]
            by_id["W-03"].poses[f] = (px, 465, px + W, 465 + H)  # feet at y=695, just inside the cell edge
        # W-15: steps out from behind the robot at frame 16 and stays.
        if f >= 16:
            by_id["W-15"].poses[f] = (650, 455, 650 + W, 455 + H)

    return render_workers(workers, n_frames, dt, seed)


def render_workers(workers: list[GroundTruthWorker], n_frames: int, dt: float = 0.25, seed: int = 7,
                   shadow: float = 0.27, noise_sigma: float = 2.0) -> Scenario:
    """Render any ground-truth worker script into a timed clip (used by the scenario and the evaluation variants)."""
    background = _background()
    frames, timestamps = [], []
    for f in range(n_frames):
        poses = [WorkerPose(bbox=w.poses[f], vest=w.vest, hardhat=w.hardhat, shadow=shadow) for w in workers if f in w.poses]
        frames.append(render_scene(poses, background, noise_sigma=noise_sigma, seed=seed + f))
        timestamps.append(round(f * dt, 3))
    return Scenario(frames=frames, timestamps=timestamps, background=background, zone=list(CELL_ZONE), workers=workers, dt=dt)
