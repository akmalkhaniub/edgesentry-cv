"""Perception robustness: camera motion, lighting drift, merged blobs, and fail-safe shapes."""
import cv2
import numpy as np

from edgesentry.agent import perceive, run_agent
from edgesentry.detector import PlateDiffDetector
from edgesentry.synthetic import GroundTruthWorker, build_scenario, render_workers


def _workers(*specs):
    """specs: (worker_id, vest, bbox) — static for 24 frames."""
    out = []
    for wid, vest, bbox in specs:
        w = GroundTruthWorker(wid, wid, vest, vest != "none", "")
        w.poses = {f: bbox for f in range(24)}
        out.append(w)
    return render_workers(out, 24)


def test_plate_is_not_mutated_by_phase_correlation():
    # OpenCV 5.0.0's phaseCorrelate windows its inputs in place; the detector must pass copies.
    sc = build_scenario()
    det = PlateDiffDetector(sc.background, realign_every=1)
    before = det._plate_small.copy()
    for f in sc.frames[:5]:
        det.detect(f)
    assert np.array_equal(det._plate_small, before)


def test_recovers_camera_rotation_and_translation():
    sc = build_scenario()
    frame = sc.frames[10]
    rotated = cv2.warpAffine(frame, cv2.getRotationMatrix2D((640, 360), 1.0, 1.0), (1280, 720), borderMode=cv2.BORDER_REPLICATE)
    warp, score = PlateDiffDetector(sc.background).estimate_camera_pose(rotated)
    assert abs(abs(np.degrees(np.arctan2(warp[1, 0], warp[0, 0]))) - 1.0) < 0.3 and score > 0.5
    shifted = cv2.warpAffine(frame, np.float32([[1, 0, 8], [0, 1, 4]]), (1280, 720), borderMode=cv2.BORDER_REPLICATE)
    det = PlateDiffDetector(sc.background)
    boxes = sorted(d.bbox for d in det.detect(shifted))
    assert len(boxes) == 2 and det.health["shift_px"] != (0.0, 0.0)
    assert abs(boxes[0][0] - 500) <= 3  # boxes land back in plate (zone) coordinates


def test_lighting_offset_is_compensated():
    sc = build_scenario()
    bright = cv2.add(sc.frames[10], np.full_like(sc.frames[10], 30))
    det = PlateDiffDetector(sc.background)
    assert len(det.detect(bright)) == 2 and abs(det.health["light_offset"] - 30) < 2
    plain = PlateDiffDetector(sc.background, register=False, compensate_light=False)
    plain.detect(bright)
    assert plain.health["light_offset"] == 0.0


def test_merged_groups_are_split_per_head():
    pair = _workers(("A", "none", (700, 460, 770, 690)), ("B", "full", (772, 460, 842, 690)))
    assert len(PlateDiffDetector(pair.background).detect(pair.frames[0])) == 2
    trio = _workers(*[(f"W{k}", "full", (640 + 72 * k, 460, 710 + 72 * k, 690)) for k in range(3)])
    dets = PlateDiffDetector(trio.background).detect(trio.frames[0])
    assert len(dets) == 3 and all(d.label == "person" for d in dets)
    merged = PlateDiffDetector(pair.background, split_merged=False).detect(pair.frames[0])
    assert len(merged) == 1


def test_unrecognized_shape_is_never_dropped_and_gets_stopped():
    sc = _workers(("K", "none", (700, 570, 800, 690)))  # kneeling: wider than tall
    dets = PlateDiffDetector(sc.background).detect(sc.frames[0])
    assert len(dets) == 1 and dets[0].label == "unknown"
    perception = perceive(sc.frames, sc.timestamps, sc.background, sc.zone)
    (tr,) = perception.tracks.values()
    assert not tr.upright and tr.public()["shape"].startswith("unrecognized")
    run, _ = run_agent(perception, prefer="rules")
    assert [d["action"] for d in run.decisions.values()] == ["trigger_estop_alert"]
