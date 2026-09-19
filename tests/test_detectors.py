"""Model-free OpenCV detectors (contour + MOG2 motion) — no trained model needed."""
import numpy as np
from edgesentry import ContourDetector, MotionDetector, EdgeSentryPipeline


def _person_scene():
    frame = np.full((540, 960, 3), 60, np.uint8)
    frame[300:520, 450:520] = 200  # tall bright person 1
    frame[300:520, 100:170] = 200  # person 2
    return frame


def test_contour_detector_finds_people():
    dets = ContourDetector(min_area=500, bg_value=60, delta=25).detect(_person_scene())
    assert len(dets) == 2
    # People are taller than wide (aspect filter kept them).
    assert all((d.bbox[3] - d.bbox[1]) > (d.bbox[2] - d.bbox[0]) for d in dets)


def test_contour_detector_ignores_wide_floor_like_regions():
    frame = np.full((540, 960, 3), 60, np.uint8)
    frame[500:520, 0:960] = 200  # a wide, short strip (floor-like) → filtered by aspect
    assert ContourDetector(min_area=500).detect(frame) == []


def test_motion_detector_tracks_moving_blob():
    md = MotionDetector(min_area=300)
    res = []
    for f in range(5):
        fr = np.full((360, 640, 3), 60, np.uint8)
        fr[150:300, 100 + f * 20:170 + f * 20] = 220
        res = md.detect(fr)
    assert len(res) >= 1  # MOG2 has learned the background and flags the mover


def test_pipeline_with_contour_detector_end_to_end():
    pipe = EdgeSentryPipeline(detector=ContourDetector(min_area=500, bg_value=60, delta=25))
    pipe.register_zone("z", [(430, 280), (540, 280), (540, 540), (430, 540)], "Cell")
    r = pipe.process_frame(_person_scene())
    assert len(r.detections) == 2
    assert len(r.violations) >= 1  # person 1 is inside the zone


def test_cli_runs_stream_with_contour_detector(tmp_path):
    """End-to-end CLI over a real synthetic .mp4 using the model-free contour detector."""
    import cv2
    from edgesentry.cli import main
    path = str(tmp_path / "clip.mp4")
    w, h = 640, 360
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    if not writer.isOpened():
        import pytest; pytest.skip("no mp4v encoder")
    for _ in range(4):
        fr = np.full((h, w, 3), 60, np.uint8)
        fr[150:320, 300:360] = 200  # a person-shaped bright region
        writer.write(fr)
    writer.release()
    rc = main(["--source", path, "--detector", "contour", "--max-frames", "3",
               "--zone", "290,140;370,140;370,330;290,330"])
    assert rc == 0
