"""Branch coverage for the pipeline, detector guards, and AWS dispatch fallback."""
import numpy as np
import pytest

from edgesentry import (
    AWSServerlessDispatcher, Detection, EdgeSentryPipeline, HOGPeopleDetector,
    estimate_hi_vis,
)
from edgesentry.detector import OnnxDetector
from edgesentry.synthetic import make_frame

ZONE = [(360, 320), (620, 320), (620, 520), (360, 520)]


class StubDetector:
    """A detector that returns pre-set detections — exercises the pipeline's detector branch."""
    def __init__(self, dets):
        self._dets = dets
    def detect(self, frame):
        return list(self._dets)


def test_pipeline_uses_injected_detector():
    dets = [Detection(id="p", bbox=(450, 300, 520, 520), has_hardhat=False, has_high_vis_vest=False)]
    pipe = EdgeSentryPipeline(detector=StubDetector(dets))
    pipe.register_zone("z", ZONE, "Cell")
    frame = make_frame(people=[{"bbox": (450, 300, 520, 520), "hi_vis": False}])
    r = pipe.process_frame(frame)  # no detections= arg → goes through self.detector
    assert len(r.detections) == 1 and len(r.violations) == 1


def test_process_frame_without_detector_raises():
    pipe = EdgeSentryPipeline()  # no detector
    with pytest.raises(RuntimeError):
        pipe.process_frame(make_frame())


def test_run_stream_requires_detector():
    pipe = EdgeSentryPipeline()
    with pytest.raises(RuntimeError):
        list(pipe.run_stream("nonexistent.mp4"))


def test_run_stream_bad_source_raises():
    pipe = EdgeSentryPipeline(detector=StubDetector([]))
    with pytest.raises(RuntimeError):
        list(pipe.run_stream("this_file_does_not_exist.mp4"))


def test_draw_zones_overlay_shape():
    pipe = EdgeSentryPipeline()
    pipe.register_zone("z", ZONE, "Cell")
    frame = make_frame()
    out = pipe.draw_zones(frame)
    assert out.shape == frame.shape and out.dtype == np.uint8


def test_hog_detector_guard_when_unavailable():
    import cv2
    if not hasattr(cv2, "HOGDescriptor"):
        with pytest.raises(RuntimeError):
            HOGPeopleDetector()


def test_onnx_detector_missing_model_raises():
    with pytest.raises(Exception):
        OnnxDetector("no_such_model.onnx")


def test_estimate_hi_vis_edges():
    # Empty ROI (degenerate bbox) → False, no crash.
    assert estimate_hi_vis(make_frame(), (10, 10, 10, 10)) is False


def test_dispatch_live_path_falls_back_without_creds():
    # Force non-mock but no real AWS → the dispatch try/except falls back to simulator.
    from edgesentry.hazard_detector import SpatialHazardDetector
    from edgesentry.temporal_filter import TemporalEventFilter
    det = SpatialHazardDetector(); det.register_zone("z", ZONE, "Cell")
    v = det.evaluate([Detection(id="a", bbox=(450, 300, 520, 520))])
    tf = TemporalEventFilter(persistence_threshold_s=1.0)
    tf.update(v, now=0.0)          # first sighting registers the track
    alerts = tf.update(v, now=2.0)  # persisted past threshold → fires
    assert len(alerts) == 1
    disp = AWSServerlessDispatcher(sns_topic_arn="arn:x", s3_bucket="b", region="us-east-1", force_mock=False)
    rec = disp.dispatch(alerts[0], frame=make_frame())
    assert rec.provider in ("aws", "simulator")  # boto3 absent/creds invalid → simulator


def test_run_stream_over_synthetic_video(tmp_path):
    """Write a tiny synthetic MP4 with cv2.VideoWriter, then run the real capture loop."""
    import cv2
    path = str(tmp_path / "clip.mp4")
    w, h = 640, 360
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    if not writer.isOpened():
        import pytest as _pt
        _pt.skip("no mp4v encoder available in this OpenCV build")
    for _ in range(4):
        writer.write(make_frame(width=w, height=h, people=[{"bbox": (300, 200, 360, 340), "hi_vis": True}]))
    writer.release()

    pipe = EdgeSentryPipeline(detector=StubDetector([Detection(id="p", bbox=(300, 200, 360, 340))]))
    pipe.register_zone("z", [(280, 180), (380, 180), (380, 350), (280, 350)], "Cell")
    out = str(tmp_path / "annotated.mp4")
    results = list(pipe.run_stream(path, max_frames=3, write_path=out))
    assert 1 <= len(results) <= 3
    assert all(r.annotated.shape[:2] == (h, w) for r in results)


def test_cli_demo_runs():
    from edgesentry.cli import main
    assert main(["--demo"]) == 0


def test_cli_demo_with_zone_arg():
    from edgesentry.cli import main
    assert main(["--demo", "--zone", "360,320;620,320;620,520;360,520"]) == 0


def test_cli_no_source_falls_back_to_demo():
    from edgesentry.cli import main
    assert main([]) == 0
