"""EdgeSentry test suite — real OpenCV, no camera required (synthetic frames)."""
import cv2
import numpy as np
import pytest

from edgesentry import (
    AWSServerlessDispatcher,
    DepthEstimator,
    Detection,
    EdgeSentryPipeline,
    HOGPeopleDetector,
    SpatialHazardDetector,
    TemporalEventFilter,
    estimate_hi_vis,
)
from edgesentry.synthetic import make_frame

ZONE = [(360, 320), (620, 320), (620, 520), (360, 520)]


def test_point_in_polygon_uses_cv2():
    det = SpatialHazardDetector()
    det.register_zone("z", ZONE, "Cell")
    inside = Detection(id="a", bbox=(450, 300, 520, 520))   # foot (485, 520) on edge/inside
    outside = Detection(id="b", bbox=(100, 300, 170, 400))  # foot (135, 400) far left
    v = det.evaluate([inside, outside])
    assert len(v) == 1
    assert v[0].entity_id == "a"


def test_ppe_classification():
    det = SpatialHazardDetector()
    det.register_zone("z", ZONE, "Cell")
    v = det.evaluate([Detection(id="a", bbox=(450, 300, 520, 520), has_hardhat=False, has_high_vis_vest=False)])
    assert v[0].violation_type == "ZONE_INTRUSION_MISSING_PPE"
    v2 = det.evaluate([Detection(id="c", bbox=(450, 300, 520, 520), has_hardhat=True, has_high_vis_vest=True)])
    assert v2[0].violation_type == "UNAUTHORIZED_ZONE_INTRUSION"


def test_temporal_filter_debounces():
    tf = TemporalEventFilter(persistence_threshold_s=1.5)
    det = SpatialHazardDetector()
    det.register_zone("z", ZONE, "Cell")
    viol = det.evaluate([Detection(id="a", bbox=(450, 300, 520, 520))])
    assert tf.update(viol, now=0.0) == []       # first sighting -> no alert yet
    assert tf.update(viol, now=0.5) == []       # still within threshold
    fired = tf.update(viol, now=2.0)            # persisted -> verified
    assert len(fired) == 1
    assert fired[0].status == "VERIFIED_PERSISTENT_HAZARD"
    assert tf.update(viol, now=2.5) == []       # does not re-fire


def test_temporal_eviction():
    tf = TemporalEventFilter(persistence_threshold_s=1.0, evict_after_s=2.0)
    det = SpatialHazardDetector()
    det.register_zone("z", ZONE, "Cell")
    viol = det.evaluate([Detection(id="a", bbox=(450, 300, 520, 520))])
    tf.update(viol, now=0.0)
    assert tf.active_track_count == 1
    tf.update([], now=5.0)  # gone long enough -> evicted
    assert tf.active_track_count == 0


def test_depth_estimator_proximity():
    depth = DepthEstimator(proximity_hazard_m=2.5)
    near = Detection(id="n", bbox=(0, 0, 100, 900))   # tall bbox -> close
    far = Detection(id="f", bbox=(0, 0, 60, 120))     # short bbox -> far
    assert depth.is_proximity_hazard(near) is True
    assert depth.is_proximity_hazard(far) is False


def test_hi_vis_hsv_detection():
    frame = make_frame(people=[{"bbox": (450, 300, 520, 520), "hi_vis": True}])
    assert estimate_hi_vis(frame, (450, 300, 520, 520)) is True
    plain = make_frame(people=[{"bbox": (100, 300, 170, 520), "hi_vis": False}])
    assert estimate_hi_vis(plain, (100, 300, 170, 520)) is False


@pytest.mark.skipif(not hasattr(cv2, "HOGDescriptor"), reason="OpenCV build lacks objdetect/HOGDescriptor")
def test_hog_detector_runs_on_frame():
    frame = make_frame(people=[{"bbox": (450, 300, 520, 520), "hi_vis": True}])
    dets = HOGPeopleDetector().detect(frame)
    assert isinstance(dets, list)  # HOG may or may not fire on synthetic art; must not crash


def test_pipeline_end_to_end_dispatch():
    pipeline = EdgeSentryPipeline(dispatcher=AWSServerlessDispatcher(force_mock=True))
    pipeline.register_zone("zone_a", ZONE, "Robot Cell A")
    dets = [Detection(id="p_intruder", bbox=(450, 300, 520, 520), has_hardhat=False, has_high_vis_vest=False)]
    frame = make_frame(people=[{"bbox": (450, 300, 520, 520), "hi_vis": False}])
    r0 = pipeline.process_frame(frame, now=0.0, detections=dets)
    assert len(r0.violations) == 1 and len(r0.verified) == 0
    r1 = pipeline.process_frame(frame, now=2.0, detections=dets)
    assert len(r1.verified) == 1
    assert len(r1.dispatched) == 1
    assert r1.dispatched[0].provider == "simulator"
    assert r1.annotated.shape == frame.shape and r1.annotated.dtype == np.uint8
