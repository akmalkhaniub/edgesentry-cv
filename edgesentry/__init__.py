"""EdgeSentry — edge computer-vision safety monitoring on OpenCV with AWS serverless dispatch."""
from .aws_dispatch import AWSServerlessDispatcher, DispatchRecord
from .depth_estimator import DepthEstimator, Spatial3D
from .detector import HOGPeopleDetector, OnnxDetector, estimate_hi_vis
from .hazard_detector import Detection, SpatialHazardDetector, Violation, Zone
from .pipeline import EdgeSentryPipeline, FrameResult
from .temporal_filter import TemporalEventFilter, VerifiedAlert

__all__ = [
    "AWSServerlessDispatcher",
    "DispatchRecord",
    "DepthEstimator",
    "Spatial3D",
    "HOGPeopleDetector",
    "OnnxDetector",
    "estimate_hi_vis",
    "Detection",
    "SpatialHazardDetector",
    "Violation",
    "Zone",
    "EdgeSentryPipeline",
    "FrameResult",
    "TemporalEventFilter",
    "VerifiedAlert",
]

__version__ = "1.0.0"
