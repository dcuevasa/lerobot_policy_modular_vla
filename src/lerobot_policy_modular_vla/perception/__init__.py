"""Perception, affordance grounding, and live overlay visualization module."""

from .types import BoundingBox, AffordanceDetection
from .detector import BaseAffordanceDetector, MockAffordanceDetector, YOLODetector, make_detector
from .conditioning import AffordanceConditioner
from .visualization import TelemetryVisualizer

__all__ = [
    "BoundingBox",
    "AffordanceDetection",
    "BaseAffordanceDetector",
    "MockAffordanceDetector",
    "YOLODetector",
    "make_detector",
    "AffordanceConditioner",
    "TelemetryVisualizer",
]
