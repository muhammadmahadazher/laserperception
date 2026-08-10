"""Framework-independent 3D detection results and geometry."""

from laserperception.detection.geometry import bev_corners
from laserperception.detection.types import Detection3D, DetectionFrame

__all__ = ["Detection3D", "DetectionFrame", "bev_corners"]
