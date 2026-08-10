"""CPU-only geometry helpers for 3D detection visualization."""

from __future__ import annotations

from math import cos, sin

import numpy as np

from laserperception.detection.types import Detection3D


def bev_corners(detection: Detection3D) -> np.ndarray:
    """Return four counter-clockwise XY corners for an oriented 3D box.

    The first corner is front-left when yaw is zero. Length is parallel to the
    box heading and width is perpendicular to it. The returned float64 array
    has shape ``(4, 2)`` and does not repeat its first point.
    """

    length, width, _ = detection.size_lwh
    local = np.array(
        [
            [length / 2.0, width / 2.0],
            [-length / 2.0, width / 2.0],
            [-length / 2.0, -width / 2.0],
            [length / 2.0, -width / 2.0],
        ],
        dtype=np.float64,
    )
    rotation = np.array(
        [
            [cos(detection.yaw_rad), -sin(detection.yaw_rad)],
            [sin(detection.yaw_rad), cos(detection.yaw_rad)],
        ],
        dtype=np.float64,
    )
    return local @ rotation.T + np.asarray(detection.center_xyz[:2], dtype=np.float64)
