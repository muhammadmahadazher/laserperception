"""Small, validated detection result types owned by LaserPerception."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any


def _finite_tuple(values: Sequence[float], *, length: int, name: str) -> tuple[float, ...]:
    if len(values) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    converted = tuple(float(value) for value in values)
    if not all(isfinite(value) for value in converted):
        raise ValueError(f"{name} must contain only finite values")
    return converted


@dataclass(frozen=True, slots=True)
class Detection3D:
    """One oriented 3D box in a documented sensor coordinate frame.

    ``center_xyz`` is the geometric box center. ``size_lwh`` is always
    ``(length, width, height)``: at yaw zero, length is parallel to positive X,
    width is parallel to positive Y, and height is parallel to positive Z.
    ``yaw_rad`` is measured counter-clockwise from positive X when viewed from
    above (toward negative Z). Velocity, when present, is ``(vx, vy)`` in the
    same coordinate frame.
    """

    center_xyz: tuple[float, float, float]
    size_lwh: tuple[float, float, float]
    yaw_rad: float
    score: float
    class_id: int
    class_name: str
    velocity_xy: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        center = _finite_tuple(self.center_xyz, length=3, name="center_xyz")
        size = _finite_tuple(self.size_lwh, length=3, name="size_lwh")
        velocity = (
            None
            if self.velocity_xy is None
            else _finite_tuple(self.velocity_xy, length=2, name="velocity_xy")
        )
        yaw = float(self.yaw_rad)
        score = float(self.score)

        if not all(value > 0.0 for value in size):
            raise ValueError("size_lwh values must be positive")
        if not isfinite(yaw):
            raise ValueError("yaw_rad must be finite")
        if not isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("score must be finite and between 0 and 1")
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, int):
            raise TypeError("class_id must be an integer")
        if self.class_id < 0:
            raise ValueError("class_id must be non-negative")
        if not isinstance(self.class_name, str) or not self.class_name.strip():
            raise ValueError("class_name must be a non-empty string")

        object.__setattr__(self, "center_xyz", center)
        object.__setattr__(self, "size_lwh", size)
        object.__setattr__(self, "velocity_xy", velocity)
        object.__setattr__(self, "yaw_rad", yaw)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "class_name", self.class_name.strip())

    def sort_key(self) -> tuple[Any, ...]:
        """Return the stable key used for exported result ordering."""

        velocity = self.velocity_xy if self.velocity_xy is not None else (float("inf"),) * 2
        return (
            -self.score,
            self.class_id,
            self.class_name,
            *self.center_xyz,
            *self.size_lwh,
            self.yaw_rad,
            *velocity,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""

        return {
            "center_xyz": list(self.center_xyz),
            "size_lwh": list(self.size_lwh),
            "yaw_rad": self.yaw_rad,
            "score": self.score,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "velocity_xy": None if self.velocity_xy is None else list(self.velocity_xy),
        }


@dataclass(frozen=True, slots=True)
class DetectionFrame:
    """Deterministically ordered detections for one LiDAR sample."""

    detections: tuple[Detection3D, ...]
    sample_id: str
    coordinate_frame: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be a non-empty string")
        if not isinstance(self.coordinate_frame, str) or not self.coordinate_frame.strip():
            raise ValueError("coordinate_frame must be a non-empty string")
        detections = tuple(self.detections)
        if not all(isinstance(detection, Detection3D) for detection in detections):
            raise TypeError("detections must contain only Detection3D values")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        object.__setattr__(self, "detections", tuple(sorted(detections, key=Detection3D.sort_key)))
        object.__setattr__(self, "sample_id", self.sample_id.strip())
        object.__setattr__(self, "coordinate_frame", self.coordinate_frame.strip())
        object.__setattr__(self, "metadata", MappingProxyType(deepcopy(dict(self.metadata))))

    def filtered(self, min_score: float) -> DetectionFrame:
        """Return a new frame containing scores at or above ``min_score``."""

        threshold = float(min_score)
        if not isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("min_score must be finite and between 0 and 1")
        return DetectionFrame(
            detections=tuple(
                detection for detection in self.detections if detection.score >= threshold
            ),
            sample_id=self.sample_id,
            coordinate_frame=self.coordinate_frame,
            metadata={**self.metadata, "score_threshold": threshold},
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible frame without backend-specific objects."""

        return {
            "schema_version": "1.0",
            "sample_id": self.sample_id,
            "coordinate_frame": self.coordinate_frame,
            "metadata": deepcopy(dict(self.metadata)),
            "detections": [detection.to_dict() for detection in self.detections],
        }
