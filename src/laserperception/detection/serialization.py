"""Strict readers for the existing DetectionFrame JSON representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from laserperception.perception.serialization import JsonRecord, load_json

from .types import Detection3D, DetectionFrame


@dataclass(frozen=True)
class _DetectionRecord(JsonRecord):
    center_xyz: tuple[float, float, float]
    size_lwh: tuple[float, float, float]
    yaw_rad: float
    score: float
    class_id: int
    class_name: str
    velocity_xy: tuple[float, float] | None


def detection_from_dict(value: object) -> Detection3D:
    record = _DetectionRecord.from_dict(value)
    return Detection3D(
        record.center_xyz,
        record.size_lwh,
        record.yaw_rad,
        record.score,
        record.class_id,
        record.class_name,
        record.velocity_xy,
    )


def detection_frame_from_dict(value: object) -> DetectionFrame:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "sample_id",
        "coordinate_frame",
        "metadata",
        "detections",
    }:
        raise ValueError("DetectionFrame fields differ from schema 1.0")
    if value["schema_version"] != "1.0" or not isinstance(value["metadata"], dict):
        raise ValueError("invalid DetectionFrame schema or metadata")
    if not isinstance(value["detections"], list):
        raise ValueError("detections must be an array")
    for name in ("sample_id", "coordinate_frame"):
        if not isinstance(value[name], str):
            raise ValueError(f"{name} must be a string")
    return DetectionFrame(
        tuple(detection_from_dict(d) for d in value["detections"]),
        value["sample_id"],
        value["coordinate_frame"],
        value["metadata"],
    )


def detection_frame_from_json(value: str) -> DetectionFrame:
    return detection_frame_from_dict(load_json(value))


@dataclass(frozen=True)
class TimedDetectionFrame:
    """Explicit integer nanoseconds and an unchanged DetectionFrame."""

    frame: DetectionFrame
    timestamp_ns: int
    schema_version: Literal["laserperception.timed-detection.v1"] = (
        "laserperception.timed-detection.v1"
    )

    def __post_init__(self) -> None:
        if not isinstance(self.frame, DetectionFrame):
            raise TypeError("frame must be DetectionFrame")
        if type(self.timestamp_ns) is not int or not 0 <= self.timestamp_ns < 2**63:
            raise ValueError("timestamp_ns must be a non-negative integer")
        if self.schema_version != "laserperception.timed-detection.v1":
            raise ValueError("unknown timed detection schema")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "timestamp_ns": self.timestamp_ns,
            "frame": self.frame.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> TimedDetectionFrame:
        if not isinstance(value, dict) or set(value) != {"schema_version", "timestamp_ns", "frame"}:
            raise ValueError("timed detection fields differ")
        if value["schema_version"] != "laserperception.timed-detection.v1":
            raise ValueError("unknown timed detection schema")
        stamp = value["timestamp_ns"]
        if type(stamp) is not int:
            raise ValueError("timestamp_ns must be an integer")
        return cls(detection_frame_from_dict(value["frame"]), stamp)

    @classmethod
    def from_json(cls, value: str) -> TimedDetectionFrame:
        return cls.from_dict(load_json(value))
