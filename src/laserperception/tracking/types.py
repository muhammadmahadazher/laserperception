"""Immutable track snapshots with observation and prediction identities separated."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Literal

from laserperception.detection.serialization import detection_from_dict
from laserperception.detection.types import Detection3D
from laserperception.perception.serialization import JsonRecord, load_json

from .config import TrackerConfig

TRACKER_IDENTITY: Literal["laserperception.constant-velocity-global-greedy.v1"] = (
    "laserperception.constant-velocity-global-greedy.v1"
)


class TrackState(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"


@dataclass(frozen=True)
class Track3D:
    track_id: int
    detection: Detection3D
    lifecycle: TrackState
    age: int
    hits: int
    missed: int
    velocity_xy: tuple[float, float] | None
    predicted_center_xyz: tuple[float, float, float]
    source_sample_id: str
    timestamp_ns: int
    observation_timestamp_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.detection, Detection3D) or not isinstance(
            self.lifecycle, TrackState
        ):
            raise TypeError("invalid detection or lifecycle")
        for name in (
            "track_id",
            "age",
            "hits",
            "missed",
            "timestamp_ns",
            "observation_timestamp_ns",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.track_id < 1 or self.hits < 1 or self.age < self.hits + self.missed:
            raise ValueError("invalid lifecycle counters")
        if (
            self.observation_timestamp_ns > self.timestamp_ns
            or not isinstance(self.source_sample_id, str)
            or not self.source_sample_id.strip()
        ):
            raise ValueError("invalid observation identity")
        if self.lifecycle is TrackState.LOST and self.missed == 0:
            raise ValueError("lost track requires missed observations")
        if self.lifecycle is not TrackState.LOST and self.missed != 0:
            raise ValueError("observed track cannot have missed observations")
        for name, length in (("predicted_center_xyz", 3), ("velocity_xy", 2)):
            value = getattr(self, name)
            if value is None and name == "predicted_center_xyz":
                raise ValueError("predicted_center_xyz must be supplied")
            if value is not None:
                values = tuple(float(v) for v in value)
                if len(values) != length or not all(isfinite(v) for v in values):
                    raise ValueError(f"{name} must have {length} finite values")
                object.__setattr__(self, name, values)

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "detection": self.detection.to_dict(),
            "lifecycle": self.lifecycle.value,
            "age": self.age,
            "hits": self.hits,
            "missed": self.missed,
            "velocity_xy": self.velocity_xy,
            "predicted_center_xyz": self.predicted_center_xyz,
            "source_sample_id": self.source_sample_id,
            "timestamp_ns": self.timestamp_ns,
            "observation_timestamp_ns": self.observation_timestamp_ns,
        }


@dataclass(frozen=True)
class _TrackRecord(JsonRecord):
    track_id: int
    lifecycle: TrackState
    age: int
    hits: int
    missed: int
    velocity_xy: tuple[float, float] | None
    predicted_center_xyz: tuple[float, float, float]
    source_sample_id: str
    timestamp_ns: int
    observation_timestamp_ns: int


def _track_from_dict(value: object) -> Track3D:
    if not isinstance(value, dict) or "detection" not in value:
        raise ValueError("track must include a detection")
    remaining = {key: item for key, item in value.items() if key != "detection"}
    record = _TrackRecord.from_dict(remaining)
    return Track3D(
        record.track_id,
        detection_from_dict(value["detection"]),
        record.lifecycle,
        record.age,
        record.hits,
        record.missed,
        record.velocity_xy,
        record.predicted_center_xyz,
        record.source_sample_id,
        record.timestamp_ns,
        record.observation_timestamp_ns,
    )


@dataclass(frozen=True)
class TrackFrame:
    sequence_id: str
    sample_id: str
    coordinate_frame: str
    timestamp_ns: int
    tracks: tuple[Track3D, ...]
    config: TrackerConfig
    tracker_identity: Literal["laserperception.constant-velocity-global-greedy.v1"] = (
        TRACKER_IDENTITY
    )
    schema_version: Literal["laserperception.track-frame.v1"] = "laserperception.track-frame.v1"

    def __post_init__(self) -> None:
        if any(
            not isinstance(v, str) or not v.strip()
            for v in (self.sequence_id, self.sample_id, self.coordinate_frame)
        ):
            raise ValueError("frame identities must be non-empty strings")
        if type(self.timestamp_ns) is not int or not 0 <= self.timestamp_ns < 2**63:
            raise ValueError("timestamp_ns must be a non-negative integer")
        if not isinstance(self.config, TrackerConfig):
            raise TypeError("config must be TrackerConfig")
        tracks = tuple(self.tracks)
        if not all(isinstance(t, Track3D) for t in tracks):
            raise TypeError("tracks must contain Track3D")
        if len({t.track_id for t in tracks}) != len(tracks):
            raise ValueError("duplicate track IDs")
        if any(t.timestamp_ns != self.timestamp_ns for t in tracks):
            raise ValueError("track timestamps differ from frame")
        if (
            self.tracker_identity != TRACKER_IDENTITY
            or self.schema_version != "laserperception.track-frame.v1"
        ):
            raise ValueError("unsupported tracker identity or schema")
        object.__setattr__(self, "tracks", tuple(sorted(tracks, key=lambda t: t.track_id)))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "tracker_identity": self.tracker_identity,
            "sequence_id": self.sequence_id,
            "sample_id": self.sample_id,
            "coordinate_frame": self.coordinate_frame,
            "timestamp_ns": self.timestamp_ns,
            "config": self.config.to_dict(),
            "tracks": [t.to_dict() for t in self.tracks],
        }

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, value: str) -> TrackFrame:
        payload = load_json(value)
        if (
            not isinstance(payload, dict)
            or set(payload)
            != {
                "schema_version",
                "tracker_identity",
                "sequence_id",
                "sample_id",
                "coordinate_frame",
                "timestamp_ns",
                "config",
                "tracks",
            }
            or not isinstance(payload["tracks"], list)
        ):
            raise ValueError("TrackFrame fields differ")
        return cls(
            payload["sequence_id"],
            payload["sample_id"],
            payload["coordinate_frame"],
            payload["timestamp_ns"],
            tuple(_track_from_dict(t) for t in payload["tracks"]),
            TrackerConfig.from_dict(payload["config"]),
            payload["tracker_identity"],
            payload["schema_version"],
        )
