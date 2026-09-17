"""Streaming constant-XY-velocity tracker; no detection or coordinate conversion."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from math import isfinite
from typing import Protocol

from laserperception.detection.serialization import TimedDetectionFrame
from laserperception.detection.types import DetectionFrame

from .association import GlobalGreedyAssociator
from .config import TrackerConfig
from .types import Track3D, TrackFrame, TrackState


class Tracker3D(Protocol):
    def update(self, frame: DetectionFrame, *, timestamp_ns: int) -> TrackFrame: ...
    def state(self) -> TrackFrame | None: ...
    def reset(self, *, sequence_id: str) -> None: ...


class ConstantVelocityTracker:
    """Global greedy engineering baseline in one caller-supplied fixed meter frame."""

    def __init__(self, sequence_id: str, config: TrackerConfig | None = None) -> None:
        self._config = config if config is not None else TrackerConfig()
        if not isinstance(self.config, TrackerConfig):
            raise TypeError("config must be TrackerConfig")
        self._associator = GlobalGreedyAssociator()
        self.reset(sequence_id=sequence_id)

    @property
    def config(self) -> TrackerConfig:
        return self._config

    @property
    def sequence_id(self) -> str:
        return self._sequence_id

    def reset(self, *, sequence_id: str) -> None:
        if not isinstance(sequence_id, str) or not sequence_id.strip():
            raise ValueError("sequence_id must be explicit")
        sequence_id = sequence_id.strip()
        if getattr(self, "_last", None) is not None and sequence_id == self.sequence_id:
            raise ValueError("reset requires a new sequence_id to preserve the ID scope")
        self._sequence_id = sequence_id
        self._last: TrackFrame | None = None
        self._next_id = 1

    def state(self) -> TrackFrame | None:
        return self._last

    def update(self, frame: DetectionFrame, *, timestamp_ns: int) -> TrackFrame:
        if not isinstance(frame, DetectionFrame):
            raise TypeError("frame must be DetectionFrame")
        if type(timestamp_ns) is not int or not 0 <= timestamp_ns < 2**63:
            raise ValueError("timestamp_ns must be a non-negative integer")
        last = self._last
        if last is not None:
            if timestamp_ns < last.timestamp_ns:
                raise ValueError("backwards timestamp; reset with a new sequence identity")
            if frame.coordinate_frame != last.coordinate_frame:
                raise ValueError("coordinate frame changed; transform explicitly before tracking")
        old = () if last is None else last.tracks
        predictions = []
        for track in old:
            dt = (timestamp_ns - track.observation_timestamp_ns) / 1e9
            vx, vy = track.velocity_xy or (0.0, 0.0)
            x, y, z = track.detection.center_xyz
            center = (x + vx * dt, y + vy * dt, z)
            if not all(isfinite(v) for v in center):
                raise ValueError("motion prediction overflow")
            predictions.append(center)
        matches = self._associator.match(old, frame.detections, predictions, self.config)
        ti_to_di = dict(matches)
        if len(ti_to_di) != len(matches) or len({di for _, di in matches}) != len(matches):
            raise ValueError("associator returned duplicate matches")
        if any(
            ti < 0 or ti >= len(old) or di < 0 or di >= len(frame.detections) for ti, di in matches
        ):
            raise ValueError("associator returned out-of-range matches")
        used = {di for _, di in matches}
        tracks = []
        for ti, track in enumerate(old):
            if ti in ti_to_di:
                detection = frame.detections[ti_to_di[ti]]
                velocity = detection.velocity_xy
                if velocity is None:
                    velocity = track.velocity_xy
                    dt = (timestamp_ns - track.observation_timestamp_ns) / 1e9
                    if dt > 0:
                        estimated = tuple(
                            (detection.center_xyz[i] - track.detection.center_xyz[i]) / dt
                            for i in (0, 1)
                        )
                        alpha = self.config.velocity_smoothing
                        previous = track.velocity_xy
                        velocity = (
                            (estimated[0], estimated[1])
                            if previous is None
                            else (
                                (1 - alpha) * previous[0] + alpha * estimated[0],
                                (1 - alpha) * previous[1] + alpha * estimated[1],
                            )
                        )
                hits = track.hits + 1
                lifecycle = (
                    TrackState.CONFIRMED if hits >= self.config.min_hits else TrackState.TENTATIVE
                )
                tracks.append(
                    Track3D(
                        track.track_id,
                        detection,
                        lifecycle,
                        track.age + 1,
                        hits,
                        0,
                        velocity,
                        detection.center_xyz,
                        frame.sample_id,
                        timestamp_ns,
                        timestamp_ns,
                    )
                )
            elif track.missed < self.config.max_missed:
                tracks.append(
                    Track3D(
                        track.track_id,
                        track.detection,
                        TrackState.LOST,
                        track.age + 1,
                        track.hits,
                        track.missed + 1,
                        track.velocity_xy,
                        predictions[ti],
                        track.source_sample_id,
                        timestamp_ns,
                        track.observation_timestamp_ns,
                    )
                )
        next_id = self._next_id
        for di, detection in enumerate(frame.detections):
            if di not in used:
                lifecycle = (
                    TrackState.CONFIRMED if self.config.min_hits == 1 else TrackState.TENTATIVE
                )
                tracks.append(
                    Track3D(
                        next_id,
                        detection,
                        lifecycle,
                        1,
                        1,
                        0,
                        detection.velocity_xy,
                        detection.center_xyz,
                        frame.sample_id,
                        timestamp_ns,
                        timestamp_ns,
                    )
                )
                next_id += 1
        result = TrackFrame(
            self.sequence_id,
            frame.sample_id,
            frame.coordinate_frame,
            timestamp_ns,
            tuple(tracks),
            self.config,
        )
        self._last = result
        self._next_id = next_id
        return result


def track_sequence(
    frames: Iterable[TimedDetectionFrame], *, sequence_id: str, config: TrackerConfig | None = None
) -> Iterator[TrackFrame]:
    """Consume one timed frame at a time; retain only active tracks."""
    tracker = ConstantVelocityTracker(sequence_id, config)
    for value in frames:
        yield tracker.update(value.frame, timestamp_ns=value.timestamp_ns)
