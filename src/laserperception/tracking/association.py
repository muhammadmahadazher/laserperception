"""Class-aware global greedy matching with explicit deterministic tie order."""

from __future__ import annotations

from collections.abc import Sequence
from math import dist, isfinite
from typing import Protocol

from laserperception.detection.types import Detection3D

from .config import TrackerConfig
from .types import Track3D


class Associator(Protocol):
    def match(
        self,
        tracks: Sequence[Track3D],
        detections: Sequence[Detection3D],
        predictions: Sequence[tuple[float, float, float]],
        config: TrackerConfig,
    ) -> tuple[tuple[int, int], ...]: ...


class GlobalGreedyAssociator:
    """Sort eligible pairs by Euclidean cost, track ID, then DetectionFrame row."""

    def match(
        self,
        tracks: Sequence[Track3D],
        detections: Sequence[Detection3D],
        predictions: Sequence[tuple[float, float, float]],
        config: TrackerConfig,
    ) -> tuple[tuple[int, int], ...]:
        if len(tracks) != len(predictions):
            raise ValueError("track and prediction counts differ")
        pairs: list[tuple[float, int, int, int]] = []
        for ti, (track, center) in enumerate(zip(tracks, predictions, strict=True)):
            for di, detection in enumerate(detections):
                if config.class_aware and (
                    track.detection.class_id != detection.class_id
                    or track.detection.class_name != detection.class_name
                ):
                    continue
                cost = dist(center, detection.center_xyz)
                if not isfinite(cost):
                    raise ValueError("non-finite association cost")
                if cost <= config.association_radius_m:
                    pairs.append((cost, track.track_id, di, ti))
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        result: list[tuple[int, int]] = []
        for _, _, di, ti in sorted(pairs):
            if ti not in used_tracks and di not in used_detections:
                used_tracks.add(ti)
                used_detections.add(di)
                result.append((ti, di))
        return tuple(result)
