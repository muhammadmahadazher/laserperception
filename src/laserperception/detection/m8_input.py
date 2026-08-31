"""Deterministic five-feature input contract for the M8 detector candidate.

This module extends the accepted multi-sweep reconstruction with the source
intensity column.  It deliberately preserves the pinned M6 physical point
population: projecting columns ``x, y, z, time_lag`` must reproduce the
historical model-ready matrix byte for byte.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np

from laserperception.detection.multisweep import (
    POINTPILLARS_POINT_CLOUD_RANGE,
    HistoricalSweep,
    MultiSweepBuilder,
    MultiSweepBuilderConfig,
    RawSweep,
)

M8_FEATURE_NAMES = ("x", "y", "z", "intensity", "time_lag")
M8_HISTORICAL_PROJECTION = (0, 1, 2, 4)
M8_POINT_CLOUD_RANGE = POINTPILLARS_POINT_CLOUD_RANGE


def m8_elapsed_seconds(current_microseconds: int, historical_microseconds: int) -> np.float32:
    """Return current-minus-historical elapsed seconds at the frozen float32 cast point."""

    for name, value in (
        ("current_microseconds", current_microseconds),
        ("historical_microseconds", historical_microseconds),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if historical_microseconds > current_microseconds:
        raise ValueError("historical timestamp must not be newer than the current timestamp")
    return np.float32(current_microseconds / 1_000_000 - historical_microseconds / 1_000_000)


@dataclass(frozen=True, slots=True)
class M8PointCloud:
    """Contiguous float32 ``[x, y, z, intensity, time_lag]`` points."""

    points: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points)
        if points.dtype != np.dtype(np.float32):
            raise TypeError("M8 points must have dtype float32")
        if points.ndim != 2 or points.shape[1] != len(M8_FEATURE_NAMES):
            raise ValueError("M8 points must have shape (N, 5)")
        if points.shape[0] == 0:
            raise ValueError("M8 points must be non-empty")
        if not np.isfinite(points).all():
            raise ValueError("M8 points must contain only finite values")
        object.__setattr__(self, "points", np.ascontiguousarray(points).copy())

    @property
    def historical_projection(self) -> np.ndarray:
        """Return contiguous XYZT in the frozen historical column order."""

        return cast(
            np.ndarray,
            np.ascontiguousarray(self.points[:, M8_HISTORICAL_PROJECTION]),
        )


class M8MultiSweepBuilder:
    """Retain raw intensity while reproducing the frozen M6 row population."""

    def __init__(self, config: MultiSweepBuilderConfig | None = None) -> None:
        self.config = config or MultiSweepBuilderConfig()

    def build(
        self,
        current: RawSweep,
        historical: Sequence[HistoricalSweep],
    ) -> M8PointCloud:
        """Build XYZIT with the accepted transform, ordering, lag, and mask semantics."""

        current_points = current.points.copy()
        current_points[:, 4] = np.float32(0.0)
        parts = [current_points]
        selected = tuple(historical[: self.config.max_historical_sweeps])

        if not selected and self.config.pad_empty_sweeps:
            duplicate = current_points
            if self.config.remove_close:
                duplicate = _remove_close(duplicate, self.config.remove_close_radius)
            parts.extend(duplicate.copy() for _ in range(self.config.max_historical_sweeps))

        for item in selected:
            if item.transform.target_id != current.source_id:
                raise ValueError("historical transform target_id must match current source_id")
            points = item.sweep.points.copy()
            if self.config.remove_close:
                points = _remove_close(points, self.config.remove_close_radius)

            # Match MultiSweepBuilder's pinned serialization/cast/write-back points.
            lidar2sensor = np.array(item.transform.lidar2sensor.tolist())
            points[:, :3] = points[:, :3] @ lidar2sensor[:3, :3]
            points[:, :3] -= lidar2sensor[:3, 3]
            points[:, 4] = m8_elapsed_seconds(
                current.timestamp_microseconds, item.sweep.timestamp_microseconds
            )
            parts.append(points)

        concatenated = np.concatenate(parts, axis=0)
        minimum = M8_POINT_CLOUD_RANGE[:3]
        maximum = M8_POINT_CLOUD_RANGE[3:]
        mask = (
            (concatenated[:, 0] > minimum[0])
            & (concatenated[:, 0] < maximum[0])
            & (concatenated[:, 1] > minimum[1])
            & (concatenated[:, 1] < maximum[1])
            & (concatenated[:, 2] > minimum[2])
            & (concatenated[:, 2] < maximum[2])
        )
        result = M8PointCloud(np.ascontiguousarray(concatenated[mask]))

        historical_reference = MultiSweepBuilder(self.config).build(current, historical).points_xyzt
        if not np.array_equal(result.historical_projection, historical_reference):
            raise RuntimeError("M8 historical XYZT projection is not byte-identical")
        return result


def _remove_close(points: np.ndarray, radius: float) -> np.ndarray:
    x_close = np.abs(points[:, 0]) < radius
    y_close = np.abs(points[:, 1]) < radius
    return cast(np.ndarray, points[np.logical_not(np.logical_and(x_close, y_close))])
