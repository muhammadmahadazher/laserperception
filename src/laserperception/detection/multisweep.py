"""Exact offline reconstruction of pinned nuScenes multi-sweep detector inputs.

The production classes in this module depend only on NumPy and LaserPerception's
existing :class:`ModelReadyPointCloud` contract. MMDetection3D is deliberately
not imported: it is an integration-test oracle, not a runtime dependency.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from laserperception.detection.ros2_contract import ModelReadyPointCloud

NUSCENES_LOAD_DIM = 5
POINTPILLARS_USE_DIM = (0, 1, 2, 4)
POINTPILLARS_POINT_CLOUD_RANGE = (-50.0, -50.0, -5.0, 50.0, 50.0, 3.0)


@dataclass(frozen=True, slots=True)
class RawSweep:
    """One raw nuScenes LIDAR_TOP acquisition and its provenance.

    ``points`` is the raw float32 ``x, y, z, intensity, ring_index`` matrix.
    ``timestamp_microseconds`` retains the integer nuScenes clock value so the
    pinned seconds conversion happens at exactly the same point as upstream.
    """

    points: np.ndarray
    timestamp_microseconds: int
    source_id: str

    def __post_init__(self) -> None:
        points = np.asarray(self.points)
        if points.dtype != np.dtype(np.float32):
            raise TypeError("raw sweep points must have dtype float32")
        if points.ndim != 2 or points.shape[1] != NUSCENES_LOAD_DIM:
            raise ValueError("raw sweep points must have shape (N, 5)")
        if points.shape[0] == 0:
            raise ValueError("raw sweep points must be non-empty")
        if not np.isfinite(points).all():
            raise ValueError("raw sweep points must contain only finite values")
        if isinstance(self.timestamp_microseconds, bool) or not isinstance(
            self.timestamp_microseconds, int
        ):
            raise TypeError("timestamp_microseconds must be an integer")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be a non-empty string")
        object.__setattr__(self, "points", np.ascontiguousarray(points).copy())
        object.__setattr__(self, "source_id", self.source_id.strip())

    @property
    def timestamp_seconds(self) -> float:
        """Return the pinned binary64 microseconds-to-seconds conversion."""

        return self.timestamp_microseconds / 1_000_000

    @classmethod
    def from_nuscenes_file(
        cls,
        path: str | Path,
        *,
        timestamp_microseconds: int,
        source_id: str,
    ) -> RawSweep:
        """Read a headerless nuScenes five-float LIDAR_TOP file."""

        raw = np.fromfile(Path(path), dtype=np.float32)
        if raw.size == 0:
            raise ValueError("raw sweep file must contain at least one point")
        if raw.size % NUSCENES_LOAD_DIM != 0:
            raise ValueError("raw sweep file length is not divisible by five float32 values")
        return cls(
            raw.reshape(-1, NUSCENES_LOAD_DIM),
            timestamp_microseconds=timestamp_microseconds,
            source_id=source_id,
        )


@dataclass(frozen=True, slots=True)
class LidarPose:
    """Float64 sensor calibration and ego pose used by the pinned converter."""

    lidar_to_ego_rotation: np.ndarray
    lidar_to_ego_translation: np.ndarray
    ego_to_global_rotation: np.ndarray
    ego_to_global_translation: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "lidar_to_ego_rotation",
            _validated_float64_array(self.lidar_to_ego_rotation, (3, 3), "lidar_to_ego_rotation"),
        )
        object.__setattr__(
            self,
            "lidar_to_ego_translation",
            _validated_float64_array(
                self.lidar_to_ego_translation, (3,), "lidar_to_ego_translation"
            ),
        )
        object.__setattr__(
            self,
            "ego_to_global_rotation",
            _validated_float64_array(self.ego_to_global_rotation, (3, 3), "ego_to_global_rotation"),
        )
        object.__setattr__(
            self,
            "ego_to_global_translation",
            _validated_float64_array(
                self.ego_to_global_translation, (3,), "ego_to_global_translation"
            ),
        )


@dataclass(frozen=True, slots=True)
class SweepTransform:
    """Pinned float32 ``lidar2sensor`` transform plus source/target identity."""

    lidar2sensor: np.ndarray
    source_id: str
    target_id: str

    def __post_init__(self) -> None:
        matrix = np.asarray(self.lidar2sensor)
        if matrix.dtype != np.dtype(np.float32):
            raise TypeError("lidar2sensor must have dtype float32")
        if matrix.shape != (4, 4):
            raise ValueError("lidar2sensor must have shape (4, 4)")
        if not np.isfinite(matrix).all():
            raise ValueError("lidar2sensor must contain only finite values")
        if not np.array_equal(matrix[3], np.array([0.0, 0.0, 0.0, 1.0], np.float32)):
            raise ValueError("lidar2sensor must have homogeneous final row [0, 0, 0, 1]")
        for name, value in (("source_id", self.source_id), ("target_id", self.target_id)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value.strip())
        object.__setattr__(self, "lidar2sensor", np.ascontiguousarray(matrix).copy())

    @classmethod
    def from_poses(
        cls,
        *,
        source_id: str,
        target_id: str,
        sweep_pose: LidarPose,
        current_pose: LidarPose,
    ) -> SweepTransform:
        """Reproduce the pinned converter arithmetic and float32 storage cast."""

        l2e_r_s_mat = sweep_pose.lidar_to_ego_rotation
        e2g_r_s_mat = sweep_pose.ego_to_global_rotation
        l2e_t_s = sweep_pose.lidar_to_ego_translation
        e2g_t_s = sweep_pose.ego_to_global_translation
        l2e_r_mat = current_pose.lidar_to_ego_rotation
        e2g_r_mat = current_pose.ego_to_global_rotation
        l2e_t = current_pose.lidar_to_ego_translation
        e2g_t = current_pose.ego_to_global_translation

        rotation = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
        )
        translation = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
        )
        translation -= (
            e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
            + l2e_t @ np.linalg.inv(l2e_r_mat).T
        )

        sensor2lidar_rotation = rotation.T
        lidar2sensor = np.eye(4)
        lidar2sensor[:3, :3] = sensor2lidar_rotation.T
        lidar2sensor[:3, 3:4] = -1 * np.matmul(sensor2lidar_rotation.T, translation.reshape(3, 1))
        return cls(
            lidar2sensor.astype(np.float32),
            source_id=source_id,
            target_id=target_id,
        )


@dataclass(frozen=True, slots=True)
class HistoricalSweep:
    """A raw historical sweep with its transform into the current sensor frame."""

    sweep: RawSweep
    transform: SweepTransform

    def __post_init__(self) -> None:
        if self.sweep.source_id != self.transform.source_id:
            raise ValueError("historical sweep and transform source_id values must match")


@dataclass(frozen=True, slots=True)
class MultiSweepBuilderConfig:
    """Pinned reference behavior for deterministic offline reconstruction."""

    max_historical_sweeps: int = 10
    remove_close: bool = False
    remove_close_radius: float = 1.0
    pad_empty_sweeps: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.max_historical_sweeps, bool) or not isinstance(
            self.max_historical_sweeps, int
        ):
            raise TypeError("max_historical_sweeps must be an integer")
        if self.max_historical_sweeps <= 0:
            raise ValueError("max_historical_sweeps must be positive")
        radius = float(self.remove_close_radius)
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("remove_close_radius must be finite and positive")
        object.__setattr__(self, "remove_close_radius", radius)


class MultiSweepBuilder:
    """Build the exact pinned PointPillars XYZT matrix from independent raw sweeps."""

    def __init__(self, config: MultiSweepBuilderConfig | None = None) -> None:
        self.config = config or MultiSweepBuilderConfig()

    def build(
        self,
        current: RawSweep,
        historical: Sequence[HistoricalSweep],
    ) -> ModelReadyPointCloud:
        """Reconstruct current plus selected history in caller-provided time order.

        ``historical`` must be ordered nearest-to-farthest, as in the pinned
        nuScenes converter. Test-mode semantics select the first configured N.
        """

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

            # Upstream serializes a float32 matrix to a list, then reloads it as
            # float64. Preserve that cast point and both float32 write-backs.
            lidar2sensor = np.array(item.transform.lidar2sensor.tolist())
            points[:, :3] = points[:, :3] @ lidar2sensor[:3, :3]
            points[:, :3] -= lidar2sensor[:3, 3]
            points[:, 4] = current.timestamp_seconds - item.sweep.timestamp_seconds
            parts.append(points)

        concatenated = np.concatenate(parts, axis=0)
        xyzt = concatenated[:, POINTPILLARS_USE_DIM]
        minimum = POINTPILLARS_POINT_CLOUD_RANGE[:3]
        maximum = POINTPILLARS_POINT_CLOUD_RANGE[3:]
        mask = (
            (xyzt[:, 0] > minimum[0])
            & (xyzt[:, 0] < maximum[0])
            & (xyzt[:, 1] > minimum[1])
            & (xyzt[:, 1] < maximum[1])
            & (xyzt[:, 2] > minimum[2])
            & (xyzt[:, 2] < maximum[2])
        )
        return ModelReadyPointCloud(np.ascontiguousarray(xyzt[mask]))


def _remove_close(points: np.ndarray, radius: float) -> np.ndarray:
    """Apply the upstream strict axis-aligned-square filter without reordering."""

    x_close = np.abs(points[:, 0]) < radius
    y_close = np.abs(points[:, 1]) < radius
    return cast(np.ndarray, points[np.logical_not(np.logical_and(x_close, y_close))])


def _validated_float64_array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must have dtype float64")
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return cast(np.ndarray, np.ascontiguousarray(array).copy())
