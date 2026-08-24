"""ROS-independent projected-pose references for M6c feasibility work."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from laserperception.datasets.kitti_raw import KittiRawSequence
from laserperception.datasets.kitti_ros_replay import model_lidar_pose_to_world_transform
from laserperception.detection.multisweep import (
    HistoricalSweep,
    LidarPose,
    MultiSweepBuilder,
    MultiSweepBuilderConfig,
    RawSweep,
    SweepTransform,
)
from laserperception.detection.ros2_contract import ModelReadyPointCloud
from laserperception.evaluation.m6c_representation import (
    builder_matrix_from_ros_transform,
    quaternion_to_rotation_matrix_xyzw,
)


@dataclass(frozen=True, slots=True)
class ProjectedWorldPose:
    """One accepted KITTI pose after unit-quaternion representability projection."""

    rotation: np.ndarray
    translation: np.ndarray
    quaternion_xyzw: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation)
        translation = np.asarray(self.translation)
        if rotation.dtype != np.dtype(np.float64) or rotation.shape != (3, 3):
            raise TypeError("projected rotation must be a float64 3x3 matrix")
        if translation.dtype != np.dtype(np.float64) or translation.shape != (3,):
            raise TypeError("projected translation must be a float64 3-vector")
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("projected pose must be finite")
        quaternion = np.asarray(self.quaternion_xyzw, dtype=np.float64)
        if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
            raise ValueError("projected quaternion must contain four finite values")
        object.__setattr__(self, "rotation", np.ascontiguousarray(rotation).copy())
        object.__setattr__(self, "translation", np.ascontiguousarray(translation).copy())


@dataclass(frozen=True, slots=True)
class ProjectedReferenceResult:
    """One offline projected reconstruction and its ordered historical transforms."""

    current_index: int
    historical_indices: tuple[int, ...]
    transforms: tuple[SweepTransform, ...]
    point_cloud: ModelReadyPointCloud


def projected_world_pose(pose: LidarPose) -> ProjectedWorldPose:
    """Project an accepted pose through the frozen unit-quaternion representation."""

    translation_values, quaternion_values = model_lidar_pose_to_world_transform(pose)
    quaternion = (
        float(quaternion_values[0]),
        float(quaternion_values[1]),
        float(quaternion_values[2]),
        float(quaternion_values[3]),
    )
    rotation = quaternion_to_rotation_matrix_xyzw(np.asarray(quaternion, dtype=np.float64))
    return ProjectedWorldPose(
        rotation=rotation,
        translation=np.asarray(translation_values, dtype=np.float64),
        quaternion_xyzw=quaternion,
    )


def projected_relative_transform(
    *,
    source_id: str,
    target_id: str,
    historical_pose: LidarPose,
    current_pose: LidarPose,
) -> SweepTransform:
    """Compose a historical-to-current builder transform without ROS or tf2."""

    historical = projected_world_pose(historical_pose)
    current = projected_world_pose(current_pose)
    relative_rotation = current.rotation.T @ historical.rotation
    relative_translation = current.rotation.T @ (historical.translation - current.translation)
    matrix = builder_matrix_from_ros_transform(relative_rotation, relative_translation)
    return SweepTransform(
        matrix.astype(np.float32),
        source_id=source_id,
        target_id=target_id,
    )


def build_projected_reference(
    sequence: KittiRawSequence,
    *,
    current_index: int,
    history_depth: int,
) -> ProjectedReferenceResult:
    """Build one projected offline model-ready input with nearest-first history."""

    if isinstance(history_depth, bool) or not isinstance(history_depth, int):
        raise TypeError("history_depth must be an integer")
    if history_depth <= 0:
        raise ValueError("history_depth must be positive")
    first_index = max(-1, current_index - history_depth - 1)
    historical_indices = tuple(range(current_index - 1, first_index, -1))
    required_indices = (current_index, *historical_indices)
    sweeps = {index: sequence.frame(index).to_raw_sweep() for index in required_indices}
    poses = {index: sequence.lidar_pose(index) for index in required_indices}
    return build_projected_reference_from_sources(
        current_index=current_index,
        history_depth=history_depth,
        sweeps=sweeps,
        poses=poses,
    )


def build_projected_reference_from_sources(
    *,
    current_index: int,
    history_depth: int,
    sweeps: Mapping[int, RawSweep],
    poses: Mapping[int, LidarPose],
) -> ProjectedReferenceResult:
    """Build a projected reference from preloaded raw sweeps and accepted poses."""

    if isinstance(history_depth, bool) or not isinstance(history_depth, int):
        raise TypeError("history_depth must be an integer")
    if history_depth <= 0:
        raise ValueError("history_depth must be positive")
    first_index = max(-1, current_index - history_depth - 1)
    historical_indices = tuple(range(current_index - 1, first_index, -1))
    required_indices = (current_index, *historical_indices)
    missing_sweeps = [index for index in required_indices if index not in sweeps]
    missing_poses = [index for index in required_indices if index not in poses]
    if missing_sweeps or missing_poses:
        raise KeyError(
            "projected-reference sources are incomplete: "
            f"missing sweeps={missing_sweeps}, missing poses={missing_poses}"
        )
    current = sweeps[current_index]
    current_pose = poses[current_index]
    transforms: list[SweepTransform] = []
    historical: list[HistoricalSweep] = []
    for index in historical_indices:
        sweep = sweeps[index]
        transform = projected_relative_transform(
            source_id=sweep.source_id,
            target_id=current.source_id,
            historical_pose=poses[index],
            current_pose=current_pose,
        )
        transforms.append(transform)
        historical.append(HistoricalSweep(sweep, transform))
    builder = MultiSweepBuilder(MultiSweepBuilderConfig(max_historical_sweeps=history_depth))
    return ProjectedReferenceResult(
        current_index=current_index,
        historical_indices=historical_indices,
        transforms=tuple(transforms),
        point_cloud=builder.build(current, historical),
    )
