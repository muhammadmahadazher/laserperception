"""ROS-independent KITTI Raw replay records for the M6c integration gate."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

from laserperception.datasets.kitti_raw import KittiRawSequence
from laserperception.detection.multisweep import LidarPose

KITTI_POSE_ROTATION_ATOL = 1e-6


@dataclass(frozen=True, slots=True)
class KittiRosReplayAcquisition:
    """One acquisition expressed at the frozen raw ROS boundary."""

    drive_id: str
    frame_index: int
    timestamp_nanoseconds: int
    points_xyz: np.ndarray
    world_translation_xyz: tuple[float, float, float]
    world_rotation_xyzw: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        points = np.asarray(self.points_xyz)
        if points.dtype != np.dtype(np.float32):
            raise TypeError("KITTI replay XYZ points must have dtype float32")
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError("KITTI replay XYZ points must have non-empty shape (N, 3)")
        if not np.isfinite(points).all():
            raise ValueError("KITTI replay XYZ points must contain only finite values")
        if self.frame_index < 0:
            raise ValueError("KITTI replay frame index must be non-negative")
        if self.timestamp_nanoseconds < 0:
            raise ValueError("KITTI replay timestamp must be non-negative")
        if not self.drive_id.strip():
            raise ValueError("KITTI replay drive identity must be non-empty")
        object.__setattr__(self, "drive_id", self.drive_id.strip())
        object.__setattr__(self, "points_xyz", np.ascontiguousarray(points).copy())

    @property
    def stamp_components(self) -> tuple[int, int]:
        """Return exact ROS ``sec`` and ``nanosec`` fields."""

        return divmod(self.timestamp_nanoseconds, 1_000_000_000)


def kitti_ros_replay_acquisition(
    sequence: KittiRawSequence,
    frame_index: int,
) -> KittiRosReplayAcquisition:
    """Adapt one official acquisition without changing its frozen geometry."""

    frame = sequence.frame(frame_index)
    raw = frame.to_raw_sweep()
    translation, rotation = model_lidar_pose_to_world_transform(sequence.lidar_pose(frame_index))
    return KittiRosReplayAcquisition(
        drive_id=sequence.drive_root.name.removesuffix("_sync"),
        frame_index=frame_index,
        timestamp_nanoseconds=frame.timestamp.nanoseconds,
        points_xyz=np.ascontiguousarray(raw.points[:, :3]),
        world_translation_xyz=translation,
        world_rotation_xyzw=rotation,
    )


def model_lidar_pose_to_world_transform(
    pose: LidarPose,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Compose the accepted model-lidar calibration and OXTS world pose."""

    rotation = pose.ego_to_global_rotation @ pose.lidar_to_ego_rotation
    translation = (
        pose.ego_to_global_rotation @ pose.lidar_to_ego_translation + pose.ego_to_global_translation
    )
    translation_xyz = (
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
    )
    return translation_xyz, rotation_matrix_to_quaternion_xyzw(rotation)


def rotation_matrix_to_quaternion_xyzw(
    rotation: np.ndarray,
) -> tuple[float, float, float, float]:
    """Convert a proper float64 rotation matrix to a canonical unit quaternion."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not np.allclose(
        matrix.T @ matrix,
        np.eye(3),
        atol=KITTI_POSE_ROTATION_ATOL,
        rtol=0.0,
    ):
        raise ValueError("rotation must be orthonormal")
    if not np.isclose(
        np.linalg.det(matrix),
        1.0,
        atol=KITTI_POSE_ROTATION_ATOL,
        rtol=0.0,
    ):
        raise ValueError("rotation determinant must equal +1")

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * sqrt(trace + 1.0)
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = 2.0 * sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = 2.0 * sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = 2.0 * sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return (
        float(quaternion[0]),
        float(quaternion[1]),
        float(quaternion[2]),
        float(quaternion[3]),
    )
