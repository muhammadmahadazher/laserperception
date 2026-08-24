from __future__ import annotations

import numpy as np
from laserperception_ros.conversion import pointcloud2_to_raw_xyz
from laserperception_ros.kitti_raw_replay_node import (
    _raw_xyz_message,
    _stamp,
    _world_to_lidar_transform,
)

from laserperception.datasets.kitti_ros_replay import KittiRosReplayAcquisition


def test_kitti_ros_replay_serializes_exact_float32_rows_and_timestamp() -> None:
    points = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype=np.float32)
    stamp = _stamp(1_316_995_200_123_456_789)
    message = _raw_xyz_message(points, stamp, "kitti_model_aligned_lidar")
    decoded = pointcloud2_to_raw_xyz(message)
    np.testing.assert_array_equal(decoded.points_xyz, points)
    assert message.header.stamp.sec == 1_316_995_200
    assert message.header.stamp.nanosec == 123_456_789
    assert message.header.frame_id == "kitti_model_aligned_lidar"


def test_kitti_pose_record_becomes_nonidentity_world_tf() -> None:
    acquisition = KittiRosReplayAcquisition(
        drive_id="2011_09_26_drive_0001",
        frame_index=2,
        timestamp_nanoseconds=1_000_000_002,
        points_xyz=np.ones((1, 3), dtype=np.float32),
        world_translation_xyz=(1.0, -2.0, 3.0),
        world_rotation_xyzw=(0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)),
    )
    stamp = _stamp(acquisition.timestamp_nanoseconds)
    transform = _world_to_lidar_transform(
        acquisition,
        stamp=stamp,
        fixed_frame="kitti_world",
        lidar_frame="kitti_model_aligned_lidar",
    )
    assert transform.header.frame_id == "kitti_world"
    assert transform.child_frame_id == "kitti_model_aligned_lidar"
    assert transform.transform.translation.x == 1.0
    assert transform.transform.translation.y == -2.0
    assert transform.transform.translation.z == 3.0
    assert transform.transform.rotation.z == np.sqrt(0.5)
    assert transform.transform.rotation.w == np.sqrt(0.5)
