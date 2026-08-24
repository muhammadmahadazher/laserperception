"""Deterministic KITTI Raw replay at the accepted M4.5b ROS boundary."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster

from laserperception.datasets.kitti_raw import KittiRawSequence
from laserperception.datasets.kitti_ros_replay import (
    KittiRosReplayAcquisition,
    kitti_ros_replay_acquisition,
)


class KittiRawReplayNode(Node):
    """Publish official KITTI acquisitions as model-axis XYZ plus OXTS TF."""

    def __init__(self, *, parameter_overrides: list[Any] | None = None) -> None:
        super().__init__(
            "laserperception_kitti_raw_replay",
            parameter_overrides=parameter_overrides,
        )
        self.declare_parameter("raw_points_topic", "/laserperception/points_raw")
        self.declare_parameter("data_root", os.environ.get("LASERPERCEPTION_KITTI_RAW_ROOT", ""))
        self.declare_parameter("date", "2011_09_26")
        self.declare_parameter("drive_id", "2011_09_26_drive_0001")
        self.declare_parameter("start_frame", 0)
        self.declare_parameter("end_frame", -1)
        self.declare_parameter("publish_period_sec", 0.25)
        self.declare_parameter("auto_start", True)
        self.declare_parameter("fixed_frame", "kitti_world")
        self.declare_parameter("lidar_frame", "kitti_model_aligned_lidar")
        self.declare_parameter("raw_qos_depth", 5)

        data_root = str(self.get_parameter("data_root").value).strip()
        if not data_root:
            raise ValueError("set LASERPERCEPTION_KITTI_RAW_ROOT or KITTI replay data_root")
        date = _required_text(self, "date")
        drive_id = _required_text(self, "drive_id")
        date_root = Path(data_root).expanduser().resolve() / date
        drive_root = date_root / f"{drive_id}_sync"
        self.sequence = KittiRawSequence(date_root, drive_root)
        self._fixed_frame = _required_text(self, "fixed_frame")
        self._lidar_frame = _required_text(self, "lidar_frame")

        self._next_index = int(self.get_parameter("start_frame").value)
        configured_end = int(self.get_parameter("end_frame").value)
        self._end_index = len(self.sequence) - 1 if configured_end < 0 else configured_end
        if not 0 <= self._next_index <= self._end_index < len(self.sequence):
            raise ValueError("KITTI replay frame range is outside the selected drive")
        period = float(self.get_parameter("publish_period_sec").value)
        if period <= 0.0:
            raise ValueError("publish_period_sec must be positive")
        depth = int(self.get_parameter("raw_qos_depth").value)
        if depth <= 0:
            raise ValueError("raw_qos_depth must be positive")

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=depth,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._publisher = self.create_publisher(
            PointCloud2,
            str(self.get_parameter("raw_points_topic").value),
            qos,
        )
        self._tf_broadcaster = TransformBroadcaster(self)
        self._timer = None
        if bool(self.get_parameter("auto_start").value):
            self._timer = self.create_timer(period, self.publish_next)
        self.published_count = 0
        self.published_indices: list[int] = []
        self.raw_point_counts: list[int] = []
        self.get_logger().info(
            "KITTI Raw replay ready; "
            f"drive={drive_id}, frames={self._next_index}-{self._end_index}, "
            f"point_frame={self._lidar_frame}, fixed_frame={self._fixed_frame}"
        )

    def publish_next(self) -> int | None:
        """Publish the next chronological acquisition and return its frame index."""

        if self._next_index > self._end_index:
            if self._timer is not None:
                self._timer.cancel()
            return None
        acquisition = kitti_ros_replay_acquisition(self.sequence, self._next_index)
        stamp = _stamp(acquisition.timestamp_nanoseconds)
        self._tf_broadcaster.sendTransform(
            _world_to_lidar_transform(
                acquisition,
                stamp=stamp,
                fixed_frame=self._fixed_frame,
                lidar_frame=self._lidar_frame,
            )
        )
        self._publisher.publish(_raw_xyz_message(acquisition.points_xyz, stamp, self._lidar_frame))
        published = self._next_index
        self.published_count += 1
        self.published_indices.append(published)
        self.raw_point_counts.append(len(acquisition.points_xyz))
        self._next_index += 1
        if self._next_index > self._end_index and self._timer is not None:
            self._timer.cancel()
        return published


def _stamp(timestamp_nanoseconds: int) -> Time:
    seconds, nanoseconds = divmod(timestamp_nanoseconds, 1_000_000_000)
    return Time(sec=seconds, nanosec=nanoseconds)


def _world_to_lidar_transform(
    acquisition: KittiRosReplayAcquisition,
    *,
    stamp: Time,
    fixed_frame: str,
    lidar_frame: str,
) -> TransformStamped:
    transform = TransformStamped()
    transform.header = Header(stamp=stamp, frame_id=fixed_frame)
    transform.child_frame_id = lidar_frame
    translation = acquisition.world_translation_xyz
    quaternion = acquisition.world_rotation_xyzw
    transform.transform.translation.x = translation[0]
    transform.transform.translation.y = translation[1]
    transform.transform.translation.z = translation[2]
    transform.transform.rotation.x = quaternion[0]
    transform.transform.rotation.y = quaternion[1]
    transform.transform.rotation.z = quaternion[2]
    transform.transform.rotation.w = quaternion[3]
    return transform


def _raw_xyz_message(points: np.ndarray, stamp: Time, frame_id: str) -> PointCloud2:
    message = PointCloud2()
    message.header = Header(stamp=stamp, frame_id=frame_id)
    message.height = 1
    message.width = len(points)
    message.fields = [
        PointField(name=name, offset=index * 4, datatype=PointField.FLOAT32, count=1)
        for index, name in enumerate(("x", "y", "z"))
    ]
    message.is_bigendian = False
    message.point_step = 12
    message.row_step = 12 * len(points)
    message.data = np.ascontiguousarray(points, dtype=np.float32).tobytes(order="C")
    message.is_dense = True
    return message


def _required_text(node: Node, name: str) -> str:
    value = str(node.get_parameter(name).value).strip()
    if not value:
        raise ValueError(f"{name} must be non-empty")
    return value


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = KittiRawReplayNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
