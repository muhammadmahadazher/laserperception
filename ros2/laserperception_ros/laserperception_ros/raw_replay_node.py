"""Development replay of actual raw nuScenes sweeps plus independent ROS TF."""

from __future__ import annotations

import os
from dataclasses import dataclass
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

W1_SAMPLE_TOKEN = "07fad91090c746ccaa1b2bdb55329e20"
SCENE_START_SAMPLE_TOKEN = "3e8750f331d7499e9b5123e9eb70f2e2"


@dataclass(frozen=True, slots=True)
class ReplayAcquisition:
    """One real nuScenes raw file and its independently sourced TF records."""

    sample_data_token: str
    timestamp_microseconds: int
    path: Path
    calibration: dict[str, Any]
    ego_pose: dict[str, Any]


class NuScenesRawMultiSweepReplayNode(Node):
    """Replay history chronologically so the live builder warms naturally."""

    def __init__(self, *, parameter_overrides: list[Any] | None = None) -> None:
        super().__init__(
            "laserperception_nuscenes_raw_replay",
            parameter_overrides=parameter_overrides,
        )
        self.declare_parameter("raw_points_topic", "/laserperception/points_raw")
        self.declare_parameter("data_root", os.environ.get("LASERPERCEPTION_NUSCENES_ROOT", ""))
        self.declare_parameter("sample_token", W1_SAMPLE_TOKEN)
        self.declare_parameter("max_historical_sweeps", 10)
        self.declare_parameter("publish_period_sec", 0.25)
        self.declare_parameter("fixed_frame", "nuscenes_map")
        self.declare_parameter("ego_frame", "nuscenes_ego")
        self.declare_parameter("lidar_frame", "nuscenes_lidar_top")
        self.declare_parameter("raw_qos_depth", 5)

        data_root = str(self.get_parameter("data_root").value).strip()
        if not data_root:
            raise ValueError("set LASERPERCEPTION_NUSCENES_ROOT or raw replay data_root")
        period = float(self.get_parameter("publish_period_sec").value)
        if period <= 0.0:
            raise ValueError("publish_period_sec must be positive")
        history = int(self.get_parameter("max_historical_sweeps").value)
        if history <= 0:
            raise ValueError("max_historical_sweeps must be positive")
        depth = int(self.get_parameter("raw_qos_depth").value)
        if depth <= 0:
            raise ValueError("raw_qos_depth must be positive")

        try:
            from nuscenes.nuscenes import NuScenes
        except ImportError as error:
            raise RuntimeError(
                "nuScenes raw replay requires nuscenes-devkit in the development environment"
            ) from error
        self._nusc = NuScenes(version="v1.0-mini", dataroot=data_root, verbose=False)
        sample_token = str(self.get_parameter("sample_token").value).strip()
        if not sample_token:
            raise ValueError("sample_token must be non-empty")
        self._acquisitions = _acquisitions(self._nusc, sample_token, history)
        self._fixed_frame = _required_frame(self, "fixed_frame")
        self._ego_frame = _required_frame(self, "ego_frame")
        self._lidar_frame = _required_frame(self, "lidar_frame")
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
        self._offset = 0
        self.published_count = 0
        self.published_tokens: list[str] = []
        self.raw_point_counts: list[int] = []
        self._timer = self.create_timer(period, self._publish_next)
        self.get_logger().info(
            f"raw nuScenes replay ready with {len(self._acquisitions)} chronological acquisitions"
        )

    def _publish_next(self) -> None:
        if self._offset >= len(self._acquisitions):
            self._timer.cancel()
            return
        acquisition = self._acquisitions[self._offset]
        stamp = _stamp(acquisition.timestamp_microseconds)
        self._tf_broadcaster.sendTransform(
            [
                _transform(
                    parent=self._fixed_frame,
                    child=self._ego_frame,
                    stamp=stamp,
                    record=acquisition.ego_pose,
                ),
                _transform(
                    parent=self._ego_frame,
                    child=self._lidar_frame,
                    stamp=stamp,
                    record=acquisition.calibration,
                ),
            ]
        )
        points = np.fromfile(acquisition.path, dtype=np.float32)
        if points.size == 0 or points.size % 5 != 0:
            raise RuntimeError("nuScenes raw LIDAR_TOP file is empty or malformed")
        xyz = np.ascontiguousarray(points.reshape(-1, 5)[:, :3])
        self._publisher.publish(_raw_xyz_message(xyz, stamp, self._lidar_frame))
        self.published_count += 1
        self.published_tokens.append(acquisition.sample_data_token)
        self.raw_point_counts.append(len(xyz))
        self._offset += 1
        if self._offset >= len(self._acquisitions):
            self._timer.cancel()


def _acquisitions(nusc: Any, sample_token: str, max_history: int) -> tuple[ReplayAcquisition, ...]:
    sample = nusc.get("sample", sample_token)
    current_token = str(sample["data"]["LIDAR_TOP"])
    current_data = nusc.get("sample_data", current_token)
    if int(sample["timestamp"]) != int(current_data["timestamp"]):
        raise RuntimeError("current sample and LIDAR_TOP acquisition timestamps differ")
    tokens = [current_token]
    previous = str(current_data["prev"])
    while previous and len(tokens) <= max_history:
        tokens.append(previous)
        previous = str(nusc.get("sample_data", previous)["prev"])
    chronological = tuple(reversed(tokens))
    result: list[ReplayAcquisition] = []
    for token in chronological:
        sample_data = nusc.get("sample_data", token)
        result.append(
            ReplayAcquisition(
                sample_data_token=token,
                timestamp_microseconds=int(sample_data["timestamp"]),
                path=Path(nusc.get_sample_data_path(token)),
                calibration=dict(
                    nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
                ),
                ego_pose=dict(nusc.get("ego_pose", sample_data["ego_pose_token"])),
            )
        )
    return tuple(result)


def _stamp(timestamp_microseconds: int) -> Time:
    seconds, microseconds = divmod(timestamp_microseconds, 1_000_000)
    return Time(sec=seconds, nanosec=microseconds * 1_000)


def _transform(
    *,
    parent: str,
    child: str,
    stamp: Time,
    record: dict[str, Any],
) -> TransformStamped:
    translation = record["translation"]
    quaternion_wxyz = record["rotation"]
    if len(translation) != 3 or len(quaternion_wxyz) != 4:
        raise RuntimeError("nuScenes transform metadata has an unexpected shape")
    message = TransformStamped()
    message.header = Header(stamp=stamp, frame_id=parent)
    message.child_frame_id = child
    message.transform.translation.x = float(translation[0])
    message.transform.translation.y = float(translation[1])
    message.transform.translation.z = float(translation[2])
    message.transform.rotation.w = float(quaternion_wxyz[0])
    message.transform.rotation.x = float(quaternion_wxyz[1])
    message.transform.rotation.y = float(quaternion_wxyz[2])
    message.transform.rotation.z = float(quaternion_wxyz[3])
    return message


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
    message.is_dense = bool(np.isfinite(points).all())
    return message


def _required_frame(node: Node, name: str) -> str:
    value = str(node.get_parameter(name).value).strip()
    if not value:
        raise ValueError(f"{name} must be non-empty")
    return value


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = NuScenesRawMultiSweepReplayNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
