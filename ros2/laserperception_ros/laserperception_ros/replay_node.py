"""nuScenes model-ready PointCloud2 replay using the verified M2 preparation path."""

from __future__ import annotations

import os
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header

from laserperception_ros.conversion import model_ready_to_pointcloud2
from laserperception_ros.runtime import create_backend, resolve_m3_assets


class NuScenesReplayNode(Node):
    """Publish exact prepared Nx4 arrays as model-ready PointCloud2 messages."""

    def __init__(self) -> None:
        super().__init__("laserperception_replay")
        self.declare_parameter("output_topic", "/laserperception/points_model_ready")
        self.declare_parameter("data_root", os.environ.get("LASERPERCEPTION_NUSCENES_ROOT", ""))
        self.declare_parameter("split", "mini_val")
        self.declare_parameter("start_index", 0)
        self.declare_parameter("sample_count", 1)
        self.declare_parameter("one_shot", False)
        self.declare_parameter("loop", True)
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("frame_id", "nuscenes_lidar_top")
        self.declare_parameter("qos_depth", 1)

        rate_hz = float(self.get_parameter("publish_rate_hz").value)
        if not rate_hz > 0.0:
            raise ValueError("publish_rate_hz must be positive")
        depth = int(self.get_parameter("qos_depth").value)
        if depth <= 0:
            raise ValueError("qos_depth must be positive")
        data_root = str(self.get_parameter("data_root").value).strip()
        if not data_root:
            raise ValueError("set LASERPERCEPTION_NUSCENES_ROOT or replay.data_root")
        self._data_root = Path(data_root).expanduser()
        self._split = str(self.get_parameter("split").value)
        self._start_index = int(self.get_parameter("start_index").value)
        self._sample_count = int(self.get_parameter("sample_count").value)
        self._one_shot = bool(self.get_parameter("one_shot").value)
        self._loop = bool(self.get_parameter("loop").value)
        self._frame_id = str(self.get_parameter("frame_id").value)
        if self._sample_count <= 0:
            raise ValueError("sample_count must be positive")

        self._backend = create_backend(resolve_m3_assets())
        split_size = self._backend.dataset_size(self._data_root, self._split)
        if self._start_index < 0 or self._start_index + self._sample_count > split_size:
            raise IndexError("configured replay range is outside the prepared split")
        self._offset = 0
        self.published_count = 0
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=depth,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._publisher = self.create_publisher(
            PointCloud2, str(self.get_parameter("output_topic").value), qos
        )
        self._timer = self.create_timer(1.0 / rate_hz, self._publish_next)
        self.get_logger().info(
            f"model-ready replay configured at {rate_hz:.3f} Hz; "
            "20 Hz is a synthetic performance stress cadence, not nuScenes keyframe timing"
        )

    def _publish_next(self) -> None:
        index = self._start_index + self._offset
        prepared = self._backend.prepare_sample(self._data_root, split=self._split, index=index)
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self._frame_id
        self._publisher.publish(model_ready_to_pointcloud2(prepared.model_ready_points(), header))
        self.published_count += 1
        self._offset += 1
        if self._one_shot or (self._offset >= self._sample_count and not self._loop):
            self._timer.cancel()
        elif self._offset >= self._sample_count:
            self._offset = 0


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = NuScenesReplayNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
