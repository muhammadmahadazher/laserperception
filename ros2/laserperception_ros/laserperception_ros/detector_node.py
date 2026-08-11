"""ROS 2 detector node for model-ready multi-sweep PointCloud2 messages."""

from __future__ import annotations

import time
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import MarkerArray

from laserperception_ros.conversion import (
    detection_frame_to_message,
    pointcloud2_to_model_ready,
)
from laserperception_ros.markers import detection_frame_to_markers
from laserperception_ros.runtime import M3DetectorRuntime


class LaserPerceptionDetectorNode(Node):
    """Bounded-QoS TensorRT detector initialized once at node startup."""

    def __init__(self, *, runtime: Any | None = None) -> None:
        super().__init__("laserperception_detector")
        self.declare_parameter("input_topic", "/laserperception/points_model_ready")
        self.declare_parameter("output_topic", "/laserperception/detections")
        self.declare_parameter("marker_topic", "/laserperception/markers")
        self.declare_parameter("publish_markers", True)
        self.declare_parameter("input_qos_depth", 1)
        self.declare_parameter("input_reliability", "best_effort")
        self.declare_parameter("output_qos_depth", 5)
        self.declare_parameter("output_reliability", "reliable")
        self.declare_parameter("engine_path", "")

        input_qos = _qos(
            depth=int(self.get_parameter("input_qos_depth").value),
            reliability=str(self.get_parameter("input_reliability").value),
        )
        output_qos = _qos(
            depth=int(self.get_parameter("output_qos_depth").value),
            reliability=str(self.get_parameter("output_reliability").value),
        )
        output_topic = str(self.get_parameter("output_topic").value)
        self._detections_publisher = self.create_publisher(
            Detection3DArray, output_topic, output_qos
        )
        self._marker_publisher = None
        if bool(self.get_parameter("publish_markers").value):
            self._marker_publisher = self.create_publisher(
                MarkerArray,
                str(self.get_parameter("marker_topic").value),
                output_qos,
            )

        self._runtime = runtime or M3DetectorRuntime(
            engine_override=str(self.get_parameter("engine_path").value)
        )
        self.callback_latencies_ms: list[float] = []
        self.received_count = 0
        self.accepted_count = 0
        self.published_count = 0
        self.rejected_count = 0
        self._subscription = self.create_subscription(
            PointCloud2,
            str(self.get_parameter("input_topic").value),
            self._on_points,
            input_qos,
        )
        engine_sha = getattr(self._runtime, "engine_sha256", "injected-test-runtime")
        self.get_logger().info(f"M3 detector ready; engine_sha256={engine_sha}")

    def _on_points(self, message: PointCloud2) -> None:
        started_ns = time.perf_counter_ns()
        self.received_count += 1
        try:
            points = pointcloud2_to_model_ready(message)
            self.accepted_count += 1
            sample_id = f"{message.header.stamp.sec}.{message.header.stamp.nanosec:09d}"
            frame = self._runtime.infer(
                points,
                sample_id=sample_id,
                coordinate_frame=message.header.frame_id,
            )
            output = detection_frame_to_message(frame, message.header)
            self._detections_publisher.publish(output)
            ended_ns = time.perf_counter_ns()
            self.callback_latencies_ms.append((ended_ns - started_ns) / 1_000_000.0)
            self.published_count += 1
            if self._marker_publisher is not None:
                self._marker_publisher.publish(detection_frame_to_markers(frame, message.header))
        except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
            self.rejected_count += 1
            self.get_logger().error(
                f"rejected model-ready PointCloud2: {error}",
                throttle_duration_sec=5.0,
            )


def _qos(*, depth: int, reliability: str) -> QoSProfile:
    if depth <= 0:
        raise ValueError("QoS depth must be positive")
    normalized = reliability.strip().lower()
    policies = {
        "best_effort": ReliabilityPolicy.BEST_EFFORT,
        "reliable": ReliabilityPolicy.RELIABLE,
    }
    if normalized not in policies:
        raise ValueError("reliability must be best_effort or reliable")
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=policies[normalized],
        durability=DurabilityPolicy.VOLATILE,
    )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LaserPerceptionDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
