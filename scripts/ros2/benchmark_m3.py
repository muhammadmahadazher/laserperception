"""Measure M3 callback processing and same-host ROS loopback latency at 20 Hz."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from laserperception_ros.conversion import model_ready_to_pointcloud2
from laserperception_ros.detector_node import LaserPerceptionDetectorNode
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from vision_msgs.msg import Detection3DArray

from laserperception.detection.benchmark import latency_statistics_ms
from laserperception.detection.runtime_metadata import nvidia_smi_value, repository_git_sha

WARMUPS = 20
MEASURED = 200
RATE_HZ = 20.0
PERIOD_SECONDS = 1.0 / RATE_HZ


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _qos(*, reliability: ReliabilityPolicy, depth: int) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=reliability,
        durability=DurabilityPolicy.VOLATILE,
    )


class _StressPublisher(Node):
    def __init__(self, points: object) -> None:
        super().__init__("laserperception_m3_benchmark_replay")
        self._points = points
        self.published_count = 0
        self.publication_perf_ns: list[int] = []
        self._publisher = self.create_publisher(
            PointCloud2,
            "/laserperception/points_model_ready",
            _qos(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1),
        )
        self._timer = self.create_timer(PERIOD_SECONDS, self._publish)

    def _publish(self) -> None:
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "nuscenes_lidar_top"
        self._publisher.publish(model_ready_to_pointcloud2(self._points, header))
        self.publication_perf_ns.append(time.perf_counter_ns())
        self.published_count += 1

    def stop(self) -> None:
        self._timer.cancel()


class _LoopbackSink(Node):
    def __init__(self) -> None:
        super().__init__("laserperception_m3_benchmark_sink")
        self.loopback_ms: list[float] = []
        self.reception_perf_ns: list[int] = []
        self._subscription = self.create_subscription(
            Detection3DArray,
            "/laserperception/detections",
            self._receive,
            _qos(reliability=ReliabilityPolicy.RELIABLE, depth=5),
        )

    def _receive(self, message: Detection3DArray) -> None:
        received_ros_ns = self.get_clock().now().nanoseconds
        source_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(
            message.header.stamp.nanosec
        )
        self.loopback_ms.append((received_ros_ns - source_ns) / 1_000_000.0)
        self.reception_perf_ns.append(time.perf_counter_ns())


def _deadline_summary(values: Sequence[float]) -> dict[str, float | int]:
    count = sum(value > 50.0 for value in values)
    return {"above_50_ms_count": count, "above_50_ms_fraction": count / len(values)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rclpy.init(args=["--ros-args", "-p", "publish_markers:=false"])
    detector = LaserPerceptionDetectorNode()
    prepared = detector._runtime.backend.prepare_sample(args.data_root, split="mini_val", index=0)
    publisher = _StressPublisher(prepared.model_ready_points())
    sink = _LoopbackSink()
    executor = MultiThreadedExecutor(num_threads=3)
    for node in (publisher, detector, sink):
        executor.add_node(node)
    target = WARMUPS + MEASURED
    deadline = time.monotonic() + args.timeout_seconds
    try:
        while len(sink.loopback_ms) < target and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        publisher.stop()
        settle_deadline = time.monotonic() + 2.0
        while time.monotonic() < settle_deadline:
            executor.spin_once(timeout_sec=0.05)
    finally:
        executor.shutdown()
        for node in (publisher, detector, sink):
            node.destroy_node()
        rclpy.shutdown()

    callback_values = detector.callback_latencies_ms[WARMUPS : WARMUPS + MEASURED]
    loopback_values = sink.loopback_ms[WARMUPS : WARMUPS + MEASURED]
    reception_times = sink.reception_perf_ns[WARMUPS : WARMUPS + MEASURED]
    if len(callback_values) != MEASURED or len(loopback_values) != MEASURED:
        raise SystemExit(
            f"M3 benchmark incomplete: callback={len(callback_values)}, "
            f"loopback={len(loopback_values)}; do not promote"
        )
    callback = {**latency_statistics_ms(callback_values), **_deadline_summary(callback_values)}
    loopback = {**latency_statistics_ms(loopback_values), **_deadline_summary(loopback_values)}
    elapsed_seconds = (reception_times[-1] - reception_times[0]) / 1_000_000_000.0
    effective_hz = (MEASURED - 1) / elapsed_seconds
    publication_times = publisher.publication_perf_ns[WARMUPS:]
    publication_elapsed_seconds = (publication_times[-1] - publication_times[0]) / 1_000_000_000.0
    effective_replay_hz = (len(publication_times) - 1) / publication_elapsed_seconds
    input_loss = publisher.published_count - detector.received_count
    rejected = detector.rejected_count
    processing_backlog = detector.accepted_count - detector.published_count
    output_transport_loss = detector.published_count - len(sink.loopback_ms)
    no_loss_or_backlog = (
        input_loss == 0 and rejected == 0 and processing_backlog == 0 and output_transport_loss == 0
    )
    replay_rate_pass = effective_replay_hz >= RATE_HZ * 0.95
    output_rate_pass = effective_hz >= RATE_HZ * 0.95
    gate_pass = (
        callback["median_ms"] <= 50.0
        and no_loss_or_backlog
        and replay_rate_pass
        and output_rate_pass
    )
    runtime = detector._runtime
    result = {
        "schema_version": "1.0",
        "milestone": "M3A",
        "status": "pass" if gate_pass else "fail_review_required",
        "measurement_commit": repository_git_sha(_root()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "ros_distro": "humble",
            "ubuntu": "22.04",
            "python": "3.10",
            "rmw_implementation": rclpy.utilities.get_rmw_implementation_identifier(),
            "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU",
            "driver": nvidia_smi_value("driver_version"),
            "tensorrt": "8.6.1",
        },
        "engine_sha256": runtime.engine_sha256,
        "input_contract": {
            "topic": "/laserperception/points_model_ready",
            "fields": ["x", "y", "z", "time_lag"],
            "semantics": "model_ready_multi_sweep_current_lidar_frame",
        },
        "protocol": {
            "sample": "nuScenes v1.0-mini mini_val index 0 repeated",
            "replay_mode": "synthetic_20_hz_performance_stress_not_native_keyframe_cadence",
            "replay_rate_hz": RATE_HZ,
            "warmup_messages": WARMUPS,
            "measured_messages": MEASURED,
            "input_qos": "keep_last_depth_1_best_effort_volatile",
            "output_qos": "keep_last_depth_5_reliable_volatile",
            "markers_enabled": False,
            "executor": "three_thread_same_process_separate_ros_nodes",
        },
        "callback_processing_latency": callback,
        "same_host_ros_loopback_latency": loopback,
        "message_counts": {
            "replay_published": publisher.published_count,
            "detector_received": detector.received_count,
            "detector_accepted": detector.accepted_count,
            "detections_published": detector.published_count,
            "sink_received": len(sink.loopback_ms),
            "detector_rejected": detector.rejected_count,
        },
        "sustained_rate": {
            "effective_replay_publish_hz": effective_replay_hz,
            "effective_output_hz": effective_hz,
            "output_sensor_rate_fraction": effective_hz / RATE_HZ,
            "input_messages_not_received": input_loss,
            "rejected_messages": rejected,
            "final_processing_backlog_messages": processing_backlog,
            "output_transport_loss_messages": output_transport_loss,
            "processing_induced_loss": input_loss > 0 or rejected > 0,
            "accumulating_backlog_observed": processing_backlog != 0,
        },
        "gate": {
            "callback_median_at_or_below_50_ms": callback["median_ms"] <= 50.0,
            "no_processing_induced_loss_or_final_backlog": no_loss_or_backlog,
            "effective_replay_at_least_95_percent_of_requested_rate": replay_rate_pass,
            "effective_output_at_least_95_percent_of_requested_rate": output_rate_pass,
            "pass": gate_pass,
        },
        "limitations": [
            "callback processing ends immediately after Detection3DArray publish returns",
            "same-host loopback is not sensor-to-actuator latency",
            "20 Hz replay is synthetic stress cadence, not native nuScenes keyframe timing",
            "input is already model-ready multi-sweep data; no TF or sweep history is included",
        ],
    }
    output = args.output or runtime.assets.engine_path.parent.parent / "m3" / "benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"external result: {output}")
    if not gate_pass:
        raise SystemExit("M3A runtime-rate gate failed; stop for M3B review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
