"""Validate live raw ROS multi-sweep output against the accepted M4.5a hashes."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import rclpy
from geometry_msgs.msg import TransformStamped
from laserperception_ros.conversion import pointcloud2_to_model_ready
from laserperception_ros.multisweep_node import LaserPerceptionMultiSweepNode
from laserperception_ros.raw_replay_node import (
    SCENE_START_SAMPLE_TOKEN,
    W1_SAMPLE_TOKEN,
    NuScenesRawMultiSweepReplayNode,
)
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer

EXPECTED = {
    0: {
        "sample_token": SCENE_START_SAMPLE_TOKEN,
        "point_count": 33_587,
        "sha256": "4da6843d2f4fcca676705ecd440047e0d0371efa53ee8d4bed305c72d8e1def4",
    },
    42: {
        "sample_token": W1_SAMPLE_TOKEN,
        "point_count": 354_182,
        "sha256": "5c1f5590ce7925d9312fe9ef755d0d849f579ff3d7ae842af75d41a0252cb29a",
    },
}


@dataclass(frozen=True, slots=True)
class CapturedCloud:
    stamp_ns: int
    message: PointCloud2


class _CaptureNode(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("m45b_model_ready_capture")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.clouds: list[CapturedCloud] = []
        self._subscription = self.create_subscription(PointCloud2, topic, self._capture, qos)

    def _capture(self, message: PointCloud2) -> None:
        stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)
        self.clouds.append(CapturedCloud(stamp_ns, message))


def _run_case(index: int, data_root: Path, *, timeout_sec: float) -> dict[str, Any]:
    expected = EXPECTED[index]
    suffix = f"s{index}"
    raw_topic = f"/laserperception/m45b_validation/{suffix}/raw"
    model_ready_topic = f"/laserperception/m45b_validation/{suffix}/model_ready"
    rclpy.init()
    replay = NuScenesRawMultiSweepReplayNode(
        parameter_overrides=[
            Parameter("data_root", value=str(data_root)),
            Parameter("sample_token", value=str(expected["sample_token"])),
            Parameter("raw_points_topic", value=raw_topic),
            Parameter("publish_period_sec", value=0.2),
        ]
    )
    builder = LaserPerceptionMultiSweepNode(
        parameter_overrides=[
            Parameter("raw_points_topic", value=raw_topic),
            Parameter("model_ready_topic", value=model_ready_topic),
            Parameter("fixed_frame", value="nuscenes_map"),
            Parameter("target_frame", value=""),
            Parameter("max_historical_sweeps", value=10),
            Parameter("transform_timeout_sec", value=0.5),
            Parameter("history_reset_gap_sec", value=0.0),
            Parameter("tf_cache_time_sec", value=10.0),
            Parameter("raw_qos_depth", value=5),
            Parameter("model_ready_qos_depth", value=1),
        ]
    )
    capture = _CaptureNode(model_ready_topic)
    executor = MultiThreadedExecutor(num_threads=3)
    for node in (replay, builder, capture):
        executor.add_node(node)
    deadline = time.monotonic() + timeout_sec
    try:
        final_stamp_us = replay._acquisitions[-1].timestamp_microseconds
        final_stamp_ns = final_stamp_us * 1_000
        final: CapturedCloud | None = None
        while time.monotonic() < deadline and final is None:
            executor.spin_once(timeout_sec=0.05)
            final = next(
                (cloud for cloud in reversed(capture.clouds) if cloud.stamp_ns == final_stamp_ns),
                None,
            )
        if final is None:
            raise RuntimeError(f"sample {index} did not produce the final model-ready cloud")
        cloud = pointcloud2_to_model_ready(final.message)
        observed_count = int(cloud.points_xyzt.shape[0])
        observed_hash = cloud.sha256
        exact = observed_count == int(expected["point_count"]) and observed_hash == str(
            expected["sha256"]
        )
        return {
            "sample_index": index,
            "sample_token": expected["sample_token"],
            "acquisition_tokens_chronological": [
                item.sample_data_token for item in replay._acquisitions
            ],
            "per_input_raw_point_counts": replay.raw_point_counts,
            "nonfinite_filtered_counts": [0] * len(replay.raw_point_counts),
            "final_history_depth": builder.current_history_depth,
            "final_point_count": observed_count,
            "dtype": str(cloud.points_xyzt.dtype),
            "shape": list(cloud.points_xyzt.shape),
            "expected_sha256": expected["sha256"],
            "observed_sha256": observed_hash,
            "exact": exact,
            "counters": {
                "raw_frames_received": builder.raw_frames_received,
                "valid_raw_frames": builder.valid_raw_frames,
                "invalid_points_filtered": builder.invalid_points_filtered,
                "model_ready_frames_published": builder.model_ready_frames_published,
                "rejected_frames": builder.rejected_frames,
                "tf_failures": builder.tf_failures,
                "history_resets": builder.history_resets,
                "current_history_depth": builder.current_history_depth,
            },
        }
    finally:
        executor.shutdown(timeout_sec=2.0)
        for node in (capture, replay):
            executor.remove_node(node)
            node.destroy_node()
        executor.remove_node(builder)
        builder.destroy_node()
        rclpy.shutdown()


def _same_frame_time_travel() -> dict[str, Any]:
    buffer = Buffer()
    first = _map_to_lidar(seconds=1, x=0.0)
    second = _map_to_lidar(seconds=2, x=1.0)
    buffer.set_transform(first, "m45b_validation")
    buffer.set_transform(second, "m45b_validation")
    result = buffer.lookup_transform_full(
        "lidar",
        Time(seconds=2),
        "lidar",
        Time(seconds=1),
        "map",
    )
    observed = float(result.transform.translation.x)
    return {
        "source_frame": "lidar",
        "source_time_sec": 1,
        "target_frame": "lidar",
        "target_time_sec": 2,
        "fixed_frame": "map",
        "ego_motion_x_m": 1.0,
        "observed_source_to_target_translation_x_m": observed,
        "expected_translation_x_m": -1.0,
        "exact": observed == -1.0,
    }


def _map_to_lidar(*, seconds: int, x: float) -> TransformStamped:
    transform = TransformStamped()
    transform.header.frame_id = "map"
    transform.header.stamp = Time(seconds=seconds).to_msg()
    transform.child_frame_id = "lidar"
    transform.transform.translation.x = x
    transform.transform.rotation.w = 1.0
    return transform


def _tf2_version() -> str:
    for distribution in ("tf2-ros", "tf2_ros"):
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    return "0.25.22 (installed Debian package)"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("LASERPERCEPTION_NUSCENES_ROOT", "")),
    )
    parser.add_argument("--timeout-sec", type=float, default=15.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if not str(args.data_root) or not args.data_root.is_dir():
        raise SystemExit("set --data-root or LASERPERCEPTION_NUSCENES_ROOT")
    scene_start = _run_case(0, args.data_root, timeout_sec=args.timeout_sec)
    w1 = _run_case(42, args.data_root, timeout_sec=args.timeout_sec)
    same_frame = _same_frame_time_travel()
    passed = bool(scene_start["exact"] and w1["exact"] and same_frame["exact"])
    record = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "ros_distro": os.environ.get("ROS_DISTRO", "unknown"),
        "rmw_implementation": rclpy.utilities.get_rmw_implementation_identifier(),
        "tf2_ros_version": _tf2_version(),
        "tf_api": "Buffer.lookup_transform_full",
        "tf_wait_strategy": "bounded blocking lookup with dedicated TransformListener thread",
        "qos": {
            "raw_subscription": "best_effort/volatile/keep_last/depth_5",
            "model_ready_publisher": "best_effort/volatile/keep_last/depth_1",
            "detector_subscription": "best_effort/volatile/keep_last/depth_1",
        },
        "history": {
            "max_historical_sweeps": 10,
            "order": "nearest_to_farthest",
            "reset_gap_sec": 0.0,
        },
        "scene_start": scene_start,
        "w1": w1,
        "same_frame_different_time": same_frame,
        "scope_guards": {
            "performance_campaign": False,
            "model_changed": False,
            "onnx_changed": False,
            "engine_changed": False,
            "exact_fast_changed": False,
        },
    }
    payload = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not passed:
        raise SystemExit("M4.5b raw ROS exact gate failed")


if __name__ == "__main__":
    main()
