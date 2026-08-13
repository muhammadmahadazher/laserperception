"""Measure final M3 W1 ROS callback and same-host loopback performance."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rclpy
from laserperception_ros.conversion import model_ready_to_pointcloud2
from laserperception_ros.detector_node import LaserPerceptionDetectorNode
from laserperception_ros.runtime import M3DetectorRuntime
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from vision_msgs.msg import Detection3DArray

from laserperception.detection.benchmark import (
    half_run_backlog_summary,
    latency_statistics_ms,
)
from laserperception.detection.measurement_telemetry import (
    NvidiaSmiSampler,
    nvidia_clock_capability,
    paired_gpu_state_eligibility,
    summarize_gpu_telemetry,
    summarize_telemetry_by_block,
)
from laserperception.detection.mmdet3d_backend import sha256_file
from laserperception.detection.runtime_metadata import repository_git_sha

WARMUPS = 20
MEASURED = 200
SAMPLE_INDEX = 42
EXPECTED_POINT_COUNT = 354_182
TELEMETRY_INTERVAL_SECONDS = 0.5
SUSTAINED_WARMUP_SECONDS = 30.0
EXPECTED_ARTIFACT_HASHES = {
    "checkpoint": "f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0",
    "onnx": "61ce22a8ca31498675c32576bfb94f0093d31dc95d2762f7254bf915a59ecc16",
    "engine": "a005f75852097cd9b193750560b214cc3d5237ae9b6c106c7fca3d4fc348714b",
}


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
    def __init__(self, points: object, *, rate_hz: float, timer_enabled: bool = True) -> None:
        super().__init__("laserperception_m3_benchmark_replay")
        self.published_count = 0
        self.publication_perf_ns: list[int] = []
        self.publication_ros_ns: list[int] = []
        header = Header()
        header.frame_id = "nuscenes_lidar_top"
        self._message = model_ready_to_pointcloud2(points, header)
        self._publisher = self.create_publisher(
            PointCloud2,
            "/laserperception/points_model_ready",
            _qos(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1),
        )
        self._timer = self.create_timer(1.0 / rate_hz, self._publish)
        if not timer_enabled:
            self._timer.cancel()

    def _publish(self) -> None:
        now = self.get_clock().now()
        self._message.header.stamp = now.to_msg()
        self._publisher.publish(self._message)
        self.publication_perf_ns.append(time.perf_counter_ns())
        self.publication_ros_ns.append(now.nanoseconds)
        self.published_count += 1

    def publish_once(self) -> None:
        self._publish()

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


def _intervals_ms(values_ns: Sequence[int]) -> list[float]:
    return [
        (second - first) / 1_000_000.0
        for first, second in zip(values_ns, values_ns[1:], strict=False)
    ]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_correctness(
    path: Path, *, commit_sha: str, runtime: M3DetectorRuntime
) -> dict[str, object]:
    record = dict(json.loads(path.read_text(encoding="utf-8")))
    if record.get("status") != "pass":
        raise RuntimeError("production correctness evidence is not passing")
    if record.get("implementation_commit") != commit_sha:
        raise RuntimeError("production correctness evidence commit does not match benchmark")
    if record.get("production_voxelization_mode") != "exact_fast":
        raise RuntimeError("production correctness evidence did not use exact_fast")
    if record.get("provenance_mode") != "live":
        raise RuntimeError("production correctness evidence did not use live provenance")
    if record.get("artifacts") != EXPECTED_ARTIFACT_HASHES:
        raise RuntimeError("production correctness evidence artifact hashes changed")
    voxel_gate = record.get("voxel_gate")
    detector_gate = record.get("detector_and_ros_gate")
    if not isinstance(voxel_gate, Mapping) or voxel_gate.get("passed") is not True:
        raise RuntimeError("81-sample production voxel gate is not passing")
    if not isinstance(detector_gate, Mapping) or detector_gate.get("passed") is not True:
        raise RuntimeError("20-sample production detector/ROS gate is not passing")
    if runtime.voxelization_mode != "exact_fast" or runtime.provenance_mode != "live":
        raise RuntimeError("runtime policy differs from production correctness evidence")
    return {
        "path_logical_name": path.name,
        "sha256": _file_sha256(path),
        "implementation_commit": commit_sha,
        "voxel_gate_passed": True,
        "detector_and_ros_gate_passed": True,
    }


def _artifact_hashes(runtime: M3DetectorRuntime) -> dict[str, str]:
    return {
        "checkpoint": sha256_file(runtime.assets.checkpoint_path),
        "onnx": sha256_file(runtime.assets.onnx_path),
        "engine": sha256_file(runtime.assets.engine_path),
    }


def _sustained_warmup(
    runtime: M3DetectorRuntime, points: object, *, duration_seconds: float
) -> dict[str, object]:
    started = time.monotonic()
    count = 0
    while time.monotonic() - started < duration_seconds:
        runtime.infer(
            points,
            sample_id=f"warmup-{count}",
            coordinate_frame="nuscenes_lidar_top",
        )
        count += 1
    elapsed = time.monotonic() - started
    return {
        "requested_seconds": duration_seconds,
        "actual_seconds": elapsed,
        "inference_count": count,
        "passed": elapsed >= duration_seconds and count > 0,
    }


def _relabel_measured_halves(
    samples: Sequence[Mapping[str, object]], *, start_ns: int, end_ns: int
) -> list[dict[str, object]]:
    midpoint = start_ns + (end_ns - start_ns) // 2
    result: list[dict[str, object]] = []
    for sample in samples:
        copied = dict(sample)
        timestamp = copied.get("monotonic_ns")
        if isinstance(timestamp, int) and start_ns <= timestamp <= end_ns:
            copied["block"] = (
                "ros_measured_first_half" if timestamp < midpoint else "ros_measured_second_half"
            )
        result.append(copied)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--correctness-result", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--host-ac-confirmed", action="store_true")
    parser.add_argument("--host-performance-mode", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.rate_hz not in {1.0, 10.0, 15.0, 20.0}:
        raise SystemExit("rate must be the 1 Hz smoke or bounded 10/15/20 Hz protocol")
    if not args.smoke_only and (not args.host_ac_confirmed or not args.host_performance_mode):
        raise SystemExit("eligible measurement requires confirmed AC power and performance mode")
    commit_sha = repository_git_sha(_root())
    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            "publish_markers:=false",
            "-p",
            "voxelization_mode:=exact_fast",
            "-p",
            "provenance_mode:=live",
        ]
    )
    runtime = M3DetectorRuntime(voxelization_mode="exact_fast", provenance_mode="live")
    detector = LaserPerceptionDetectorNode(runtime=runtime)
    prepared = runtime.backend.prepare_sample(args.data_root, split="mini_val", index=SAMPLE_INDEX)
    points = prepared.model_ready_points()
    point_count = int(points.points_xyzt.shape[0])
    if point_count != EXPECTED_POINT_COUNT:
        raise SystemExit(
            f"W1 index 42 point count changed: expected {EXPECTED_POINT_COUNT}, found {point_count}"
        )
    artifacts = _artifact_hashes(runtime)
    if artifacts != EXPECTED_ARTIFACT_HASHES:
        raise SystemExit(f"frozen artifact SHA256 mismatch: {artifacts}")
    correctness: dict[str, object] | None = None
    if not args.smoke_only:
        if args.correctness_result is None:
            raise SystemExit("final measurement requires --correctness-result")
        correctness = _load_correctness(
            args.correctness_result.resolve(), commit_sha=commit_sha, runtime=runtime
        )

    publisher = _StressPublisher(points, rate_hz=args.rate_hz, timer_enabled=not args.smoke_only)
    sink = _LoopbackSink()
    executor = MultiThreadedExecutor(num_threads=3)
    for node in (publisher, detector, sink):
        executor.add_node(node)
    sampler = NvidiaSmiSampler(interval_seconds=TELEMETRY_INTERVAL_SECONDS)
    sampler.start()
    warmup_record: dict[str, object] | None = None
    ros_label = f"ros_{args.rate_hz:g}_hz"
    try:
        if args.smoke_only:
            discovery_deadline = time.monotonic() + 1.0
            while time.monotonic() < discovery_deadline:
                executor.spin_once(timeout_sec=0.05)
            publisher.publish_once()
            target = 1
        else:
            sampler.begin_block("sustained_gpu_warmup")
            warmup_record = _sustained_warmup(
                runtime, points, duration_seconds=SUSTAINED_WARMUP_SECONDS
            )
            sampler.end_block("sustained_gpu_warmup")
            target = WARMUPS + MEASURED
        sampler.begin_block(ros_label)
        deadline = time.monotonic() + args.timeout_seconds
        while len(sink.loopback_ms) < target and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        publisher.stop()
        settle_deadline = time.monotonic() + 2.0
        while time.monotonic() < settle_deadline:
            executor.spin_once(timeout_sec=0.05)
        sampler.end_block(ros_label)
    finally:
        sampler.stop()
        executor.shutdown()
        for node in (publisher, detector, sink):
            node.destroy_node()
        rclpy.shutdown()

    telemetry_samples = list(sampler.samples)
    if args.smoke_only:
        smoke_passed = (
            detector.accepted_count >= 1
            and detector.published_count >= 1
            and len(sink.loopback_ms) >= 1
            and detector.rejected_count == 0
        )
        result = {
            "schema_version": "1.0",
            "milestone": "M3",
            "status": "pass" if smoke_passed else "fail",
            "gate": "low_rate_production_correctness_smoke",
            "measurement_commit": commit_sha,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "sample": {"split": "mini_val", "index": SAMPLE_INDEX, "point_count": point_count},
            "voxelization_mode": runtime.voxelization_mode,
            "provenance_mode": runtime.provenance_mode,
            "artifacts": artifacts,
            "message_counts": {
                "published_input": publisher.published_count,
                "accepted_callbacks": detector.accepted_count,
                "published_detections": detector.published_count,
                "sink_received": len(sink.loopback_ms),
                "rejected": detector.rejected_count,
            },
        }
        output = args.output or runtime.assets.engine_path.parent.parent / "m3" / "smoke.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        print(f"external result: {output}")
        return 0 if smoke_passed else 1

    callback_values = detector.callback_latencies_ms[WARMUPS : WARMUPS + MEASURED]
    loopback_values = sink.loopback_ms[WARMUPS : WARMUPS + MEASURED]
    reception_times = sink.reception_perf_ns[WARMUPS : WARMUPS + MEASURED]
    entry_times = detector.callback_entry_perf_ns[WARMUPS : WARMUPS + MEASURED]
    accepted_stamps = detector.accepted_source_stamp_ns[WARMUPS : WARMUPS + MEASURED]
    if any(
        len(values) != MEASURED
        for values in (
            callback_values,
            loopback_values,
            reception_times,
            entry_times,
            accepted_stamps,
        )
    ):
        raise SystemExit(
            "M3 benchmark incomplete: "
            f"callback={len(callback_values)}, loopback={len(loopback_values)}, "
            f"entries={len(entry_times)}; do not promote"
        )
    callback = {**latency_statistics_ms(callback_values), **_deadline_summary(callback_values)}
    loopback = {**latency_statistics_ms(loopback_values), **_deadline_summary(loopback_values)}
    output_elapsed_seconds = (reception_times[-1] - reception_times[0]) / 1_000_000_000.0
    effective_output_hz = (MEASURED - 1) / output_elapsed_seconds

    first_stamp, last_stamp = accepted_stamps[0], accepted_stamps[-1]
    offered_indices = [
        index
        for index, stamp in enumerate(publisher.publication_ros_ns)
        if first_stamp <= stamp <= last_stamp
    ]
    offered_stamps = [publisher.publication_ros_ns[index] for index in offered_indices]
    offered_perf = [publisher.publication_perf_ns[index] for index in offered_indices]
    if len(offered_stamps) < MEASURED:
        raise SystemExit("accepted callbacks are not contained in the measured offered window")
    effective_offered_hz = (len(offered_perf) - 1) / (
        (offered_perf[-1] - offered_perf[0]) / 1_000_000_000.0
    )
    midpoint = len(offered_stamps) // 2
    accepted_set = set(accepted_stamps)
    first_drops = sum(stamp not in accepted_set for stamp in offered_stamps[:midpoint])
    second_drops = sum(stamp not in accepted_set for stamp in offered_stamps[midpoint:])
    measured_input_drops = first_drops + second_drops
    half_behavior = half_run_backlog_summary(
        _intervals_ms(entry_times),
        first_half_drops=first_drops,
        second_half_drops=second_drops,
    )

    relabeled = _relabel_measured_halves(
        telemetry_samples, start_ns=entry_times[0], end_ns=entry_times[-1]
    )
    eligibility = paired_gpu_state_eligibility(
        relabeled,
        [
            (
                "ros_measured_first_vs_second_half",
                "ros_measured_first_half",
                "ros_measured_second_half",
            )
        ],
    )
    total_input_loss = publisher.published_count - detector.received_count
    processing_backlog = detector.accepted_count - detector.published_count
    output_transport_loss = detector.published_count - len(sink.loopback_ms)
    no_loss_or_backlog = (
        measured_input_drops == 0
        and detector.rejected_count == 0
        and processing_backlog == 0
        and output_transport_loss == 0
    )
    replay_rate_pass = effective_offered_hz >= args.rate_hz * 0.95
    output_rate_pass = effective_output_hz >= args.rate_hz * 0.95
    stable_halves = not bool(half_behavior["falling_behind_between_halves"])
    sustainable_at_rate = (
        no_loss_or_backlog
        and replay_rate_pass
        and output_rate_pass
        and stable_halves
        and bool(eligibility["eligible"])
    )
    twenty_hz_pass = args.rate_hz == 20.0 and callback["median_ms"] <= 50.0 and sustainable_at_rate
    if not bool(eligibility["eligible"]):
        status = "ineligible_measurement"
    elif args.rate_hz == 20.0:
        status = (
            "representative_sensor_rate_demonstrated"
            if twenty_hz_pass
            else "representative_20_hz_not_sustained"
        )
    else:
        status = (
            "bounded_characterization_rate_sustained"
            if sustainable_at_rate
            else "bounded_characterization_rate_not_sustained"
        )

    result: dict[str, Any] = {
        "schema_version": "2.0",
        "milestone": "M3",
        "status": status,
        "measurement_commit": commit_sha,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "ros_distro": "humble",
            "ubuntu": "22.04",
            "python": "3.10",
            "rmw_implementation": rclpy.utilities.get_rmw_implementation_identifier(),
            "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU",
            "backend_versions": dict(runtime.backend.versions),
            "host_ac_power_confirmed_external": args.host_ac_confirmed,
            "host_performance_mode_confirmed_external": args.host_performance_mode,
        },
        "artifacts": artifacts,
        "correctness_prerequisite": correctness,
        "production_policy": {
            "voxelization_mode": runtime.voxelization_mode,
            "provenance_mode": runtime.provenance_mode,
            "fallback_allowed": False,
        },
        "workload": {
            "dataset": "nuScenes v1.0-mini",
            "split": "mini_val",
            "sample_index": SAMPLE_INDEX,
            "history": "10 historical sweeps plus current keyframe",
            "point_count": point_count,
        },
        "protocol": {
            "replay_rate_hz": args.rate_hz,
            "warmup_messages": WARMUPS,
            "measured_accepted_output_opportunities": MEASURED,
            "input_qos": "keep_last_depth_1_best_effort_volatile",
            "output_qos": "keep_last_depth_5_reliable_volatile",
            "markers_enabled": False,
            "executor": "three_thread_same_process_separate_ros_nodes",
            "replay_payload_construction": (
                "PointCloud2 payload built once before timing; only its source timestamp is "
                "refreshed immediately before each publish"
            ),
            "callback_boundary": (
                "PointCloud2 callback entry through Detection3DArray publish return"
            ),
            "loopback_boundary": (
                "input publisher ROS stamp through Detection3DArray sink reception"
            ),
        },
        "sustained_gpu_warmup": warmup_record,
        "callback_processing_latency": callback,
        "same_host_ros_loopback_latency": loopback,
        "callback_entry_and_drop_halves": half_behavior,
        "message_counts": {
            "replay_published_total": publisher.published_count,
            "measured_offered_messages": len(offered_stamps),
            "detector_received_total": detector.received_count,
            "detector_accepted_total": detector.accepted_count,
            "detections_published_total": detector.published_count,
            "sink_received_total": len(sink.loopback_ms),
            "detector_rejected_total": detector.rejected_count,
        },
        "sustained_rate": {
            "requested_offered_hz": args.rate_hz,
            "effective_offered_hz": effective_offered_hz,
            "effective_detector_output_hz": effective_output_hz,
            "output_offered_rate_fraction": effective_output_hz / args.rate_hz,
            "measured_input_drops": measured_input_drops,
            "first_half_input_drops": first_drops,
            "second_half_input_drops": second_drops,
            "total_input_messages_not_received": total_input_loss,
            "final_processing_backlog_messages": processing_backlog,
            "detector_to_sink_drops": output_transport_loss,
            "falling_behind_between_halves": not stable_halves,
            "sustainable_at_offered_rate": sustainable_at_rate,
        },
        "measurement_session": {
            "telemetry_interval_seconds": TELEMETRY_INTERVAL_SECONDS,
            "telemetry": {
                "summary": summarize_gpu_telemetry(telemetry_samples),
                "by_block": summarize_telemetry_by_block(relabeled),
                "raw_samples": telemetry_samples,
            },
            "eligibility": eligibility,
            "clock_capability": nvidia_clock_capability(),
        },
        "gate": {
            "callback_median_at_or_below_50_ms": callback["median_ms"] <= 50.0,
            "no_loss_or_final_backlog": no_loss_or_backlog,
            "effective_offered_at_least_95_percent": replay_rate_pass,
            "effective_output_at_least_95_percent": output_rate_pass,
            "no_first_to_second_half_deterioration": stable_halves,
            "session_eligible": bool(eligibility["eligible"]),
            "twenty_hz_operation_demonstrated": twenty_hz_pass,
        },
        "limitations": [
            "same-host loopback is not sensor-to-actuator latency",
            "input is already model-ready multi-sweep data; no TF or sweep accumulation is timed",
            "loopback excludes one-time construction of the immutable replay PointCloud2 payload",
            "telemetry establishes eligibility but does not prove clock causality",
        ],
    }
    output = args.output or (
        runtime.assets.engine_path.parent.parent / "m3" / f"ros_w1_{args.rate_hz:g}hz.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "measurement_session"},
            indent=2,
            sort_keys=True,
        )
    )
    print(f"external result: {output}")
    return 2 if status == "ineligible_measurement" else 0


if __name__ == "__main__":
    raise SystemExit(main())
