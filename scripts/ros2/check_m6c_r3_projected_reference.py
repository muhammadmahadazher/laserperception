"""Check whether an M6c projected offline reference is byte-exact through real ROS/tf2."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from laserperception_ros.conversion import pointcloud2_to_model_ready
from laserperception_ros.kitti_raw_replay_node import KittiRawReplayNode
from laserperception_ros.multisweep_node import LaserPerceptionMultiSweepNode, _time_message
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2

from laserperception.datasets.kitti_raw import KittiRawSequence
from laserperception.detection.live_multisweep import LiveRawSweep, sweep_transform_from_ros
from laserperception.detection.mmdet3d_backend import sha256_file
from laserperception.detection.multisweep import HistoricalSweep, SweepTransform
from laserperception.evaluation.m6c_projected_reference import build_projected_reference
from laserperception.evaluation.m6c_representation import array_sha256, compare_float32_arrays

BASE_MAIN_SHA = "ebbbc0bbc4423e3be476abcd1165f75a136fa54c"
STARTING_HEAD = "e64d80ff46bc735b7bec4ad568fa015731ada9eb"
R2_PROTOCOL_COMMIT = "0a8419978d265571b51f943ffc797b5fcc78c4ca"
M6A_SHA256 = "a62da9808079994d971c4d47bbdf04f2c50d44ddcfd0c7958f7534512552155b"
M6B_SHA256 = "b9d47120aabb38733d79f987f16277ebab626d2b7c13159c40c30a68b76c1d26"
M6B_LEDGER_SHA256 = "2413224808b0140856a0e00f884c53bc9b49ec6b545172f5e7fa57d00802dc15"
R2_FAILURE_SHA256 = "fd319dfcb13da570c62d83b4475531bbbd735845808225ebffee4afe03f1cea4"
D1_TRANSFORM_SHA256 = "07ea0434fb5833c96d8e6c619a8459cb43c30bbde97d5cfdba96ac8288f3db5d"
D1_DOWNSTREAM_SHA256 = "6346a9d0f9916ea4c6e2abb4e7f9c58587a49a5f3b4cbe7ac9d2a6b4b2c3cd3c"
FIXED_FRAME = "kitti_world"
LIDAR_FRAME = "kitti_model_aligned_lidar"
H10_CONDITIONS = (
    ("2011_09_26_drive_0001", 10, 10),
    ("2011_09_26_drive_0001", 107, 10),
    ("2011_09_26_drive_0091", 10, 10),
)
H5_CONDITION = ("2011_09_26_drive_0091", 10, 5)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=_root(), check=True, capture_output=True, text=True
    ).stdout.strip()


def _require_identity(implementation_commit: str) -> None:
    if _git("rev-parse", "HEAD") != implementation_commit:
        raise RuntimeError("R3 feasibility must run at the exact implementation commit")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("R3 feasibility requires a clean tracked worktree")
    for ancestor in (BASE_MAIN_SHA, STARTING_HEAD, R2_PROTOCOL_COMMIT):
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, implementation_commit],
            cwd=_root(),
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"implementation does not descend from required commit {ancestor}")


def _verify_frozen_evidence() -> dict[str, str]:
    paths = {
        "m6a": (
            _root() / "benchmarks/m6a/results/kitti_raw_offline_reconstruction.json",
            M6A_SHA256,
        ),
        "m6b": (
            _root() / "benchmarks/m6b/results/kitti_raw_cross_domain_characterization.json",
            M6B_SHA256,
        ),
        "m6b_input_ledger": (
            _root() / "benchmarks/m6b/diagnostics/pre_inference_input_ledger.json",
            M6B_LEDGER_SHA256,
        ),
        "r2_failure": (
            _root() / "benchmarks/m6c/diagnostics/gate_a_failure_frame_0000000001.json",
            R2_FAILURE_SHA256,
        ),
        "d1_transform": (
            _root() / "benchmarks/m6c/diagnostics/post_failure_tf_representation.json",
            D1_TRANSFORM_SHA256,
        ),
        "d1_downstream": (
            _root() / "benchmarks/m6c/diagnostics/post_failure_downstream_frame10_h10.json",
            D1_DOWNSTREAM_SHA256,
        ),
    }
    verified: dict[str, str] = {}
    for name, (path, expected) in paths.items():
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"frozen {name} identity mismatch")
        verified[name] = observed
    return verified


class _CaptureNode(Node):
    def __init__(self, topic: str, suffix: str) -> None:
        super().__init__(f"m6c_r3_feasibility_capture_{suffix}")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._condition = threading.Condition()
        self._messages: deque[PointCloud2] = deque()
        self._subscription = self.create_subscription(PointCloud2, topic, self._capture, qos)

    def _capture(self, message: PointCloud2) -> None:
        with self._condition:
            self._messages.append(message)
            self._condition.notify_all()

    def wait(self, timeout_sec: float) -> PointCloud2:
        deadline = time.monotonic() + timeout_sec
        with self._condition:
            while not self._messages:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("timed out waiting for projected-reference ROS output")
                self._condition.wait(remaining)
            return self._messages.popleft()


class _TransformCaptureBuilder(LaserPerceptionMultiSweepNode):
    """Capture real tf2 results while preserving the production conversion and builder."""

    def __init__(self, *, parameter_overrides: list[Any]) -> None:
        super().__init__(parameter_overrides=parameter_overrides)
        self._active_transforms: list[SweepTransform] = []
        self.last_transforms: tuple[SweepTransform, ...] = ()

    def _on_raw_points(self, message: PointCloud2) -> None:
        self._active_transforms = []
        super()._on_raw_points(message)
        self.last_transforms = tuple(self._active_transforms)

    def _historical_sweep(
        self,
        historical: LiveRawSweep,
        current: LiveRawSweep,
        target_frame: str,
    ) -> HistoricalSweep:
        transform = self._tf_buffer.lookup_transform_full(
            target_frame,
            Time.from_msg(_time_message(current)),
            historical.frame_id,
            Time.from_msg(_time_message(historical)),
            self._fixed_frame,
            timeout=Duration(seconds=self._transform_timeout_sec),
        )
        encoded = sweep_transform_from_ros(
            translation_xyz=(
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ),
            quaternion_xyzw=(
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ),
            source_id=historical.sweep.source_id,
            target_id=current.sweep.source_id,
        )
        self._active_transforms.append(encoded)
        return HistoricalSweep(historical.sweep, encoded)


def _stamp_nanoseconds(message: PointCloud2) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


def _capture_live_condition(
    data_root: Path,
    *,
    drive: str,
    current_index: int,
    history_depth: int,
    timeout_sec: float,
) -> tuple[PointCloud2, tuple[SweepTransform, ...], dict[str, int]]:
    suffix = f"{drive[-4:]}_{current_index}_{history_depth}"
    raw_topic = f"/laserperception/m6c/r3_feasibility/{suffix}/raw"
    model_topic = f"/laserperception/m6c/r3_feasibility/{suffix}/model_ready"
    start_index = current_index - history_depth
    replay = KittiRawReplayNode(
        parameter_overrides=[
            Parameter("raw_points_topic", value=raw_topic),
            Parameter("data_root", value=str(data_root)),
            Parameter("drive_id", value=drive),
            Parameter("start_frame", value=start_index),
            Parameter("end_frame", value=current_index),
            Parameter("auto_start", value=False),
            Parameter("fixed_frame", value=FIXED_FRAME),
            Parameter("lidar_frame", value=LIDAR_FRAME),
            Parameter("raw_qos_depth", value=5),
        ]
    )
    builder = _TransformCaptureBuilder(
        parameter_overrides=[
            Parameter("raw_points_topic", value=raw_topic),
            Parameter("model_ready_topic", value=model_topic),
            Parameter("fixed_frame", value=FIXED_FRAME),
            Parameter("target_frame", value=LIDAR_FRAME),
            Parameter("max_historical_sweeps", value=history_depth),
            Parameter("transform_timeout_sec", value=0.5),
            Parameter("tf_cache_time_sec", value=60.0),
            Parameter("raw_qos_depth", value=5),
            Parameter("model_ready_qos_depth", value=1),
        ]
    )
    capture = _CaptureNode(model_topic, suffix)
    executor = MultiThreadedExecutor(num_threads=3)
    for node in (replay, builder, capture):
        executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        discovery_deadline = time.monotonic() + 5.0
        while replay._publisher.get_subscription_count() == 0:
            if time.monotonic() >= discovery_deadline:
                raise TimeoutError("raw replay did not discover the feasibility builder")
            time.sleep(0.02)
        time.sleep(0.1)
        target: PointCloud2 | None = None
        for frame_index in range(start_index, current_index + 1):
            if replay.publish_next() != frame_index:
                raise RuntimeError("feasibility replay lost chronological frame identity")
            target = capture.wait(timeout_sec)
        if target is None:
            raise RuntimeError("feasibility replay produced no final model-ready message")
        counters = {
            "raw_frames_received": builder.raw_frames_received,
            "valid_raw_frames": builder.valid_raw_frames,
            "model_ready_outputs": builder.model_ready_frames_published,
            "rejected_frames": builder.rejected_frames,
            "tf_failures": builder.tf_failures,
            "history_resets": builder.history_resets,
            "history_depth": builder.current_history_depth,
        }
        if counters["rejected_frames"] or counters["tf_failures"]:
            raise RuntimeError(f"feasibility ROS condition rejected input: {counters}")
        if len(builder.last_transforms) != history_depth:
            raise RuntimeError("final ROS output did not expose the expected history depth")
        return target, builder.last_transforms, counters
    finally:
        executor.shutdown(timeout_sec=2.0)
        thread.join(timeout=2.0)
        capture.destroy_node()
        builder.destroy_node()
        replay.destroy_node()


def _transform_record(
    projected: SweepTransform,
    live: SweepTransform,
    *,
    current_index: int,
    historical_index: int,
) -> dict[str, object]:
    comparison = compare_float32_arrays(projected.lidar2sensor, live.lidar2sensor)
    return {
        "current_frame": f"{current_index:010d}",
        "historical_frame": f"{historical_index:010d}",
        "projected_sha256": array_sha256(projected.lidar2sensor),
        "live_tf2_sha256": array_sha256(live.lidar2sensor),
        "rotation_exact": np.array_equal(projected.lidar2sensor[:3, :3], live.lidar2sensor[:3, :3]),
        "translation_exact": np.array_equal(
            projected.lidar2sensor[:3, 3], live.lidar2sensor[:3, 3]
        ),
        "complete_transform_exact": bool(comparison["exact"]),
        "differing_float32_elements": int(comparison["differing_elements"]),
        "maximum_absolute_delta": float(comparison["maximum_absolute_difference"]),
        "ulp_distance": comparison["ulp_distance"],
        "differing_positions": comparison["differing_positions"],
    }


def _first_model_ready_difference(
    *,
    timestamp_exact: bool,
    history_exact: bool,
    point_count_exact: bool,
    shape_exact: bool,
    dtype_exact: bool,
    bytes_exact: bool,
) -> str | None:
    for name, exact in (
        ("timestamp", timestamp_exact),
        ("history_depth", history_exact),
        ("point_count", point_count_exact),
        ("shape", shape_exact),
        ("dtype", dtype_exact),
        ("row_order_or_xyzt_values", bytes_exact),
    ):
        if not exact:
            return name
    return None


def _run_condition(
    data_root: Path,
    *,
    drive: str,
    current_index: int,
    history_depth: int,
    timeout_sec: float,
) -> dict[str, object]:
    date_root = data_root / "2011_09_26"
    sequence = KittiRawSequence(date_root, date_root / f"{drive}_sync")
    projected = build_projected_reference(
        sequence, current_index=current_index, history_depth=history_depth
    )
    message, live_transforms, counters = _capture_live_condition(
        data_root,
        drive=drive,
        current_index=current_index,
        history_depth=history_depth,
        timeout_sec=timeout_sec,
    )
    live_cloud = pointcloud2_to_model_ready(message)
    transform_records = [
        _transform_record(
            projected_transform,
            live_transform,
            current_index=current_index,
            historical_index=historical_index,
        )
        for historical_index, projected_transform, live_transform in zip(
            projected.historical_indices,
            projected.transforms,
            live_transforms,
            strict=True,
        )
    ]
    expected_timestamp = sequence.timestamps[current_index].nanoseconds
    observed_timestamp = _stamp_nanoseconds(message)
    reference_points = projected.point_cloud.points_xyzt
    live_points = live_cloud.points_xyzt
    timestamp_exact = observed_timestamp == expected_timestamp
    history_exact = counters["history_depth"] == history_depth
    point_count_exact = len(reference_points) == len(live_points)
    shape_exact = reference_points.shape == live_points.shape
    dtype_exact = reference_points.dtype == live_points.dtype == np.dtype(np.float32)
    bytes_exact = shape_exact and np.array_equal(reference_points, live_points)
    model_exact = (
        timestamp_exact
        and history_exact
        and point_count_exact
        and shape_exact
        and dtype_exact
        and bytes_exact
        and message.header.frame_id == LIDAR_FRAME
    )
    record: dict[str, object] = {
        "drive": drive,
        "frame": f"{current_index:010d}",
        "condition": f"H{history_depth}",
        "selection_basis": "frozen before feasibility outcome",
        "historical_indices": [f"{value:010d}" for value in projected.historical_indices],
        "transforms": transform_records,
        "transform_summary": {
            "required": history_depth,
            "exact": sum(bool(item["complete_transform_exact"]) for item in transform_records),
            "non_exact": sum(
                not bool(item["complete_transform_exact"]) for item in transform_records
            ),
            "maximum_float32_delta": max(
                (float(item["maximum_absolute_delta"]) for item in transform_records),
                default=0.0,
            ),
        },
        "model_ready": {
            "exact": model_exact,
            "projected_sha256": projected.point_cloud.sha256,
            "live_ros_sha256": live_cloud.sha256,
            "timestamp_exact": timestamp_exact,
            "expected_timestamp_nanoseconds": expected_timestamp,
            "observed_timestamp_nanoseconds": observed_timestamp,
            "history_depth_exact": history_exact,
            "expected_history_depth": history_depth,
            "observed_history_depth": counters["history_depth"],
            "point_count_exact": point_count_exact,
            "projected_point_count": len(reference_points),
            "live_point_count": len(live_points),
            "shape_exact": shape_exact,
            "projected_shape": list(reference_points.shape),
            "live_shape": list(live_points.shape),
            "dtype_exact": dtype_exact,
            "projected_dtype": str(reference_points.dtype),
            "live_dtype": str(live_points.dtype),
            "row_order_and_xyzt_bytes_exact": bytes_exact,
            "frame_id_exact": message.header.frame_id == LIDAR_FRAME,
            "first_different_boundary": _first_model_ready_difference(
                timestamp_exact=timestamp_exact,
                history_exact=history_exact,
                point_count_exact=point_count_exact,
                shape_exact=shape_exact,
                dtype_exact=dtype_exact,
                bytes_exact=bytes_exact,
            ),
        },
        "ros_counters": counters,
    }
    print(
        f"{drive}/{current_index:010d}|H{history_depth}: "
        f"transforms={record['transform_summary']['exact']}/{history_depth} "  # type: ignore[index]
        f"model_ready_exact={model_exact}",
        flush=True,
    )
    return record


def _atomic_write(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--message-timeout-sec", type=float, default=15.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _require_identity(args.implementation_commit)
    source_evidence = _verify_frozen_evidence()
    data_root = args.data_root.expanduser().resolve()
    h10_records: list[dict[str, object]] = []
    h5_record: dict[str, object] | None = None
    rclpy.init()
    try:
        for drive, frame, depth in H10_CONDITIONS:
            h10_records.append(
                _run_condition(
                    data_root,
                    drive=drive,
                    current_index=frame,
                    history_depth=depth,
                    timeout_sec=args.message_timeout_sec,
                )
            )
        h10_transform_exact = sum(
            int(record["transform_summary"]["exact"])  # type: ignore[index]
            for record in h10_records
        )
        h10_model_exact = sum(bool(record["model_ready"]["exact"]) for record in h10_records)  # type: ignore[index]
        if h10_transform_exact == 30 and h10_model_exact == 3:
            drive, frame, depth = H5_CONDITION
            h5_record = _run_condition(
                data_root,
                drive=drive,
                current_index=frame,
                history_depth=depth,
                timeout_sec=args.message_timeout_sec,
            )
    finally:
        rclpy.shutdown()

    h10_transform_exact = sum(
        int(record["transform_summary"]["exact"])  # type: ignore[index]
        for record in h10_records
    )
    h10_model_exact = sum(bool(record["model_ready"]["exact"]) for record in h10_records)  # type: ignore[index]
    h5_exact = bool(
        h5_record is not None
        and h5_record["transform_summary"]["exact"] == 5  # type: ignore[index]
        and h5_record["model_ready"]["exact"]  # type: ignore[index]
    )
    if h10_transform_exact < 30:
        classification = "PROJECTED_REFERENCE_NOT_TRANSFORM_EXACT_AT_FULL_HISTORY"
    elif h10_model_exact < 3:
        classification = "PROJECTED_REFERENCE_TRANSFORM_EXACT_BUT_MODEL_READY_NOT_EXACT"
    elif h5_record is None:
        classification = "INCONCLUSIVE"
    elif h5_exact:
        classification = "PROJECTED_REFERENCE_BYTE_GATE_FEASIBLE"
    else:
        classification = "INCONCLUSIVE"
    record: dict[str, object] = {
        "schema_version": 1,
        "status": classification,
        "diagnostic_only": True,
        "draft_protocol_frozen": False,
        "branch": "feat/m6c-kitti-ros-exactness",
        "base_main": BASE_MAIN_SHA,
        "starting_head": STARTING_HEAD,
        "implementation_commit": args.implementation_commit,
        "r2_status": "FAILED",
        "r2_protocol_commit": R2_PROTOCOL_COMMIT,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "ros_distro": "Humble",
            "rmw_implementation": "rmw_fastrtps_cpp",
        },
        "source_evidence": source_evidence,
        "reference_independence": {
            "shared": [
                "accepted KITTI decoding and source poses",
                "frozen matrix-to-unit-quaternion conversion definition",
                "MultiSweepBuilder mathematical contract",
            ],
            "projected_reference_uses_tf2": False,
            "projected_reference_uses_ros_messages": False,
            "projected_reference_calls_live_builder_node": False,
            "live_path_uses_only_published_PointCloud2_and_tf2_after_replay": True,
        },
        "h10": {
            "conditions": h10_records,
            "transform_comparisons_required": 30,
            "transform_comparisons_exact": h10_transform_exact,
            "transform_comparisons_non_exact": 30 - h10_transform_exact,
            "model_ready_frames_required": 3,
            "model_ready_frames_exact": h10_model_exact,
            "model_ready_frames_non_exact": 3 - h10_model_exact,
            "maximum_float32_transform_delta": max(
                (
                    float(record["transform_summary"]["maximum_float32_delta"])  # type: ignore[index]
                    for record in h10_records
                ),
                default=0.0,
            ),
        },
        "optional_h5": {
            "eligible": h10_transform_exact == 30 and h10_model_exact == 3,
            "executed": h5_record is not None,
            "result": h5_record,
            "exact": h5_exact if h5_record is not None else None,
        },
        "scope": {
            "gpu_initialized": False,
            "tensorrt_initialized": False,
            "detector_inference_performed": False,
            "gate_a_rerun": False,
            "gate_b_started": False,
            "detector_sentinels_run": False,
            "performance_campaign": False,
        },
    }
    _atomic_write(args.output.expanduser().resolve(), record)
    print(json.dumps({"status": classification, "h10_exact": h10_model_exact, "h5": h5_exact}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
