"""Run resumable M6c Gates A/B through real PointCloud2 and time-aware tf2."""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rclpy
from laserperception_ros.conversion import pointcloud2_to_model_ready
from laserperception_ros.kitti_raw_replay_node import KittiRawReplayNode
from laserperception_ros.multisweep_node import LaserPerceptionMultiSweepNode
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2

from laserperception.detection.m6c_contract import M6cInputProgress, M6cProgressIdentity
from laserperception.detection.mmdet3d_backend import sha256_file

BASE_MAIN_SHA = "ebbbc0bbc4423e3be476abcd1165f75a136fa54c"
M6A_SHA256 = "a62da9808079994d971c4d47bbdf04f2c50d44ddcfd0c7958f7534512552155b"
M6B_LEDGER_SHA256 = "2413224808b0140856a0e00f884c53bc9b49ec6b545172f5e7fa57d00802dc15"
DRIVE_ENDS = {"2011_09_26_drive_0001": 107, "2011_09_26_drive_0091": 339}
CONDITION_DEPTHS = {"H10": 10, "H5": 5}
M6A_DRIVE = "2011_09_26_drive_0001"
FIXED_FRAME = "kitti_world"
LIDAR_FRAME = "kitti_model_aligned_lidar"


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON mapping in {path.name}")
    return value


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=_root(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_measurement_identity(protocol_commit: str, implementation_commit: str) -> None:
    if _git("rev-parse", "HEAD") != protocol_commit:
        raise RuntimeError("M6c input gate must run at the exact protocol commit")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("M6c input gate requires a clean tracked worktree")
    if not _git("merge-base", "--is-ancestor", BASE_MAIN_SHA, protocol_commit):
        raise RuntimeError("M6c protocol commit does not descend from the frozen base")
    if not _git("merge-base", "--is-ancestor", implementation_commit, protocol_commit):
        raise RuntimeError("M6c protocol does not descend from the frozen implementation")
    protocol_path = _root() / "docs/m6/M6C_PROTOCOL.md"
    if _git("log", "-1", "--format=%H", "--", str(protocol_path)) != protocol_commit:
        raise RuntimeError("M6c protocol file was not frozen by the claimed protocol commit")


class _CaptureNode(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("m6c_model_ready_capture")
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
                    raise TimeoutError("timed out waiting for model-ready PointCloud2")
                self._condition.wait(timeout=remaining)
            return self._messages.popleft()


def _stamp_ns(message: PointCloud2) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


def _atomic_write(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _counter_record(node: LaserPerceptionMultiSweepNode) -> dict[str, int]:
    return {
        "raw_frames_received": node.raw_frames_received,
        "valid_raw_frames": node.valid_raw_frames,
        "invalid_points_filtered": node.invalid_points_filtered,
        "model_ready_outputs": node.model_ready_frames_published,
        "rejected_frames": node.rejected_frames,
        "tf_failures": node.tf_failures,
        "history_resets": node.history_resets,
        "final_history_depth": node.current_history_depth,
    }


def _exact_record(
    message: PointCloud2,
    *,
    expected_sha256: str,
    expected_point_count: int,
    expected_history_depth: int,
    expected_timestamp_ns: int,
    observed_history_depth: int,
) -> tuple[dict[str, object], bool]:
    cloud = pointcloud2_to_model_ready(message)
    record: dict[str, object] = {
        "expected_sha256": expected_sha256,
        "observed_sha256": cloud.sha256,
        "expected_point_count": expected_point_count,
        "observed_point_count": len(cloud.points_xyzt),
        "expected_shape": [expected_point_count, 4],
        "observed_shape": list(cloud.points_xyzt.shape),
        "expected_dtype": "float32",
        "observed_dtype": str(cloud.points_xyzt.dtype),
        "expected_history_depth": expected_history_depth,
        "observed_history_depth": observed_history_depth,
        "expected_timestamp_nanoseconds": expected_timestamp_ns,
        "observed_timestamp_nanoseconds": _stamp_ns(message),
        "frame_id": str(message.header.frame_id),
    }
    exact = (
        cloud.sha256 == expected_sha256
        and len(cloud.points_xyzt) == expected_point_count
        and cloud.points_xyzt.shape == (expected_point_count, 4)
        and str(cloud.points_xyzt.dtype) == "float32"
        and observed_history_depth == expected_history_depth
        and _stamp_ns(message) == expected_timestamp_ns
        and message.header.frame_id == LIDAR_FRAME
    )
    record["exact"] = exact
    return record, exact


def _pass_targets(
    drive: str,
    condition: str,
    gate_a: M6cInputProgress,
    gate_b: M6cInputProgress,
    gate_a_expected: Mapping[str, Mapping[str, object]],
    gate_b_expected: Mapping[str, Mapping[str, object]],
) -> list[int]:
    targets: set[int] = set()
    for key in gate_b_expected:
        if key.startswith(f"{drive}/") and key.endswith(f"|{condition}") and not gate_b.passed(key):
            targets.add(int(key.split("/", 1)[1].split("|", 1)[0]))
    if drive == M6A_DRIVE and condition == "H10":
        for key in gate_a_expected:
            if not gate_a.passed(key):
                targets.add(int(key.split("/", 1)[1].split("|", 1)[0]))
    return sorted(targets)


def _run_pass(
    *,
    data_root: Path,
    drive: str,
    condition: str,
    targets: Sequence[int],
    gate_a: M6cInputProgress,
    gate_b: M6cInputProgress,
    gate_a_expected: Mapping[str, Mapping[str, object]],
    gate_b_expected: Mapping[str, Mapping[str, object]],
    timeout_sec: float,
) -> dict[str, object]:
    depth = CONDITION_DEPTHS[condition]
    start_frame = max(0, min(targets) - depth)
    end_frame = max(targets)
    suffix = f"{drive[-4:]}_{condition.lower()}"
    raw_topic = f"/laserperception/m6c/{suffix}/raw"
    model_topic = f"/laserperception/m6c/{suffix}/model_ready"
    replay = KittiRawReplayNode(
        parameter_overrides=[
            Parameter("raw_points_topic", value=raw_topic),
            Parameter("data_root", value=str(data_root)),
            Parameter("drive_id", value=drive),
            Parameter("start_frame", value=start_frame),
            Parameter("end_frame", value=end_frame),
            Parameter("auto_start", value=False),
            Parameter("fixed_frame", value=FIXED_FRAME),
            Parameter("lidar_frame", value=LIDAR_FRAME),
        ]
    )
    builder = LaserPerceptionMultiSweepNode(
        parameter_overrides=[
            Parameter("raw_points_topic", value=raw_topic),
            Parameter("model_ready_topic", value=model_topic),
            Parameter("fixed_frame", value=FIXED_FRAME),
            Parameter("target_frame", value=LIDAR_FRAME),
            Parameter("max_historical_sweeps", value=depth),
            Parameter("transform_timeout_sec", value=0.5),
            Parameter("tf_cache_time_sec", value=60.0),
            Parameter("raw_qos_depth", value=5),
            Parameter("model_ready_qos_depth", value=1),
        ]
    )
    capture = _CaptureNode(model_topic)
    executor = MultiThreadedExecutor(num_threads=3)
    for node in (replay, builder, capture):
        executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    verified = 0
    thread.start()
    try:
        discovery_deadline = time.monotonic() + 5.0
        while replay._publisher.get_subscription_count() == 0:
            if time.monotonic() >= discovery_deadline:
                raise TimeoutError("raw replay publisher did not discover the live builder")
            time.sleep(0.02)
        target_set = set(targets)
        for frame_index in range(start_frame, end_frame + 1):
            published = replay.publish_next()
            if published != frame_index:
                raise RuntimeError("KITTI replay did not preserve chronological frame identity")
            message = capture.wait(timeout_sec)
            expected_timestamp_ns = replay.sequence.timestamps[frame_index].nanoseconds
            if _stamp_ns(message) != expected_timestamp_ns:
                raise RuntimeError(
                    f"output timestamp mismatch before gate comparison: {drive}/{frame_index}"
                )
            if frame_index not in target_set:
                continue
            frame_text = f"{frame_index:010d}"
            key = f"{drive}/{frame_text}|{condition}"
            elapsed = time.monotonic() - started
            if key in gate_b_expected and not gate_b.passed(key):
                expected = gate_b_expected[key]
                record, exact = _exact_record(
                    message,
                    expected_sha256=str(expected["model_ready_input_sha256"]),
                    expected_point_count=int(expected["point_count"]),
                    expected_history_depth=int(expected["history_depth"]),
                    expected_timestamp_ns=expected_timestamp_ns,
                    observed_history_depth=builder.current_history_depth,
                )
                gate_b.mark(
                    key,
                    status="PASS" if exact else "FAIL",
                    expected_sha256=str(record["expected_sha256"]),
                    observed_sha256=str(record["observed_sha256"]),
                    point_count=int(record["observed_point_count"]),
                    history_depth=builder.current_history_depth,
                    timestamp_nanoseconds=int(record["observed_timestamp_nanoseconds"]),
                    elapsed_seconds=elapsed,
                )
                if not exact:
                    raise RuntimeError(f"M6c Gate B exactness failed: {key}: {record}")
                verified += 1
            if key in gate_a_expected and not gate_a.passed(key):
                expected = gate_a_expected[key]
                record, exact = _exact_record(
                    message,
                    expected_sha256=str(expected["output_sha256"]),
                    expected_point_count=int(expected["output_row_count"]),
                    expected_history_depth=int(expected["history_depth"]),
                    expected_timestamp_ns=int(expected["timestamp_ns"]),
                    observed_history_depth=builder.current_history_depth,
                )
                gate_a.mark(
                    key,
                    status="PASS" if exact else "FAIL",
                    expected_sha256=str(record["expected_sha256"]),
                    observed_sha256=str(record["observed_sha256"]),
                    point_count=int(record["observed_point_count"]),
                    history_depth=builder.current_history_depth,
                    timestamp_nanoseconds=int(record["observed_timestamp_nanoseconds"]),
                    elapsed_seconds=elapsed,
                )
                if not exact:
                    raise RuntimeError(f"M6c Gate A exactness failed: {key}: {record}")
                verified += 1
            if verified == 1 or verified % 25 == 0 or frame_index == end_frame:
                print(
                    f"M6c input progress: pass={drive} {condition} frame={frame_index}; "
                    f"session_verified={verified}; wall_seconds={elapsed:.1f}",
                    flush=True,
                )
        counters = _counter_record(builder)
        if counters["tf_failures"] or counters["rejected_frames"]:
            raise RuntimeError(f"canonical ROS input pass had rejected frames: {counters}")
        return {
            "drive": drive,
            "condition": condition,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "published_raw_frames": replay.published_count,
            "newly_verified_conditions": verified,
            "started_at_utc": started_utc,
            "wall_clock_progress_seconds": time.monotonic() - started,
            "wall_clock_note": "test-orchestration progress only; not a performance measurement",
            "counters": counters,
        }
    finally:
        executor.shutdown(timeout_sec=2.0)
        thread.join(timeout=2.0)
        capture.destroy_node()
        builder.destroy_node()
        replay.destroy_node()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol-commit", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--progress-root", type=Path, default=_root() / ".local/m6c")
    parser.add_argument("--message-timeout-sec", type=float, default=10.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _require_measurement_identity(args.protocol_commit, args.implementation_commit)
    root = _root()
    m6a_path = root / "benchmarks/m6a/results/kitti_raw_offline_reconstruction.json"
    m6b_path = root / "benchmarks/m6b/diagnostics/pre_inference_input_ledger.json"
    if sha256_file(m6a_path) != M6A_SHA256 or sha256_file(m6b_path) != M6B_LEDGER_SHA256:
        raise RuntimeError("frozen M6a/M6b input evidence identity mismatch")
    m6a = _load(m6a_path)
    m6b = _load(m6b_path)
    gate_a_expected = {
        f"{M6A_DRIVE}/{int(record['frame_index']):010d}|H10": record
        for record in m6a["offline_reconstruction"]["frames"]
    }
    gate_b_expected = {
        f"{record['drive']}/{record['frame']}|{record['condition']}": record
        for record in m6b["conditions"]
    }
    if len(gate_a_expected) != 24 or len(gate_b_expected) != 856:
        raise RuntimeError("frozen M6c input corpus cardinality changed")
    identity = M6cProgressIdentity(
        protocol_commit=args.protocol_commit,
        implementation_commit=args.implementation_commit,
        m6a_evidence_sha256=M6A_SHA256,
        m6b_input_ledger_sha256=M6B_LEDGER_SHA256,
    )
    progress_root = args.progress_root.expanduser().resolve()
    gate_a = M6cInputProgress(
        progress_root / "gate_a_progress.json", identity, sorted(gate_a_expected)
    )
    gate_b = M6cInputProgress(
        progress_root / "gate_b_progress.json", identity, sorted(gate_b_expected)
    )
    if gate_a.totals()["fail"] or gate_b.totals()["fail"]:
        raise RuntimeError("preserved failed M6c input condition requires owner review")

    rclpy.init()
    sessions: list[dict[str, object]] = []
    campaign_started = time.monotonic()
    try:
        for condition in ("H10", "H5"):
            for drive in DRIVE_ENDS:
                targets = _pass_targets(
                    drive,
                    condition,
                    gate_a,
                    gate_b,
                    gate_a_expected,
                    gate_b_expected,
                )
                if not targets:
                    continue
                sessions.append(
                    _run_pass(
                        data_root=args.data_root.expanduser().resolve(),
                        drive=drive,
                        condition=condition,
                        targets=targets,
                        gate_a=gate_a,
                        gate_b=gate_b,
                        gate_a_expected=gate_a_expected,
                        gate_b_expected=gate_b_expected,
                        timeout_sec=args.message_timeout_sec,
                    )
                )
    finally:
        rclpy.shutdown()
    gate_a_totals = gate_a.totals()
    gate_b_totals = gate_b.totals()
    overall_pass = gate_a_totals == {"pass": 24, "fail": 0, "pending": 0} and (
        gate_b_totals == {"pass": 856, "fail": 0, "pending": 0}
    )
    summary = {
        "schema_version": 1,
        "status": "PASS" if overall_pass else "INCOMPLETE",
        "protocol_commit": args.protocol_commit,
        "implementation_commit": args.implementation_commit,
        "source_evidence": {
            "m6a": {"sha256": M6A_SHA256},
            "m6b_input_ledger": {"sha256": M6B_LEDGER_SHA256},
        },
        "gate_a": {"required": 24, "totals": gate_a_totals, "exact": gate_a_totals["pass"] == 24},
        "gate_b": {
            "required": 856,
            "H10_required": 428,
            "H5_required": 428,
            "totals": gate_b_totals,
            "exact": gate_b_totals["pass"] == 856,
            "corpus_reduced_to_fit_session": False,
        },
        "sessions_this_invocation": sessions,
        "wall_clock_progress_seconds_this_invocation": time.monotonic() - campaign_started,
        "wall_clock_note": "test-orchestration progress only; not a performance measurement",
        "independence": {
            "replay_adapter_uses_KittiRawSequence": True,
            "builder_node_uses_only_PointCloud2_and_tf2": True,
            "builder_node_calls_KittiRawSequence": False,
        },
        "velocity_contract": "not exposed by current Detection3DArray conversion",
    }
    _atomic_write(progress_root / "input_gate_summary.json", summary)
    print(
        json.dumps({"status": summary["status"], "gate_a": gate_a_totals, "gate_b": gate_b_totals})
    )
    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
