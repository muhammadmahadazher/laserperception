"""Run final M6c R3 Gate 1 against frozen projected-reference identities."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import threading
import time
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from laserperception_ros.conversion import pointcloud2_to_model_ready
from laserperception_ros.kitti_raw_replay_node import KittiRawReplayNode
from laserperception_ros.multisweep_node import LaserPerceptionMultiSweepNode
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2

from laserperception.detection.m6c_contract import M6cR3InputProgress, M6cR3ProgressIdentity
from laserperception.detection.mmdet3d_backend import sha256_file

BASE_MAIN_SHA = "ebbbc0bbc4423e3be476abcd1165f75a136fa54c"
REFERENCE_GENERATION_COMMIT = "03ce7729bea0d76028783234dee559fe32cf21db"
PROJECTED_MANIFEST_SHA256 = "c06cddc6884fef87de99d1c68ec2b5c1f1945f7f9e5ecae6fcb3e4275dd952a2"
FIXED_FRAME = "kitti_world"
LIDAR_FRAME = "kitti_model_aligned_lidar"


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=_root(), check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_is_ancestor(ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=_root(),
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def _atomic_write(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _require_measurement_identity(
    *,
    protocol_commit: str,
    implementation_commit: str,
    manifest_path: Path,
    manifest_sha256: str,
) -> str:
    if _git("rev-parse", "HEAD") != protocol_commit:
        raise RuntimeError("M6c R3 Gate 1 must run at the exact protocol commit")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("M6c R3 Gate 1 requires a clean tracked worktree")
    for ancestor in (BASE_MAIN_SHA, REFERENCE_GENERATION_COMMIT, implementation_commit):
        if not _git_is_ancestor(ancestor, protocol_commit):
            raise RuntimeError(f"R3 protocol does not descend from required commit {ancestor}")
    protocol_relative = "docs/m6/M6C_PROTOCOL_R3.md"
    if _git("log", "-1", "--format=%H", "--", protocol_relative) != protocol_commit:
        raise RuntimeError("final R3 protocol was not frozen by the claimed protocol commit")
    if manifest_sha256 != PROJECTED_MANIFEST_SHA256:
        raise RuntimeError("projected manifest CLI identity differs from the frozen protocol")
    observed_manifest_sha = sha256_file(manifest_path)
    if observed_manifest_sha != manifest_sha256:
        raise RuntimeError("projected-reference manifest SHA256 mismatch")
    return sha256_file(_root() / protocol_relative)


class _CaptureNode(Node):
    def __init__(self, topic: str, suffix: str) -> None:
        super().__init__(f"m6c_r3_gate1_capture_{suffix}")
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
                    raise TimeoutError("timed out waiting for final R3 model-ready PointCloud2")
                self._condition.wait(timeout=remaining)
            return self._messages.popleft()


def _stamp_ns(message: PointCloud2) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


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


def _record_passed(progress: M6cR3InputProgress, record: Mapping[str, object]) -> bool:
    return progress.passed(
        str(record["key"]),
        expected_sha256=str(record["model_ready_sha256"]),
        expected_point_count=int(record["point_count"]),
        expected_history_depth=int(record["expected_actual_history_depth"]),
        expected_timestamp_nanoseconds=int(record["official_timestamp_nanoseconds"]),
    )


def _compare_and_mark(
    message: PointCloud2,
    *,
    expected: Mapping[str, object],
    observed_history_depth: int,
    progress: M6cR3InputProgress,
    elapsed_seconds: float,
) -> bool:
    cloud = pointcloud2_to_model_ready(message)
    expected_sha = str(expected["model_ready_sha256"])
    expected_count = int(expected["point_count"])
    expected_history = int(expected["expected_actual_history_depth"])
    expected_timestamp = int(expected["official_timestamp_nanoseconds"])
    observed_timestamp = _stamp_ns(message)
    exact = all(
        (
            cloud.sha256 == expected_sha,
            len(cloud.points_xyzt) == expected_count,
            cloud.points_xyzt.shape == (expected_count, 4),
            cloud.points_xyzt.dtype == np.dtype(np.float32),
            observed_history_depth == expected_history,
            observed_timestamp == expected_timestamp,
            message.header.frame_id == LIDAR_FRAME,
        )
    )
    progress.mark(
        str(expected["key"]),
        status="PASS" if exact else "FAIL",
        expected_sha256=expected_sha,
        observed_sha256=cloud.sha256,
        expected_point_count=expected_count,
        observed_point_count=len(cloud.points_xyzt),
        expected_history_depth=expected_history,
        observed_history_depth=observed_history_depth,
        expected_timestamp_nanoseconds=expected_timestamp,
        observed_timestamp_nanoseconds=observed_timestamp,
        elapsed_seconds=elapsed_seconds,
    )
    return exact


def _run_pass(
    *,
    data_root: Path,
    drive: str,
    condition: str,
    targets: Sequence[Mapping[str, object]],
    progress: M6cR3InputProgress,
    timeout_sec: float,
) -> dict[str, object]:
    depth = int(str(condition).removeprefix("H"))
    target_by_frame = {int(str(record["frame"])): record for record in targets}
    start_frame = max(0, min(target_by_frame) - depth)
    end_frame = max(target_by_frame)
    suffix = f"drive_{drive[-4:]}_{condition.lower()}"
    raw_topic = f"/laserperception/m6c/r3/{suffix}/raw"
    model_topic = f"/laserperception/m6c/r3/{suffix}/model_ready"
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
            Parameter("raw_qos_depth", value=5),
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
    capture = _CaptureNode(model_topic, suffix)
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
                raise TimeoutError("R3 replay publisher did not discover the live builder")
            time.sleep(0.02)
        time.sleep(0.1)
        for frame_index in range(start_frame, end_frame + 1):
            if replay.publish_next() != frame_index:
                raise RuntimeError("R3 chronological replay lost frame identity")
            message = capture.wait(timeout_sec)
            expected = target_by_frame.get(frame_index)
            if expected is None:
                continue
            elapsed = time.monotonic() - started
            if not _compare_and_mark(
                message,
                expected=expected,
                observed_history_depth=builder.current_history_depth,
                progress=progress,
                elapsed_seconds=elapsed,
            ):
                raise RuntimeError(f"M6c R3 Gate 1 exactness failed: {expected['key']}")
            verified += 1
            if verified == 1 or verified % 25 == 0 or frame_index == end_frame:
                print(
                    f"R3 Gate 1 progress: {drive} {condition} frame={frame_index}; "
                    f"session_verified={verified}/{len(targets)}; wall_seconds={elapsed:.1f}",
                    flush=True,
                )
        counters = _counter_record(builder)
        if counters["tf_failures"] or counters["rejected_frames"]:
            raise RuntimeError(f"R3 live pass had rejected frames: {counters}")
        return {
            "drive": drive,
            "condition": condition,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "published_raw_frames": replay.published_count,
            "newly_verified_conditions": verified,
            "started_at_utc": started_utc,
            "wall_clock_progress_seconds": time.monotonic() - started,
            "wall_clock_note": "orchestration metadata only; not performance evidence",
            "counters": counters,
        }
    finally:
        executor.shutdown(timeout_sec=2.0)
        thread.join(timeout=2.0)
        capture.destroy_node()
        builder.destroy_node()
        replay.destroy_node()


def _gate_totals(
    records: Sequence[Mapping[str, object]],
    progress: M6cR3InputProgress,
    membership: str,
) -> dict[str, int]:
    statuses = [
        str(progress.conditions()[str(record["key"])]["status"])  # type: ignore[index]
        for record in records
        if membership in record["gate_membership"]
    ]
    return {name.lower(): statuses.count(name) for name in ("PASS", "FAIL", "PENDING")}


def _compact_conditions(
    records: Sequence[Mapping[str, object]], progress: M6cR3InputProgress
) -> list[dict[str, object]]:
    compact: list[dict[str, object]] = []
    ledger = progress.conditions()
    for record in records:
        state = ledger[str(record["key"])]
        assert isinstance(state, Mapping)
        compact.append(
            {
                "key": record["key"],
                "gate_membership": record["gate_membership"],
                "status": state["status"],
                "expected_sha256": record["model_ready_sha256"],
                "observed_sha256": state.get("observed_sha256"),
                "point_count": state.get("observed_point_count"),
                "history_depth": state.get("observed_history_depth"),
                "timestamp_nanoseconds": state.get("observed_timestamp_nanoseconds"),
            }
        )
    return compact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol-commit", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_root() / "benchmarks/m6c/preregistration/projected_reference_manifest.json",
    )
    parser.add_argument("--manifest-sha256", default=PROJECTED_MANIFEST_SHA256)
    parser.add_argument("--progress-root", type=Path, default=_root() / ".local/m6c-r3")
    parser.add_argument("--result-output", type=Path)
    parser.add_argument("--message-timeout-sec", type=float, default=20.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    protocol_file_sha = _require_measurement_identity(
        protocol_commit=args.protocol_commit,
        implementation_commit=args.implementation_commit,
        manifest_path=manifest_path,
        manifest_sha256=args.manifest_sha256,
    )
    manifest = _load(manifest_path)
    if manifest.get("status") != "FROZEN_PROJECTED_REFERENCE_IDENTITIES_BEFORE_LIVE_R3":
        raise RuntimeError("projected-reference manifest is not frozen")
    records = manifest.get("conditions")
    if not isinstance(records, Sequence) or len(records) != 860:
        raise RuntimeError("projected-reference manifest must contain 860 conditions")
    condition_records = [dict(record) for record in records if isinstance(record, Mapping)]
    if len(condition_records) != 860:
        raise RuntimeError("projected-reference condition record is malformed")
    identity = M6cR3ProgressIdentity(
        protocol_commit=args.protocol_commit,
        implementation_commit=args.implementation_commit,
        projected_manifest_sha256=args.manifest_sha256,
    )
    progress_root = args.progress_root.expanduser().resolve()
    progress = M6cR3InputProgress(
        progress_root / "gate1_progress.json",
        identity,
        [str(record["key"]) for record in condition_records],
    )
    if progress.totals()["fail"]:
        raise RuntimeError("preserved failed R3 Gate 1 condition requires final negative closure")

    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for record in condition_records:
        if not _record_passed(progress, record):
            grouped[(str(record["drive"]), str(record["condition"]))].append(record)
    sessions: list[dict[str, object]] = []
    failure: str | None = None
    campaign_started = time.monotonic()
    rclpy.init()
    try:
        for drive, condition in (
            ("2011_09_26_drive_0001", "H10"),
            ("2011_09_26_drive_0091", "H10"),
            ("2011_09_26_drive_0001", "H5"),
            ("2011_09_26_drive_0091", "H5"),
        ):
            targets = grouped.get((drive, condition), [])
            if targets:
                sessions.append(
                    _run_pass(
                        data_root=args.data_root.expanduser().resolve(),
                        drive=drive,
                        condition=condition,
                        targets=targets,
                        progress=progress,
                        timeout_sec=args.message_timeout_sec,
                    )
                )
    except Exception as error:  # preserve the first fail-closed condition before returning
        failure = f"{type(error).__name__}: {error}"
    finally:
        rclpy.shutdown()

    unique_totals = progress.totals()
    gate_1a = _gate_totals(condition_records, progress, "Gate1A")
    gate_1b = _gate_totals(condition_records, progress, "Gate1B")
    passed = (
        gate_1a == {"pass": 24, "fail": 0, "pending": 0}
        and gate_1b == {"pass": 856, "fail": 0, "pending": 0}
        and unique_totals == {"pass": 860, "fail": 0, "pending": 0}
    )
    status = "PASS" if passed else ("FAIL" if unique_totals["fail"] else "INCOMPLETE")
    aggregate_counters = {
        name: sum(int(session["counters"][name]) for session in sessions)  # type: ignore[index]
        for name in (
            "raw_frames_received",
            "valid_raw_frames",
            "invalid_points_filtered",
            "model_ready_outputs",
            "rejected_frames",
            "tf_failures",
            "history_resets",
        )
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "status": status,
        "scientific_classification": (
            "GATE_1_PROJECTED_ROS_INPUT_EXACTNESS_PASS"
            if passed
            else "GATE_1_PROJECTED_ROS_INPUT_EXACTNESS_NOT_PASS"
        ),
        "protocol_commit": args.protocol_commit,
        "protocol_file_sha256": protocol_file_sha,
        "measurement_implementation_commit": args.implementation_commit,
        "projected_manifest_sha256": args.manifest_sha256,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "ros_distro": "Humble",
            "rmw_implementation": "rmw_fastrtps_cpp",
        },
        "gate_1a": {"required": 24, "totals": gate_1a, "exact": gate_1a["pass"] == 24},
        "gate_1b": {"required": 856, "totals": gate_1b, "exact": gate_1b["pass"] == 856},
        "unique_conditions": {
            "required": 860,
            "totals": unique_totals,
            "exact": unique_totals["pass"] == 860,
        },
        "overlap": {
            "gate_1a_gate_1b_shared_conditions": 20,
            "redundantly_replayed": False,
        },
        "ros_counters_this_invocation": aggregate_counters,
        "sessions_this_invocation": sessions,
        "wall_clock_progress_seconds_this_invocation": time.monotonic() - campaign_started,
        "wall_clock_note": "orchestration metadata only; not performance evidence",
        "failure": failure,
        "conditions": _compact_conditions(condition_records, progress),
    }
    output = (
        args.result_output.expanduser().resolve()
        if args.result_output is not None
        else progress_root / "input_gate_summary.json"
    )
    _atomic_write(output, result)
    if output != progress_root / "input_gate_summary.json":
        _atomic_write(progress_root / "input_gate_summary.json", result)
    print(json.dumps({"status": status, "gate_1a": gate_1a, "gate_1b": gate_1b}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
