"""Run M6c Gate D on ten preregistered detector conditions through ROS 2."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from laserperception_ros.conversion import detection_frame_to_message, pointcloud2_to_model_ready
from laserperception_ros.detector_node import LaserPerceptionDetectorNode
from laserperception_ros.kitti_raw_replay_node import KittiRawReplayNode
from laserperception_ros.multisweep_node import LaserPerceptionMultiSweepNode
from laserperception_ros.runtime import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_ONNX_SHA256,
    resolve_m3_assets,
)
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from vision_msgs.msg import Detection3DArray

from laserperception.detection.m2_backend import M2Backend
from laserperception.detection.m6c_contract import M6C_ENGINE_SHA256, require_file_sha256
from laserperception.detection.mmdet3d_backend import sha256_file
from laserperception.detection.ros2_contract import ModelReadyPointCloud
from laserperception.detection.types import Detection3D, DetectionFrame

BASE_MAIN_SHA = "ebbbc0bbc4423e3be476abcd1165f75a136fa54c"
FIXED_FRAME = "kitti_world"
LIDAR_FRAME = "kitti_model_aligned_lidar"
RAW_NAMES = ("cls_score", "bbox_pred", "dir_cls_pred")


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
        raise RuntimeError(f"expected a JSON mapping in {path.name}")
    return value


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _raw_hashes(raw: Mapping[str, list[Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in RAW_NAMES:
        values = raw[name]
        if len(values) != 1:
            raise RuntimeError(f"raw output {name} must contain one feature level")
        result[name] = _array_sha256(values[0].detach().cpu().contiguous().numpy())
    return result


def _require_measurement_identity(protocol_commit: str, implementation_commit: str) -> None:
    if _git("rev-parse", "HEAD") != protocol_commit:
        raise RuntimeError("M6c detector gate must run at the exact protocol commit")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("M6c detector gate requires a clean tracked worktree")
    if not _git_is_ancestor(BASE_MAIN_SHA, protocol_commit):
        raise RuntimeError("M6c protocol commit does not descend from the frozen base")
    if not _git_is_ancestor(implementation_commit, protocol_commit):
        raise RuntimeError("M6c protocol does not descend from the frozen implementation")
    protocol_relative = "docs/m6/M6C_PROTOCOL.md"
    if _git("log", "-1", "--format=%H", "--", protocol_relative) != protocol_commit:
        raise RuntimeError("M6c protocol file was not frozen by the claimed protocol commit")


def _detection_from_dict(record: Mapping[str, object]) -> Detection3D:
    velocity = record.get("velocity_xy")
    return Detection3D(
        center_xyz=tuple(float(value) for value in record["center_xyz"]),  # type: ignore[arg-type]
        size_lwh=tuple(float(value) for value in record["size_lwh"]),  # type: ignore[arg-type]
        yaw_rad=float(record["yaw_rad"]),
        score=float(record["score"]),
        class_id=int(record["class_id"]),
        class_name=str(record["class_name"]),
        velocity_xy=(
            None if velocity is None else tuple(float(value) for value in velocity)  # type: ignore[arg-type]
        ),
    )


def _frame_from_dict(record: Mapping[str, object]) -> DetectionFrame:
    detections = record.get("detections")
    metadata = record.get("metadata")
    if not isinstance(detections, Sequence) or not isinstance(metadata, Mapping):
        raise RuntimeError("frozen M6b DetectionFrame payload is malformed")
    return DetectionFrame(
        detections=tuple(_detection_from_dict(value) for value in detections),  # type: ignore[arg-type]
        sample_id=str(record["sample_id"]),
        coordinate_frame=str(record["coordinate_frame"]),
        metadata=dict(metadata),
    )


class _M6cDetectorRuntime:
    """Dedicated exactness runtime using the frozen 40k engine and shared backend."""

    def __init__(self, engine_path: Path) -> None:
        self.assets = resolve_m3_assets(engine_override=str(engine_path))
        require_file_sha256(
            self.assets.checkpoint_path,
            EXPECTED_CHECKPOINT_SHA256,
            artifact_name="frozen PointPillars checkpoint",
        )
        require_file_sha256(
            self.assets.onnx_path,
            EXPECTED_ONNX_SHA256,
            artifact_name="frozen PointPillars ONNX",
        )
        self.engine_sha256 = require_file_sha256(
            self.assets.engine_path,
            M6C_ENGINE_SHA256,
            artifact_name="M6 structural 40k TensorRT engine",
        )
        checkpoint_sha = str(self.assets.m1_manifest["model"]["checkpoint"]["sha256"])
        self.backend = M2Backend(
            self.assets.config_path,
            self.assets.checkpoint_path,
            self.assets.deploy_config_path,
            checkpoint_sha256=checkpoint_sha,
            voxelization_mode="exact_fast",
        )
        self.backend.initialize()
        self.backend._backend_model(self.assets.engine_path)
        self.target_sample_id = ""
        self.last_frame: DetectionFrame | None = None
        self.last_evidence: dict[str, object] | None = None

    def infer(
        self,
        points: ModelReadyPointCloud,
        *,
        sample_id: str,
        coordinate_frame: str,
    ) -> DetectionFrame:
        del sample_id
        if not self.target_sample_id:
            raise RuntimeError("M6c detector target identity was not frozen before callback")
        prepared = self.backend.prepare_model_ready_points(
            points,
            sample_id=self.target_sample_id,
            coordinate_frame=coordinate_frame,
        )
        voxelized = self.backend.voxelize(prepared)
        raw = self.backend.run_tensorrt_raw(voxelized, self.assets.engine_path)
        frame = self.backend.postprocess_raw(
            raw,
            voxelized,
            backend_name="tensorrt",
            precision="fp16",
            provenance_mode="full",
        )
        self.last_frame = frame
        self.last_evidence = {
            "model_ready_sha256": points.sha256,
            "voxel_count": voxelized.voxel_count,
            "voxel_hashes": voxelized.hashes(),
            "raw_output_hashes": _raw_hashes(raw),
            "detection_frame_sha256": _canonical_sha256(frame.to_dict()),
            "detection_count": len(frame.detections),
        }
        return frame


class _PointCapture(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("m6c_detector_model_ready_capture")
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
                    raise TimeoutError("timed out waiting for M6c model-ready target")
                self._condition.wait(remaining)
            return self._messages.popleft()


class _DetectionCapture(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("m6c_detection_capture")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._condition = threading.Condition()
        self._messages: deque[Detection3DArray] = deque()
        self._subscription = self.create_subscription(Detection3DArray, topic, self._capture, qos)

    def _capture(self, message: Detection3DArray) -> None:
        with self._condition:
            self._messages.append(message)
            self._condition.notify_all()

    def wait(self, timeout_sec: float) -> Detection3DArray:
        deadline = time.monotonic() + timeout_sec
        with self._condition:
            while not self._messages:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("timed out waiting for M6c Detection3DArray")
                self._condition.wait(remaining)
            return self._messages.popleft()


class _DetectorInputPublisher(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("m6c_detector_input_publisher")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.publisher = self.create_publisher(PointCloud2, topic, qos)


def _capture_model_ready(
    executor: MultiThreadedExecutor,
    *,
    data_root: Path,
    drive: str,
    frame_index: int,
    history_depth: int,
    case_index: int,
    timeout_sec: float,
) -> tuple[PointCloud2, dict[str, int]]:
    start_frame = max(0, frame_index - history_depth)
    suffix = f"case_{case_index}"
    raw_topic = f"/laserperception/m6c/{suffix}/raw"
    model_topic = f"/laserperception/m6c/{suffix}/model_ready"
    replay = KittiRawReplayNode(
        parameter_overrides=[
            Parameter("raw_points_topic", value=raw_topic),
            Parameter("data_root", value=str(data_root)),
            Parameter("drive_id", value=drive),
            Parameter("start_frame", value=start_frame),
            Parameter("end_frame", value=frame_index),
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
            Parameter("max_historical_sweeps", value=history_depth),
            Parameter("transform_timeout_sec", value=0.5),
            Parameter("tf_cache_time_sec", value=60.0),
        ]
    )
    capture = _PointCapture(model_topic)
    for node in (replay, builder, capture):
        executor.add_node(node)
    try:
        deadline = time.monotonic() + 5.0
        while replay._publisher.get_subscription_count() == 0:
            if time.monotonic() >= deadline:
                raise TimeoutError("KITTI replay did not discover the M6c builder")
            time.sleep(0.02)
        target: PointCloud2 | None = None
        for expected_index in range(start_frame, frame_index + 1):
            if replay.publish_next() != expected_index:
                raise RuntimeError("KITTI detector replay lost chronological identity")
            target = capture.wait(timeout_sec)
        if target is None:
            raise RuntimeError("KITTI detector replay produced no model-ready target")
        counters = {
            "raw_frames_received": builder.raw_frames_received,
            "valid_raw_frames": builder.valid_raw_frames,
            "invalid_points_filtered": builder.invalid_points_filtered,
            "model_ready_outputs": builder.model_ready_frames_published,
            "rejected_frames": builder.rejected_frames,
            "tf_failures": builder.tf_failures,
            "history_resets": builder.history_resets,
            "history_depth": builder.current_history_depth,
        }
        if counters["rejected_frames"] or counters["tf_failures"]:
            raise RuntimeError(f"M6c detector replay had rejected inputs: {counters}")
        return target, counters
    finally:
        for node in (capture, builder, replay):
            executor.remove_node(node)
        capture.destroy_node()
        builder.destroy_node()
        replay.destroy_node()


def _atomic_write(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--protocol-commit", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--sentinel-sha256", required=True)
    parser.add_argument(
        "--sentinels",
        type=Path,
        default=_root() / "benchmarks/m6c/preregistration/detector_sentinels.json",
    )
    parser.add_argument("--progress-root", type=Path, default=_root() / ".local/m6c")
    parser.add_argument("--message-timeout-sec", type=float, default=30.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _require_measurement_identity(args.protocol_commit, args.implementation_commit)
    progress_root = args.progress_root.expanduser().resolve()
    input_summary = _load(progress_root / "input_gate_summary.json")
    if input_summary.get("status") != "PASS":
        raise RuntimeError("M6c Gates A/B must pass before detector execution")
    if input_summary.get("protocol_commit") != args.protocol_commit:
        raise RuntimeError("M6c input gate protocol identity differs from detector gate")
    sentinel_path = args.sentinels.expanduser().resolve()
    if sha256_file(sentinel_path) != args.sentinel_sha256:
        raise RuntimeError("M6c preregistered detector sentinel SHA256 mismatch")
    preregistration = _load(sentinel_path)
    if preregistration.get("status") != "FROZEN_BEFORE_M6C_DETECTOR_EXECUTION":
        raise RuntimeError("M6c detector sentinels were not frozen before inference")
    sentinels = preregistration["sentinels"]
    if not isinstance(sentinels, Sequence) or len(sentinels) != 10:
        raise RuntimeError("M6c detector sentinel cardinality changed")
    if preregistration["ros_output_contract"]["velocity_exposed"] is not False:
        raise RuntimeError("M6c ROS velocity contract changed after protocol freeze")

    engine = args.engine.expanduser().resolve()
    runtime = _M6cDetectorRuntime(engine)
    detector_input_topic = "/laserperception/m6c/detector_input"
    detector_output_topic = "/laserperception/m6c/detections"
    detector = LaserPerceptionDetectorNode(
        runtime=runtime,
        parameter_overrides=[
            Parameter("input_topic", value=detector_input_topic),
            Parameter("output_topic", value=detector_output_topic),
            Parameter("publish_markers", value=False),
            Parameter("voxelization_mode", value="exact_fast"),
            Parameter("provenance_mode", value="full"),
            Parameter("engine_path", value=str(engine)),
        ],
    )
    publisher = _DetectorInputPublisher(detector_input_topic)
    detection_capture = _DetectionCapture(detector_output_topic)
    executor = MultiThreadedExecutor(num_threads=4)
    for node in (detector, publisher, detection_capture):
        executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    records: list[dict[str, object]] = []
    state_path = progress_root / "detector_gate_progress.json"
    campaign_started = time.monotonic()
    try:
        deadline = time.monotonic() + 5.0
        while publisher.publisher.get_subscription_count() == 0:
            if time.monotonic() >= deadline:
                raise TimeoutError("detector input publisher did not discover the detector node")
            time.sleep(0.02)
        for case_index, sentinel in enumerate(sentinels):
            if not isinstance(sentinel, Mapping):
                raise RuntimeError("M6c detector sentinel record is malformed")
            frame_id = str(sentinel["frame_id"])
            drive = str(sentinel["drive"])
            frame_index = int(sentinel["frame"])
            condition = str(sentinel["condition"])
            history_depth = int(sentinel["history_depth"])
            expected = sentinel["expected"]
            frame_payload = sentinel["detection_frame"]
            if not isinstance(expected, Mapping) or not isinstance(frame_payload, Mapping):
                raise RuntimeError("M6c detector expected payload is malformed")
            model_message, builder_counters = _capture_model_ready(
                executor,
                data_root=args.data_root.expanduser().resolve(),
                drive=drive,
                frame_index=frame_index,
                history_depth=history_depth,
                case_index=case_index,
                timeout_sec=args.message_timeout_sec,
            )
            model_cloud = pointcloud2_to_model_ready(model_message)
            runtime.target_sample_id = frame_id
            runtime.last_frame = None
            runtime.last_evidence = None
            publisher.publisher.publish(model_message)
            observed_message = detection_capture.wait(args.message_timeout_sec)
            if runtime.last_frame is None or runtime.last_evidence is None:
                raise RuntimeError("M6c detector callback returned without runtime evidence")
            expected_frame = _frame_from_dict(frame_payload)
            expected_message = detection_frame_to_message(expected_frame, model_message.header)
            comparisons = {
                "model_ready_input_exact": model_cloud.sha256 == expected["model_ready_sha256"],
                "voxel_count_exact": runtime.last_evidence["voxel_count"]
                == expected["voxel_count"],
                "voxel_tensors_exact": runtime.last_evidence["voxel_hashes"]
                == expected["voxel_hashes"],
                "raw_tensorrt_outputs_exact": (
                    runtime.last_evidence["raw_output_hashes"] == expected["raw_output_hashes"]
                ),
                "detection_frame_exact": (
                    runtime.last_evidence["detection_frame_sha256"]
                    == expected["detection_frame_sha256"]
                    and runtime.last_frame.to_dict() == dict(frame_payload)
                ),
                "detection_array_semantic_geometric_exact": observed_message == expected_message,
                "header_exact": observed_message.header == model_message.header,
                "per_detection_headers_exact": all(
                    detection.header == model_message.header
                    for detection in observed_message.detections
                ),
                "velocity_field_exposed": False,
            }
            exact = all(
                value for key, value in comparisons.items() if key != "velocity_field_exposed"
            )
            record = {
                "frame_id": frame_id,
                "condition": condition,
                "expected": dict(expected),
                "observed": dict(runtime.last_evidence),
                "observed_model_ready_sha256": model_cloud.sha256,
                "observed_detection_count": len(observed_message.detections),
                "comparisons": comparisons,
                "builder_counters": builder_counters,
                "status": "PASS" if exact else "FAIL",
            }
            records.append(record)
            _atomic_write(
                state_path,
                {
                    "schema_version": 1,
                    "protocol_commit": args.protocol_commit,
                    "implementation_commit": args.implementation_commit,
                    "engine_sha256": runtime.engine_sha256,
                    "sentinel_preregistration_sha256": args.sentinel_sha256,
                    "records": records,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            if not exact:
                raise RuntimeError(f"M6c Gate D exactness failed: {frame_id} {condition}")
            print(
                f"M6c detector exactness {case_index + 1}/10 PASS: {frame_id} {condition}",
                flush=True,
            )
    finally:
        executor.shutdown(timeout_sec=2.0)
        thread.join(timeout=2.0)
        detection_capture.destroy_node()
        publisher.destroy_node()
        detector.destroy_node()
        rclpy.shutdown()
    overall_pass = len(records) == 10 and all(record["status"] == "PASS" for record in records)
    result = {
        "schema_version": 1,
        "status": "PASS" if overall_pass else "FAIL",
        "protocol_commit": args.protocol_commit,
        "implementation_commit": args.implementation_commit,
        "engine_sha256": runtime.engine_sha256,
        "sentinel_preregistration": {
            "sha256": args.sentinel_sha256,
            "condition_count": 10,
        },
        "detector_gate": {
            "required": 10,
            "passed": sum(record["status"] == "PASS" for record in records),
            "exact": overall_pass,
            "records": records,
        },
        "detector_node_counters": {
            "received": detector.received_count,
            "accepted": detector.accepted_count,
            "published": detector.published_count,
            "rejected": detector.rejected_count,
        },
        "velocity_contract": {
            "exposed": False,
            "note": "The current Detection3DArray conversion does not overload velocity_xy.",
        },
        "wall_clock_progress_seconds": time.monotonic() - campaign_started,
        "wall_clock_note": "test-orchestration progress only; not a performance measurement",
    }
    _atomic_write(progress_root / "detector_gate_summary.json", result)
    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
