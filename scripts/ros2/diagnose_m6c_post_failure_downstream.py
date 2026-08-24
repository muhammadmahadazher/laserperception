"""Run the one preregistered M6c D1 frame-10/H10 downstream diagnostic."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from laserperception_ros.conversion import detection_frame_to_message, pointcloud2_to_model_ready
from laserperception_ros.detector_node import LaserPerceptionDetectorNode
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter
from validate_m6c_kitti_detector import (
    _array_sha256,
    _canonical_sha256,
    _capture_model_ready,
    _DetectionCapture,
    _DetectorInputPublisher,
    _frame_from_dict,
    _M6cDetectorRuntime,
    _raw_hashes,
)

from laserperception.datasets.kitti_raw import KittiRawSequence
from laserperception.detection.mmdet3d_backend import sha256_file
from laserperception.detection.multisweep import MultiSweepBuilder, MultiSweepBuilderConfig
from laserperception.detection.ros2_contract import ModelReadyPointCloud
from laserperception.detection.types import DetectionFrame
from laserperception.evaluation.m6b_input_oracle import reconstruct_from_frozen_transforms
from laserperception.evaluation.m6c_representation import (
    compare_float32_arrays,
    compare_voxel_structures,
    voxel_structure,
)

BASE_MAIN_SHA = "ebbbc0bbc4423e3be476abcd1165f75a136fa54c"
PLAN_COMMIT = "6a00cdc8fc2fa950ca7f8a4bf4261fdeeefbc6d9"
TRANSFORM_DIAGNOSTIC_COMMIT = "34d976f22acec713ac756ba48dc226d61d9a1142"
R2_PROTOCOL_COMMIT = "0a8419978d265571b51f943ffc797b5fcc78c4ca"
SENTINEL_SHA256 = "e80e803fe8923a52ced1bdecd17841a4cf915352a56ea5427607746409cf3be3"
FULL_RESULT_SHA256 = "87870b2aa0cc2a91d39331afc8154fdad0c8c796f1cabfb4f8530a3eb106de27"
FULL_LEDGER_SHA256 = "e25b3d62113cc7e8c1fcf736caa68b1ab698f965f007c758ff91d3e498ca6caa"
DRIVE = "2011_09_26_drive_0001"
FRAME_INDEX = 10
CONDITION = "H10"
FRAME_ID = f"{DRIVE}/{FRAME_INDEX:010d}"
RAW_NAMES = ("cls_score", "bbox_pred", "dir_cls_pred")


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=_root(), check=True, capture_output=True, text=True
    ).stdout.strip()


def _require_identity(diagnostic_commit: str) -> None:
    if _git("rev-parse", "HEAD") != diagnostic_commit:
        raise RuntimeError("D1 downstream must run at the exact committed implementation")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("D1 downstream requires a clean tracked worktree")
    for ancestor in (BASE_MAIN_SHA, R2_PROTOCOL_COMMIT, PLAN_COMMIT, TRANSFORM_DIAGNOSTIC_COMMIT):
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, diagnostic_commit],
            cwd=_root(),
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"D1 downstream commit does not descend from {ancestor}")


class _DiagnosticRuntime(_M6cDetectorRuntime):
    """Retain compact-comparison arrays without changing the shared detector pipeline."""

    def __init__(self, engine_path: Path) -> None:
        super().__init__(engine_path)
        self.last_arrays: dict[str, np.ndarray] | None = None

    def infer(
        self,
        points: ModelReadyPointCloud,
        *,
        sample_id: str,
        coordinate_frame: str,
    ) -> DetectionFrame:
        del sample_id
        if not self.target_sample_id:
            raise RuntimeError("D1 detector target identity was not frozen before inference")
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
        arrays = {
            "voxels": voxelized.voxels.detach().cpu().contiguous().numpy(),
            "num_points": voxelized.num_points.detach().cpu().contiguous().numpy(),
            "coors": voxelized.coors.detach().cpu().contiguous().numpy(),
        }
        for name in RAW_NAMES:
            values = raw[name]
            if len(values) != 1:
                raise RuntimeError(f"raw output {name} must have one feature level")
            arrays[name] = values[0].detach().cpu().contiguous().numpy()
        self.last_arrays = arrays
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


def _load_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON mapping in {path.name}")
    return value


def _sentinel_record(path: Path) -> dict[str, Any]:
    preregistration = _load_mapping(path)
    sentinels = preregistration.get("sentinels")
    if not isinstance(sentinels, Sequence) or len(sentinels) != 10:
        raise RuntimeError("frozen detector sentinel manifest is malformed")
    matches = [
        record
        for record in sentinels
        if isinstance(record, Mapping)
        and record.get("drive") == DRIVE
        and record.get("frame") == f"{FRAME_INDEX:010d}"
        and record.get("condition") == CONDITION
    ]
    if len(matches) != 1:
        raise RuntimeError("frozen frame-10/H10 detector sentinel is not unique")
    return dict(matches[0])


def _frozen_model_ready(
    data_root: Path,
    full_ledger: Mapping[str, object],
) -> ModelReadyPointCloud:
    frames = full_ledger.get("frames")
    if not isinstance(frames, Sequence):
        raise RuntimeError("full M6b input ledger frame list is malformed")
    matches = [
        record
        for record in frames
        if isinstance(record, Mapping) and record.get("frame_id") == FRAME_ID
    ]
    if len(matches) != 1:
        raise RuntimeError("full M6b ledger frame-10 identity is not unique")
    record = matches[0]
    transforms = record.get("frozen_sweep_transforms")
    h10 = record.get("h10")
    if not isinstance(transforms, Sequence) or not isinstance(h10, Mapping):
        raise RuntimeError("full M6b frame-10 reconstruction record is malformed")
    date_root = data_root / "2011_09_26"
    sequence = KittiRawSequence(date_root, date_root / f"{DRIVE}_sync")
    reconstruction = reconstruct_from_frozen_transforms(
        sequence,
        FRAME_INDEX,
        transforms,  # type: ignore[arg-type]
        builder=MultiSweepBuilder(MultiSweepBuilderConfig(max_historical_sweeps=10)),
    )
    expected_sha256 = str(h10["model_ready_sha256"])
    if reconstruction.point_cloud.sha256 != expected_sha256:
        raise RuntimeError("full-ledger frozen frame-10 input did not reproduce its identity")
    return reconstruction.point_cloud


def _copy_arrays(runtime: _DiagnosticRuntime) -> dict[str, np.ndarray]:
    if runtime.last_arrays is None:
        raise RuntimeError("diagnostic runtime did not retain comparison arrays")
    return {name: np.ascontiguousarray(value).copy() for name, value in runtime.last_arrays.items()}


def _canonical_control(
    runtime: _DiagnosticRuntime,
    frozen_cloud: ModelReadyPointCloud,
    sentinel: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, np.ndarray], DetectionFrame]:
    expected = sentinel.get("expected")
    frame_payload = sentinel.get("detection_frame")
    if not isinstance(expected, Mapping) or not isinstance(frame_payload, Mapping):
        raise RuntimeError("frozen sentinel expected payload is malformed")
    runtime.target_sample_id = FRAME_ID
    frame = runtime.infer(
        frozen_cloud,
        sample_id=FRAME_ID,
        coordinate_frame="kitti_model_aligned_lidar",
    )
    if runtime.last_evidence is None:
        raise RuntimeError("canonical control produced no runtime evidence")
    checks = {
        "model_ready": frozen_cloud.sha256 == expected["model_ready_sha256"],
        "voxel_count": runtime.last_evidence["voxel_count"] == expected["voxel_count"],
        "voxel_hashes": runtime.last_evidence["voxel_hashes"] == expected["voxel_hashes"],
        "raw_output_hashes": (
            runtime.last_evidence["raw_output_hashes"] == expected["raw_output_hashes"]
        ),
        "detection_frame_sha256": (
            runtime.last_evidence["detection_frame_sha256"] == expected["detection_frame_sha256"]
        ),
        "detection_frame_payload": frame.to_dict() == dict(frame_payload),
    }
    return (
        {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "observed": dict(runtime.last_evidence),
            "expected": dict(expected),
        },
        _copy_arrays(runtime),
        frame,
    )


def _detection_frame_comparison(
    expected: DetectionFrame,
    observed: DetectionFrame,
) -> dict[str, object]:
    exact = expected.to_dict() == observed.to_dict()
    record: dict[str, object] = {
        "exact": exact,
        "expected_sha256": _canonical_sha256(expected.to_dict()),
        "observed_sha256": _canonical_sha256(observed.to_dict()),
        "expected_count": len(expected.detections),
        "observed_count": len(observed.detections),
    }
    if len(expected.detections) != len(observed.detections):
        return record
    record["class_identity_exact"] = all(
        (left.class_id, left.class_name) == (right.class_id, right.class_name)
        for left, right in zip(expected.detections, observed.detections, strict=True)
    )
    fields = {
        "score": np.asarray([item.score for item in expected.detections], dtype=np.float32),
        "center": np.asarray([item.center_xyz for item in expected.detections], dtype=np.float32),
        "dimensions": np.asarray([item.size_lwh for item in expected.detections], dtype=np.float32),
        "axis_yaw": np.asarray([item.yaw_rad for item in expected.detections], dtype=np.float32),
        "velocity_xy": np.asarray(
            [item.velocity_xy for item in expected.detections], dtype=np.float32
        ),
    }
    observed_fields = {
        "score": np.asarray([item.score for item in observed.detections], dtype=np.float32),
        "center": np.asarray([item.center_xyz for item in observed.detections], dtype=np.float32),
        "dimensions": np.asarray([item.size_lwh for item in observed.detections], dtype=np.float32),
        "axis_yaw": np.asarray([item.yaw_rad for item in observed.detections], dtype=np.float32),
        "velocity_xy": np.asarray(
            [item.velocity_xy for item in observed.detections], dtype=np.float32
        ),
    }
    record["index_aligned_descriptive_deltas"] = {
        name: compare_float32_arrays(values, observed_fields[name])
        for name, values in fields.items()
    }
    return record


def _atomic_write(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument(
        "--sentinels",
        type=Path,
        default=_root() / "benchmarks/m6c/preregistration/detector_sentinels.json",
    )
    parser.add_argument("--full-ledger", type=Path, required=True)
    parser.add_argument("--full-result", type=Path, required=True)
    parser.add_argument("--diagnostic-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--message-timeout-sec", type=float, default=45.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _require_identity(args.diagnostic_commit)
    sentinel_path = args.sentinels.expanduser().resolve()
    full_ledger_path = args.full_ledger.expanduser().resolve()
    full_result_path = args.full_result.expanduser().resolve()
    if sha256_file(sentinel_path) != SENTINEL_SHA256:
        raise RuntimeError("frozen M6c sentinel manifest identity mismatch")
    if sha256_file(full_ledger_path) != FULL_LEDGER_SHA256:
        raise RuntimeError("full M6b input ledger identity mismatch")
    if sha256_file(full_result_path) != FULL_RESULT_SHA256:
        raise RuntimeError("full M6b result identity mismatch")

    sentinel = _sentinel_record(sentinel_path)
    expected_record = sentinel.get("expected")
    frame_payload = sentinel.get("detection_frame")
    if not isinstance(expected_record, Mapping) or not isinstance(frame_payload, Mapping):
        raise RuntimeError("frozen frame-10 sentinel payload is malformed")
    frozen_cloud = _frozen_model_ready(
        args.data_root.expanduser().resolve(), _load_mapping(full_ledger_path)
    )
    if frozen_cloud.sha256 != expected_record["model_ready_sha256"]:
        raise RuntimeError("frozen model-ready control identity differs from sentinel")

    runtime = _DiagnosticRuntime(args.engine.expanduser().resolve())
    control, control_arrays, control_frame = _canonical_control(runtime, frozen_cloud, sentinel)
    result: dict[str, object] = {
        "schema_version": 1,
        "diagnostic_only": True,
        "r2_status": "FAILED",
        "r2_protocol_commit": R2_PROTOCOL_COMMIT,
        "diagnostic_plan_commit": PLAN_COMMIT,
        "diagnostic_implementation_commit": args.diagnostic_commit,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "condition": {"drive": DRIVE, "frame": f"{FRAME_INDEX:010d}", "condition": CONDITION},
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "ros_distro": "Humble",
            "rmw_implementation": "rmw_fastrtps_cpp",
            "engine_sha256": runtime.engine_sha256,
        },
        "source_evidence": {
            "sentinel_sha256": SENTINEL_SHA256,
            "full_result_sha256": FULL_RESULT_SHA256,
            "full_input_ledger_sha256": FULL_LEDGER_SHA256,
        },
        "canonical_control": control,
        "scope": {
            "authorized_detector_conditions": 1,
            "detector_conditions_executed": 1,
            "gate_b_started": False,
            "remaining_sentinels_run": False,
            "performance_campaign": False,
        },
    }
    if control["status"] != "PASS":
        result["status"] = "CANONICAL_DOWNSTREAM_CONTROL_FAILED"
        _atomic_write(args.output.expanduser().resolve(), result)
        print(json.dumps({"status": result["status"]}))
        return 2

    rclpy.init()
    executor = MultiThreadedExecutor(num_threads=4)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    detector: LaserPerceptionDetectorNode | None = None
    publisher: _DetectorInputPublisher | None = None
    detection_capture: _DetectionCapture | None = None
    try:
        model_message, builder_counters = _capture_model_ready(
            executor,
            data_root=args.data_root.expanduser().resolve(),
            drive=DRIVE,
            frame_index=FRAME_INDEX,
            history_depth=10,
            case_index=0,
            timeout_sec=args.message_timeout_sec,
        )
        ros_cloud = pointcloud2_to_model_ready(model_message)
        expected_structure = voxel_structure(frozen_cloud.points_xyzt, include_values=False)
        ros_structure = voxel_structure(ros_cloud.points_xyzt, include_values=False)
        structure_comparison = compare_voxel_structures(expected_structure, ros_structure)

        input_topic = "/laserperception/m6c/d1/detector_input"
        output_topic = "/laserperception/m6c/d1/detections"
        detector = LaserPerceptionDetectorNode(
            runtime=runtime,
            parameter_overrides=[
                Parameter("input_topic", value=input_topic),
                Parameter("output_topic", value=output_topic),
                Parameter("publish_markers", value=False),
                Parameter("voxelization_mode", value="exact_fast"),
                Parameter("provenance_mode", value="full"),
                Parameter("engine_path", value=str(args.engine.expanduser().resolve())),
            ],
        )
        publisher = _DetectorInputPublisher(input_topic)
        detection_capture = _DetectionCapture(output_topic)
        for node in (detector, publisher, detection_capture):
            executor.add_node(node)
        deadline = time.monotonic() + 5.0
        while publisher.publisher.get_subscription_count() == 0:
            if time.monotonic() >= deadline:
                raise TimeoutError("D1 detector publisher did not discover the detector node")
            time.sleep(0.02)
        runtime.target_sample_id = FRAME_ID
        runtime.last_arrays = None
        runtime.last_evidence = None
        runtime.last_frame = None
        publisher.publisher.publish(model_message)
        observed_message = detection_capture.wait(args.message_timeout_sec)
        if runtime.last_evidence is None or runtime.last_frame is None:
            raise RuntimeError("D1 ROS detector callback produced no evidence")
        ros_arrays = _copy_arrays(runtime)
        expected_frame = _frame_from_dict(frame_payload)
        expected_message = detection_frame_to_message(expected_frame, model_message.header)
        raw_comparison = {
            name: compare_float32_arrays(control_arrays[name], ros_arrays[name])
            for name in RAW_NAMES
        }
        voxel_feature_comparison = compare_float32_arrays(
            control_arrays["voxels"], ros_arrays["voxels"]
        )
        coors_exact = np.array_equal(control_arrays["coors"], ros_arrays["coors"])
        num_points_exact = np.array_equal(control_arrays["num_points"], ros_arrays["num_points"])
        result.update(
            {
                "status": "DOWNSTREAM_DIAGNOSTIC_COMPLETE",
                "model_ready": compare_float32_arrays(
                    frozen_cloud.points_xyzt, ros_cloud.points_xyzt
                ),
                "voxel_structure": structure_comparison,
                "exact_fast_outputs": {
                    "expected_voxel_count": int(len(control_arrays["coors"])),
                    "observed_voxel_count": int(len(ros_arrays["coors"])),
                    "coors_exact": coors_exact,
                    "coors_expected_sha256": _array_sha256(control_arrays["coors"]),
                    "coors_observed_sha256": _array_sha256(ros_arrays["coors"]),
                    "num_points_exact": num_points_exact,
                    "num_points_expected_sha256": _array_sha256(control_arrays["num_points"]),
                    "num_points_observed_sha256": _array_sha256(ros_arrays["num_points"]),
                    "voxel_feature_values": voxel_feature_comparison,
                },
                "raw_tensorrt_outputs": raw_comparison,
                "detection_frame": _detection_frame_comparison(control_frame, runtime.last_frame),
                "detection_array": {
                    "semantic_exact": observed_message == expected_message,
                    "header_exact": observed_message.header == model_message.header,
                    "per_detection_headers_exact": all(
                        detection.header == model_message.header
                        for detection in observed_message.detections
                    ),
                    "expected_count": len(expected_message.detections),
                    "observed_count": len(observed_message.detections),
                    "velocity_exposed": False,
                },
                "ros_builder_counters": builder_counters,
                "ros_detector_counters": {
                    "received": detector.received_count,
                    "accepted": detector.accepted_count,
                    "published": detector.published_count,
                    "rejected": detector.rejected_count,
                },
                "observed_hashes": {
                    "model_ready": ros_cloud.sha256,
                    "voxel": dict(runtime.last_evidence["voxel_hashes"]),
                    "raw": dict(runtime.last_evidence["raw_output_hashes"]),
                    "detection_frame": runtime.last_evidence["detection_frame_sha256"],
                },
                "scope": {
                    "authorized_detector_conditions": 1,
                    "detector_conditions_executed": 1,
                    "network_executions": 2,
                    "execution_note": "one frozen control plus one ROS variant for one condition",
                    "gate_b_started": False,
                    "remaining_sentinels_run": False,
                    "performance_campaign": False,
                },
            }
        )
    finally:
        if detector is not None and publisher is not None and detection_capture is not None:
            for node in (detection_capture, publisher, detector):
                executor.remove_node(node)
            detection_capture.destroy_node()
            publisher.destroy_node()
            detector.destroy_node()
        executor.shutdown(timeout_sec=2.0)
        thread.join(timeout=2.0)
        rclpy.shutdown()
    _atomic_write(args.output.expanduser().resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "model_ready_exact": result["model_ready"]["exact"],  # type: ignore[index]
                "detection_frame_exact": result["detection_frame"]["exact"],  # type: ignore[index]
                "detection_array_exact": result["detection_array"]["semantic_exact"],  # type: ignore[index]
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
