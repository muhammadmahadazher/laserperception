"""Run the final exact M4.5b raw-ROS-to-detector correctness gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
import yaml
from laserperception_ros.conversion import (
    detection_frame_to_message,
    pointcloud2_to_model_ready,
)
from laserperception_ros.runtime import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_ENGINE_SHA256,
    EXPECTED_ONNX_SHA256,
    M3DetectorRuntime,
)
from validate_m45b_raw_ros import _capture_case, _tf2_version

from laserperception.detection.mmdet3d_backend import sha256_file
from laserperception.detection.types import DetectionFrame

BASE_MAIN_SHA = "9c0fecbb45ebb1d0c65e61a99f13b72558327527"
EXPECTED_REPAIR_EVIDENCE_SHA256 = "078ceb041bf0123cc82b0e2ca1c97e6f47cf081eaa702254564f9c13150e2a66"
EXPECTED_FAILURE_SHA256 = "d912eaa94cdb38ee1c8b6c6f4fc59831c31f37d33152b23d1d2a9f334a2fc8d6"
EXPECTED_LEDGER_SHA256 = "0363fd23ff426aca7a9d88518203062a8e7440b0155a49879f639b3c96c18f2d"
RAW_OUTPUT_NAMES = ("cls_score", "bbox_pred", "dir_cls_pred")


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object in {path.name}")
    return payload


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(payload)


def _git_output(*args: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(_root()), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or f"git {' '.join(args)} failed")
    return process.stdout.strip()


def _require_clean_measurement_tree(
    *,
    explicit_commit: str | None,
    explicit_branch: str | None,
) -> tuple[str, str, str]:
    if (explicit_commit is None) != (explicit_branch is None):
        raise RuntimeError("explicit measurement commit and branch must be supplied together")
    if explicit_commit is not None and explicit_branch is not None:
        if re.fullmatch(r"[0-9a-f]{40}", explicit_commit) is None:
            raise RuntimeError("explicit measurement commit must be a lowercase full SHA")
        if explicit_branch != "feat/m45b-ros-multisweep":
            raise RuntimeError(f"unexpected explicit measurement branch: {explicit_branch}")
        return (
            explicit_commit,
            explicit_branch,
            "Windows Git clean-tree check immediately before WSL invocation",
        )
    status = _git_output("status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError("final detector-chain measurement requires a clean working tree")
    return (
        _git_output("rev-parse", "HEAD"),
        _git_output("branch", "--show-current"),
        "native Git clean-tree check inside measurement process",
    )


def _first_array_difference(expected: np.ndarray, observed: np.ndarray) -> dict[str, object]:
    if expected.shape != observed.shape:
        return {"kind": "shape", "expected": list(expected.shape), "observed": list(observed.shape)}
    if expected.dtype != observed.dtype:
        return {"kind": "dtype", "expected": str(expected.dtype), "observed": str(observed.dtype)}
    differing = np.argwhere(expected != observed)
    if differing.size == 0:
        return {"kind": "none"}
    index = tuple(int(value) for value in differing[0])
    return {
        "kind": "value",
        "index": list(index),
        "expected": float(expected[index]),
        "observed": float(observed[index]),
    }


def _numpy_comparison(
    expected: np.ndarray,
    observed: np.ndarray,
    *,
    accepted_sha256: str | None = None,
) -> dict[str, object]:
    expected_array = np.ascontiguousarray(expected)
    observed_array = np.ascontiguousarray(observed)
    expected_hash = _sha256_bytes(expected_array.tobytes(order="C"))
    observed_hash = _sha256_bytes(observed_array.tobytes(order="C"))
    exact = bool(
        expected_array.shape == observed_array.shape
        and expected_array.dtype == observed_array.dtype
        and np.array_equal(expected_array, observed_array)
    )
    accepted_exact = accepted_sha256 is None or (
        expected_hash == accepted_sha256 and observed_hash == accepted_sha256
    )
    record: dict[str, object] = {
        "shape": list(expected_array.shape),
        "dtype": str(expected_array.dtype),
        "expected_sha256": expected_hash,
        "observed_sha256": observed_hash,
        "values_exact": exact,
        "accepted_reference_sha256": accepted_sha256,
        "accepted_reference_exact": accepted_exact,
        "exact": exact and accepted_exact,
    }
    if not exact:
        record["first_difference"] = _first_array_difference(expected_array, observed_array)
    return record


def _tensor_comparison(
    expected: Any,
    observed: Any,
    *,
    accepted_sha256: str,
) -> dict[str, object]:
    expected_array = expected.detach().cpu().contiguous().numpy()
    observed_array = observed.detach().cpu().contiguous().numpy()
    return _numpy_comparison(
        expected_array,
        observed_array,
        accepted_sha256=accepted_sha256,
    )


def _raw_comparison(
    expected: Mapping[str, list[Any]],
    observed: Mapping[str, list[Any]],
    accepted: Mapping[str, Any],
) -> tuple[dict[str, object], bool]:
    records: dict[str, object] = {}
    passed = True
    if set(expected) != set(RAW_OUTPUT_NAMES) or set(observed) != set(RAW_OUTPUT_NAMES):
        raise RuntimeError("raw TensorRT output names differ from the frozen contract")
    for name in RAW_OUTPUT_NAMES:
        if len(expected[name]) != 1 or len(observed[name]) != 1:
            raise RuntimeError(f"raw output {name} must contain exactly one tensor")
        accepted_hash = str(accepted[name]["official_sha256"])
        record = _tensor_comparison(
            expected[name][0],
            observed[name][0],
            accepted_sha256=accepted_hash,
        )
        records[name] = record
        passed = passed and bool(record["exact"])
    return records, passed


def _frame_difference(expected: DetectionFrame, observed: DetectionFrame) -> dict[str, object]:
    expected_payload = expected.to_dict()
    observed_payload = observed.to_dict()
    for field in ("sample_id", "coordinate_frame", "metadata"):
        if expected_payload[field] != observed_payload[field]:
            return {
                "kind": "frame_field",
                "field": field,
                "expected": expected_payload[field],
                "observed": observed_payload[field],
            }
    expected_detections = expected_payload["detections"]
    observed_detections = observed_payload["detections"]
    if len(expected_detections) != len(observed_detections):
        return {
            "kind": "detection_count",
            "expected": len(expected_detections),
            "observed": len(observed_detections),
        }
    for index, (expected_detection, observed_detection) in enumerate(
        zip(expected_detections, observed_detections, strict=True)
    ):
        if expected_detection != observed_detection:
            return {
                "kind": "detection",
                "index": index,
                "expected": expected_detection,
                "observed": observed_detection,
            }
    return {"kind": "none"}


def _frame_comparison(expected: DetectionFrame, observed: DetectionFrame) -> dict[str, object]:
    expected_payload = expected.to_dict()
    observed_payload = observed.to_dict()
    exact = expected_payload == observed_payload
    record: dict[str, object] = {
        "expected_sha256": _canonical_json_sha256(expected_payload),
        "observed_sha256": _canonical_json_sha256(observed_payload),
        "expected_detection_count": len(expected.detections),
        "observed_detection_count": len(observed.detections),
        "exact": exact,
    }
    if not exact:
        record["first_difference"] = _frame_difference(expected, observed)
    return record


def _header_record(header: Any) -> dict[str, object]:
    return {
        "frame_id": str(header.frame_id),
        "stamp": {"sec": int(header.stamp.sec), "nanosec": int(header.stamp.nanosec)},
    }


def _message_comparison(
    expected_frame: DetectionFrame,
    observed_frame: DetectionFrame,
    source_message: Any,
) -> dict[str, object]:
    expected = detection_frame_to_message(expected_frame, source_message.header)
    observed = detection_frame_to_message(observed_frame, source_message.header)
    semantic_exact = expected.detections == observed.detections
    source_header_exact = observed.header == source_message.header and all(
        detection.header == source_message.header for detection in observed.detections
    )
    return {
        "semantic_and_geometric_content_exact": semantic_exact,
        "normalized_full_message_exact": expected == observed,
        "source_header_preserved_exact": source_header_exact,
        "observed_header": _header_record(observed.header),
        "older_model_ready_replay_header_exactness_required": False,
        "older_replay_header_note": (
            "The older model-ready replay stamps messages at publication time; the raw path "
            "preserves the acquisition header. Semantic content is compared with a shared header."
        ),
        "exact": semantic_exact and source_header_exact,
    }


def _expected_records() -> tuple[list[int], dict[int, Any], dict[int, Any]]:
    root = _root()
    m45a = _load_json(root / "benchmarks/m45a/results/nuscenes_mini_multisweep_parity.json")
    parity = dict(
        yaml.safe_load((root / "configs/detection/m2_parity_v2.yaml").read_text(encoding="utf-8"))
    )
    indices = [int(value) for value in parity["dataset"]["sample_indices"]]
    if len(indices) != 20:
        raise RuntimeError("frozen detector suite must contain exactly 20 samples")
    tier = {int(item["sample_index"]): item for item in m45a["tier_a"]["samples"]}
    detector = {
        int(item["sample_index"]): item for item in m45a["detector_verification"]["samples"]
    }
    if indices != [int(value) for value in m45a["detector_verification"]["frozen_sample_indices"]]:
        raise RuntimeError("M4.5a and M2 frozen detector sample sets differ")
    return indices, tier, detector


def _artifact_hashes(runtime: M3DetectorRuntime) -> dict[str, str]:
    actual = {
        "checkpoint": sha256_file(runtime.assets.checkpoint_path),
        "onnx": sha256_file(runtime.assets.onnx_path),
        "engine": sha256_file(runtime.assets.engine_path),
    }
    expected = {
        "checkpoint": EXPECTED_CHECKPOINT_SHA256,
        "onnx": EXPECTED_ONNX_SHA256,
        "engine": EXPECTED_ENGINE_SHA256,
    }
    if actual != expected:
        raise RuntimeError(f"frozen artifact hashes changed: {actual}")
    return actual


def _repair_evidence() -> tuple[dict[str, Any], dict[str, str]]:
    root = _root()
    repair_path = root / "benchmarks/m45b/results/tf_adapter_repair_exactness.json"
    failure_path = root / "benchmarks/m45b/diagnostics/w1_raw_ros_hash_failure.json"
    ledger_path = root / "benchmarks/m45b/diagnostics/w1_tf_transform_ledger.json"
    hashes = {
        "original_w1_failure": sha256_file(failure_path),
        "tf_transform_ledger": sha256_file(ledger_path),
        "adapter_repair_exactness": sha256_file(repair_path),
    }
    expected = {
        "original_w1_failure": EXPECTED_FAILURE_SHA256,
        "tf_transform_ledger": EXPECTED_LEDGER_SHA256,
        "adapter_repair_exactness": EXPECTED_REPAIR_EVIDENCE_SHA256,
    }
    if hashes != expected:
        raise RuntimeError(f"M4.5b chronology evidence hashes changed: {hashes}")
    repair = _load_json(repair_path)
    if repair.get("status") != "passed":
        raise RuntimeError("adapter repair evidence is not passed")
    return repair, hashes


def _load_legacy_smoke(path: Path, measurement_commit: str) -> dict[str, Any]:
    record = _load_json(path)
    if record.get("status") != "pass":
        raise RuntimeError("legacy model-ready M3 smoke did not pass")
    if record.get("measurement_commit") != measurement_commit:
        raise RuntimeError("legacy M3 smoke was not measured at the detector-chain commit")
    return record


def _sample_gate(
    *,
    index: int,
    data_root: Path,
    timeout_sec: float,
    runtime: M3DetectorRuntime,
    tier_reference: Mapping[str, Any],
    detector_reference: Mapping[str, Any],
) -> tuple[dict[str, object], bool]:
    backend = runtime.backend
    reference_prepared = backend.prepare_sample(data_root, split="mini_val", index=index)
    reference_points = reference_prepared.model_ready_points()
    expected_point = tier_reference["official"]
    expected = {
        "sample_token": str(tier_reference["sample_token"]),
        "point_count": int(tier_reference["final_point_count"]),
        "sha256": str(expected_point["sha256"]),
    }
    ros_record, captured = _capture_case(
        index,
        data_root,
        timeout_sec=timeout_sec,
        expected=expected,
    )
    observed_points = pointcloud2_to_model_ready(captured.message)
    input_comparison = _numpy_comparison(
        reference_points.points_xyzt,
        observed_points.points_xyzt,
        accepted_sha256=str(expected_point["sha256"]),
    )
    sample: dict[str, object] = {
        "sample_index": index,
        "sample_token": str(tier_reference["sample_token"]),
        "history_depth": int(ros_record["final_history_depth"]),
        "point_count": int(ros_record["final_point_count"]),
        "raw_ros_counters": ros_record["counters"],
        "model_ready_input": input_comparison,
    }
    if not bool(input_comparison["exact"]):
        sample["status"] = "failed_model_ready_input"
        sample["downstream"] = "not_run"
        return sample, False

    observed_prepared = backend.prepare_model_ready_points(
        observed_points,
        sample_id=reference_prepared.sample_id,
        coordinate_frame="nuscenes_lidar_top",
    )
    reference_voxels = backend.voxelize_official(reference_prepared)
    observed_voxels = backend.voxelize(observed_prepared)
    voxel_records: dict[str, object] = {}
    voxels_exact = True
    accepted_voxels = detector_reference["voxel_hashes"]["official"]
    for name in ("voxels", "num_points", "coors"):
        record = _tensor_comparison(
            getattr(reference_voxels, name),
            getattr(observed_voxels, name),
            accepted_sha256=str(accepted_voxels[name]),
        )
        voxel_records[name] = record
        voxels_exact = voxels_exact and bool(record["exact"])
    sample["voxelization"] = {"tensors": voxel_records, "exact": voxels_exact}
    if not voxels_exact:
        sample["status"] = "failed_voxelization"
        sample["downstream"] = "not_run"
        return sample, False

    reference_raw = backend.run_tensorrt_raw(reference_voxels, runtime.assets.engine_path)
    observed_raw = backend.run_tensorrt_raw(observed_voxels, runtime.assets.engine_path)
    raw_records, raw_exact = _raw_comparison(
        reference_raw,
        observed_raw,
        detector_reference["raw_tensorrt_outputs"],
    )
    sample["raw_tensorrt_outputs"] = {"tensors": raw_records, "exact": raw_exact}
    if not raw_exact:
        sample["status"] = "failed_raw_tensorrt_outputs"
        sample["downstream"] = "not_run"
        return sample, False

    reference_frame = backend.postprocess_raw(
        reference_raw,
        reference_voxels,
        backend_name="tensorrt",
        precision="fp16",
        provenance_mode="live",
    )
    observed_frame = backend.postprocess_raw(
        observed_raw,
        observed_voxels,
        backend_name="tensorrt",
        precision="fp16",
        provenance_mode="live",
    )
    public_runtime_frame = runtime.infer(
        observed_points,
        sample_id=reference_prepared.sample_id,
        coordinate_frame="nuscenes_lidar_top",
    )
    frame_record = _frame_comparison(reference_frame, observed_frame)
    public_runtime_record = _frame_comparison(observed_frame, public_runtime_frame)
    frames_exact = bool(frame_record["exact"] and public_runtime_record["exact"])
    sample["detection_frame"] = {
        "reference_vs_raw_ros": frame_record,
        "stepped_vs_public_runtime": public_runtime_record,
        "exact": frames_exact,
    }
    if not frames_exact:
        sample["status"] = "failed_detection_frame"
        sample["downstream"] = "not_run"
        return sample, False

    message_record = _message_comparison(reference_frame, public_runtime_frame, captured.message)
    sample["detection3darray"] = message_record
    passed = bool(message_record["exact"])
    sample["status"] = "pass" if passed else "failed_detection3darray"
    return sample, passed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("LASERPERCEPTION_NUSCENES_ROOT"))
    parser.add_argument("--legacy-smoke-result", type=Path, required=True)
    parser.add_argument("--measurement-commit")
    parser.add_argument("--measurement-branch")
    parser.add_argument("--timeout-sec", type=float, default=25.0)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.data_root:
        raise SystemExit("set LASERPERCEPTION_NUSCENES_ROOT or pass --data-root")
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit("nuScenes data root does not exist")
    if args.timeout_sec <= 0.0:
        raise SystemExit("timeout must be positive")

    measurement_commit, branch, clean_tree_verification = _require_clean_measurement_tree(
        explicit_commit=args.measurement_commit,
        explicit_branch=args.measurement_branch,
    )
    if branch != "feat/m45b-ros-multisweep":
        raise SystemExit(f"unexpected measurement branch: {branch}")
    repair, chronology_hashes = _repair_evidence()
    legacy_smoke = _load_legacy_smoke(args.legacy_smoke_result, measurement_commit)
    indices, tier_records, detector_records = _expected_records()

    runtime = M3DetectorRuntime(voxelization_mode="exact_fast", provenance_mode="live")
    artifacts = _artifact_hashes(runtime)
    samples: list[dict[str, object]] = []
    passed = True
    for index in indices:
        sample, sample_passed = _sample_gate(
            index=index,
            data_root=data_root,
            timeout_sec=args.timeout_sec,
            runtime=runtime,
            tier_reference=tier_records[index],
            detector_reference=detector_records[index],
        )
        samples.append(sample)
        print(f"raw ROS detector index {index:02d}: {sample['status']}", flush=True)
        if not sample_passed:
            passed = False
            break

    all_model_ready = passed and all(
        bool(sample["model_ready_input"]["exact"]) for sample in samples
    )
    all_voxels = passed and all(bool(sample["voxelization"]["exact"]) for sample in samples)
    all_raw = passed and all(bool(sample["raw_tensorrt_outputs"]["exact"]) for sample in samples)
    all_frames = passed and all(bool(sample["detection_frame"]["exact"]) for sample in samples)
    all_messages = passed and all(bool(sample["detection3darray"]["exact"]) for sample in samples)
    passed = passed and len(samples) == 20

    repair_samples = {
        "scene_start": repair["scene_start"],
        "w1": repair["w1"],
        "rotation_stratified_sentinels": repair["additional_full_history_sentinels"],
    }
    result: dict[str, object] = {
        "schema_version": "1.0",
        "milestone": "M4.5b",
        "status": "pass" if passed else "fail",
        "identity": {
            "base_main_sha": BASE_MAIN_SHA,
            "measurement_commit": measurement_commit,
            "branch": branch,
            "clean_tree_verification": clean_tree_verification,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "ros_distro": os.environ.get("ROS_DISTRO", "unknown"),
            "rmw_implementation": rclpy.utilities.get_rmw_implementation_identifier(),
            "tf2_ros_version": _tf2_version(),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "backend_versions": dict(runtime.backend.versions),
            "gpu": str(runtime.backend._runtime.torch.cuda.get_device_name(0)),
            "torch_cuda": str(runtime.backend._runtime.torch.version.cuda),
        },
        "artifacts": artifacts,
        "architecture": {
            "raw_input_topic": "/laserperception/points_raw",
            "model_ready_output_topic": "/laserperception/points_model_ready",
            "detector_output_topic": "/laserperception/detections",
            "fixed_frame": "nuscenes_map",
            "target_frame_behavior": "empty means the current raw message frame",
            "tf_api": "tf2_ros.Buffer.lookup_transform_full",
            "tf_signature": (
                "target_frame/current_stamp, source_frame/historical_stamp, fixed_frame, timeout"
            ),
            "measurement_transform_timeout_sec": 0.5,
            "packaged_transform_timeout_sec": 0.2,
            "tf_listener": "TransformListener(buffer, None, spin_thread=True)",
            "validation_executor": "MultiThreadedExecutor(num_threads=3)",
            "wait_strategy": "bounded lookup timeout; required historical TF fails closed",
            "qos": {
                "raw": "best_effort/volatile/keep_last/depth_5",
                "model_ready": "best_effort/volatile/keep_last/depth_1",
                "detector_output": "reliable/volatile/keep_last/depth_5",
            },
        },
        "transform_convention": {
            "ros_column_vector": "p_target = R @ p_source + t",
            "sweep_rotation_storage": "R.T",
            "sweep_translation_storage": "-R.T @ t",
            "failed_translation_storage": "-t",
            "fail_first_rotation_translation_regression": True,
            "authoritative_evidence": (
                "actual raw nuScenes sweeps through PointCloud2, time-aware tf2, repaired adapter, "
                "and MultiSweepBuilder match the accepted M4.5a oracle byte-for-byte"
            ),
        },
        "regression_history": {
            "evidence_sha256": chronology_hashes,
            "failed_formula": "translation_storage = -t",
            "corrected_formula": "translation_storage = -R.T @ t",
            "repair_commit": repair["repair_commit"],
            "fail_first_regression_passed": bool(
                repair["rotation_bearing_unit_regression"]["exact"]
            ),
        },
        "correctness": {
            "repair_exactness": repair_samples,
            "legacy_model_ready_m3_smoke": legacy_smoke,
            "frozen_detector_chain": {
                "required_sample_count": 20,
                "completed_sample_count": len(samples),
                "sample_indices": indices,
                "model_ready_inputs_exact": all_model_ready,
                "voxel_tensors_exact": all_voxels,
                "raw_tensorrt_outputs_exact": all_raw,
                "detection_frames_exact": all_frames,
                "detection3darray_semantics_exact": all_messages,
                "passed": passed,
                "samples": samples,
            },
        },
        "canonical_w1_counters": repair["w1"]["counters"],
        "scope_guards": {
            "model_changed": False,
            "onnx_changed": False,
            "engine_changed": False,
            "exact_fast_changed": False,
            "threshold_changed": False,
            "voxel_geometry_changed": False,
            "performance_campaign_run": False,
        },
    }
    output = args.output or _root() / "benchmarks/m45b/results/raw_ros_multisweep_correctness.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "measurement_commit": measurement_commit,
                "completed_samples": len(samples),
                "model_ready_inputs_exact": all_model_ready,
                "voxel_tensors_exact": all_voxels,
                "raw_tensorrt_outputs_exact": all_raw,
                "detection_frames_exact": all_frames,
                "detection3darray_semantics_exact": all_messages,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"result: {output}")
    if not passed:
        raise SystemExit("M4.5b detector-chain correctness failed; stop before finalization")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
