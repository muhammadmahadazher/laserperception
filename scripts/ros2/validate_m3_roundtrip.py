"""Run final production exact-fast voxel, detector, and ROS correctness gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from builtin_interfaces.msg import Time
from laserperception_ros.conversion import (
    detection_frame_to_message,
    model_ready_to_pointcloud2,
    pointcloud2_to_model_ready,
)
from laserperception_ros.runtime import M3DetectorRuntime
from std_msgs.msg import Header

from laserperception.detection.mmdet3d_backend import sha256_file
from laserperception.detection.runtime_metadata import repository_git_sha

EXPECTED_CHECKPOINT_SHA256 = "f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0"
EXPECTED_ONNX_SHA256 = "61ce22a8ca31498675c32576bfb94f0093d31dc95d2762f7254bf915a59ecc16"
EXPECTED_ENGINE_SHA256 = "a005f75852097cd9b193750560b214cc3d5237ae9b6c106c7fca3d4fc348714b"


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _manifest(name: str) -> dict[str, Any]:
    return dict(yaml.safe_load((_root() / "configs/detection" / name).read_text()))


def _raw_records(raw: Mapping[str, list[Any]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, values in raw.items():
        if len(values) != 1:
            raise RuntimeError(f"raw output {name} must contain one tensor")
        array = values[0].detach().cpu().contiguous().numpy()
        numeric = array.astype(np.float64, copy=False)
        result[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            "minimum": float(numeric.min()),
            "maximum": float(numeric.max()),
            "mean": float(numeric.mean()),
        }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("LASERPERCEPTION_NUSCENES_ROOT"))
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.data_root:
        raise SystemExit("set LASERPERCEPTION_NUSCENES_ROOT or pass --data-root")
    parity = _manifest("m2_parity_v2.yaml")
    runtime = M3DetectorRuntime(voxelization_mode="exact_fast", provenance_mode="live")
    backend = runtime.backend
    engine = runtime.assets.engine_path
    artifact_hashes = {
        "checkpoint": sha256_file(runtime.assets.checkpoint_path),
        "onnx": sha256_file(runtime.assets.onnx_path),
        "engine": sha256_file(engine),
    }
    expected_hashes = {
        "checkpoint": EXPECTED_CHECKPOINT_SHA256,
        "onnx": EXPECTED_ONNX_SHA256,
        "engine": EXPECTED_ENGINE_SHA256,
    }
    if artifact_hashes != expected_hashes:
        raise SystemExit(f"frozen artifact SHA256 mismatch: {artifact_hashes}; M3 fidelity refused")

    output = args.output or engine.parent.parent / "m3" / "production_correctness.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    split_size = backend.dataset_size(args.data_root, "mini_val")
    if split_size != 81:
        raise SystemExit(f"production exact gate requires 81 mini_val samples, found {split_size}")
    voxel_records: list[dict[str, object]] = []
    all_voxels_exact = True
    for index in range(split_size):
        prepared = backend.prepare_sample(args.data_root, split="mini_val", index=index)
        official = backend.voxelize_official(prepared)
        exact_fast = backend.voxelize(prepared)
        tensors: dict[str, object] = {}
        sample_exact = True
        for name in ("voxels", "num_points", "coors"):
            official_array = getattr(official, name).detach().cpu().contiguous().numpy()
            exact_array = getattr(exact_fast, name).detach().cpu().contiguous().numpy()
            tensor_exact = bool(np.array_equal(official_array, exact_array))
            sample_exact = sample_exact and tensor_exact
            tensors[name] = {
                "exact": tensor_exact,
                "shape": list(official_array.shape),
                "dtype": str(official_array.dtype),
                "official_sha256": hashlib.sha256(official_array.tobytes(order="C")).hexdigest(),
                "exact_fast_sha256": hashlib.sha256(exact_array.tobytes(order="C")).hexdigest(),
            }
        all_voxels_exact = all_voxels_exact and sample_exact
        voxel_records.append(
            {
                "sample_index": index,
                "sample_id": prepared.sample_id,
                "point_count": int(prepared.points_xyzt.shape[0]),
                "voxel_count": official.voxel_count,
                "exact": sample_exact,
                "tensors": tensors,
            }
        )
        print(f"production voxel index {index:02d}: exact={sample_exact}", flush=True)

    if not all_voxels_exact:
        failed = {
            "schema_version": "1.0",
            "milestone": "M3",
            "status": "fail_stop_before_detector_and_benchmark",
            "implementation_commit": repository_git_sha(_root()),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": artifact_hashes,
            "production_voxelization_mode": runtime.voxelization_mode,
            "provenance_mode": runtime.provenance_mode,
            "voxel_gate": {
                "required_sample_count": 81,
                "passed": False,
                "samples": voxel_records,
            },
        }
        output.write_text(json.dumps(failed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise SystemExit("production exact-fast voxel gate failed; benchmark refused")

    records: list[dict[str, object]] = []
    all_passed = True
    indices = [int(value) for value in parity["dataset"]["sample_indices"]]
    if len(indices) != 20:
        raise SystemExit("frozen parity-v2 suite must contain exactly 20 samples")
    for index in indices:
        original_prepared = backend.prepare_sample(args.data_root, split="mini_val", index=index)
        source = original_prepared.model_ready_points()
        message = model_ready_to_pointcloud2(
            source,
            Header(stamp=Time(sec=index + 1, nanosec=index), frame_id="nuscenes_lidar_top"),
        )
        transported = pointcloud2_to_model_ready(message)
        adapter_prepared = backend.prepare_model_ready_points(
            transported,
            sample_id=original_prepared.sample_id,
            coordinate_frame="nuscenes_lidar_top",
        )
        original_voxels = backend.voxelize_official(original_prepared)
        adapter_voxels = backend.voxelize(adapter_prepared)
        original_raw = backend.run_tensorrt_raw(original_voxels, engine)
        adapter_raw = backend.run_tensorrt_raw(adapter_voxels, engine)
        original_frame = backend.postprocess_raw(
            original_raw, original_voxels, backend_name="tensorrt", precision="fp16"
        )
        adapter_frame = runtime.infer(
            transported,
            sample_id=original_prepared.sample_id,
            coordinate_frame="nuscenes_lidar_top",
        )
        point_equal = bool(np.array_equal(source.points_xyzt, transported.points_xyzt))
        voxel_hashes_equal = original_voxels.hashes() == adapter_voxels.hashes()
        original_raw_record = _raw_records(original_raw)
        adapter_raw_record = _raw_records(adapter_raw)
        raw_equal = original_raw_record == adapter_raw_record
        detections_equal = [item.to_dict() for item in original_frame.detections] == [
            item.to_dict() for item in adapter_frame.detections
        ]
        reference_message = detection_frame_to_message(original_frame, message.header)
        adapter_message = detection_frame_to_message(adapter_frame, message.header)
        ros_message_equal = reference_message == adapter_message
        sample_passed = (
            point_equal
            and voxel_hashes_equal
            and raw_equal
            and detections_equal
            and ros_message_equal
        )
        all_passed = all_passed and sample_passed
        records.append(
            {
                "sample_index": index,
                "sample_id": original_prepared.sample_id,
                "point_count": int(source.points_xyzt.shape[0]),
                "point_shape": list(source.points_xyzt.shape),
                "point_dtype": str(source.points_xyzt.dtype),
                "source_point_sha256": source.sha256,
                "ros_roundtrip_point_sha256": transported.sha256,
                "point_values_exact": point_equal,
                "original_voxel_count": original_voxels.voxel_count,
                "adapter_voxel_count": adapter_voxels.voxel_count,
                "original_voxel_hashes": original_voxels.hashes(),
                "adapter_voxel_hashes": adapter_voxels.hashes(),
                "voxel_hashes_exact": voxel_hashes_equal,
                "original_raw_outputs": original_raw_record,
                "adapter_raw_outputs": adapter_raw_record,
                "raw_outputs_exact": raw_equal,
                "original_detection_count": len(original_frame.detections),
                "adapter_detection_count": len(adapter_frame.detections),
                "final_detections_exact": detections_equal,
                "ros_detection3darray_exact": ros_message_equal,
                "status": "pass" if sample_passed else "fail",
            }
        )
    result = {
        "schema_version": "1.0",
        "milestone": "M3",
        "gate": "production_exact_fast_and_ros_correctness",
        "status": "pass" if all_passed else "fail",
        "implementation_commit": repository_git_sha(_root()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifact_hashes,
        "production_voxelization_mode": runtime.voxelization_mode,
        "provenance_mode": runtime.provenance_mode,
        "voxel_gate": {
            "required_sample_count": 81,
            "completed_sample_count": len(voxel_records),
            "passed": all_voxels_exact,
            "samples": voxel_records,
        },
        "detector_and_ros_gate": {
            "sample_count": len(records),
            "passed": all_passed,
            "point_contract": ["x", "y", "z", "time_lag"],
            "samples": records,
        },
    }

    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "status": result["status"],
        "implementation_commit": result["implementation_commit"],
        "artifacts": result["artifacts"],
        "production_voxelization_mode": result["production_voxelization_mode"],
        "provenance_mode": result["provenance_mode"],
        "voxel_samples": len(voxel_records),
        "detector_and_ros_samples": len(records),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"external result: {output}")
    if not all_passed:
        raise SystemExit("M3 production correctness failed; stop without benchmarking")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
