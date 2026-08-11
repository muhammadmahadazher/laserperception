"""Validate all 20 frozen M2 samples through the exact M3 PointCloud2 round trip."""

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
from laserperception_ros.conversion import model_ready_to_pointcloud2, pointcloud2_to_model_ready
from std_msgs.msg import Header

from laserperception.detection.m1_assets import resolve_m1_asset_paths
from laserperception.detection.m2_assets import resolve_m2_asset_paths
from laserperception.detection.m2_backend import M2Backend
from laserperception.detection.mmdet3d_backend import sha256_file
from laserperception.detection.runtime_metadata import repository_git_sha

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
    m1 = _manifest("m1_pointpillars_nuscenes.yaml")
    m2 = _manifest("m2_pointpillars_tensorrt.yaml")
    parity = _manifest("m2_parity_v2.yaml")
    a1 = resolve_m1_asset_paths(m1)
    a2 = resolve_m2_asset_paths(m2)
    engine = a2.engine_directory / "pointpillars_fp16.engine"
    engine_sha = sha256_file(engine)
    if engine_sha != EXPECTED_ENGINE_SHA256:
        raise SystemExit("frozen TensorRT engine SHA256 mismatch; M3 fidelity refused")
    backend = M2Backend(
        a1.mmdet3d_root / str(m1["model"]["upstream_config"]),
        a1.checkpoint_path,
        a2.mmdeploy_root / str(m2["deployment"]["official_deployment_config"]),
        checkpoint_sha256=str(m1["model"]["checkpoint"]["sha256"]),
    )
    backend.initialize()
    backend._backend_model(engine)
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
        original_voxels = backend.voxelize(original_prepared)
        adapter_voxels = backend.voxelize(adapter_prepared)
        original_raw = backend.run_tensorrt_raw(original_voxels, engine)
        adapter_raw = backend.run_tensorrt_raw(adapter_voxels, engine)
        original_frame = backend.postprocess_raw(
            original_raw, original_voxels, backend_name="tensorrt", precision="fp16"
        )
        adapter_frame = backend.postprocess_raw(
            adapter_raw, adapter_voxels, backend_name="tensorrt", precision="fp16"
        )
        point_equal = bool(np.array_equal(source.points_xyzt, transported.points_xyzt))
        voxel_hashes_equal = original_voxels.hashes() == adapter_voxels.hashes()
        original_raw_record = _raw_records(original_raw)
        adapter_raw_record = _raw_records(adapter_raw)
        raw_equal = original_raw_record == adapter_raw_record
        detections_equal = [item.to_dict() for item in original_frame.detections] == [
            item.to_dict() for item in adapter_frame.detections
        ]
        sample_passed = point_equal and voxel_hashes_equal and raw_equal and detections_equal
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
                "status": "pass" if sample_passed else "fail",
            }
        )
    result = {
        "schema_version": "1.0",
        "milestone": "M3A",
        "gate": "ros_pointcloud2_roundtrip_fidelity",
        "status": "pass" if all_passed else "fail",
        "implementation_commit": repository_git_sha(_root()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(records),
        "engine_sha256": engine_sha,
        "point_contract": ["x", "y", "z", "time_lag"],
        "samples": records,
    }
    output = args.output or a2.artifact_directory / "m3" / "roundtrip_fidelity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))
    print(f"external result: {output}")
    if not all_passed:
        raise SystemExit("M3 round-trip fidelity failed; stop without benchmarking")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
