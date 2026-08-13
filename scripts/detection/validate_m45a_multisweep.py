"""Validate offline raw-sweep reconstruction against the pinned official oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from laserperception.detection.m1_assets import resolve_m1_asset_paths
from laserperception.detection.m2_assets import resolve_m2_asset_paths
from laserperception.detection.m2_backend import M2Backend
from laserperception.detection.mmdet3d_backend import sha256_file
from laserperception.detection.multisweep import (
    HistoricalSweep,
    LidarPose,
    MultiSweepBuilder,
    RawSweep,
    SweepTransform,
)
from laserperception.detection.runtime_metadata import repository_git_sha

EXPECTED_CHECKPOINT_SHA256 = "f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0"
EXPECTED_ONNX_SHA256 = "61ce22a8ca31498675c32576bfb94f0093d31dc95d2762f7254bf915a59ecc16"
EXPECTED_ENGINE_SHA256 = "a005f75852097cd9b193750560b214cc3d5237ae9b6c106c7fca3d4fc348714b"
DETECTOR_INDICES = (0, 4, 8, 12, 16, 21, 25, 29, 33, 37, 42, 46, 50, 54, 58, 63, 67, 71, 75, 80)
RAW_OUTPUT_NAMES = ("cls_score", "bbox_pred", "dir_cls_pred")


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _manifest(name: str) -> dict[str, Any]:
    path = _root() / "configs" / "detection" / name
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _array_record(array: np.ndarray) -> dict[str, object]:
    return {
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "contiguous": bool(array.flags.c_contiguous),
        "sha256": _array_sha256(array),
    }


def _artifact_record(path: Path, expected_sha256: str) -> dict[str, object]:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"frozen artifact hash mismatch for {path.name}")
    return {"logical_name": path.name, "sha256": actual, "size_bytes": path.stat().st_size}


def _lidar_pose(nusc: Any, sample_data_token: str, quaternion_type: Any) -> LidarPose:
    sample_data = nusc.get("sample_data", sample_data_token)
    calibration = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
    return LidarPose(
        np.asarray(quaternion_type(calibration["rotation"]).rotation_matrix, dtype=np.float64),
        np.asarray(calibration["translation"], dtype=np.float64),
        np.asarray(quaternion_type(ego_pose["rotation"]).rotation_matrix, dtype=np.float64),
        np.asarray(ego_pose["translation"], dtype=np.float64),
    )


def _candidate_from_raw(
    nusc: Any,
    sample_token: str,
    builder: MultiSweepBuilder,
    quaternion_type: Any,
) -> tuple[Any, dict[str, object], RawSweep, tuple[HistoricalSweep, ...]]:
    sample = nusc.get("sample", sample_token)
    current_token = str(sample["data"]["LIDAR_TOP"])
    current_data = nusc.get("sample_data", current_token)
    current = RawSweep.from_nuscenes_file(
        nusc.get_sample_data_path(current_token),
        timestamp_microseconds=int(sample["timestamp"]),
        source_id=current_token,
    )
    current_pose = _lidar_pose(nusc, current_token, quaternion_type)
    history: list[HistoricalSweep] = []
    previous = str(current_data["prev"])
    while previous and len(history) < builder.config.max_historical_sweeps:
        sample_data = nusc.get("sample_data", previous)
        raw = RawSweep.from_nuscenes_file(
            nusc.get_sample_data_path(previous),
            timestamp_microseconds=int(sample_data["timestamp"]),
            source_id=previous,
        )
        transform = SweepTransform.from_poses(
            source_id=previous,
            target_id=current_token,
            sweep_pose=_lidar_pose(nusc, previous, quaternion_type),
            current_pose=current_pose,
        )
        history.append(HistoricalSweep(raw, transform))
        previous = str(sample_data["prev"])

    cloud = builder.build(current, history)
    source_records = [
        {
            "role": "current",
            "sample_data_token": current.source_id,
            "raw_point_count": int(current.points.shape[0]),
        }
    ]
    source_records.extend(
        {
            "role": "historical",
            "history_index": index,
            "sample_data_token": item.sweep.source_id,
            "raw_point_count": int(item.sweep.points.shape[0]),
        }
        for index, item in enumerate(history)
    )
    details = {
        "current_sample_data_token": current_token,
        "historical_sweep_tokens": [item.sweep.source_id for item in history],
        "historical_sweep_count": len(history),
        "source_point_counts": source_records,
        "final_point_count": int(cloud.points_xyzt.shape[0]),
        "unique_time_lag_count": int(np.unique(cloud.points_xyzt[:, 3]).size),
    }
    return cloud, details, current, tuple(history)


def _ordered_float32(value: np.float32) -> int:
    signed = int(np.asarray(value, dtype=np.float32).view(np.int32))
    return 0x80000000 - signed if signed < 0 else signed + 0x80000000


def _provenance_rows(
    current: RawSweep, history: Sequence[HistoricalSweep]
) -> list[dict[str, object]]:
    """Rebuild row identities only for a failure diagnostic."""

    parts: list[np.ndarray] = []
    identities: list[dict[str, object]] = []
    current_points = current.points.copy()
    current_points[:, 4] = np.float32(0.0)
    parts.append(current_points)
    identities.extend(
        {
            "sweep_identity": current.source_id,
            "sweep_role": "current",
            "source_point_index": index,
        }
        for index in range(len(current_points))
    )
    for history_index, item in enumerate(history):
        points = item.sweep.points.copy()
        matrix = np.array(item.transform.lidar2sensor.tolist())
        points[:, :3] = points[:, :3] @ matrix[:3, :3]
        points[:, :3] -= matrix[:3, 3]
        points[:, 4] = current.timestamp_seconds - item.sweep.timestamp_seconds
        parts.append(points)
        identities.extend(
            {
                "sweep_identity": item.sweep.source_id,
                "sweep_role": "historical",
                "history_index": history_index,
                "source_point_index": index,
            }
            for index in range(len(points))
        )
    concatenated = np.concatenate(parts, axis=0)
    mask = (
        (concatenated[:, 0] > -50.0)
        & (concatenated[:, 0] < 50.0)
        & (concatenated[:, 1] > -50.0)
        & (concatenated[:, 1] < 50.0)
        & (concatenated[:, 2] > -5.0)
        & (concatenated[:, 2] < 3.0)
    )
    return [record for record, retained in zip(identities, mask, strict=True) if bool(retained)]


def _first_failure(
    official: np.ndarray,
    candidate: np.ndarray,
    *,
    sample_index: int,
    sample_token: str,
    current: RawSweep,
    history: Sequence[HistoricalSweep],
) -> dict[str, object]:
    base: dict[str, object] = {
        "sample_index": sample_index,
        "sample_token": sample_token,
        "official": _array_record(official),
        "candidate": _array_record(candidate),
        "remove_close": False,
        "operation_order": (
            "float32_points_matmul_float64_rotation_assign_float32_then_"
            "inplace_subtract_float64_translation"
        ),
        "stage_dtypes": {
            "raw_points": "float32",
            "stored_transform": "float32",
            "reloaded_transform": "float64",
            "matmul_expression": "float64",
            "post_rotation_points": "float32",
            "post_translation_points": "float32",
            "timestamp_seconds": "binary64",
            "final_points": "float32",
        },
    }
    if official.shape != candidate.shape:
        return {**base, "classification": "G. other", "reason": "shape_mismatch"}
    differing = np.argwhere(official != candidate)
    if differing.size == 0:
        return {**base, "classification": "G. other", "reason": "byte_or_layout_mismatch"}
    row, column = (int(value) for value in differing[0])
    official_value = np.float32(official[row, column])
    candidate_value = np.float32(candidate[row, column])
    provenance = _provenance_rows(current, history)
    source = provenance[row] if row < len(provenance) else {"sweep_identity": "unresolved"}
    source_id = str(source["sweep_identity"])
    source_sweep = next(
        (item.sweep for item in history if item.sweep.source_id == source_id), current
    )
    field = ("x", "y", "z", "time_lag")[column]
    return {
        **base,
        "classification": "G. other_pending_diagnosis",
        "first_differing_row": row,
        "field": field,
        "official_value": float(official_value),
        "candidate_value": float(candidate_value),
        "absolute_difference": float(abs(official_value - candidate_value)),
        "ulp_difference": abs(_ordered_float32(official_value) - _ordered_float32(candidate_value)),
        **source,
        "timestamp_intermediates": {
            "current_microseconds": current.timestamp_microseconds,
            "source_microseconds": source_sweep.timestamp_microseconds,
            "current_seconds_binary64": current.timestamp_seconds,
            "source_seconds_binary64": source_sweep.timestamp_seconds,
            "lag_seconds_binary64": current.timestamp_seconds - source_sweep.timestamp_seconds,
        },
    }


def _numpy_raw(raw: Mapping[str, list[Any]], name: str) -> np.ndarray:
    values = raw[name]
    if len(values) != 1:
        raise RuntimeError(f"raw output {name} must have one feature level")
    return values[0].detach().cpu().contiguous().numpy()


def _detector_gate(
    backend: M2Backend,
    nusc: Any,
    data_root: Path,
    engine: Path,
    builder: MultiSweepBuilder,
    quaternion_type: Any,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for index in DETECTOR_INDICES:
        official = backend.prepare_sample(data_root, split="mini_val", index=index)
        cloud, _, _, _ = _candidate_from_raw(nusc, official.sample_id, builder, quaternion_type)
        candidate = backend.prepare_model_ready_points(
            cloud,
            sample_id=official.sample_id,
            coordinate_frame=official.coordinate_frame,
        )
        candidate = replace(candidate, sample_index=index, split="mini_val")
        official_voxels = backend.voxelize(official)
        candidate_voxels = backend.voxelize(candidate)
        voxel_hashes_official = official_voxels.hashes()
        voxel_hashes_candidate = candidate_voxels.hashes()
        voxels_exact = voxel_hashes_official == voxel_hashes_candidate
        if not voxels_exact:
            raise RuntimeError(f"frozen detector gate found voxel mismatch at sample {index}")
        official_raw = backend.run_tensorrt_raw(official_voxels, engine)
        candidate_raw = backend.run_tensorrt_raw(candidate_voxels, engine)
        raw_records: dict[str, object] = {}
        raw_exact = True
        for name in RAW_OUTPUT_NAMES:
            official_array = _numpy_raw(official_raw, name)
            candidate_array = _numpy_raw(candidate_raw, name)
            exact = bool(np.array_equal(official_array, candidate_array))
            raw_exact = raw_exact and exact
            raw_records[name] = {
                "official_sha256": _array_sha256(official_array),
                "candidate_sha256": _array_sha256(candidate_array),
                "exact": exact,
            }
        official_frame = backend.postprocess_raw(
            official_raw, official_voxels, backend_name="tensorrt", precision="fp16"
        )
        candidate_frame = backend.postprocess_raw(
            candidate_raw, candidate_voxels, backend_name="tensorrt", precision="fp16"
        )
        frame_exact = official_frame.to_dict() == candidate_frame.to_dict()
        record = {
            "sample_index": index,
            "sample_token": official.sample_id,
            "voxel_hashes": {
                "official": voxel_hashes_official,
                "candidate": voxel_hashes_candidate,
            },
            "voxel_tensors_exact": voxels_exact,
            "raw_tensorrt_outputs": raw_records,
            "raw_tensorrt_outputs_exact": raw_exact,
            "detection_count": len(official_frame.detections),
            "detection_frame_exact": frame_exact,
        }
        records.append(record)
        print(
            f"detector {index:02d}: voxels={voxels_exact} raw={raw_exact} frame={frame_exact}",
            flush=True,
        )
        if not raw_exact or not frame_exact:
            raise RuntimeError(f"frozen detector output mismatch at sample {index}")
    return {
        "frozen_sample_indices": list(DETECTOR_INDICES),
        "sample_count": len(records),
        "all_voxel_tensors_exact": all(bool(item["voxel_tensors_exact"]) for item in records),
        "all_raw_tensorrt_outputs_exact": all(
            bool(item["raw_tensorrt_outputs_exact"]) for item in records
        ),
        "all_detection_frames_exact": all(bool(item["detection_frame_exact"]) for item in records),
        "passed": len(records) == len(DETECTOR_INDICES),
        "samples": records,
    }


def _write_json(path: Path, record: Mapping[str, object]) -> None:
    encoded = json.dumps(record, indent=2, sort_keys=True) + "\n"
    forbidden = (str(Path.home()), "J:\\", "/root/")
    if any(value and value in encoded for value in forbidden):
        raise RuntimeError("refusing to write evidence containing a private absolute path")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("LASERPERCEPTION_NUSCENES_ROOT"))
    parser.add_argument(
        "--output",
        type=Path,
        default=_root() / "benchmarks/m45a/results/nuscenes_mini_multisweep_parity.json",
    )
    parser.add_argument("--parity-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.data_root:
        raise SystemExit("set LASERPERCEPTION_NUSCENES_ROOT or pass --data-root")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=_root(),
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise SystemExit("commit the implementation and start from a clean tree before measuring")

    try:
        from nuscenes.nuscenes import NuScenes
        from pyquaternion import Quaternion
    except ImportError as error:
        raise SystemExit("the pinned nuScenes devkit and pyquaternion are required") from error

    data_root = Path(args.data_root).expanduser().resolve()
    m1 = _manifest("m1_pointpillars_nuscenes.yaml")
    m2 = _manifest("m2_pointpillars_tensorrt.yaml")
    m1_assets = resolve_m1_asset_paths(m1)
    m2_assets = resolve_m2_asset_paths(m2)
    onnx = m2_assets.artifact_directory / "pointpillars.onnx"
    engine = m2_assets.engine_directory / "pointpillars_fp16.engine"
    artifacts = {
        "checkpoint": _artifact_record(m1_assets.checkpoint_path, EXPECTED_CHECKPOINT_SHA256),
        "onnx": _artifact_record(onnx, EXPECTED_ONNX_SHA256),
        "tensorrt_engine": _artifact_record(engine, EXPECTED_ENGINE_SHA256),
    }
    backend = M2Backend(
        m1_assets.mmdet3d_root / str(m1["model"]["upstream_config"]),
        m1_assets.checkpoint_path,
        m2_assets.mmdeploy_root / str(m2["deployment"]["official_deployment_config"]),
        checkpoint_sha256=str(m1["model"]["checkpoint"]["sha256"]),
        voxelization_mode="exact_fast",
    )
    backend.initialize()
    nusc = NuScenes(version="v1.0-mini", dataroot=str(data_root), verbose=False)
    builder = MultiSweepBuilder()
    dataset_size = backend.dataset_size(data_root, "mini_val")
    if dataset_size != 81:
        raise SystemExit(f"M4.5a requires exactly 81 mini_val samples, found {dataset_size}")

    samples: list[dict[str, object]] = []
    first_failure: dict[str, object] | None = None
    scene_start = 0
    full_history = 0
    for index in range(dataset_size):
        official_sample = backend.prepare_sample(data_root, split="mini_val", index=index)
        official = official_sample.model_ready_points().points_xyzt
        candidate_cloud, details, current, history = _candidate_from_raw(
            nusc, official_sample.sample_id, builder, Quaternion
        )
        candidate = candidate_cloud.points_xyzt
        exact = bool(
            official.dtype == candidate.dtype
            and official.shape == candidate.shape
            and official.flags.c_contiguous
            and candidate.flags.c_contiguous
            and official.tobytes(order="C") == candidate.tobytes(order="C")
        )
        if int(details["historical_sweep_count"]) == 0:
            scene_start += 1
        if int(details["historical_sweep_count"]) == 10:
            full_history += 1
        sample_record = {
            "sample_index": index,
            "sample_token": official_sample.sample_id,
            **details,
            "official": _array_record(official),
            "candidate": _array_record(candidate),
            "exact": exact,
        }
        samples.append(sample_record)
        print(
            f"parity {index:02d}: history={details['historical_sweep_count']} "
            f"points={details['final_point_count']} exact={exact}",
            flush=True,
        )
        if not exact:
            first_failure = _first_failure(
                official,
                candidate,
                sample_index=index,
                sample_token=official_sample.sample_id,
                current=current,
                history=history,
            )
            break

    tier_a_passed = len(samples) == dataset_size and first_failure is None
    commit_sha = repository_git_sha(_root())
    record: dict[str, object] = {
        "schema_version": 1,
        "milestone": "M4.5a",
        "candidate_commit": commit_sha,
        "base_main_commit": "320e146d4cb0d272e8e569a914fbc6fdb450875b",
        "pinned_upstream": {
            "mmdetection3d_commit": "fe25f7a51d36e3702f961e198894580d83c4387b",
            "mmcv_version": "2.1.0",
            "mmcv_tag_commit": "57c4e25e06e2d4f8a9357c84bcd24089a284dc88",
            "nuscenes_devkit_version": "1.2.0",
            "nuscenes_devkit_tag_commit": "eff381829dc86fa75caf7dbbbe862d2091dacf64",
            "pyquaternion_version": "0.9.9",
            "pyquaternion_tag_commit": "2ccfdd5ec6b214092efcbebacd74eabba5c072e1",
        },
        "config": {
            "identity": "configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py",
            "laserperception_manifest": "configs/detection/m1_pointpillars_nuscenes.yaml",
            "laserperception_manifest_sha256": sha256_file(
                _root() / "configs/detection/m1_pointpillars_nuscenes.yaml"
            ),
        },
        "upstream_semantics": {
            "load_dim": 5,
            "initial_use_dim": [0, 1, 2, 3, 4],
            "final_use_dim": [0, 1, 2, 4],
            "final_features": ["x", "y", "z", "time_lag"],
            "sweeps_num": 10,
            "sweeps_num_meaning": "up_to_ten_historical_plus_current",
            "test_selection": "first_N_prepared_sweeps_nearest_to_farthest",
            "remove_close": False,
            "remove_close_dormant_rule": (
                "strict_abs_x_lt_1_and_abs_y_lt_1_before_history_transform"
            ),
            "pad_empty_sweeps": False,
            "scene_start": "current_only",
            "timestamp_source_units": "integer_microseconds",
            "time_lag": "float32((current_us/1e6)-(historical_us/1e6))",
            "transform": (
                "float32_points_matmul_float64_reloaded_quantized_rotation_"
                "assign_float32_then_inplace_subtract_float64_translation"
            ),
            "concatenation": "current_then_history_nearest_to_farthest_source_row_order",
            "range_filter": "strict_-50_x_50_-50_y_50_-5_z_3",
        },
        "artifacts": artifacts,
        "tier_a": {
            "required_samples": dataset_size,
            "completed_samples": len(samples),
            "exact_samples": sum(bool(sample["exact"]) for sample in samples),
            "scene_start_samples": scene_start,
            "full_history_samples": full_history,
            "passed": tier_a_passed,
            "samples": samples,
        },
        "first_failure": first_failure,
        "tier_b": {"status": "not_run_tier_a_passed" if tier_a_passed else "not_run"},
        "scope_guards": {
            "production_builder_calls_mmdetection3d": False,
            "ros_implemented": False,
            "model_changed": False,
            "onnx_changed": False,
            "engine_changed": False,
            "exact_fast_changed": False,
            "thresholds_changed": False,
            "voxel_geometry_changed": False,
        },
    }
    if not tier_a_passed:
        record["status"] = "M4.5a PARITY FAIL"
        _write_json(args.output, record)
        print(f"wrote failed evidence to {args.output}")
        return 2
    if args.parity_only:
        record["detector_verification"] = {"status": "not_run_parity_only"}
        record["status"] = "M4.5a EXACT PARITY PASS — detector verification pending"
        _write_json(args.output, record)
        return 0

    detector = _detector_gate(backend, nusc, data_root, engine, builder, Quaternion)
    record["detector_verification"] = detector
    if not bool(detector["passed"]):
        record["status"] = "M4.5a PARITY FAIL"
        _write_json(args.output, record)
        return 3
    record["status"] = "M4.5a EXACT PARITY PASS"
    _write_json(args.output, record)
    print(f"wrote M4.5a evidence to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
