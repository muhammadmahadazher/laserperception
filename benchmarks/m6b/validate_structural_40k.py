"""Freeze and validate the prospective M6b-R1 structural-40k engine candidate."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from laserperception.evaluation.pillar_analysis import analyze_pillars

from laserperception.datasets.kitti_raw import KittiRawSequence
from laserperception.detection.artifacts import ExternalArtifactMetadata
from laserperception.detection.m1_assets import resolve_m1_asset_paths
from laserperception.detection.m2_assets import resolve_m2_asset_paths
from laserperception.detection.m2_backend import M2Backend
from laserperception.detection.m6b_engine_remediation import (
    NON_EVALUATION_DRIVE,
    load_engine_manifest,
    reject_evaluation_drive,
    select_repeatability_frames,
    select_third_drive_frames,
    validate_candidate_manifest,
)
from laserperception.detection.mmdet3d_backend import sha256_file
from laserperception.detection.multisweep import MultiSweepBuilder, MultiSweepBuilderConfig
from laserperception.detection.parity_v2 import (
    aggregate_acceptance_v2,
    distribution_statistics,
    raw_tensor_difference_statistics,
)
from laserperception.detection.parity_validation import analyze_sample
from laserperception.evaluation.m6b_input_oracle import (
    freeze_sweep_transforms,
    reconstruct_from_frozen_transforms,
)

RAW_NAMES = ("cls_score", "bbox_pred", "dir_cls_pred")
HISTORY = 10


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"YAML root is not a mapping: {path}")
    return dict(value)


def _clean_measurement_tree(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip():
        raise RuntimeError("M6b-R1 measurement requires a clean Git worktree")
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _to_numpy(value: object) -> np.ndarray:
    current = value
    for method_name in ("detach", "cpu"):
        method = getattr(current, method_name, None)
        if callable(method):
            current = method()
    numpy_method = getattr(current, "numpy", None)
    if callable(numpy_method):
        current = numpy_method()
    return np.asarray(current)


def _raw_array(raw: Mapping[str, list[Any]], name: str) -> np.ndarray:
    values = raw.get(name)
    if not isinstance(values, list) or len(values) != 1:
        raise RuntimeError(f"raw output {name} must contain exactly one tensor")
    return _to_numpy(values[0])


def _raw_hashes(raw: Mapping[str, list[Any]]) -> dict[str, str]:
    return {name: _array_sha256(_raw_array(raw, name)) for name in RAW_NAMES}


def _aggregate_raw(
    per_sample: Sequence[Mapping[str, object]],
    differences: Mapping[str, Sequence[np.ndarray]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in RAW_NAMES:
        records = [record["tensors"][name] for record in per_sample]  # type: ignore[index]
        result[name] = {
            "sample_count": len(records),
            "shape_consistent_across_samples_and_runtimes": all(
                bool(record["shape_consistent"])
                for record in records  # type: ignore[index]
            ),
            "dtype_consistent_across_samples_and_runtimes": all(
                bool(record["dtype_consistent"])
                for record in records  # type: ignore[index]
            ),
            "shapes": sorted({tuple(record["shape"]) for record in records}),  # type: ignore[index]
            "pytorch_dtypes": sorted(
                {str(record["pytorch_dtype"]) for record in records}  # type: ignore[index]
            ),
            "tensorrt_dtypes": sorted(
                {str(record["tensorrt_dtype"]) for record in records}  # type: ignore[index]
            ),
            "absolute_difference": distribution_statistics(
                np.concatenate(tuple(differences[name]))
            ),
        }
    return result


def _write_json(path: Path, record: Mapping[str, object]) -> None:
    encoded = json.dumps(record, indent=2, sort_keys=True) + "\n"
    forbidden = (str(Path.home()), "J:\\", "/root/")
    if any(value and value in encoded for value in forbidden):
        raise RuntimeError("refusing to write evidence containing a private absolute path")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _distribution(values: Sequence[int]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "minimum": int(np.min(array)),
        "median": float(np.median(array)),
        "p90_nearest_rank": int(np.sort(array)[max(0, int(np.ceil(0.9 * len(array))) - 1)]),
        "p95_nearest_rank": int(np.sort(array)[max(0, int(np.ceil(0.95 * len(array))) - 1)]),
        "maximum": int(np.max(array)),
        "at_or_below_30000": sum(value <= 30000 for value in values),
        "above_30000": sum(value > 30000 for value in values),
        "at_or_above_39000": sum(value >= 39000 for value in values),
        "at_40000": sum(value == 40000 for value in values),
    }


def _selection(args: argparse.Namespace) -> int:
    root = _root()
    measurement_commit = _clean_measurement_tree(root)
    reject_evaluation_drive(args.drive_id)
    date_root = args.data_root.expanduser().resolve() / "2011_09_30"
    sequence = KittiRawSequence(date_root, date_root / f"{args.drive_id}_sync")
    frames: list[dict[str, object]] = []
    for frame_index in range(HISTORY, len(sequence)):
        reconstruction = sequence.reconstruct(
            frame_index,
            builder=MultiSweepBuilder(MultiSweepBuilderConfig(max_historical_sweeps=HISTORY)),
        )
        points = reconstruction.point_cloud.points_xyzt
        audit = analyze_pillars(points)
        frames.append(
            {
                "frame_id": f"{args.drive_id}/{frame_index:010d}",
                "frame_index": frame_index,
                "point_count": int(len(points)),
                "model_ready_sha256": _array_sha256(points),
                "voxel_count": audit.retained_count,
                "candidate_voxel_count": audit.candidate_count,
            }
        )
        if len(frames) == 1 or len(frames) % 50 == 0 or frame_index == len(sequence) - 1:
            print(f"input-only census {len(frames)}/{len(sequence) - HISTORY}", flush=True)
    selected = select_third_drive_frames(frames)
    for record in selected:
        frame_index = int(record["frame_index"])
        frozen = freeze_sweep_transforms(sequence, frame_index)
        reproduction = reconstruct_from_frozen_transforms(
            sequence,
            frame_index,
            frozen,
            builder=MultiSweepBuilder(MultiSweepBuilderConfig(max_historical_sweeps=HISTORY)),
        )
        if _array_sha256(reproduction.point_cloud.points_xyzt) != record["model_ready_sha256"]:
            raise RuntimeError("frozen third-drive transforms changed a selected model-ready input")
        record["frozen_sweep_transforms"] = list(frozen)
    record = {
        "schema_version": 1,
        "milestone": "M6b-R1",
        "status": "non_evaluation_input_only_selection_frozen",
        "measurement_commit": measurement_commit,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "network_output_produced": False,
        "evaluation_drives_used": [],
        "drive_id": args.drive_id,
        "history": {"historical_sweeps": HISTORY, "current_plus_history": HISTORY + 1},
        "eligible_frame_count": len(frames),
        "voxel_count_distribution": _distribution([int(frame["voxel_count"]) for frame in frames]),
        "selection_policy": {
            "nearest_rank_quantiles_percent": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
            "tie_break": "lower_frame_index",
            "dedup_fill": "greatest_unused_voxel_count_distance_then_lower_frame_index",
        },
        "coverage": _distribution([int(frame["voxel_count"]) for frame in selected]),
        "selected_frames": selected,
        "all_input_only_frames": frames,
    }
    _write_json(args.output, record)
    print(json.dumps({"selected_frames": selected, "coverage": record["coverage"]}, indent=2))
    return 0


def _backend(root: Path, candidate: Mapping[str, Any]) -> tuple[M2Backend, Path, Path]:
    m1 = _load_yaml(root / "configs/detection/m1_pointpillars_nuscenes.yaml")
    assets = resolve_m1_asset_paths(m1)
    m2_assets = resolve_m2_asset_paths(candidate)
    model = m1["model"]
    backend = M2Backend(
        assets.mmdet3d_root / str(model["upstream_config"]),
        assets.checkpoint_path,
        m2_assets.mmdeploy_root / str(candidate["deployment"]["official_deployment_config"]),
        checkpoint_sha256=str(model["checkpoint"]["sha256"]),
        voxelization_mode="exact_fast",
    )
    return backend, m2_assets.artifact_directory / "pointpillars.onnx", assets.checkpoint_path


def _stage_1(reports: Sequence[Mapping[str, Any]], parity: Mapping[str, Any]) -> dict[str, object]:
    config = parity["stage_1_acceptance"]
    matched = config["matched_high_confidence"]
    counts = config["count_guards"]
    coverage = config["high_confidence_match_coverage"]
    return aggregate_acceptance_v2(
        reports,
        minimum_coverage=float(coverage["pytorch_to_tensorrt_minimum"]),
        minimum_metric_pass_fraction=float(matched["minimum_per_detection_pass_fraction"]),
        maximum_xy_m=float(matched["maximum_xy_center_displacement_m"]),
        maximum_z_m=float(matched["maximum_absolute_z_center_difference_m"]),
        maximum_dimension_relative_error=float(matched["maximum_relative_error_per_lwh_dimension"]),
        maximum_axis_yaw_degrees=float(matched["maximum_axis_yaw_difference_degrees_modulo_pi"]),
        maximum_score_difference=float(matched["maximum_absolute_score_difference"]),
        minimum_direction_agreement=float(matched["minimum_heading_direction_agreement"]),
        maximum_aggregate_count_relative_difference=float(
            counts["aggregate_maximum_relative_difference"]
        ),
    )


def _kitti_parity(args: argparse.Namespace) -> int:
    root = _root()
    measurement_commit = _clean_measurement_tree(root)
    candidate_path = root / "configs/detection/m6_pointpillars_tensorrt_40k.yaml"
    historical_path = root / "configs/detection/m2_pointpillars_tensorrt.yaml"
    parity_path = root / "configs/detection/m2_parity_v2.yaml"
    candidate = load_engine_manifest(candidate_path)
    historical = load_engine_manifest(historical_path)
    validate_candidate_manifest(candidate, historical)
    parity = _load_yaml(parity_path)
    if sha256_file(parity_path) != candidate["validation"]["parity_protocol_sha256"]:
        raise RuntimeError("frozen M2 parity-v2 protocol SHA256 changed")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection["status"] != "non_evaluation_input_only_selection_frozen":
        raise RuntimeError("third-drive selection is not frozen input-only evidence")
    if selection["measurement_commit"] != measurement_commit:
        raise RuntimeError("third-drive selection was not generated at this measurement commit")
    drive_id = str(selection["drive_id"])
    reject_evaluation_drive(drive_id)
    selected = selection["selected_frames"]
    if len(selected) != 12:
        raise RuntimeError("third-drive parity requires exactly 12 frozen frames")
    recomputed = select_third_drive_frames(selection["all_input_only_frames"])
    expected_identity = [
        (int(record["frame_index"]), int(record["voxel_count"])) for record in recomputed
    ]
    selected_identity = [
        (int(record["frame_index"]), int(record["voxel_count"])) for record in selected
    ]
    if selected_identity != expected_identity:
        raise RuntimeError("frozen third-drive frame set differs from deterministic selection")

    engine_artifact = ExternalArtifactMetadata.from_file(
        args.engine, logical_name=str(candidate["artifacts"]["engine"]["logical_name"])
    )
    if engine_artifact.sha256 != args.engine_sha256:
        raise RuntimeError("candidate engine SHA256 differs from the build evidence")
    backend, onnx_path, checkpoint_path = _backend(root, candidate)
    onnx_artifact = ExternalArtifactMetadata.from_file(
        onnx_path, logical_name=str(candidate["artifacts"]["onnx"]["logical_name"])
    )
    if onnx_artifact.sha256 != candidate["artifacts"]["onnx"]["sha256"]:
        raise RuntimeError("source ONNX SHA256 changed")
    checkpoint_artifact = ExternalArtifactMetadata.from_file(
        checkpoint_path, logical_name="checkpoints/pointpillars_nuscenes.pth"
    )
    if checkpoint_artifact.sha256 != candidate["source_model"]["checkpoint_sha256"]:
        raise RuntimeError("checkpoint SHA256 changed")

    date_root = args.data_root.expanduser().resolve() / "2011_09_30"
    sequence = KittiRawSequence(date_root, date_root / f"{drive_id}_sync")
    reports: list[dict[str, object]] = []
    raw_records: list[dict[str, object]] = []
    differences: dict[str, list[np.ndarray]] = {name: [] for name in RAW_NAMES}
    executions: list[dict[str, object]] = []
    voxelized_by_index: dict[int, object] = {}
    exported_threshold = float(parity["thresholds"]["exported_detection"])
    high_threshold = float(parity["thresholds"]["high_confidence_guard"])
    minimum_iou = float(parity["matching"]["minimum_bev_iou"])
    for record in selected:
        frame_index = int(record["frame_index"])
        reproduction = reconstruct_from_frozen_transforms(
            sequence,
            frame_index,
            record["frozen_sweep_transforms"],
            builder=MultiSweepBuilder(MultiSweepBuilderConfig(max_historical_sweeps=HISTORY)),
        )
        points = reproduction.point_cloud.points_xyzt
        if _array_sha256(points) != record["model_ready_sha256"]:
            raise RuntimeError("selected model-ready input hash changed before network execution")
        prepared = backend.prepare_model_ready_points(
            reproduction.point_cloud,
            sample_id=str(record["frame_id"]),
            coordinate_frame="kitti_model_aligned_lidar",
        )
        voxelized = backend.voxelize(prepared)
        if voxelized.voxel_count != int(record["voxel_count"]):
            raise RuntimeError("production exact_fast voxel count differs from frozen input census")
        voxelized_by_index[frame_index] = voxelized
        pytorch_raw = backend.run_rewritten_pytorch_raw(voxelized)
        tensorrt_raw = backend.run_tensorrt_raw(voxelized, args.engine)
        tensor_records: dict[str, object] = {}
        for name in RAW_NAMES:
            comparison, difference = raw_tensor_difference_statistics(
                _raw_array(pytorch_raw, name), _raw_array(tensorrt_raw, name)
            )
            tensor_records[name] = comparison
            differences[name].append(difference.astype(np.float32, copy=False))
        raw_records.append(
            {"frame_id": record["frame_id"], "frame_index": frame_index, "tensors": tensor_records}
        )
        pytorch_frame = backend.postprocess_raw(
            pytorch_raw,
            voxelized,
            backend_name="mmdeploy_rewritten_pytorch",
            precision="fp32",
            provenance_mode="full",
        )
        tensorrt_frame = backend.postprocess_raw(
            tensorrt_raw,
            voxelized,
            backend_name="structural_40k_tensorrt",
            precision="fp16",
            provenance_mode="full",
        )
        report = analyze_sample(
            pytorch_frame,
            tensorrt_frame,
            sample_index=frame_index,
            exported_threshold=exported_threshold,
            high_confidence_threshold=high_threshold,
            minimum_bev_iou=minimum_iou,
        )
        reports.append(report)
        executions.append(
            {
                "frame_id": record["frame_id"],
                "frame_index": frame_index,
                "point_count": int(len(points)),
                "model_ready_sha256": record["model_ready_sha256"],
                "voxel_count": voxelized.voxel_count,
                "voxel_hashes": voxelized.hashes(),
                "pytorch_raw_hashes": _raw_hashes(pytorch_raw),
                "tensorrt_raw_hashes": _raw_hashes(tensorrt_raw),
                "pytorch_detection_frame_sha256": _json_sha256(pytorch_frame.to_dict()),
                "tensorrt_detection_frame_sha256": _json_sha256(tensorrt_frame.to_dict()),
            }
        )
        print(f"KITTI parity {record['frame_id']} voxels={voxelized.voxel_count}", flush=True)
    stage_1 = _stage_1(reports, parity)
    result = {
        "schema_version": 1,
        "milestone": "M6b-R1",
        "status": "pass" if stage_1["overall_pass"] else "fail",
        "measurement_commit": measurement_commit,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "non_evaluation_KITTI_rewritten_PyTorch_FP32_vs_structural_40k_TensorRT_FP16",
        "drive_id": drive_id,
        "evaluation_drives_used": [],
        "ground_truth_metrics_run": False,
        "selection": {
            "path_class": "external_local_input_only_evidence",
            "sha256": sha256_file(args.selection),
            "frame_count": len(selected),
            "frame_ids": [record["frame_id"] for record in selected],
            "voxel_counts": [record["voxel_count"] for record in selected],
        },
        "artifacts": {
            "checkpoint": checkpoint_artifact.to_dict(),
            "onnx": onnx_artifact.to_dict(),
            "candidate_engine": engine_artifact.to_dict(),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "backend_versions": dict(backend.versions),
            "mmdeploy": str(importlib.import_module("mmdeploy").__version__),
            "tensorrt": str(importlib.import_module("tensorrt").__version__),
        },
        "protocol": {
            "parity_config": "configs/detection/m2_parity_v2.yaml",
            "parity_config_sha256": sha256_file(parity_path),
            "thresholds": parity["thresholds"],
            "matching": parity["matching"],
            "stage_1_acceptance": parity["stage_1_acceptance"],
            "thresholds_mutated": False,
        },
        "executions": executions,
        "raw_network_comparisons": {
            "aggregate": _aggregate_raw(raw_records, differences),
            "per_sample": raw_records,
        },
        "samples": reports,
        "stage_1": stage_1,
        "overall_pass": bool(stage_1["overall_pass"]),
        "M6b_evaluation_started": False,
    }
    _write_json(args.output, result)
    if not stage_1["overall_pass"]:
        print(json.dumps(stage_1, indent=2, sort_keys=True))
        return 2

    repeat_frames = select_repeatability_frames(selected)
    repeat_results: dict[str, object] = {}
    for role, record in repeat_frames.items():
        frame_index = int(record["frame_index"])
        voxelized = voxelized_by_index[frame_index]
        hashes = [_raw_hashes(backend.run_tensorrt_raw(voxelized, args.engine)) for _ in range(5)]
        exact = all(value == hashes[0] for value in hashes)
        repeat_results[role] = {
            "frame_id": record["frame_id"],
            "frame_index": frame_index,
            "voxel_count": record["voxel_count"],
            "repetitions": 5,
            "raw_output_hashes": hashes,
            "exact": exact,
        }
        if not exact:
            break
    repeatability = {
        "schema_version": 1,
        "milestone": "M6b-R1",
        "status": "pass"
        if len(repeat_results) == 2
        and all(
            bool(value["exact"])
            for value in repeat_results.values()  # type: ignore[index]
        )
        else "fail",
        "measurement_commit": measurement_commit,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_engine_sha256": engine_artifact.sha256,
        "drive_id": drive_id,
        "evaluation_drives_used": [],
        "samples": repeat_results,
    }
    _write_json(args.repeatability_output, repeatability)
    print(json.dumps({"stage_1": stage_1, "repeatability": repeatability}, indent=2))
    return 0 if repeatability["status"] == "pass" else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select", help="freeze input-only drive-0016 census and set")
    select.add_argument("--data-root", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--drive-id", default=NON_EVALUATION_DRIVE)
    parity = subparsers.add_parser("kitti-parity", help="run Gate 2 and repeatability")
    parity.add_argument("--data-root", type=Path, required=True)
    parity.add_argument("--selection", type=Path, required=True)
    parity.add_argument("--engine", type=Path, required=True)
    parity.add_argument("--engine-sha256", required=True)
    parity.add_argument("--output", type=Path, required=True)
    parity.add_argument("--repeatability-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "select":
        return _selection(args)
    if args.command == "kitti-parity":
        return _kitti_parity(args)
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
