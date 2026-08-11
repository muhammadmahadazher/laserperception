"""Run the preregistered M2 parity-v2 protocol on the unchanged FP16 engine."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from laserperception.detection.artifacts import ExternalArtifactMetadata
from laserperception.detection.m1_assets import resolve_m1_asset_paths
from laserperception.detection.m2_assets import resolve_m2_asset_paths
from laserperception.detection.m2_backend import M2Backend
from laserperception.detection.mmdet3d_backend import sha256_file
from laserperception.detection.parity_v2 import (
    aggregate_acceptance_v2,
    direction_disagreement_records,
    direction_population_summary,
    distribution_statistics,
    official_nms_pre_union,
    raw_tensor_difference_statistics,
    reshape_anchor_logits,
)
from laserperception.detection.parity_validation import analyze_sample
from laserperception.detection.runtime_metadata import (
    nvidia_smi_value,
    repository_git_sha,
)

_RAW_TENSOR_NAMES = ("cls_score", "bbox_pred", "dir_cls_pred")


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _manifest(name: str) -> dict[str, Any]:
    path = _repository_root() / "configs" / "detection" / name
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def _data_root(value: str | None) -> Path:
    raw = value or os.environ.get("LASERPERCEPTION_NUSCENES_ROOT")
    if not raw:
        raise ValueError("set LASERPERCEPTION_NUSCENES_ROOT or pass --data-root")
    return Path(raw).expanduser()


def _profile_bounds(manifest: Mapping[str, Any]) -> tuple[int, int]:
    profile = manifest["profile"]
    return (
        int(profile["selected_min_shapes"]["voxels"][0]),
        int(profile["selected_max_shapes"]["voxels"][0]),
    )


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


def _aggregate_raw_differences(
    sample_records: Sequence[Mapping[str, Any]],
    differences: Mapping[str, Sequence[np.ndarray]],
) -> dict[str, object]:
    summary: dict[str, object] = {}
    for name in _RAW_TENSOR_NAMES:
        tensor_records = [
            record["tensors"][name]
            for record in sample_records
            if isinstance(record.get("tensors"), Mapping)
        ]
        combined = np.concatenate(tuple(differences[name]))
        pytorch_shapes = {tuple(record["shape"]) for record in tensor_records}
        tensorrt_shapes = {tuple(record["shape"]) for record in tensor_records}
        pytorch_dtypes = {str(record["pytorch_dtype"]) for record in tensor_records}
        tensorrt_dtypes = {str(record["tensorrt_dtype"]) for record in tensor_records}
        summary[name] = {
            "sample_count": len(tensor_records),
            "shape_consistent_across_samples_and_runtimes": (
                len(pytorch_shapes) == 1
                and pytorch_shapes == tensorrt_shapes
                and all(bool(record["shape_consistent"]) for record in tensor_records)
            ),
            "shapes": [list(shape) for shape in sorted(pytorch_shapes)],
            "dtype_consistent_across_samples_and_runtimes": (
                len(pytorch_dtypes) == 1
                and pytorch_dtypes == tensorrt_dtypes
                and all(bool(record["dtype_consistent"]) for record in tensor_records)
            ),
            "pytorch_dtypes": sorted(pytorch_dtypes),
            "tensorrt_dtypes": sorted(tensorrt_dtypes),
            "absolute_difference": distribution_statistics(combined),
        }
    return summary


def _artifact_or_stop(
    path: Path,
    *,
    logical_name: str,
    expected_sha256: str,
    kind: str,
) -> ExternalArtifactMetadata:
    artifact = ExternalArtifactMetadata.from_file(path, logical_name=logical_name)
    if artifact.sha256 != expected_sha256:
        raise SystemExit(
            f"error: {kind} SHA256 differs from the frozen parity-v2 artifact; do not rebuild"
        )
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="nuScenes root; defaults to environment variable")
    parser.add_argument("--engine", type=Path, help="override the external frozen TensorRT engine")
    parser.add_argument("--output", type=Path, help="override the external sanitized JSON")
    parser.add_argument(
        "--diagnostic-index",
        type=int,
        help="run one index for debugging without claiming frozen-suite parity",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository_root = _repository_root()
    m1_manifest = _manifest("m1_pointpillars_nuscenes.yaml")
    m2_manifest = _manifest("m2_pointpillars_tensorrt.yaml")
    parity_manifest = _manifest("m2_parity_v2.yaml")
    if int(parity_manifest.get("protocol_version", 0)) != 2:
        raise SystemExit("error: parity runner requires protocol_version 2")

    m1_assets = resolve_m1_asset_paths(m1_manifest)
    m2_assets = resolve_m2_asset_paths(m2_manifest)
    engine_path = args.engine or m2_assets.engine_directory / "pointpillars_fp16.engine"
    onnx_path = m2_assets.artifact_directory / "pointpillars.onnx"
    frozen_indices = [int(value) for value in parity_manifest["dataset"]["sample_indices"]]
    diagnostic_only = args.diagnostic_index is not None
    indices = [int(args.diagnostic_index)] if diagnostic_only else frozen_indices
    if not diagnostic_only and len(indices) != 20:
        raise SystemExit("error: frozen M2 parity-v2 set must contain exactly 20 indices")
    output = args.output or m2_assets.artifact_directory / (
        f"diagnostic_v2_index_{indices[0]}.json" if diagnostic_only else "parity_v2.json"
    )

    frozen_artifacts = parity_manifest["frozen_artifacts"]
    onnx_artifact = _artifact_or_stop(
        onnx_path,
        logical_name=str(m2_manifest["artifacts"]["onnx"]["logical_name"]),
        expected_sha256=str(frozen_artifacts["onnx_sha256"]),
        kind="ONNX",
    )
    engine_artifact = _artifact_or_stop(
        engine_path,
        logical_name=str(m2_manifest["artifacts"]["engine"]["logical_name"]),
        expected_sha256=str(frozen_artifacts["tensorrt_fp16_engine_sha256"]),
        kind="TensorRT engine",
    )

    model_info = m1_manifest["model"]
    checkpoint_info = model_info["checkpoint"]
    if str(checkpoint_info["sha256"]) != str(frozen_artifacts["checkpoint_sha256"]):
        raise SystemExit("error: checkpoint manifest differs from the frozen parity-v2 protocol")
    deploy_relative = str(m2_manifest["deployment"]["official_deployment_config"])
    backend = M2Backend(
        m1_assets.mmdet3d_root / str(model_info["upstream_config"]),
        m1_assets.checkpoint_path,
        m2_assets.mmdeploy_root / deploy_relative,
        checkpoint_sha256=str(checkpoint_info["sha256"]),
    )
    data_root = _data_root(args.data_root)
    split_size = backend.dataset_size(data_root, "mini_val")
    if split_size != int(parity_manifest["dataset"]["observed_split_size"]):
        raise SystemExit("error: prepared mini_val split size differs from the frozen protocol")

    nms_pre = backend.official_nms_pre
    exported_threshold = float(parity_manifest["thresholds"]["exported_detection"])
    high_threshold = float(parity_manifest["thresholds"]["high_confidence_guard"])
    minimum_iou = float(parity_manifest["matching"]["minimum_bev_iou"])
    profile_minimum, profile_maximum = _profile_bounds(m2_manifest)
    reports: list[dict[str, object]] = []
    raw_sample_records: list[dict[str, object]] = []
    raw_differences: dict[str, list[np.ndarray]] = {name: [] for name in _RAW_TENSOR_NAMES}
    input_diagnostics: list[dict[str, object]] = []
    all_pytorch_direction: list[np.ndarray] = []
    all_tensorrt_direction: list[np.ndarray] = []
    relevant_pytorch_direction: list[np.ndarray] = []
    relevant_tensorrt_direction: list[np.ndarray] = []
    relevant_disagreements: list[dict[str, object]] = []

    for index in indices:
        prepared = backend.prepare_sample(data_root, split="mini_val", index=index)
        voxelized = backend.voxelize(prepared)
        if not profile_minimum <= voxelized.voxel_count <= profile_maximum:
            raise SystemExit(
                f"error: index {index} voxel count {voxelized.voxel_count} is outside the "
                "frozen TensorRT profile; classify as profile_or_binding_failure"
            )
        pytorch_raw = backend.run_rewritten_pytorch_raw(voxelized)
        tensorrt_raw = backend.run_tensorrt_raw(voxelized, engine_path)

        sample_tensor_records: dict[str, object] = {}
        raw_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in _RAW_TENSOR_NAMES:
            pytorch_array = _raw_array(pytorch_raw, name)
            tensorrt_array = _raw_array(tensorrt_raw, name)
            comparison, difference = raw_tensor_difference_statistics(pytorch_array, tensorrt_array)
            sample_tensor_records[name] = comparison
            raw_differences[name].append(difference.astype(np.float32, copy=False))
            raw_arrays[name] = (pytorch_array, tensorrt_array)
        raw_sample_records.append(
            {
                "sample_index": index,
                "sample_id": prepared.sample_id,
                "tensors": sample_tensor_records,
            }
        )

        pytorch_direction_raw, tensorrt_direction_raw = raw_arrays["dir_cls_pred"]
        pytorch_class_raw, tensorrt_class_raw = raw_arrays["cls_score"]
        anchors_per_position = int(pytorch_direction_raw.shape[1]) // 2
        if anchors_per_position <= 0 or int(pytorch_class_raw.shape[1]) % anchors_per_position:
            raise RuntimeError("raw PointPillars head channels do not share an anchor layout")
        class_count = int(pytorch_class_raw.shape[1]) // anchors_per_position
        pytorch_direction = reshape_anchor_logits(pytorch_direction_raw, values_per_anchor=2)
        tensorrt_direction = reshape_anchor_logits(tensorrt_direction_raw, values_per_anchor=2)
        pytorch_classes = reshape_anchor_logits(pytorch_class_raw, values_per_anchor=class_count)
        tensorrt_classes = reshape_anchor_logits(tensorrt_class_raw, values_per_anchor=class_count)
        if len(pytorch_classes) != len(pytorch_direction):
            raise RuntimeError("class and direction heads expose different anchor counts")
        relevant_indices = official_nms_pre_union(
            pytorch_classes, tensorrt_classes, nms_pre=nms_pre
        )
        all_pytorch_direction.append(pytorch_direction.astype(np.float32, copy=False))
        all_tensorrt_direction.append(tensorrt_direction.astype(np.float32, copy=False))
        relevant_pytorch_direction.append(
            pytorch_direction[relevant_indices].astype(np.float32, copy=False)
        )
        relevant_tensorrt_direction.append(
            tensorrt_direction[relevant_indices].astype(np.float32, copy=False)
        )
        relevant_disagreements.extend(
            direction_disagreement_records(
                pytorch_direction,
                tensorrt_direction,
                sample_index=index,
                sample_id=prepared.sample_id,
                anchor_indices=relevant_indices,
            )
        )

        pytorch_frame = backend.postprocess_raw(
            pytorch_raw,
            voxelized,
            backend_name="mmdeploy_rewritten_pytorch",
            precision="fp32",
        )
        tensorrt_frame = backend.postprocess_raw(
            tensorrt_raw,
            voxelized,
            backend_name="tensorrt",
            precision="fp16",
        )
        report = analyze_sample(
            pytorch_frame,
            tensorrt_frame,
            sample_index=index,
            exported_threshold=exported_threshold,
            high_confidence_threshold=high_threshold,
            minimum_bev_iou=minimum_iou,
        )
        reports.append(report)
        input_diagnostics.append(
            {
                "sample_index": index,
                "sample_id": prepared.sample_id,
                "prepared_point_count": int(prepared.points_xyz.shape[0]),
                "shared_object_identity": True,
                "voxel_count": voxelized.voxel_count,
                "voxel_tensors": voxelized.tensor_statistics(),
                "profile_fit": True,
            }
        )
        counts = report["counts"]
        print(
            f"index {index}: exported PyTorch={counts['pytorch_exported']} "
            f"TensorRT={counts['tensorrt_exported']} matches={len(report['matches'])}"
        )

    stage_1_config = parity_manifest["stage_1_acceptance"]
    matched_config = stage_1_config["matched_high_confidence"]
    count_config = stage_1_config["count_guards"]
    coverage_config = stage_1_config["high_confidence_match_coverage"]
    stage_1 = aggregate_acceptance_v2(
        reports,
        minimum_coverage=float(coverage_config["pytorch_to_tensorrt_minimum"]),
        minimum_metric_pass_fraction=float(matched_config["minimum_per_detection_pass_fraction"]),
        maximum_xy_m=float(matched_config["maximum_xy_center_displacement_m"]),
        maximum_z_m=float(matched_config["maximum_absolute_z_center_difference_m"]),
        maximum_dimension_relative_error=float(
            matched_config["maximum_relative_error_per_lwh_dimension"]
        ),
        maximum_axis_yaw_degrees=float(
            matched_config["maximum_axis_yaw_difference_degrees_modulo_pi"]
        ),
        maximum_score_difference=float(matched_config["maximum_absolute_score_difference"]),
        minimum_direction_agreement=float(matched_config["minimum_heading_direction_agreement"]),
        maximum_aggregate_count_relative_difference=float(
            count_config["aggregate_maximum_relative_difference"]
        ),
    )
    raw_network_summary = _aggregate_raw_differences(raw_sample_records, raw_differences)
    all_direction_summary = direction_population_summary(
        np.concatenate(all_pytorch_direction),
        np.concatenate(all_tensorrt_direction),
    )
    relevant_direction_summary = direction_population_summary(
        np.concatenate(relevant_pytorch_direction),
        np.concatenate(relevant_tensorrt_direction),
    )
    direction_population_diagnostic = {
        "official_nms_pre": nms_pre,
        "decision_relevant_definition": (
            "union of anchors in the official nms_pre top candidate pool in either runtime"
        ),
        "all_anchors": all_direction_summary,
        "decision_relevant_anchors": {
            **relevant_direction_summary,
            "disagreements": relevant_disagreements,
        },
    }

    stage_1_pass = bool(stage_1["overall_pass"])
    full_heading = stage_1["full_heading_diagnostics"]
    distinct_outliers = stage_1["distinct_high_confidence_continuous_outliers"]
    stage_2 = (
        {
            "required": False,
            "status": "not_required_stage_1_passed",
            "investigations_performed": [],
            "confirmed_nms_survivor_swaps": [],
            "other_discrete_divergences": [],
            "unexplained_outliers": [],
            "recommended_remediation": None,
        }
        if stage_1_pass
        else {
            "required": True,
            "status": "targeted_low_cost_forensics_recorded",
            "investigations_performed": [
                "aggregate raw-network absolute-difference statistics",
                "all-anchor direction argmax and margin analysis",
                "official nms_pre decision-relevant direction argmax and margin analysis",
                "final matched-detection axis-yaw and full-heading separation",
            ],
            "confirmed_nms_survivor_swaps": [],
            "other_discrete_divergences": full_heading["direction_flips"],
            "unexplained_outliers": distinct_outliers["detections"],
            "recommended_remediation": stage_1["recommended_next_experiment"],
        }
    )

    torch = importlib.import_module("torch")
    mmdeploy = importlib.import_module("mmdeploy")
    tensorrt = importlib.import_module("tensorrt")
    onnx = importlib.import_module("onnx")
    protocol_path = repository_root / "configs" / "detection" / "m2_parity_v2.yaml"
    v1_protocol_path = repository_root / "configs" / "detection" / "m2_parity_v1.yaml"
    status = "diagnostic" if diagnostic_only else ("pass" if stage_1_pass else "fail")
    result = {
        "schema_version": "2.0",
        "protocol_version": 2,
        "status": status,
        "diagnostic_only": diagnostic_only,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit_sha": repository_git_sha(repository_root),
        "milestone": "M2",
        "reference_to": {
            **parity_manifest["reference_to"],
            "v1_config_sha256": sha256_file(v1_protocol_path),
        },
        "rationale": str(parity_manifest["rationale"]),
        "dataset": {
            "name": "nuScenes",
            "version": "v1.0-mini",
            "split": "mini_val",
            "observed_split_size": split_size,
            "sample_indices": indices,
        },
        "model": {
            "checkpoint_sha256": str(checkpoint_info["sha256"]),
            "mmdet3d_commit": str(m2_manifest["source_model"]["framework_commit"]),
        },
        "artifacts": {
            "onnx": onnx_artifact.to_dict(),
            "engine": engine_artifact.to_dict(),
            "engine_rebuilt_for_v2": False,
            "layer_precision_changes": False,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            **dict(backend.versions),
            "mmdeploy": str(mmdeploy.__version__),
            "mmdeploy_commit": str(m2_manifest["deployment"]["exporter_commit"]),
            "onnx": str(onnx.__version__),
            "tensorrt": str(tensorrt.__version__),
            "torch_cuda_runtime": str(torch.version.cuda),
            "nvidia_driver": nvidia_smi_value("driver_version"),
            "gpu_name": str(torch.cuda.get_device_name(0)),
        },
        "protocol": {
            "parity_config": "configs/detection/m2_parity_v2.yaml",
            "parity_config_sha256": sha256_file(protocol_path),
            "thresholds": parity_manifest["thresholds"],
            "matching": parity_manifest["matching"],
            "stage_1_acceptance": parity_manifest["stage_1_acceptance"],
            "threshold_edge_policy": parity_manifest["threshold_edge_policy"],
        },
        "shared_inputs": input_diagnostics,
        "raw_network_comparisons": {
            "aggregate": raw_network_summary,
            "per_sample": raw_sample_records,
        },
        "direction_population_diagnostic": direction_population_diagnostic,
        "samples": reports,
        "stage_1": stage_1,
        "stage_2": stage_2,
        "confirmed_nms_survivor_swaps": stage_2["confirmed_nms_survivor_swaps"],
        "other_discrete_divergences": stage_2["other_discrete_divergences"],
        "threshold_edge_crossings": stage_1["threshold_edge_disagreements"],
        "unexplained_outliers": stage_2["unexplained_outliers"],
        "overall_pass": stage_1_pass,
        "benchmark_run": False,
        "failure_categories": [] if stage_1_pass else list(stage_1["failed_checks"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(stage_1, indent=2, sort_keys=True))
    print(json.dumps(direction_population_diagnostic, indent=2, sort_keys=True))
    print("Sanitized parity-v2 evidence written outside the repository.")
    if not diagnostic_only and not stage_1_pass:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
