"""Validate frozen 20-sample PyTorch FP32 versus TensorRT FP16 detection parity."""

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

import yaml

from laserperception.detection.artifacts import ExternalArtifactMetadata
from laserperception.detection.m1_assets import resolve_m1_asset_paths
from laserperception.detection.m2_assets import resolve_m2_asset_paths
from laserperception.detection.m2_backend import M2Backend
from laserperception.detection.mmdet3d_backend import sha256_file
from laserperception.detection.parity_validation import (
    aggregate_acceptance,
    analyze_sample,
)
from laserperception.detection.runtime_metadata import (
    nvidia_smi_value,
    repository_git_sha,
)


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


def _raw_tensor_comparison(reference: Mapping[str, list[Any]], candidate: Mapping[str, list[Any]]):
    records: dict[str, dict[str, object]] = {}
    for name in ("cls_score", "bbox_pred", "dir_cls_pred"):
        reference_tensor = reference[name][0].detach().float()
        candidate_tensor = candidate[name][0].detach().float()
        if tuple(reference_tensor.shape) != tuple(candidate_tensor.shape):
            raise RuntimeError(f"raw output shape mismatch for {name}")
        difference = (candidate_tensor - reference_tensor).abs()
        denominator_mask = reference_tensor.abs() > 1e-6
        maximum_relative = (
            float((difference[denominator_mask] / reference_tensor[denominator_mask].abs()).max())
            if bool(denominator_mask.any())
            else 0.0
        )
        records[name] = {
            "shape": [int(value) for value in reference_tensor.shape],
            "pytorch_dtype": str(reference[name][0].dtype),
            "tensorrt_dtype": str(candidate[name][0].dtype),
            "maximum_absolute_difference": float(difference.max()),
            "mean_absolute_difference": float(difference.mean()),
            "maximum_relative_difference_where_abs_reference_gt_1e_6": maximum_relative,
            "pytorch_minimum": float(reference_tensor.min()),
            "pytorch_maximum": float(reference_tensor.max()),
            "tensorrt_minimum": float(candidate_tensor.min()),
            "tensorrt_maximum": float(candidate_tensor.max()),
        }
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="nuScenes root; defaults to environment variable")
    parser.add_argument("--engine", type=Path, help="override external TensorRT engine")
    parser.add_argument("--output", type=Path, help="override external sanitized JSON")
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
    parity_manifest = _manifest("m2_parity.yaml")
    m1_assets = resolve_m1_asset_paths(m1_manifest)
    m2_assets = resolve_m2_asset_paths(m2_manifest)
    engine_path = args.engine or m2_assets.engine_directory / "pointpillars_fp16.engine"
    onnx_path = m2_assets.artifact_directory / "pointpillars.onnx"
    frozen_indices = [int(value) for value in parity_manifest["dataset"]["sample_indices"]]
    diagnostic_only = args.diagnostic_index is not None
    indices = [int(args.diagnostic_index)] if diagnostic_only else frozen_indices
    if not diagnostic_only and len(indices) != 20:
        raise SystemExit("error: frozen M2 parity set must contain exactly 20 indices")
    output = args.output or m2_assets.artifact_directory / (
        f"diagnostic_index_{indices[0]}.json" if diagnostic_only else "parity.json"
    )

    model_info = m1_manifest["model"]
    checkpoint_info = model_info["checkpoint"]
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

    exported_threshold = float(parity_manifest["thresholds"]["exported_detection"])
    high_threshold = float(parity_manifest["thresholds"]["high_confidence_guard"])
    minimum_iou = float(parity_manifest["matching"]["minimum_bev_iou"])
    profile_minimum, profile_maximum = _profile_bounds(m2_manifest)
    reports: list[dict[str, object]] = []
    raw_comparisons: list[dict[str, object]] = []
    input_diagnostics: list[dict[str, object]] = []

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
        raw_comparisons.append(
            {
                "sample_index": index,
                "sample_id": prepared.sample_id,
                "tensors": _raw_tensor_comparison(pytorch_raw, tensorrt_raw),
            }
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

    acceptance_config = parity_manifest["acceptance"]
    matched_config = acceptance_config["matched_high_confidence"]
    count_config = acceptance_config["exported_threshold_counts"]
    summary = aggregate_acceptance(
        reports,
        minimum_coverage=float(
            acceptance_config["high_confidence_match_coverage"]["pytorch_to_tensorrt_minimum"]
        ),
        maximum_xy_m=float(matched_config["maximum_xy_center_displacement_m"]),
        maximum_z_m=float(matched_config["maximum_absolute_z_center_difference_m"]),
        maximum_dimension_relative_error=float(
            matched_config["maximum_relative_error_per_lwh_dimension"]
        ),
        maximum_yaw_degrees=float(matched_config["maximum_circular_yaw_difference_degrees"]),
        maximum_score_difference=float(matched_config["maximum_absolute_score_difference"]),
        maximum_aggregate_count_relative_difference=float(
            count_config["aggregate_maximum_relative_difference"]
        ),
    )
    onnx_artifact = ExternalArtifactMetadata.from_file(
        onnx_path, logical_name=str(m2_manifest["artifacts"]["onnx"]["logical_name"])
    )
    engine_artifact = ExternalArtifactMetadata.from_file(
        engine_path, logical_name=str(m2_manifest["artifacts"]["engine"]["logical_name"])
    )
    torch = importlib.import_module("torch")
    mmdeploy = importlib.import_module("mmdeploy")
    tensorrt = importlib.import_module("tensorrt")
    onnx = importlib.import_module("onnx")
    result = {
        "schema_version": "1.0",
        "status": "diagnostic"
        if diagnostic_only
        else ("pass" if bool(summary["overall_pass"]) else "fail"),
        "diagnostic_only": diagnostic_only,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit_sha": repository_git_sha(repository_root),
        "milestone": "M2",
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
            "parity_config_sha256": sha256_file(
                repository_root / "configs" / "detection" / "m2_parity.yaml"
            ),
            "thresholds": parity_manifest["thresholds"],
            "matching": parity_manifest["matching"],
            "acceptance": parity_manifest["acceptance"],
            "threshold_edge_policy": parity_manifest["threshold_edge_policy"],
        },
        "shared_inputs": input_diagnostics,
        "raw_network_comparisons": raw_comparisons,
        "samples": reports,
        "acceptance_summary": summary,
        "failure_categories": []
        if bool(summary["overall_pass"])
        else ["network_numerical_difference"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("Sanitized parity evidence written outside the repository.")
    if not diagnostic_only and not bool(summary["overall_pass"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
