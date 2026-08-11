"""Diagnose M2 native/export fidelity and runtime components without promotion."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from laserperception.detection.benchmark import latency_statistics_ms
from laserperception.detection.m1_assets import resolve_m1_asset_paths
from laserperception.detection.m2_assets import resolve_m2_asset_paths
from laserperception.detection.m2_backend import M2Backend
from laserperception.detection.m2_diagnostics import (
    RAW_OUTPUT_NAMES,
    assert_raw_outputs_cuda0,
)
from laserperception.detection.mmdet3d_backend import sha256_file
from laserperception.detection.parity_v2 import (
    aggregate_acceptance_v2,
    distribution_statistics,
    raw_tensor_difference_statistics,
)
from laserperception.detection.parity_validation import analyze_sample
from laserperception.detection.runtime_metadata import repository_git_sha


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


def _raw_array(raw: Mapping[str, Sequence[Any]], name: str) -> np.ndarray:
    levels = raw.get(name)
    if not isinstance(levels, Sequence) or isinstance(levels, (str, bytes)) or len(levels) != 1:
        raise RuntimeError(f"raw output {name} must contain exactly one tensor")
    return _to_numpy(levels[0])


def _aggregate_raw_differences(
    records: Sequence[Mapping[str, Any]], differences: Mapping[str, Sequence[np.ndarray]]
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in RAW_OUTPUT_NAMES:
        tensors = [record["tensors"][name] for record in records]
        combined = np.concatenate(tuple(differences[name]))
        native_shapes = {tuple(record["shape"]) for record in tensors}
        native_dtypes = {str(record["pytorch_dtype"]) for record in tensors}
        rewritten_dtypes = {str(record["tensorrt_dtype"]) for record in tensors}
        result[name] = {
            "sample_count": len(tensors),
            "shapes": [list(shape) for shape in sorted(native_shapes)],
            "shape_consistent": all(bool(record["shape_consistent"]) for record in tensors),
            "native_dtypes": sorted(native_dtypes),
            "rewritten_dtypes": sorted(rewritten_dtypes),
            "dtype_consistent": all(bool(record["dtype_consistent"]) for record in tensors),
            "absolute_difference": distribution_statistics(combined),
        }
    return result


def _cuda_block(
    torch: Any,
    operation: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, float | int], object]:
    result: object = None
    for _ in range(warmup):
        result = operation()
    torch.cuda.synchronize(0)
    latencies: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = operation()
        end.record()
        end.synchronize()
        latencies.append(float(start.elapsed_time(end)))
    return latency_statistics_ms(latencies), result


def _wall_block(
    torch: Any,
    operation: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
    synchronize_cuda: bool,
) -> tuple[dict[str, float | int], object]:
    result: object = None
    for _ in range(warmup):
        result = operation()
        if synchronize_cuda:
            torch.cuda.synchronize(0)
    latencies: list[float] = []
    for _ in range(iterations):
        if synchronize_cuda:
            torch.cuda.synchronize(0)
        started = time.perf_counter()
        result = operation()
        if synchronize_cuda:
            torch.cuda.synchronize(0)
        latencies.append((time.perf_counter() - started) * 1000.0)
    return latency_statistics_ms(latencies), result


def _gpu_telemetry() -> dict[str, object]:
    fields = (
        "name",
        "driver_version",
        "temperature.gpu",
        "power.draw",
        "power.limit",
        "clocks.sm",
        "clocks.mem",
        "utilization.gpu",
        "utilization.memory",
    )
    process = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0 or not process.stdout.strip():
        return {"available": False, "error": process.stderr.strip() or "nvidia-smi failed"}
    values = [value.strip() for value in process.stdout.splitlines()[0].split(",")]
    return {
        "available": True,
        **dict(zip(fields, values, strict=True)),
    }


def _file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _m1_reproduction(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    environment = payload["environment"]
    return {
        "source_path": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "commit_sha": payload["commit_sha"],
        "warmups": payload["warmup_iterations_per_boundary"],
        "measurements": payload["measured_iterations_per_boundary"],
        "environment": {
            key: environment[key]
            for key in (
                "gpu_name",
                "nvidia_driver",
                "torch",
                "torch_cuda_runtime",
                "mmcv",
                "mmengine",
                "mmdet",
                "mmdet3d",
            )
        },
        "model_test_step": payload["measurements"]["model_gpu"],
        "end_to_end": payload["measurements"]["end_to_end"],
        "historical_reference_ms": {
            "model_test_step_median": 52.896,
            "end_to_end_median": 55.097,
        },
    }


def _fidelity_diagnostic(
    backend: M2Backend,
    data_root: Path,
    indices: Sequence[int],
    parity_manifest: Mapping[str, Any],
) -> dict[str, object]:
    thresholds = parity_manifest["thresholds"]
    matching = parity_manifest["matching"]
    reports: list[dict[str, object]] = []
    raw_records: list[dict[str, object]] = []
    differences: dict[str, list[np.ndarray]] = {name: [] for name in RAW_OUTPUT_NAMES}
    device_records: dict[str, object] | None = None

    for index in indices:
        prepared = backend.prepare_sample(data_root, split="mini_val", index=index)
        voxelized = backend.voxelize(prepared)
        inputs = backend.assert_shared_cuda_inputs(voxelized)
        native_raw = backend.run_native_pytorch_raw(voxelized)
        rewritten_raw = backend.run_rewritten_pytorch_raw(voxelized)
        native_devices = assert_raw_outputs_cuda0(
            native_raw, runtime_name="native_pytorch", expected_dtype="torch.float32"
        )
        rewritten_devices = assert_raw_outputs_cuda0(
            rewritten_raw, runtime_name="rewritten_pytorch", expected_dtype="torch.float32"
        )
        if device_records is None:
            device_records = {
                "inputs": inputs,
                "native_outputs": native_devices,
                "rewritten_outputs": rewritten_devices,
            }

        tensor_records: dict[str, object] = {}
        for name in RAW_OUTPUT_NAMES:
            native_array = _raw_array(native_raw, name)
            rewritten_array = _raw_array(rewritten_raw, name)
            comparison, difference = raw_tensor_difference_statistics(native_array, rewritten_array)
            tensor_records[name] = comparison
            differences[name].append(difference.astype(np.float32, copy=False))
        raw_records.append(
            {
                "sample_index": index,
                "sample_id": prepared.sample_id,
                "tensors": tensor_records,
            }
        )

        native_frame = backend.postprocess_raw(
            native_raw,
            voxelized,
            backend_name="native_mmdetection3d_pytorch",
            precision="fp32",
        )
        rewritten_frame = backend.postprocess_raw(
            rewritten_raw,
            voxelized,
            backend_name="mmdeploy_rewritten_pytorch",
            precision="fp32",
        )
        report = analyze_sample(
            native_frame,
            rewritten_frame,
            sample_index=index,
            exported_threshold=float(thresholds["exported_detection"]),
            high_confidence_threshold=float(thresholds["high_confidence_guard"]),
            minimum_bev_iou=float(matching["minimum_bev_iou"]),
        )
        reports.append(report)
        print(
            f"fidelity index {index}: native={len(native_frame.detections)} "
            f"rewritten={len(rewritten_frame.detections)} matches={len(report['matches'])}"
        )

    stage = parity_manifest["stage_1_acceptance"]
    matched = stage["matched_high_confidence"]
    coverage = stage["high_confidence_match_coverage"]
    counts = stage["count_guards"]
    acceptance = aggregate_acceptance_v2(
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
    raw_summary = _aggregate_raw_differences(raw_records, differences)
    raw_consistent = all(
        bool(record["shape_consistent"]) and bool(record["dtype_consistent"])
        for record in raw_summary.values()
    )
    return {
        "purpose": "export_rewrite_fidelity_diagnostic_not_parity_v2",
        "roles": {
            "reference": "native_mmdetection3d_pytorch_fp32",
            "candidate": "mmdeploy_rewritten_pytorch_fp32",
        },
        "shared_identical_voxel_tensors": True,
        "sample_indices": list(indices),
        "sample_count": len(indices),
        "device_assertions": device_records,
        "raw_outputs": raw_summary,
        "final_detections": acceptance,
        "diagnostic_yardstick": (
            "frozen parity-v2 tolerances reused without changing the parity-v2 result or protocol"
        ),
        "materially_equivalent": bool(acceptance["overall_pass"]) and raw_consistent,
        "per_sample": reports,
    }


def _component_profile(
    backend: M2Backend,
    data_root: Path,
    engine_path: Path,
    *,
    index: int,
    warmup: int,
    iterations: int,
    torch: Any,
) -> dict[str, object]:
    prepared = backend.prepare_sample(data_root, split="mini_val", index=index)
    voxelized = backend.voxelize(prepared)
    native_raw = backend.run_native_pytorch_raw(voxelized)
    prediction = backend.run_official_postprocess_raw(native_raw, voxelized)

    prepare_stats, _ = _wall_block(
        torch,
        lambda: backend.prepare_sample(data_root, split="mini_val", index=index),
        warmup=warmup,
        iterations=iterations,
        synchronize_cuda=False,
    )
    voxel_stats, _ = _wall_block(
        torch,
        lambda: backend.voxelize(prepared),
        warmup=warmup,
        iterations=iterations,
        synchronize_cuda=True,
    )
    native_stats, native_result = _cuda_block(
        torch,
        lambda: backend.run_native_pytorch_raw(voxelized),
        warmup=warmup,
        iterations=iterations,
    )
    rewritten_stats, rewritten_result = _cuda_block(
        torch,
        lambda: backend.run_rewritten_pytorch_raw(voxelized),
        warmup=warmup,
        iterations=iterations,
    )
    tensorrt_stats, tensorrt_result = _cuda_block(
        torch,
        lambda: backend.run_tensorrt_raw(voxelized, engine_path),
        warmup=warmup,
        iterations=iterations,
    )
    postprocess_stats, postprocess_result = _wall_block(
        torch,
        lambda: backend.run_official_postprocess_raw(native_raw, voxelized),
        warmup=warmup,
        iterations=iterations,
        synchronize_cuda=True,
    )
    conversion_stats, _ = _wall_block(
        torch,
        lambda: backend.convert_postprocessed_prediction(
            prediction,
            voxelized,
            backend_name="native_mmdetection3d_pytorch",
            precision="fp32",
        ),
        warmup=warmup,
        iterations=iterations,
        synchronize_cuda=True,
    )

    models = importlib.import_module("mmengine.registry").MODELS
    head_config = backend._model.cfg.model["pts_bbox_head"]
    head_build_stats, _ = _wall_block(
        torch,
        lambda: models.build(head_config),
        warmup=warmup,
        iterations=iterations,
        synchronize_cuda=False,
    )
    if native_result is None or rewritten_result is None or tensorrt_result is None:
        raise RuntimeError("component profiler produced no raw network outputs")
    if postprocess_result is None:
        raise RuntimeError("component profiler produced no postprocessed prediction")
    return {
        "diagnostic_only": True,
        "sample_index": index,
        "sample_id": prepared.sample_id,
        "warmups_per_component": warmup,
        "measurements_per_component": iterations,
        "timing": {
            "prepare_wall_ms": prepare_stats,
            "voxelize_synchronized_wall_ms": voxel_stats,
            "native_pytorch_raw_cuda_event_ms": native_stats,
            "rewritten_pytorch_raw_cuda_event_ms": rewritten_stats,
            "tensorrt_raw_cuda_event_ms": tensorrt_stats,
            "current_mmdeploy_postprocess_synchronized_wall_ms": postprocess_stats,
            "bbox_head_construction_wall_ms": head_build_stats,
            "detection_frame_conversion_synchronized_wall_ms": conversion_stats,
        },
        "timing_boundaries": {
            "prepare": "official dataset index access and multi-sweep sample pipeline",
            "voxelize": "prepared sample through official MMDetection3D data preprocessor",
            "raw_networks": "identical precomputed voxels through three raw head tensors",
            "postprocess": (
                "existing MMDeploy VoxelDetectionModel.postprocess including per-call head build"
            ),
            "conversion": "postprocessed MMDetection3D prediction through DetectionFrame",
        },
        "cached_postprocess_candidate": {
            "implemented": False,
            "measured": False,
            "equivalence_checked": False,
            "reason": "explicitly prohibited for this M2 diagnostic pass",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="nuScenes root; defaults to environment variable")
    parser.add_argument("--engine", type=Path, help="override frozen external TensorRT engine")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--m1-diagnostic", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.warmup < 0 or args.iterations <= 0:
        raise SystemExit("error: warmup must be non-negative and iterations positive")

    repository_root = _repository_root()
    m1_manifest = _manifest("m1_pointpillars_nuscenes.yaml")
    m2_manifest = _manifest("m2_pointpillars_tensorrt.yaml")
    parity_manifest = _manifest("m2_parity_v2.yaml")
    m1_assets = resolve_m1_asset_paths(m1_manifest)
    m2_assets = resolve_m2_asset_paths(m2_manifest)
    output = args.output or m2_assets.artifact_directory / "diagnostics" / "m2_diagnosis.json"
    m1_path = args.m1_diagnostic or (
        m2_assets.artifact_directory / "diagnostics" / "m1_in_m2_env.json"
    )
    engine_path = args.engine or m2_assets.engine_directory / "pointpillars_fp16.engine"
    onnx_path = m2_assets.artifact_directory / "pointpillars.onnx"

    frozen = parity_manifest["frozen_artifacts"]
    actual_hashes = {
        "checkpoint": sha256_file(m1_assets.checkpoint_path),
        "onnx": sha256_file(onnx_path),
        "engine": sha256_file(engine_path),
    }
    expected_hashes = {
        "checkpoint": str(frozen["checkpoint_sha256"]),
        "onnx": str(frozen["onnx_sha256"]),
        "engine": str(frozen["tensorrt_fp16_engine_sha256"]),
    }
    if actual_hashes != expected_hashes:
        raise SystemExit("error: frozen checkpoint/ONNX/engine hashes differ; do not proceed")

    model = m1_manifest["model"]
    deploy_relative = str(m2_manifest["deployment"]["official_deployment_config"])
    backend = M2Backend(
        m1_assets.mmdet3d_root / str(model["upstream_config"]),
        m1_assets.checkpoint_path,
        m2_assets.mmdeploy_root / deploy_relative,
        checkpoint_sha256=str(model["checkpoint"]["sha256"]),
    )
    data_root = _data_root(args.data_root)
    indices = [int(value) for value in parity_manifest["dataset"]["sample_indices"]]
    if len(indices) != 20:
        raise SystemExit("error: fidelity diagnostic requires the frozen 20 samples")

    telemetry_before = _gpu_telemetry()
    fidelity = _fidelity_diagnostic(backend, data_root, indices, parity_manifest)
    if not fidelity["materially_equivalent"]:
        raise SystemExit(
            "error: native-vs-rewritten fidelity failed; stop M2 performance benchmarking"
        )
    torch = importlib.import_module("torch")
    profile = _component_profile(
        backend,
        data_root,
        engine_path,
        index=int(m2_manifest["benchmark"]["sample_index"]),
        warmup=args.warmup,
        iterations=args.iterations,
        torch=torch,
    )
    telemetry_after = _gpu_telemetry()
    result = {
        "schema_version": "1.0",
        "status": "diagnostic_measurement_not_canonical",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit_sha": repository_git_sha(repository_root),
        "milestone": "M2",
        "publication_eligible": False,
        "canonical_benchmark_run": False,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            **dict(backend.versions),
            "mmdeploy": str(importlib.import_module("mmdeploy").__version__),
            "onnx": str(importlib.import_module("onnx").__version__),
            "tensorrt": str(importlib.import_module("tensorrt").__version__),
            "torch_cuda_runtime": str(torch.version.cuda),
            "gpu_name": str(torch.cuda.get_device_name(0)),
        },
        "artifacts": {
            "checkpoint": _file_record(m1_assets.checkpoint_path),
            "onnx": _file_record(onnx_path),
            "engine": _file_record(engine_path),
        },
        "m1_reproduction_in_m2_environment": _m1_reproduction(m1_path),
        "native_vs_rewritten_fidelity": fidelity,
        "component_profile": profile,
        "gpu_telemetry": {"before": telemetry_before, "after": telemetry_after},
        "diagnosis": {
            "rejected_commit": "e2f9b6babb541d52beaa0bcd58e841a0a56cc851",
            "primary_error": (
                "MMDeploy-rewritten eager PyTorch was incorrectly used as the performance baseline"
            ),
            "rewriter_context_scope": (
                "the rejected runner entered RewriterContext inside every timed eager call"
            ),
            "performance_baseline": "native MMDetection3D PyTorch FP32",
            "parity_reference": "MMDeploy-rewritten PyTorch FP32",
            "parity_v2_invalidated": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(profile["timing"], indent=2, sort_keys=True))
    print(f"External non-canonical diagnosis written to {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
