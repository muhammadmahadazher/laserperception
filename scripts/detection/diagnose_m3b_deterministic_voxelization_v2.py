"""Gate an experimental exact deterministic hard-voxelization prototype."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import platform
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from laserperception.detection.benchmark import latency_statistics_ms
from laserperception.detection.exact_voxelization import ExactDeterministicVoxelizer
from laserperception.detection.m1_assets import resolve_m1_asset_paths
from laserperception.detection.m2_assets import resolve_m2_asset_paths
from laserperception.detection.m2_backend import (
    M2Backend,
    ProvenanceMode,
    VoxelizedM2Sample,
)
from laserperception.detection.measurement_telemetry import (
    NvidiaSmiSampler,
    nvidia_clock_capability,
    paired_gpu_state_eligibility,
    summarize_gpu_telemetry,
    summarize_telemetry_by_block,
)
from laserperception.detection.mmdet3d_backend import PreparedMmdet3dSample, sha256_file
from laserperception.detection.parity_v2 import (
    aggregate_acceptance_v2,
    raw_tensor_difference_statistics,
)
from laserperception.detection.parity_validation import analyze_sample
from laserperception.detection.runtime_metadata import repository_git_sha
from laserperception.detection.types import DetectionFrame
from laserperception.detection.voxel_fidelity import first_exact_array_mismatch

EXPECTED_CHECKPOINT_SHA256 = "f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0"
EXPECTED_ONNX_SHA256 = "61ce22a8ca31498675c32576bfb94f0093d31dc95d2762f7254bf915a59ecc16"
EXPECTED_ENGINE_SHA256 = "a005f75852097cd9b193750560b214cc3d5237ae9b6c106c7fca3d4fc348714b"
RAW_OUTPUT_NAMES = ("cls_score", "bbox_pred", "dir_cls_pred")


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _manifest(name: str) -> dict[str, Any]:
    path = _root() / "configs/detection" / name
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def _numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().contiguous().numpy()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _array_record(array: np.ndarray) -> dict[str, object]:
    return {
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "sha256": _array_sha256(array),
    }


def _unused_slots_zero(voxels: np.ndarray, num_points: np.ndarray) -> bool:
    if voxels.ndim != 3 or num_points.shape != (voxels.shape[0],):
        return False
    unused = np.arange(voxels.shape[1])[None, :] >= num_points[:, None]
    return bool(np.all(voxels[unused] == 0))


def _raw_arrays(raw: Mapping[str, list[Any]]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name in RAW_OUTPUT_NAMES:
        values = raw.get(name)
        if not isinstance(values, list) or len(values) != 1:
            raise RuntimeError(f"raw output {name} must contain exactly one tensor")
        result[name] = _numpy(values[0])
    return result


def _raw_hashes(raw: Mapping[str, list[Any]]) -> dict[str, str]:
    return {name: _array_sha256(value) for name, value in _raw_arrays(raw).items()}


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _artifact_record(path: Path, *, logical_name: str) -> dict[str, object]:
    return {
        "logical_name": logical_name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


class ExactVoxelizationExperiment:
    """Official deterministic reference plus experimental exact tensor candidate."""

    def __init__(self, backend: M2Backend, protocol: Mapping[str, Any]) -> None:
        backend.initialize()
        self.backend = backend
        self.protocol = protocol
        self.torch = backend._runtime.torch
        self.functional = importlib.import_module("torch.nn.functional")
        self.preprocessor = backend._model.data_preprocessor
        self.official_layer = self.preprocessor.voxel_layer
        self.candidate_layer = ExactDeterministicVoxelizer(self.official_layer).eval()
        self._validate_configuration()

    @property
    def settings(self) -> dict[str, object]:
        return {
            "voxel_type": str(self.preprocessor.voxel_type),
            "voxel_size": [float(value) for value in self.official_layer.voxel_size],
            "point_cloud_range": [float(value) for value in self.official_layer.point_cloud_range],
            "max_num_points": int(self.official_layer.max_num_points),
            "max_voxels_training": int(self.official_layer.max_voxels[0]),
            "max_voxels_test": int(self.official_layer.max_voxels[1]),
            "official_deterministic": bool(self.official_layer.deterministic),
            "candidate_coordinate_operation": "pinned_mmcv_dynamic_voxel_coordinate_cuda",
            "candidate_grouping": "pytorch_composite_key_sort_and_indexing",
            "candidate_custom_cuda": False,
            "model_training": bool(self.backend._model.training),
            "preprocessor_training": bool(self.preprocessor.training),
            "candidate_scope": "experimental_not_production",
        }

    def _validate_configuration(self) -> None:
        expected = self.protocol["voxelization"]
        actual = self.settings
        checks = {
            "voxel_type": actual["voxel_type"] == str(expected["type"]),
            "voxel_size": actual["voxel_size"]
            == [float(value) for value in expected["voxel_size"]],
            "point_cloud_range": actual["point_cloud_range"]
            == [float(value) for value in expected["point_cloud_range"]],
            "max_num_points": actual["max_num_points"] == int(expected["max_num_points"]),
            "max_voxels_training": actual["max_voxels_training"]
            == int(expected["max_voxels_training"]),
            "max_voxels_test": actual["max_voxels_test"] == int(expected["max_voxels_test"]),
            "official_deterministic": actual["official_deterministic"] is True,
            "evaluation_mode": not bool(actual["model_training"])
            and not bool(actual["preprocessor_training"]),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(f"runtime voxelizer differs from frozen V2 protocol: {failed}")

    def source_provenance(self) -> dict[str, object]:
        voxelize_module = importlib.import_module("mmcv.ops.voxelize")
        wrapper = Path(str(inspect.getsourcefile(voxelize_module))).resolve()
        ops_root = wrapper.parent
        paths = {
            "python_wrapper": wrapper,
            "pytorch_dispatch": ops_root / "csrc/pytorch/voxelization.cpp",
            "cuda_launcher": ops_root / "csrc/pytorch/cuda/voxelization_cuda.cu",
            "cuda_kernels": ops_root / "csrc/common/cuda/voxelization_cuda_kernel.cuh",
        }
        result: dict[str, object] = {}
        for name, path in paths.items():
            expected = str(self.protocol["reference_source"][name]["sha256"])
            actual = sha256_file(path)
            if actual != expected:
                raise RuntimeError(f"pinned MMCV source hash mismatch for {name}")
            result[name] = {
                "logical_name": str(self.protocol["reference_source"][name]["logical_name"]),
                "sha256": actual,
            }
        return result

    def collate(self, prepared: PreparedMmdet3dSample) -> tuple[list[Any], list[Any]]:
        data = self.preprocessor.collate_data(prepared.batch)
        try:
            points = list(data["inputs"]["points"])
            data_samples = list(data["data_samples"])
        except (KeyError, TypeError) as error:
            raise RuntimeError(
                "MMDetection3D collate_data returned malformed point data"
            ) from error
        if len(points) != 1 or len(data_samples) != 1:
            raise RuntimeError("M3B-V2 requires batch size one")
        return points, data_samples

    def voxelize(
        self,
        prepared: PreparedMmdet3dSample,
        *,
        validate_cuda: bool = True,
    ) -> VoxelizedM2Sample:
        points, data_samples = self.collate(prepared)
        return self.from_gpu_points(
            prepared,
            points[0],
            data_samples,
            validate_cuda=validate_cuda,
        )

    def from_gpu_points(
        self,
        prepared: PreparedMmdet3dSample,
        points: Any,
        data_samples: list[Any],
        *,
        validate_cuda: bool = True,
    ) -> VoxelizedM2Sample:
        voxels, coors, num_points = self.candidate_layer(points)
        centers = (coors[:, [2, 1, 0]] + 0.5) * voxels.new_tensor(
            self.candidate_layer.voxel_size
        ) + voxels.new_tensor(self.candidate_layer.point_cloud_range[0:3])
        padded = self.functional.pad(coors, (1, 0), mode="constant", value=0)
        result = VoxelizedM2Sample(
            prepared=prepared,
            voxels=self.torch.cat([voxels], dim=0).contiguous(),
            num_points=self.torch.cat([num_points], dim=0).contiguous(),
            coors=self.torch.cat([padded], dim=0).contiguous(),
            data_samples=tuple(data_samples),
        )
        self.torch.cat([centers], dim=0).contiguous()
        if tuple(result.voxels.shape[1:]) != (64, 4):
            raise RuntimeError("candidate voxels violate the frozen (N, 64, 4) shape")
        if validate_cuda:
            self.backend.assert_shared_cuda_inputs(result)
        return result


def _exact_voxel_comparison(
    reference: VoxelizedM2Sample,
    candidate: VoxelizedM2Sample,
) -> dict[str, object]:
    tensor_records: dict[str, object] = {}
    first_mismatch: dict[str, object] | None = None
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in ("voxels", "coors", "num_points"):
        reference_array = _numpy(getattr(reference, name))
        candidate_array = _numpy(getattr(candidate, name))
        arrays[name] = (reference_array, candidate_array)
        mismatch = first_exact_array_mismatch(reference_array, candidate_array, name=name)
        tensor_records[name] = {
            "reference": _array_record(reference_array),
            "candidate": _array_record(candidate_array),
            "exact": mismatch is None,
        }
        if first_mismatch is None and mismatch is not None:
            first_mismatch = mismatch
    reference_zero = _unused_slots_zero(arrays["voxels"][0], arrays["num_points"][0])
    candidate_zero = _unused_slots_zero(arrays["voxels"][1], arrays["num_points"][1])
    return {
        "exact": first_mismatch is None and reference_zero and candidate_zero,
        "voxel_count_exact": reference.voxel_count == candidate.voxel_count,
        "zero_filled_unused_slots": {
            "reference": reference_zero,
            "candidate": candidate_zero,
        },
        "tensors": tensor_records,
        "first_mismatch": first_mismatch,
    }


def _exact_fidelity_gate(
    experiment: ExactVoxelizationExperiment,
    data_root: Path,
    indices: Sequence[int],
) -> dict[str, object]:
    backend = experiment.backend
    expected_size = int(experiment.protocol["dataset"]["expected_sample_count"])
    actual_size = backend.dataset_size(data_root, "mini_val")
    if actual_size != expected_size or list(indices) != list(range(expected_size)):
        raise RuntimeError("V2 exact gate requires the complete ordered 81-sample mini_val split")
    samples: list[dict[str, object]] = []
    first_mismatch: dict[str, object] | None = None
    for index in indices:
        prepared = backend.prepare_sample(data_root, split="mini_val", index=index)
        reference = backend.voxelize(prepared)
        candidate = experiment.voxelize(prepared)
        comparison = _exact_voxel_comparison(reference, candidate)
        sample_record = {
            "sample_index": index,
            "sample_id": prepared.sample_id,
            "point_count": int(prepared.points_xyzt.shape[0]),
            "voxel_count": reference.voxel_count,
            "comparison": comparison,
        }
        samples.append(sample_record)
        print(
            f"exact index {index:02d}: points={prepared.points_xyzt.shape[0]} "
            f"voxels={reference.voxel_count} exact={comparison['exact']}",
            flush=True,
        )
        if not bool(comparison["exact"]):
            first_mismatch = {
                "sample_index": index,
                "sample_id": prepared.sample_id,
                **dict(comparison["first_mismatch"] or {}),
            }
            break
    return {
        "required_sample_count": expected_size,
        "completed_sample_count": len(samples),
        "passed": first_mismatch is None and len(samples) == expected_size,
        "first_mismatch": first_mismatch,
        "samples": samples,
    }


def _repeatability_gate(
    experiment: ExactVoxelizationExperiment,
    data_root: Path,
    engine: Path,
    indices: Sequence[int],
    *,
    runs: int,
) -> dict[str, object]:
    if runs < 30:
        raise ValueError("V2 repeatability requires at least 30 runs")
    backend = experiment.backend
    result: dict[str, object] = {}
    overall_pass = True
    for index in indices:
        prepared = backend.prepare_sample(data_root, split="mini_val", index=index)
        reference = backend.voxelize(prepared)
        reference_hashes = reference.hashes()
        input_hashes: list[dict[str, str]] = []
        raw_hashes: list[dict[str, str]] = []
        first_input_mismatch: dict[str, object] | None = None
        for run in range(runs):
            candidate = experiment.voxelize(prepared)
            candidate_hashes = candidate.hashes()
            input_hashes.append(candidate_hashes)
            if first_input_mismatch is None and candidate_hashes != reference_hashes:
                comparison = _exact_voxel_comparison(reference, candidate)
                first_input_mismatch = {
                    "run": run,
                    **dict(comparison["first_mismatch"] or {}),
                }
            raw_hashes.append(_raw_hashes(backend.run_tensorrt_raw(candidate, engine)))
        input_exact = first_input_mismatch is None and all(
            value == reference_hashes for value in input_hashes
        )
        raw_exact = all(value == raw_hashes[0] for value in raw_hashes)
        overall_pass = overall_pass and input_exact
        result[str(index)] = {
            "sample_id": prepared.sample_id,
            "runs": runs,
            "reference_input_hashes": reference_hashes,
            "candidate_input_hashes": input_hashes,
            "candidate_inputs_exact": input_exact,
            "first_input_mismatch": first_input_mismatch,
            "raw_tensorrt_output_hashes": raw_hashes,
            "raw_tensorrt_outputs_repeatable": raw_exact,
        }
        print(
            f"repeatability index {index}: inputs_exact={input_exact} raw_exact={raw_exact}",
            flush=True,
        )
        if not input_exact:
            break
    return {"passed": overall_pass and len(result) == len(indices), "samples": result}


def _raw_fidelity(
    reference: Mapping[str, list[Any]], candidate: Mapping[str, list[Any]]
) -> tuple[dict[str, object], bool]:
    result: dict[str, object] = {}
    exact = True
    reference_arrays = _raw_arrays(reference)
    candidate_arrays = _raw_arrays(candidate)
    for name in RAW_OUTPUT_NAMES:
        reference_array = reference_arrays[name]
        candidate_array = candidate_arrays[name]
        mismatch = first_exact_array_mismatch(reference_array, candidate_array, name=name)
        difference, _ = raw_tensor_difference_statistics(reference_array, candidate_array)
        result[name] = {
            "reference": _array_record(reference_array),
            "candidate": _array_record(candidate_array),
            "exact": mismatch is None,
            "first_mismatch": mismatch,
            "absolute_difference": difference["absolute_difference"],
        }
        exact = exact and mismatch is None
    return result, exact


def _detector_acceptance(
    reports: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, object]:
    gate = protocol["detector_fidelity"]
    return aggregate_acceptance_v2(
        reports,
        minimum_coverage=float(gate["minimum_bidirectional_high_confidence_coverage"]),
        minimum_metric_pass_fraction=float(gate["minimum_per_metric_pass_fraction"]),
        maximum_xy_m=float(gate["maximum_xy_center_displacement_m"]),
        maximum_z_m=float(gate["maximum_absolute_z_center_difference_m"]),
        maximum_dimension_relative_error=float(gate["maximum_relative_error_per_lwh_dimension"]),
        maximum_axis_yaw_degrees=float(gate["maximum_axis_yaw_difference_degrees_modulo_pi"]),
        maximum_score_difference=float(gate["maximum_absolute_score_difference"]),
        minimum_direction_agreement=float(gate["minimum_heading_direction_agreement"]),
        maximum_aggregate_count_relative_difference=float(
            gate["aggregate_maximum_exported_count_relative_difference"]
        ),
    )


def _detector_fidelity_gate(
    experiment: ExactVoxelizationExperiment,
    data_root: Path,
    engine: Path,
    indices: Sequence[int],
) -> dict[str, object]:
    backend = experiment.backend
    gate = experiment.protocol["detector_fidelity"]
    reports: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    for index in indices:
        prepared = backend.prepare_sample(data_root, split="mini_val", index=index)
        reference_voxels = backend.voxelize(prepared)
        candidate_voxels = experiment.voxelize(prepared)
        voxel_comparison = _exact_voxel_comparison(reference_voxels, candidate_voxels)
        if not bool(voxel_comparison["exact"]):
            raise RuntimeError(f"detector gate found non-exact voxel tensors at index {index}")
        reference_raw = backend.run_tensorrt_raw(reference_voxels, engine)
        candidate_raw = backend.run_tensorrt_raw(candidate_voxels, engine)
        raw_fidelity, raw_exact = _raw_fidelity(reference_raw, candidate_raw)
        reference_frame = backend.postprocess_raw(
            reference_raw,
            reference_voxels,
            backend_name="tensorrt_exact_voxel_fidelity",
            precision="fp16",
        )
        candidate_frame = backend.postprocess_raw(
            candidate_raw,
            candidate_voxels,
            backend_name="tensorrt_exact_voxel_fidelity",
            precision="fp16",
        )
        frame_exact = reference_frame.to_dict() == candidate_frame.to_dict()
        report = analyze_sample(
            reference_frame,
            candidate_frame,
            sample_index=index,
            exported_threshold=float(gate["exported_detection_threshold"]),
            high_confidence_threshold=float(gate["high_confidence_threshold"]),
            minimum_bev_iou=float(gate["matching_minimum_bev_iou"]),
        )
        reports.append(report)
        samples.append(
            {
                "sample_index": index,
                "sample_id": prepared.sample_id,
                "raw_tensorrt_outputs": raw_fidelity,
                "raw_tensorrt_outputs_exact": raw_exact,
                "final_detection_frame_exact": frame_exact,
                "counts": report["counts"],
            }
        )
        print(
            f"detector index {index}: raw_exact={raw_exact} frame_exact={frame_exact}",
            flush=True,
        )
    acceptance = _detector_acceptance(reports, experiment.protocol)
    return {
        "passed": bool(acceptance["overall_pass"]),
        "all_raw_outputs_exact": all(
            bool(sample["raw_tensorrt_outputs_exact"]) for sample in samples
        ),
        "all_final_detection_frames_exact": all(
            bool(sample["final_detection_frame_exact"]) for sample in samples
        ),
        "yardstick": acceptance,
        "samples": samples,
    }


def _time_block(
    torch: Any,
    operation: Callable[[], object],
    *,
    warmups: int,
    measurements: int,
    sampler: NvidiaSmiSampler,
    label: str,
) -> tuple[dict[str, float | int], object]:
    result: object = None
    sampler.begin_block(label)
    try:
        for _ in range(warmups):
            result = operation()
            torch.cuda.synchronize(0)
        values: list[float] = []
        for _ in range(measurements):
            torch.cuda.synchronize(0)
            started = time.perf_counter()
            result = operation()
            torch.cuda.synchronize(0)
            values.append((time.perf_counter() - started) * 1000.0)
    finally:
        sampler.end_block(label)
    return latency_statistics_ms(values), result


def _signed_statistics(values: Sequence[float]) -> dict[str, float | int]:
    samples = np.asarray(tuple(values), dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("signed timing residuals must be finite and one-dimensional")
    return {
        "count": int(samples.size),
        "mean_ms": float(np.mean(samples)),
        "median_ms": float(np.median(samples)),
        "p90_ms": float(np.percentile(samples, 90)),
        "p95_ms": float(np.percentile(samples, 95)),
        "min_ms": float(np.min(samples)),
        "max_ms": float(np.max(samples)),
        "population_std_ms": float(np.std(samples, ddof=0)),
    }


def _host_power_state() -> dict[str, object]:
    executable = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    if not executable.is_file():
        return {"available": False, "error": "Windows PowerShell bridge is unavailable"}
    command = (
        "$battery = @(Get-CimInstance -Namespace root/wmi -Class BatteryStatus "
        "-ErrorAction SilentlyContinue | Select-Object -ExpandProperty PowerOnline); "
        "$scheme = ((powercfg /getactivescheme) -join ' ').Trim(); "
        "$online = if ($battery.Count -eq 0) {$null} else {"
        "[bool]($battery | Where-Object {$_ -eq $false}).Count -eq 0}; "
        "[pscustomobject]@{ac_power=$online; active_power_scheme=$scheme} "
        "| ConvertTo-Json -Compress"
    )
    try:
        process = subprocess.run(
            [str(executable), "-NoProfile", "-Command", command],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "error": str(error)}
    if process.returncode != 0:
        return {
            "available": False,
            "error": process.stderr.strip() or "power-state query failed",
        }
    try:
        payload = json.loads(process.stdout.strip())
    except json.JSONDecodeError as error:
        return {"available": False, "error": str(error), "raw": process.stdout.strip()}
    if not isinstance(payload, dict):
        return {"available": False, "error": "power-state query returned malformed JSON"}
    return {"available": True, **payload}


def _run_direct(
    experiment: ExactVoxelizationExperiment,
    source: Any,
    *,
    sample_id: str,
    coordinate_frame: str,
    engine: Path,
    candidate: bool,
    provenance_mode: ProvenanceMode,
) -> DetectionFrame:
    backend = experiment.backend
    prepared = backend.prepare_model_ready_points(
        source,
        sample_id=sample_id,
        coordinate_frame=coordinate_frame,
    )
    voxelized = (
        experiment.voxelize(prepared, validate_cuda=False)
        if candidate
        else backend.voxelize(prepared)
    )
    raw = backend.run_tensorrt_raw(voxelized, engine)
    prediction = backend.run_official_postprocess_raw(raw, voxelized)
    return backend.convert_postprocessed_prediction(
        prediction,
        voxelized,
        backend_name="tensorrt",
        precision="fp16",
        provenance_mode=provenance_mode,
    )


def _component_iteration(
    experiment: ExactVoxelizationExperiment,
    source: Any,
    *,
    sample_id: str,
    coordinate_frame: str,
    engine: Path,
    provenance_mode: ProvenanceMode,
) -> tuple[dict[str, float], DetectionFrame]:
    backend = experiment.backend
    torch = experiment.torch
    torch.cuda.synchronize(0)
    total_started = time.perf_counter()

    started = time.perf_counter()
    prepared = backend.prepare_model_ready_points(
        source,
        sample_id=sample_id,
        coordinate_frame=coordinate_frame,
    )
    torch.cuda.synchronize(0)
    preparation_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    points, data_samples = experiment.collate(prepared)
    torch.cuda.synchronize(0)
    collation_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    voxelized = experiment.from_gpu_points(
        prepared,
        points[0],
        data_samples,
        validate_cuda=False,
    )
    torch.cuda.synchronize(0)
    voxelization_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    raw = backend.run_tensorrt_raw(voxelized, engine)
    torch.cuda.synchronize(0)
    raw_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    prediction = backend.run_official_postprocess_raw(raw, voxelized)
    torch.cuda.synchronize(0)
    postprocess_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    semantic_frame = backend.convert_prediction(prediction, voxelized.prepared)
    torch.cuda.synchronize(0)
    semantic_conversion_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    final_frame = backend.attach_runtime_metadata(
        semantic_frame,
        voxelized,
        backend_name="tensorrt",
        precision="fp16",
        provenance_mode=provenance_mode,
    )
    torch.cuda.synchronize(0)
    provenance_ms = (time.perf_counter() - started) * 1000.0

    total_ms = (time.perf_counter() - total_started) * 1000.0
    attributed = (
        preparation_ms
        + collation_ms
        + voxelization_ms
        + raw_ms
        + postprocess_ms
        + semantic_conversion_ms
        + provenance_ms
    )
    return (
        {
            "model_ready_preparation_wall_ms": preparation_ms,
            "cpu_to_cuda_collation_synchronized_wall_ms": collation_ms,
            "candidate_voxelization_synchronized_wall_ms": voxelization_ms,
            "tensorrt_raw_synchronized_wall_ms": raw_ms,
            "official_postprocess_synchronized_wall_ms": postprocess_ms,
            "detection_frame_semantic_conversion_synchronized_wall_ms": semantic_conversion_ms,
            "provenance_synchronized_wall_ms": provenance_ms,
            "unattributed_residual_wall_ms": total_ms - attributed,
            "total_e2e_synchronized_wall_ms": total_ms,
        },
        final_frame,
    )


def _component_ledger(
    experiment: ExactVoxelizationExperiment,
    source: Any,
    *,
    sample_id: str,
    coordinate_frame: str,
    engine: Path,
    provenance_mode: ProvenanceMode,
    warmups: int,
    measurements: int,
    sampler: NvidiaSmiSampler,
    label: str,
) -> tuple[dict[str, object], DetectionFrame]:
    frame: DetectionFrame | None = None
    sampler.begin_block(label)
    try:
        for _ in range(warmups):
            _, frame = _component_iteration(
                experiment,
                source,
                sample_id=sample_id,
                coordinate_frame=coordinate_frame,
                engine=engine,
                provenance_mode=provenance_mode,
            )
        records: list[dict[str, float]] = []
        for _ in range(measurements):
            record, frame = _component_iteration(
                experiment,
                source,
                sample_id=sample_id,
                coordinate_frame=coordinate_frame,
                engine=engine,
                provenance_mode=provenance_mode,
            )
            records.append(record)
    finally:
        sampler.end_block(label)
    if frame is None or not records:
        raise RuntimeError("component ledger produced no measured DetectionFrame")
    result: dict[str, object] = {}
    for name in records[0]:
        values = [record[name] for record in records]
        result[name] = (
            _signed_statistics(values)
            if name == "unattributed_residual_wall_ms"
            else latency_statistics_ms(values)
        )
    return result, frame


def _detection_values_equal(first: DetectionFrame, second: DetectionFrame) -> bool:
    return [detection.to_dict() for detection in first.detections] == [
        detection.to_dict() for detection in second.detections
    ]


def _sustained_gpu_warmup(
    experiment: ExactVoxelizationExperiment,
    data_root: Path,
    engine: Path,
    workloads: Mapping[str, Any],
    *,
    seconds: float,
    sampler: NvidiaSmiSampler,
) -> dict[str, object]:
    backend = experiment.backend
    selected = [
        backend.prepare_sample(
            data_root,
            split="mini_val",
            index=int(workloads[name]["sample_index"]),
        )
        for name in ("W1", "W2")
    ]
    label = "session.sustained_gpu_warmup"
    sampler.begin_block(label)
    started = time.perf_counter()
    iterations = 0
    try:
        while time.perf_counter() - started < seconds:
            prepared = selected[iterations % len(selected)]
            backend.run_tensorrt_raw(backend.voxelize(prepared), engine)
            backend.run_tensorrt_raw(
                experiment.voxelize(prepared, validate_cuda=False),
                engine,
            )
            experiment.torch.cuda.synchronize(0)
            iterations += 1
    finally:
        sampler.end_block(label)
    elapsed = time.perf_counter() - started
    return {
        "required_seconds": seconds,
        "actual_seconds": elapsed,
        "alternating_reference_candidate_cycles": iterations,
        "workloads": ["W1", "W2"],
        "passed": elapsed >= seconds and iterations > 0,
    }


def _performance_profile(
    experiment: ExactVoxelizationExperiment,
    data_root: Path,
    engine: Path,
    workloads: Mapping[str, Any],
    *,
    warmups: int,
    measurements: int,
    sustained_warmup_seconds: float,
    sampler: NvidiaSmiSampler,
) -> tuple[dict[str, object], list[tuple[str, str, str]]]:
    if warmups != 20 or measurements != 100 or sustained_warmup_seconds < 30.0:
        raise RuntimeError("V2 performance requires 20 warmups, 100 measurements, and 30s warmup")
    backend = experiment.backend
    torch = experiment.torch
    sustained = _sustained_gpu_warmup(
        experiment,
        data_root,
        engine,
        workloads,
        seconds=sustained_warmup_seconds,
        sampler=sampler,
    )
    timings: dict[str, object] = {}
    eligibility_pairs: list[tuple[str, str, str]] = []
    for name, workload in workloads.items():
        index = int(workload["sample_index"])
        dataset_prepared = backend.prepare_sample(data_root, split="mini_val", index=index)
        source = dataset_prepared.model_ready_points()
        prepared = backend.prepare_model_ready_points(
            source,
            sample_id=dataset_prepared.sample_id,
            coordinate_frame=dataset_prepared.coordinate_frame,
        )
        gpu_points, _ = experiment.collate(prepared)

        reference_layer_label = f"{name}.hard_voxel_layer.reference"
        candidate_layer_label = f"{name}.hard_voxel_layer.candidate"
        reference_layer, _ = _time_block(
            torch,
            partial(experiment.official_layer, gpu_points[0]),
            warmups=warmups,
            measurements=measurements,
            sampler=sampler,
            label=reference_layer_label,
        )
        candidate_layer, _ = _time_block(
            torch,
            partial(experiment.candidate_layer, gpu_points[0]),
            warmups=warmups,
            measurements=measurements,
            sampler=sampler,
            label=candidate_layer_label,
        )

        eligibility_pairs.append(
            (
                f"{name}.hard_voxel_layer",
                reference_layer_label,
                candidate_layer_label,
            )
        )

        reference_preprocessing_label = f"{name}.complete_preprocessing.reference"
        candidate_preprocessing_label = f"{name}.complete_preprocessing.candidate"
        reference_preprocessing, reference_voxels = _time_block(
            torch,
            partial(backend.voxelize, prepared),
            warmups=warmups,
            measurements=measurements,
            sampler=sampler,
            label=reference_preprocessing_label,
        )
        candidate_preprocessing, candidate_voxels = _time_block(
            torch,
            partial(experiment.voxelize, prepared, validate_cuda=False),
            warmups=warmups,
            measurements=measurements,
            sampler=sampler,
            label=candidate_preprocessing_label,
        )
        eligibility_pairs.append(
            (
                f"{name}.complete_preprocessing",
                reference_preprocessing_label,
                candidate_preprocessing_label,
            )
        )
        if not isinstance(reference_voxels, VoxelizedM2Sample) or not isinstance(
            candidate_voxels, VoxelizedM2Sample
        ):
            raise RuntimeError("timed preprocessing did not return voxelized samples")
        if not bool(_exact_voxel_comparison(reference_voxels, candidate_voxels)["exact"]):
            raise RuntimeError(f"timed V2 preprocessing lost exact equality at {name}")

        direct: dict[str, object] = {}
        direct_frames: dict[str, DetectionFrame] = {}
        for provenance_mode in ("full", "live"):
            reference_label = f"{name}.direct_{provenance_mode}.reference"
            candidate_label = f"{name}.direct_{provenance_mode}.candidate"
            reference_stats, reference_frame = _time_block(
                torch,
                partial(
                    _run_direct,
                    experiment,
                    source,
                    sample_id=dataset_prepared.sample_id,
                    coordinate_frame=dataset_prepared.coordinate_frame,
                    engine=engine,
                    candidate=False,
                    provenance_mode=provenance_mode,
                ),
                warmups=warmups,
                measurements=measurements,
                sampler=sampler,
                label=reference_label,
            )
            candidate_stats, candidate_frame = _time_block(
                torch,
                partial(
                    _run_direct,
                    experiment,
                    source,
                    sample_id=dataset_prepared.sample_id,
                    coordinate_frame=dataset_prepared.coordinate_frame,
                    engine=engine,
                    candidate=True,
                    provenance_mode=provenance_mode,
                ),
                warmups=warmups,
                measurements=measurements,
                sampler=sampler,
                label=candidate_label,
            )
            if not isinstance(reference_frame, DetectionFrame) or not isinstance(
                candidate_frame, DetectionFrame
            ):
                raise RuntimeError("timed direct path did not return DetectionFrame")
            direct_frames[f"reference_{provenance_mode}"] = reference_frame
            direct_frames[f"candidate_{provenance_mode}"] = candidate_frame
            direct[provenance_mode] = {
                "reference": reference_stats,
                "candidate": candidate_stats,
                "candidate_speedup": (
                    float(reference_stats["median_ms"]) / float(candidate_stats["median_ms"])
                ),
                "reference_candidate_detection_values_exact": _detection_values_equal(
                    reference_frame, candidate_frame
                ),
            }
            eligibility_pairs.append(
                (
                    f"{name}.direct_{provenance_mode}",
                    reference_label,
                    candidate_label,
                )
            )

        ledgers: dict[str, object] = {}
        if name in {"W1", "W2"}:
            for provenance_mode in ("full", "live"):
                ledger, ledger_frame = _component_ledger(
                    experiment,
                    source,
                    sample_id=dataset_prepared.sample_id,
                    coordinate_frame=dataset_prepared.coordinate_frame,
                    engine=engine,
                    provenance_mode=provenance_mode,
                    warmups=warmups,
                    measurements=measurements,
                    sampler=sampler,
                    label=f"{name}.component_ledger.candidate_{provenance_mode}",
                )
                ledgers[provenance_mode] = ledger
                if not _detection_values_equal(
                    ledger_frame, direct_frames[f"candidate_{provenance_mode}"]
                ):
                    raise RuntimeError(f"{name} component ledger changed detection values")

        full_candidate = float(direct["full"]["candidate"]["median_ms"])
        live_candidate = float(direct["live"]["candidate"]["median_ms"])
        timings[name] = {
            "sample_index": index,
            "sample_id": dataset_prepared.sample_id,
            "history": str(workload["history"]),
            "point_count": int(source.points_xyzt.shape[0]),
            "voxel_count": reference_voxels.voxel_count,
            "hard_voxel_layer_synchronized_wall_ms": {
                "reference": reference_layer,
                "candidate": candidate_layer,
                "candidate_speedup": (
                    float(reference_layer["median_ms"]) / float(candidate_layer["median_ms"])
                ),
            },
            "complete_preprocessing_synchronized_wall_ms": {
                "reference": reference_preprocessing,
                "candidate": candidate_preprocessing,
                "candidate_speedup": (
                    float(reference_preprocessing["median_ms"])
                    / float(candidate_preprocessing["median_ms"])
                ),
            },
            "direct_tensorrt_e2e_synchronized_wall_ms": direct,
            "candidate_component_ledger": ledgers,
            "candidate_full_live_detection_values_exact": _detection_values_equal(
                direct_frames["candidate_full"], direct_frames["candidate_live"]
            ),
            "reference_full_live_detection_values_exact": _detection_values_equal(
                direct_frames["reference_full"], direct_frames["reference_live"]
            ),
            "candidate_direct_e2e_classification": {
                "full_provenance": _classify_direct_latency(full_candidate),
                "live_provenance": _classify_direct_latency(live_candidate),
            },
            "timing_protocol": {
                "warmups": warmups,
                "measurements": measurements,
                "method": "synchronized_time_perf_counter_wall_clock",
                "blocks": "isolated_reference_and_candidate_blocks_in_one_session",
            },
        }
        print(
            f"{name}: layer={reference_layer['median_ms']:.3f}/"
            f"{candidate_layer['median_ms']:.3f} ms "
            f"candidate_e2e_full/live={full_candidate:.3f}/{live_candidate:.3f} ms",
            flush=True,
        )
    return (
        {
            "status": "diagnostic_measurement_not_production",
            "sustained_gpu_warmup": sustained,
            "workloads": timings,
        },
        eligibility_pairs,
    )


def _classify_direct_latency(value: float) -> str:
    if value <= 50.0:
        return "direct_20_hz_feasibility_demonstrated"
    if value <= 75.0:
        return "close_subsequent_measured_bottleneck_optimization_may_be_justified"
    if value <= 100.0:
        return "meaningful_acceleration_20_hz_not_demonstrated"
    return "insufficient_for_current_20_hz_goal"


def _session_eligibility(
    samples: Sequence[Mapping[str, object]],
    pairs: Sequence[tuple[str, str, str]],
    power_before: Mapping[str, object],
    power_after: Mapping[str, object],
) -> dict[str, object]:
    result = paired_gpu_state_eligibility(samples, pairs)
    rejection_reasons = list(result["rejection_reasons"])
    for label, state in (("before", power_before), ("after", power_after)):
        if state.get("available") is True and state.get("ac_power") is False:
            rejection_reasons.append(f"{label}:host_not_on_ac_power")
    before_scheme = power_before.get("active_power_scheme")
    after_scheme = power_after.get("active_power_scheme")
    if before_scheme and after_scheme and before_scheme != after_scheme:
        rejection_reasons.append("host_power_scheme_changed_during_session")
    result["power_state_before"] = dict(power_before)
    result["power_state_after"] = dict(power_after)
    result["rejection_reasons"] = rejection_reasons
    result["eligible"] = not rejection_reasons
    return result


def _base_record(
    experiment: ExactVoxelizationExperiment,
    protocol_path: Path,
    commit_sha: str,
    artifacts: Mapping[str, object],
) -> dict[str, object]:
    torch = experiment.torch
    return {
        "schema_version": 1,
        "milestone": "M3B-V2",
        "status": "diagnostic_in_progress_not_production",
        "publication_role": "diagnostic_evidence_not_canonical_performance",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "measurement_commit": commit_sha,
        "protocol": {
            "logical_name": "configs/detection/m3b_deterministic_voxelization_v2.yaml",
            "sha256": sha256_file(protocol_path),
            "status": str(experiment.protocol["status"]),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "versions": dict(experiment.backend.versions),
            "torch_cuda": str(torch.version.cuda),
            "gpu": str(torch.cuda.get_device_name(0)),
        },
        "artifacts": dict(artifacts),
        "reference_source": experiment.source_provenance(),
        "verified_reference_semantics": dict(
            experiment.protocol["reference_source"]["verified_semantics"]
        ),
        "runtime_voxelizer_settings": experiment.settings,
        "scope_guards": {
            "deterministic_false_adopted": False,
            "production_candidate_adopted": False,
            "custom_cuda_added": False,
            "model_changed": False,
            "engine_rebuilt": False,
            "onnx_exported": False,
            "postprocess_optimized": False,
            "ros_or_dds_optimized": False,
            "m4_started": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("LASERPERCEPTION_NUSCENES_ROOT"))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--phase",
        choices=("exact", "gates", "performance"),
        default="performance",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.data_root:
        raise SystemExit("set LASERPERCEPTION_NUSCENES_ROOT or pass --data-root")
    protocol_path = _root() / "configs/detection/m3b_deterministic_voxelization_v2.yaml"
    protocol = dict(yaml.safe_load(protocol_path.read_text(encoding="utf-8")))
    m1 = _manifest("m1_pointpillars_nuscenes.yaml")
    m2 = _manifest("m2_pointpillars_tensorrt.yaml")
    m1_assets = resolve_m1_asset_paths(m1)
    m2_assets = resolve_m2_asset_paths(m2)
    engine = m2_assets.engine_directory / "pointpillars_fp16.engine"
    onnx = m2_assets.artifact_directory / "pointpillars.onnx"
    artifacts = {
        "checkpoint": _artifact_record(
            m1_assets.checkpoint_path, logical_name=m1_assets.checkpoint_path.name
        ),
        "onnx": _artifact_record(onnx, logical_name="m2/pointpillars.onnx"),
        "engine": _artifact_record(engine, logical_name="m2/engines/pointpillars_fp16.engine"),
    }
    actual_hashes = {name: str(record["sha256"]) for name, record in artifacts.items()}
    expected_hashes = {
        "checkpoint": EXPECTED_CHECKPOINT_SHA256,
        "onnx": EXPECTED_ONNX_SHA256,
        "engine": EXPECTED_ENGINE_SHA256,
    }
    if actual_hashes != expected_hashes:
        raise SystemExit("frozen checkpoint/ONNX/engine hash mismatch; V2 refused")
    backend = M2Backend(
        m1_assets.mmdet3d_root / str(m1["model"]["upstream_config"]),
        m1_assets.checkpoint_path,
        m2_assets.mmdeploy_root / str(m2["deployment"]["official_deployment_config"]),
        checkpoint_sha256=str(m1["model"]["checkpoint"]["sha256"]),
    )
    experiment = ExactVoxelizationExperiment(backend, protocol)
    backend._backend_model(engine)
    data_root = Path(args.data_root).expanduser().resolve()
    commit_sha = repository_git_sha(_root())
    output = args.output or (
        m2_assets.artifact_directory / "m3" / f"deterministic_voxelization_v2_{commit_sha[:7]}.json"
    )
    record = _base_record(experiment, protocol_path, commit_sha, artifacts)
    exact = _exact_fidelity_gate(
        experiment,
        data_root,
        [int(value) for value in protocol["dataset"]["exact_fidelity_indices"]],
    )
    record["exact_voxel_fidelity"] = exact
    if not bool(exact["passed"]):
        record["status"] = "failed_exact_voxel_fidelity"
        _write_json(output, record)
        print(f"wrote failed V2 evidence to {output}")
        return 2
    if args.phase == "exact":
        record["status"] = "exact_voxel_gate_passed_diagnostic_incomplete"
        _write_json(output, record)
        print(f"wrote exact-gate V2 evidence to {output}")
        return 0
    repeatability = _repeatability_gate(
        experiment,
        data_root,
        engine,
        [int(value) for value in protocol["dataset"]["repeatability_indices"]],
        runs=int(protocol["repeatability"]["runs_per_sample"]),
    )
    record["repeatability"] = repeatability
    if not bool(repeatability["passed"]):
        record["status"] = "failed_exact_voxel_repeatability"
        _write_json(output, record)
        print(f"wrote failed V2 evidence to {output}")
        return 3
    detector = _detector_fidelity_gate(
        experiment,
        data_root,
        engine,
        [int(value) for value in protocol["dataset"]["detector_fidelity_indices"]],
    )
    record["detector_fidelity"] = detector
    if not bool(detector["passed"]):
        record["status"] = "failed_detector_fidelity"
        _write_json(output, record)
        print(f"wrote failed V2 evidence to {output}")
        return 4
    if args.phase == "gates":
        record["status"] = "correctness_gates_passed_diagnostic_not_production"
        _write_json(output, record)
        print(f"wrote V2 correctness evidence to {output}")
        return 0

    performance_protocol = protocol["performance"]
    power_before = _host_power_state()
    clock_capability = nvidia_clock_capability()
    sampler = NvidiaSmiSampler(
        interval_seconds=float(performance_protocol["telemetry_interval_seconds"])
    )
    sampler.start()
    try:
        performance, eligibility_pairs = _performance_profile(
            experiment,
            data_root,
            engine,
            protocol["dataset"]["timing_workloads"],
            warmups=int(performance_protocol["warmups"]),
            measurements=int(performance_protocol["measurements"]),
            sustained_warmup_seconds=float(performance_protocol["sustained_gpu_warmup_seconds"]),
            sampler=sampler,
        )
    finally:
        sampler.stop()
    power_after = _host_power_state()
    telemetry_samples = list(sampler.samples)
    eligibility = _session_eligibility(
        telemetry_samples,
        eligibility_pairs,
        power_before,
        power_after,
    )
    performance["correctness_prerequisites"] = {
        "exact_voxel_fidelity_passed": bool(exact["passed"]),
        "repeatability_passed": bool(repeatability["passed"]),
        "detector_fidelity_passed": bool(detector["passed"]),
        "measurement_commit": commit_sha,
    }
    performance["measurement_session"] = {
        "clock_capability": clock_capability,
        "telemetry_interval_seconds": float(performance_protocol["telemetry_interval_seconds"]),
        "telemetry": {
            "summary": summarize_gpu_telemetry(telemetry_samples),
            "by_measured_block": summarize_telemetry_by_block(telemetry_samples),
            "raw_samples": telemetry_samples,
        },
        "eligibility": eligibility,
    }
    record["performance"] = performance
    if not bool(eligibility["eligible"]):
        record["status"] = "rejected_performance_session_gpu_state_mismatch"
        _write_json(output, record)
        print(f"wrote rejected V2 performance session to {output}")
        return 5
    record["status"] = "diagnostic_measurement_not_production"
    _write_json(output, record)
    print(f"wrote V2 diagnostic measurement to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
