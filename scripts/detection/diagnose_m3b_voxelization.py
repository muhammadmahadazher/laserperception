"""Diagnose M3B hard voxelization and evaluate an in-memory experimental fast layer."""

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
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from laserperception.detection.benchmark import latency_statistics_ms
from laserperception.detection.m1_assets import resolve_m1_asset_paths
from laserperception.detection.m2_assets import resolve_m2_asset_paths
from laserperception.detection.m2_backend import M2Backend, VoxelizedM2Sample
from laserperception.detection.mmdet3d_backend import PreparedMmdet3dSample, sha256_file
from laserperception.detection.parity_v2 import (
    aggregate_acceptance_v2,
    distribution_statistics,
    raw_tensor_difference_statistics,
)
from laserperception.detection.parity_validation import analyze_sample
from laserperception.detection.runtime_metadata import repository_git_sha
from laserperception.detection.types import DetectionFrame
from laserperception.detection.voxel_fidelity import (
    compare_canonical_voxels,
    saturation_statistics,
)

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


def _raw_array(raw: Mapping[str, list[Any]], name: str) -> np.ndarray:
    values = raw[name]
    if len(values) != 1:
        raise RuntimeError(f"raw output {name} must contain exactly one tensor")
    return _numpy(values[0])


def _time_block(
    torch: Any,
    operation: Callable[[], object],
    *,
    warmups: int,
    measurements: int,
    synchronize_cuda: bool,
) -> tuple[dict[str, float | int], object]:
    result: object = None
    for _ in range(warmups):
        result = operation()
        if synchronize_cuda:
            torch.cuda.synchronize(0)
    values: list[float] = []
    for _ in range(measurements):
        if synchronize_cuda:
            torch.cuda.synchronize(0)
        started = time.perf_counter()
        result = operation()
        if synchronize_cuda:
            torch.cuda.synchronize(0)
        values.append((time.perf_counter() - started) * 1000.0)
    return latency_statistics_ms(values), result


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
        ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0 or not process.stdout.strip():
        return {"available": False, "error": process.stderr.strip() or "nvidia-smi failed"}
    values = [value.strip() for value in process.stdout.splitlines()[0].split(",")]
    return {"available": True, **dict(zip(fields, values, strict=True))}


class _VoxelizationExperiment:
    """Exact official hard path plus an in-memory deterministic=False candidate."""

    def __init__(self, backend: M2Backend, protocol: Mapping[str, Any]) -> None:
        backend.initialize()
        self.backend = backend
        self.protocol = protocol
        self.torch = backend._runtime.torch
        self.functional = importlib.import_module("torch.nn.functional")
        self.preprocessor = backend._model.data_preprocessor
        self.official_layer = self.preprocessor.voxel_layer
        voxelization = importlib.import_module("mmcv.ops").Voxelization
        self.fast_layer = voxelization(
            voxel_size=list(self.official_layer.voxel_size),
            point_cloud_range=list(self.official_layer.point_cloud_range),
            max_num_points=int(self.official_layer.max_num_points),
            max_voxels=tuple(int(value) for value in self.official_layer.max_voxels),
            deterministic=False,
        )
        self.fast_layer.eval()
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
            "experimental_deterministic": bool(self.fast_layer.deterministic),
            "model_training": bool(self.backend._model.training),
            "preprocessor_training": bool(self.preprocessor.training),
            "candidate_scope": "in_memory_experimental_not_production",
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
            "experimental_deterministic": actual["experimental_deterministic"] is False,
            "evaluation_mode": not bool(actual["model_training"])
            and not bool(actual["preprocessor_training"]),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(f"runtime voxelizer differs from frozen protocol: {failed}")

    def _collate(self, prepared: PreparedMmdet3dSample) -> tuple[list[Any], list[Any]]:
        data = self.preprocessor.collate_data(prepared.batch)
        try:
            points = list(data["inputs"]["points"])
            data_samples = list(data["data_samples"])
        except (KeyError, TypeError) as error:
            raise RuntimeError(
                "MMDetection3D collate_data returned malformed point data"
            ) from error
        if len(points) != 1 or len(data_samples) != 1:
            raise RuntimeError("M3B diagnostic requires batch size one")
        return points, data_samples

    def voxelize(self, prepared: PreparedMmdet3dSample, *, fast: bool) -> VoxelizedM2Sample:
        points, data_samples = self._collate(prepared)
        layer = self.fast_layer if fast else self.official_layer
        result = self._from_gpu_points(prepared, points, data_samples, layer=layer)
        self._validate_output(result)
        return result

    def _from_gpu_points(
        self,
        prepared: PreparedMmdet3dSample,
        points: list[Any],
        data_samples: list[Any],
        *,
        layer: Any,
    ) -> VoxelizedM2Sample:
        voxels: list[Any] = []
        coors: list[Any] = []
        num_points: list[Any] = []
        voxel_centers: list[Any] = []
        for batch_index, point_tensor in enumerate(points):
            res_voxels, res_coors, res_num_points = layer(point_tensor)
            centers = (res_coors[:, [2, 1, 0]] + 0.5) * res_voxels.new_tensor(
                layer.voxel_size
            ) + res_voxels.new_tensor(layer.point_cloud_range[0:3])
            padded = self.functional.pad(res_coors, (1, 0), mode="constant", value=batch_index)
            voxels.append(res_voxels)
            coors.append(padded)
            num_points.append(res_num_points)
            voxel_centers.append(centers)
        concatenated_voxels = self.torch.cat(voxels, dim=0).contiguous()
        concatenated_coors = self.torch.cat(coors, dim=0).contiguous()
        concatenated_counts = self.torch.cat(num_points, dim=0).contiguous()
        self.torch.cat(voxel_centers, dim=0).contiguous()
        return VoxelizedM2Sample(
            prepared=prepared,
            voxels=concatenated_voxels,
            num_points=concatenated_counts,
            coors=concatenated_coors,
            data_samples=tuple(data_samples),
        )

    def _validate_output(self, sample: VoxelizedM2Sample) -> None:
        if tuple(sample.voxels.shape[1:]) != (64, 4):
            raise RuntimeError("experimental voxel output violates the frozen (N, 64, 4) shape")
        self.backend.assert_shared_cuda_inputs(sample)

    def assert_official_replication(self, prepared: PreparedMmdet3dSample) -> None:
        official = self.backend.voxelize(prepared)
        decomposed = self.voxelize(prepared, fast=False)
        if official.hashes() != decomposed.hashes():
            raise RuntimeError("decomposed deterministic path is not exact official preprocessing")

    def decompose(
        self,
        source: Any,
        *,
        sample_id: str,
        coordinate_frame: str,
        warmups: int,
        measurements: int,
    ) -> dict[str, object]:
        for _ in range(warmups):
            self._decomposition_iteration(
                source, sample_id=sample_id, coordinate_frame=coordinate_frame
            )
        records = [
            self._decomposition_iteration(
                source, sample_id=sample_id, coordinate_frame=coordinate_frame
            )
            for _ in range(measurements)
        ]
        names = tuple(records[0])
        return {
            name: latency_statistics_ms([float(record[name]) for record in records])
            for name in names
        }

    def _decomposition_iteration(
        self, source: Any, *, sample_id: str, coordinate_frame: str
    ) -> dict[str, float]:
        self.torch.cuda.synchronize(0)
        start = time.perf_counter()
        prepared = self.backend.prepare_model_ready_points(
            source, sample_id=sample_id, coordinate_frame=coordinate_frame
        )
        cpu_batch_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        points, data_samples = self._collate(prepared)
        self.torch.cuda.synchronize(0)
        cast_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        res_voxels, res_coors, res_num_points = self.official_layer(points[0])
        self.torch.cuda.synchronize(0)
        layer_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        centers = (res_coors[:, [2, 1, 0]] + 0.5) * res_voxels.new_tensor(
            self.official_layer.voxel_size
        ) + res_voxels.new_tensor(self.official_layer.point_cloud_range[0:3])
        self.torch.cuda.synchronize(0)
        centers_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        padded = self.functional.pad(res_coors, (1, 0), mode="constant", value=0)
        self.torch.cuda.synchronize(0)
        padding_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        voxelized = VoxelizedM2Sample(
            prepared=prepared,
            voxels=self.torch.cat([res_voxels], dim=0).contiguous(),
            num_points=self.torch.cat([res_num_points], dim=0).contiguous(),
            coors=self.torch.cat([padded], dim=0).contiguous(),
            data_samples=tuple(data_samples),
        )
        self.torch.cat([centers], dim=0).contiguous()
        self.torch.cuda.synchronize(0)
        bookkeeping_ms = (time.perf_counter() - start) * 1000.0
        if voxelized.voxel_count <= 0:
            raise RuntimeError("decomposition produced no voxels")
        return {
            "cpu_model_ready_batch_construction_wall_ms": cpu_batch_ms,
            "cast_data_cpu_to_cuda_synchronized_wall_ms": cast_ms,
            "hard_voxel_layer_synchronized_wall_ms": layer_ms,
            "voxel_center_construction_synchronized_wall_ms": centers_ms,
            "coordinate_batch_padding_synchronized_wall_ms": padding_ms,
            "cat_contiguous_bookkeeping_synchronized_wall_ms": bookkeeping_ms,
            "decomposed_preprocessing_sum_synchronized_wall_ms": (
                cast_ms + layer_ms + centers_ms + padding_ms + bookkeeping_ms
            ),
        }


def _no_hash_conversion(
    backend: M2Backend,
    prediction: Any,
    voxelized: VoxelizedM2Sample,
) -> DetectionFrame:
    frame = backend.convert_prediction(prediction, voxelized.prepared)
    return DetectionFrame(
        detections=frame.detections,
        sample_id=frame.sample_id,
        coordinate_frame=frame.coordinate_frame,
        metadata={
            **frame.metadata,
            "backend": "tensorrt",
            "precision": "fp16",
            "voxel_count": voxelized.voxel_count,
            "diagnostic_provenance_hashing": "disabled_experimental_live_path",
        },
    )


def _run_direct(
    experiment: _VoxelizationExperiment,
    source: Any,
    *,
    sample_id: str,
    coordinate_frame: str,
    engine: Path,
    fast: bool,
    hashing: bool,
) -> DetectionFrame:
    backend = experiment.backend
    prepared = backend.prepare_model_ready_points(
        source, sample_id=sample_id, coordinate_frame=coordinate_frame
    )
    voxelized = experiment.voxelize(prepared, fast=True) if fast else backend.voxelize(prepared)
    raw = backend.run_tensorrt_raw(voxelized, engine)
    prediction = backend.run_official_postprocess_raw(raw, voxelized)
    if hashing:
        return backend.convert_postprocessed_prediction(
            prediction, voxelized, backend_name="tensorrt", precision="fp16"
        )
    return _no_hash_conversion(backend, prediction, voxelized)


def _performance_profile(
    experiment: _VoxelizationExperiment,
    data_root: Path,
    engine: Path,
    workloads: Mapping[str, Any],
    *,
    warmups: int,
    measurements: int,
) -> dict[str, object]:
    backend = experiment.backend
    torch = experiment.torch
    result: dict[str, object] = {}
    for name, workload in workloads.items():
        index = int(workload["sample_index"])
        dataset_prepared = backend.prepare_sample(data_root, split="mini_val", index=index)
        source = dataset_prepared.model_ready_points()
        prepared = backend.prepare_model_ready_points(
            source,
            sample_id=dataset_prepared.sample_id,
            coordinate_frame=dataset_prepared.coordinate_frame,
        )
        experiment.assert_official_replication(prepared)
        gpu_points, _ = experiment._collate(prepared)
        layer_reference, _ = _time_block(
            torch,
            partial(experiment.official_layer, gpu_points[0]),
            warmups=warmups,
            measurements=measurements,
            synchronize_cuda=True,
        )
        layer_fast, _ = _time_block(
            torch,
            partial(experiment.fast_layer, gpu_points[0]),
            warmups=warmups,
            measurements=measurements,
            synchronize_cuda=True,
        )
        preprocessing_reference, reference_voxelized = _time_block(
            torch,
            partial(backend.voxelize, prepared),
            warmups=warmups,
            measurements=measurements,
            synchronize_cuda=True,
        )
        preprocessing_fast, fast_voxelized = _time_block(
            torch,
            partial(experiment.voxelize, prepared, fast=True),
            warmups=warmups,
            measurements=measurements,
            synchronize_cuda=True,
        )
        decomposition = experiment.decompose(
            source,
            sample_id=dataset_prepared.sample_id,
            coordinate_frame=dataset_prepared.coordinate_frame,
            warmups=warmups,
            measurements=measurements,
        )
        direct_reference, reference_frame = _time_block(
            torch,
            partial(
                _run_direct,
                experiment,
                source,
                sample_id=dataset_prepared.sample_id,
                coordinate_frame=dataset_prepared.coordinate_frame,
                engine=engine,
                fast=False,
                hashing=True,
            ),
            warmups=warmups,
            measurements=measurements,
            synchronize_cuda=True,
        )
        direct_fast, _ = _time_block(
            torch,
            partial(
                _run_direct,
                experiment,
                source,
                sample_id=dataset_prepared.sample_id,
                coordinate_frame=dataset_prepared.coordinate_frame,
                engine=engine,
                fast=True,
                hashing=True,
            ),
            warmups=warmups,
            measurements=measurements,
            synchronize_cuda=True,
        )
        no_hash: dict[str, float | int] | None = None
        no_hash_semantic_exact: bool | None = None
        if name in {"W1", "W2"}:
            no_hash, _ = _time_block(
                torch,
                partial(
                    _run_direct,
                    experiment,
                    source,
                    sample_id=dataset_prepared.sample_id,
                    coordinate_frame=dataset_prepared.coordinate_frame,
                    engine=engine,
                    fast=True,
                    hashing=False,
                ),
                warmups=warmups,
                measurements=measurements,
                synchronize_cuda=True,
            )

        if not isinstance(reference_voxelized, VoxelizedM2Sample) or not isinstance(
            fast_voxelized, VoxelizedM2Sample
        ):
            raise RuntimeError("timed preprocessing did not return voxelized samples")
        if name in {"W1", "W2"}:
            semantic_raw = backend.run_tensorrt_raw(fast_voxelized, engine)
            semantic_prediction = backend.run_official_postprocess_raw(semantic_raw, fast_voxelized)
            hashed_frame = backend.convert_postprocessed_prediction(
                semantic_prediction,
                fast_voxelized,
                backend_name="tensorrt",
                precision="fp16",
            )
            no_hash_frame = _no_hash_conversion(backend, semantic_prediction, fast_voxelized)
            no_hash_semantic_exact = [item.to_dict() for item in hashed_frame.detections] == [
                item.to_dict() for item in no_hash_frame.detections
            ]
        voxel_config = experiment.protocol["voxelization"]
        workload_saturation = {
            "official_deterministic": saturation_statistics(
                source.points_xyzt,
                _numpy(reference_voxelized.num_points),
                point_cloud_range=voxel_config["point_cloud_range"],
                max_num_points=int(voxel_config["max_num_points"]),
                max_voxels=int(voxel_config["max_voxels_test"]),
            ),
            "experimental_fast": saturation_statistics(
                source.points_xyzt,
                _numpy(fast_voxelized.num_points),
                point_cloud_range=voxel_config["point_cloud_range"],
                max_num_points=int(voxel_config["max_num_points"]),
                max_voxels=int(voxel_config["max_voxels_test"]),
            ),
        }
        result[name] = {
            "sample_index": index,
            "sample_id": dataset_prepared.sample_id,
            "history": str(workload["history"]),
            "point_count": int(source.points_xyzt.shape[0]),
            "saturation": workload_saturation,
            "decomposition": decomposition,
            "official_complete_preprocessing_synchronized_wall_ms": preprocessing_reference,
            "official_hard_voxel_layer_synchronized_wall_ms": layer_reference,
            "experimental_fast_hard_voxel_layer_synchronized_wall_ms": layer_fast,
            "experimental_complete_preprocessing_synchronized_wall_ms": preprocessing_fast,
            "hard_voxel_layer_median_speedup": (
                float(layer_reference["median_ms"]) / float(layer_fast["median_ms"])
            ),
            "current_deterministic_direct_e2e_with_hashing_ms": direct_reference,
            "experimental_fast_direct_e2e_with_hashing_ms": direct_fast,
            "projected_experimental_fast_direct_e2e_without_hashing_ms": no_hash,
            "same_prediction_hash_disabled_detection_values_exact": no_hash_semantic_exact,
            "timing_protocol": {
                "warmups": warmups,
                "measurements": measurements,
                "method": "synchronized_time_perf_counter_wall_clock",
                "blocks": "isolated_reference_then_experimental",
            },
            "reference_frame_produced": isinstance(reference_frame, DetectionFrame),
        }
        print(
            f"{name}: points={source.points_xyzt.shape[0]} "
            f"layer={layer_reference['median_ms']:.3f}/{layer_fast['median_ms']:.3f} ms "
            f"e2e={direct_reference['median_ms']:.3f}/{direct_fast['median_ms']:.3f} ms"
        )
    return result


def _raw_difference(
    reference: Mapping[str, list[Any]], candidate: Mapping[str, list[Any]]
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in RAW_OUTPUT_NAMES:
        record, _ = raw_tensor_difference_statistics(
            _raw_array(reference, name), _raw_array(candidate, name)
        )
        result[name] = {
            "shape": record["shape"],
            "shape_consistent": record["shape_consistent"],
            "reference_dtype": record["pytorch_dtype"],
            "candidate_dtype": record["tensorrt_dtype"],
            "dtype_consistent": record["dtype_consistent"],
            "absolute_difference": record["absolute_difference"],
        }
    return result


def _detector_acceptance(
    reports: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, object]:
    yardstick = protocol["detector_fidelity"]["diagnostic_yardstick_reused_from_parity_v2"]
    return aggregate_acceptance_v2(
        reports,
        minimum_coverage=float(yardstick["minimum_bidirectional_high_confidence_coverage"]),
        minimum_metric_pass_fraction=float(yardstick["minimum_per_metric_pass_fraction"]),
        maximum_xy_m=float(yardstick["maximum_xy_center_displacement_m"]),
        maximum_z_m=float(yardstick["maximum_absolute_z_center_difference_m"]),
        maximum_dimension_relative_error=float(
            yardstick["maximum_relative_error_per_lwh_dimension"]
        ),
        maximum_axis_yaw_degrees=float(yardstick["maximum_axis_yaw_difference_degrees_modulo_pi"]),
        maximum_score_difference=float(yardstick["maximum_absolute_score_difference"]),
        minimum_direction_agreement=float(yardstick["minimum_heading_direction_agreement"]),
        maximum_aggregate_count_relative_difference=float(
            yardstick["aggregate_maximum_exported_count_relative_difference"]
        ),
    )


def _fidelity_profile(
    experiment: _VoxelizationExperiment,
    data_root: Path,
    engine: Path,
    indices: Sequence[int],
    protocol: Mapping[str, Any],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    backend = experiment.backend
    voxel_config = protocol["voxelization"]
    detector_config = protocol["detector_fidelity"]
    matching = detector_config["matching"]
    reports: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    for index in indices:
        prepared = backend.prepare_sample(data_root, split="mini_val", index=index)
        reference_voxels = backend.voxelize(prepared)
        candidate_voxels = experiment.voxelize(prepared, fast=True)
        reference_saturation = saturation_statistics(
            prepared.model_ready_points().points_xyzt,
            _numpy(reference_voxels.num_points),
            point_cloud_range=voxel_config["point_cloud_range"],
            max_num_points=int(voxel_config["max_num_points"]),
            max_voxels=int(voxel_config["max_voxels_test"]),
        )
        candidate_saturation = saturation_statistics(
            prepared.model_ready_points().points_xyzt,
            _numpy(candidate_voxels.num_points),
            point_cloud_range=voxel_config["point_cloud_range"],
            max_num_points=int(voxel_config["max_num_points"]),
            max_voxels=int(voxel_config["max_voxels_test"]),
        )
        voxel_comparison = compare_canonical_voxels(
            _numpy(reference_voxels.voxels),
            _numpy(reference_voxels.num_points),
            _numpy(reference_voxels.coors),
            _numpy(candidate_voxels.voxels),
            _numpy(candidate_voxels.num_points),
            _numpy(candidate_voxels.coors),
            max_num_points=int(voxel_config["max_num_points"]),
        )
        reference_raw = backend.run_tensorrt_raw(reference_voxels, engine)
        candidate_raw = backend.run_tensorrt_raw(candidate_voxels, engine)
        reference_frame = backend.postprocess_raw(
            reference_raw,
            reference_voxels,
            backend_name="tensorrt_deterministic_voxelization",
            precision="fp16",
        )
        candidate_frame = backend.postprocess_raw(
            candidate_raw,
            candidate_voxels,
            backend_name="tensorrt_experimental_fast_voxelization",
            precision="fp16",
        )
        report = analyze_sample(
            reference_frame,
            candidate_frame,
            sample_index=index,
            exported_threshold=float(detector_config["exported_detection_threshold"]),
            high_confidence_threshold=float(detector_config["high_confidence_threshold"]),
            minimum_bev_iou=float(matching["minimum_bev_iou"]),
        )
        reports.append(report)
        samples.append(
            {
                "sample_index": index,
                "sample_id": prepared.sample_id,
                "point_count": int(prepared.points_xyzt.shape[0]),
                "official_saturation": reference_saturation,
                "experimental_saturation": candidate_saturation,
                "coordinate_canonical_voxel_fidelity": voxel_comparison,
                "raw_tensorrt_output_fidelity": _raw_difference(reference_raw, candidate_raw),
                "detection_counts": report["counts"],
                "per_class_exported_counts": report["per_class_exported_counts"],
            }
        )
        print(
            f"fidelity index {index}: voxels={reference_voxels.voxel_count}/"
            f"{candidate_voxels.voxel_count} exported="
            f"{report['counts']['pytorch_exported']}/{report['counts']['tensorrt_exported']}"
        )
    return (
        {
            "roles": {
                "reference": "official_hard_voxelization_deterministic_true",
                "candidate": "experimental_hard_voxelization_deterministic_false",
                "network_and_postprocess": "identical_frozen_tensorrt_fp16_and_mmdeploy",
            },
            "sample_count": len(samples),
            "sample_indices": list(indices),
            "diagnostic_yardstick": _detector_acceptance(reports, protocol),
            "samples": samples,
        },
        reports,
    )


def _summarize_raw_runs(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in RAW_OUTPUT_NAMES:
        maxima = [float(record[name]["absolute_difference"]["maximum"]) for record in records]
        means = [float(record[name]["absolute_difference"]["mean"]) for record in records]
        result[name] = {
            "per_run_maximum_absolute_difference": distribution_statistics(maxima),
            "per_run_mean_absolute_difference": distribution_statistics(means),
        }
    return result


def _repeatability_profile(
    experiment: _VoxelizationExperiment,
    data_root: Path,
    engine: Path,
    indices: Sequence[int],
    *,
    runs: int,
    protocol: Mapping[str, Any],
) -> dict[str, object]:
    backend = experiment.backend
    detector_config = protocol["detector_fidelity"]
    matching = detector_config["matching"]
    voxel_config = protocol["voxelization"]
    result: dict[str, object] = {}
    for index in indices:
        prepared = backend.prepare_sample(data_root, split="mini_val", index=index)
        deterministic_voxels = backend.voxelize(prepared)
        deterministic_raw = backend.run_tensorrt_raw(deterministic_voxels, engine)
        deterministic_frame = backend.postprocess_raw(
            deterministic_raw,
            deterministic_voxels,
            backend_name="tensorrt_deterministic_voxelization",
            precision="fp16",
        )
        fast_runs: list[tuple[VoxelizedM2Sample, Mapping[str, list[Any]], DetectionFrame]] = []
        for _ in range(runs):
            voxelized = experiment.voxelize(prepared, fast=True)
            raw = backend.run_tensorrt_raw(voxelized, engine)
            frame = backend.postprocess_raw(
                raw,
                voxelized,
                backend_name="tensorrt_experimental_fast_voxelization",
                precision="fp16",
            )
            fast_runs.append((voxelized, raw, frame))
        first_voxels, first_raw, first_frame = fast_runs[0]
        first_comparisons: list[dict[str, object]] = []
        deterministic_comparisons: list[dict[str, object]] = []
        first_raw_records: list[dict[str, object]] = []
        deterministic_raw_records: list[dict[str, object]] = []
        first_detection_reports: list[dict[str, object]] = []
        deterministic_detection_reports: list[dict[str, object]] = []
        counts: list[int] = []
        for run_index, (voxelized, raw, frame) in enumerate(fast_runs):
            first_comparisons.append(
                compare_canonical_voxels(
                    _numpy(first_voxels.voxels),
                    _numpy(first_voxels.num_points),
                    _numpy(first_voxels.coors),
                    _numpy(voxelized.voxels),
                    _numpy(voxelized.num_points),
                    _numpy(voxelized.coors),
                    max_num_points=int(voxel_config["max_num_points"]),
                )
            )
            deterministic_comparisons.append(
                compare_canonical_voxels(
                    _numpy(deterministic_voxels.voxels),
                    _numpy(deterministic_voxels.num_points),
                    _numpy(deterministic_voxels.coors),
                    _numpy(voxelized.voxels),
                    _numpy(voxelized.num_points),
                    _numpy(voxelized.coors),
                    max_num_points=int(voxel_config["max_num_points"]),
                )
            )
            first_raw_records.append(_raw_difference(first_raw, raw))
            deterministic_raw_records.append(_raw_difference(deterministic_raw, raw))
            first_detection_reports.append(
                analyze_sample(
                    first_frame,
                    frame,
                    sample_index=index,
                    exported_threshold=float(detector_config["exported_detection_threshold"]),
                    high_confidence_threshold=float(detector_config["high_confidence_threshold"]),
                    minimum_bev_iou=float(matching["minimum_bev_iou"]),
                )
            )
            deterministic_detection_reports.append(
                analyze_sample(
                    deterministic_frame,
                    frame,
                    sample_index=index,
                    exported_threshold=float(detector_config["exported_detection_threshold"]),
                    high_confidence_threshold=float(detector_config["high_confidence_threshold"]),
                    minimum_bev_iou=float(matching["minimum_bev_iou"]),
                )
            )
            counts.append(voxelized.voxel_count)
            print(f"repeatability index {index}: run {run_index + 1}/{runs}")
        result[str(index)] = {
            "sample_index": index,
            "sample_id": prepared.sample_id,
            "runs": runs,
            "voxel_count": {
                "minimum": min(counts),
                "maximum": max(counts),
                "distinct_values": sorted(set(counts)),
            },
            "each_run_vs_first_fast_voxelization": _summarize_voxel_runs(first_comparisons),
            "each_run_vs_deterministic_voxelization": _summarize_voxel_runs(
                deterministic_comparisons
            ),
            "raw_outputs_each_run_vs_first_fast": _summarize_raw_runs(first_raw_records),
            "raw_outputs_each_run_vs_deterministic": _summarize_raw_runs(deterministic_raw_records),
            "detections_each_run_vs_first_fast": _detector_acceptance(
                first_detection_reports, protocol
            ),
            "detections_each_run_vs_deterministic": _detector_acceptance(
                deterministic_detection_reports, protocol
            ),
        }
    return result


def _summarize_voxel_runs(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    keys = (
        "reference_to_candidate_coordinate_coverage",
        "candidate_to_reference_coordinate_coverage",
        "coordinate_jaccard",
        "common_num_points_equal_fraction",
        "non_saturated_point_multisets_equal_fraction",
        "saturated_retained_subset_difference_fraction",
    )
    result = {
        key: distribution_statistics([float(record[key]) for record in records]) for key in keys
    }
    result["coordinate_order_exact_runs"] = sum(
        bool(record["coordinate_order_exact"]) for record in records
    )
    result["runs"] = len(records)
    return result


def _feasibility(timings: Mapping[str, Any]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in ("W1", "W2"):
        workload = timings[name]
        current = float(workload["current_deterministic_direct_e2e_with_hashing_ms"]["median_ms"])
        fast = float(workload["experimental_fast_direct_e2e_with_hashing_ms"]["median_ms"])
        no_hash = float(
            workload["projected_experimental_fast_direct_e2e_without_hashing_ms"]["median_ms"]
        )
        result[name] = {
            "current_deterministic_e2e_median_ms": current,
            "experimental_fast_e2e_median_ms": fast,
            "projected_experimental_fast_no_hash_e2e_median_ms": no_hash,
            "classification_with_hashing": _classify_latency(fast),
            "classification_without_hashing": _classify_latency(no_hash),
        }
    return result


def _classify_latency(value: float) -> str:
    if value <= 50.0:
        return "20_hz_direct_feasibility_demonstrated_experimentally"
    if value <= 100.0:
        return "meaningful_acceleration_but_below_20_hz_roughly_10_to_20_hz"
    return "additional_detector_or_runtime_optimization_required"


def _artifact_record(path: Path, *, logical_name: str) -> dict[str, object]:
    return {
        "logical_name": logical_name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("LASERPERCEPTION_NUSCENES_ROOT"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--measurements", type=int)
    parser.add_argument("--repeatability-runs", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.data_root:
        raise SystemExit("set LASERPERCEPTION_NUSCENES_ROOT or pass --data-root")
    protocol_path = _root() / "configs/detection/m3b_voxelization_fidelity_v1.yaml"
    protocol = dict(yaml.safe_load(protocol_path.read_text(encoding="utf-8")))
    timing_protocol = protocol["timing"]
    repeatability_protocol = protocol["repeatability"]
    warmups = int(args.warmups or timing_protocol["warmups"])
    measurements = int(args.measurements or timing_protocol["measurements"])
    repeatability_runs = int(args.repeatability_runs or repeatability_protocol["runs_per_sample"])
    if warmups != 20 or measurements != 100 or repeatability_runs < 30:
        raise SystemExit("M3B-V1 requires 20 warmups, 100 measurements, and at least 30 runs")

    m1 = _manifest("m1_pointpillars_nuscenes.yaml")
    m2 = _manifest("m2_pointpillars_tensorrt.yaml")
    m1_assets = resolve_m1_asset_paths(m1)
    m2_assets = resolve_m2_asset_paths(m2)
    engine = m2_assets.engine_directory / "pointpillars_fp16.engine"
    onnx = m2_assets.artifact_directory / "pointpillars.onnx"
    actual_hashes = {
        "checkpoint": sha256_file(m1_assets.checkpoint_path),
        "onnx": sha256_file(onnx),
        "engine": sha256_file(engine),
    }
    expected_hashes = {
        "checkpoint": EXPECTED_CHECKPOINT_SHA256,
        "onnx": EXPECTED_ONNX_SHA256,
        "engine": EXPECTED_ENGINE_SHA256,
    }
    if actual_hashes != expected_hashes:
        raise SystemExit("frozen checkpoint/ONNX/engine hash mismatch; diagnostic refused")
    backend = M2Backend(
        m1_assets.mmdet3d_root / str(m1["model"]["upstream_config"]),
        m1_assets.checkpoint_path,
        m2_assets.mmdeploy_root / str(m2["deployment"]["official_deployment_config"]),
        checkpoint_sha256=str(m1["model"]["checkpoint"]["sha256"]),
    )
    experiment = _VoxelizationExperiment(backend, protocol)
    backend._backend_model(engine)
    torch = experiment.torch
    data_root = Path(args.data_root).expanduser().resolve()
    commit_sha = repository_git_sha(_root())
    output = args.output or (
        m2_assets.artifact_directory / "m3" / f"voxelization_v1_{commit_sha[:7]}.json"
    )

    telemetry_before = _gpu_telemetry()
    timings = _performance_profile(
        experiment,
        data_root,
        engine,
        protocol["dataset"]["timing_workloads"],
        warmups=warmups,
        measurements=measurements,
    )
    indices = [int(value) for value in protocol["dataset"]["sample_indices"]]
    if len(indices) != 20:
        raise SystemExit("frozen M3B-V1 detector fidelity suite must contain 20 samples")
    fidelity, detailed_reports = _fidelity_profile(experiment, data_root, engine, indices, protocol)
    repeatability = _repeatability_profile(
        experiment,
        data_root,
        engine,
        [int(value) for value in repeatability_protocol["sample_indices"]],
        runs=repeatability_runs,
        protocol=protocol,
    )
    telemetry_after = _gpu_telemetry()
    common = {
        "schema_version": "1.0",
        "milestone": "M3B-V1",
        "status": "diagnostic_measurement_not_production",
        "publication_role": "diagnostic_evidence_not_canonical_performance",
        "measurement_commit": commit_sha,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "logical_name": "configs/detection/m3b_voxelization_fidelity_v1.yaml",
            "sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            "status": protocol["status"],
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            **dict(backend.versions),
            "mmdeploy": str(importlib.import_module("mmdeploy").__version__),
            "tensorrt": str(importlib.import_module("tensorrt").__version__),
            "torch_cuda_runtime": str(torch.version.cuda),
            "gpu_name": str(torch.cuda.get_device_name(0)),
        },
        "artifacts": {
            "checkpoint": _artifact_record(
                m1_assets.checkpoint_path, logical_name=str(m1["model"]["checkpoint"]["filename"])
            ),
            "onnx": _artifact_record(onnx, logical_name="m2/pointpillars.onnx"),
            "engine": _artifact_record(engine, logical_name="m2/engines/pointpillars_fp16.engine"),
        },
        "runtime_voxelizer_settings": experiment.settings,
        "timings": timings,
        "saturation_voxel_and_detector_fidelity": fidelity,
        "repeatability": repeatability,
        "feasibility": _feasibility(timings),
        "gpu_telemetry": {"before": telemetry_before, "after": telemetry_after},
        "scope_guards": {
            "production_fast_voxelizer_adopted": False,
            "official_config_edited": False,
            "model_changed": False,
            "engine_rebuilt": False,
            "onnx_exported": False,
            "postprocess_optimized": False,
            "ros_or_dds_optimized": False,
            "m4_started": False,
        },
    }
    full = {**common, "detailed_detector_matching": detailed_reports}
    _write_json(output, full)
    if args.summary_output:
        _write_json(args.summary_output, common)
    print(json.dumps({"output": str(output), "feasibility": common["feasibility"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
