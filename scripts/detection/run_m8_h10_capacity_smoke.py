#!/usr/bin/env python3
"""Run the owner-authorized highest-pillar M8 H10 structural smoke.

Ground truth is never loaded. Prediction values are never transferred,
serialized, or inspected; the complete model result is immediately reduced
to its structural count and discarded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np

from laserperception.datasets.kitti_raw import KittiRawSequence
from laserperception.detection.m8_backend import DsvtBackend
from laserperception.detection.m8_capacity import (
    candidate_dynamic_pillar_coordinates,
    candidate_dynamic_pillar_coordinates_cuda,
)
from laserperception.detection.m8_input import M8MultiSweepBuilder
from laserperception.detection.multisweep import (
    HistoricalSweep,
    MultiSweepBuilderConfig,
    SweepTransform,
)


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _load_mapping(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _historical_sweeps(
    sequence: KittiRawSequence,
    current_index: int,
    records: Sequence[Mapping[str, object]],
) -> tuple[HistoricalSweep, ...]:
    current = sequence.frame(current_index).to_raw_sweep()
    expected_indices = tuple(range(current_index - 1, max(-1, current_index - 11), -1))
    if len(records) != len(expected_indices):
        raise ValueError("frozen transform count does not match available history")
    result = []
    for expected_index, record in zip(expected_indices, records, strict=True):
        if record.get("source_index") != expected_index:
            raise ValueError("frozen transform order mismatch")
        source = sequence.frame(expected_index).to_raw_sweep()
        matrix = np.asarray(record.get("lidar2sensor"), dtype=np.float32)
        if _sha256_array(matrix) != record.get("lidar2sensor_sha256"):
            raise ValueError("frozen transform SHA256 mismatch")
        result.append(
            HistoricalSweep(source, SweepTransform(matrix, source.source_id, current.source_id))
        )
    return tuple(result)


def _reconstruct_selected(*, full_ledger: Path, date_root: Path, condition_id: str) -> np.ndarray:
    source = _load_mapping(full_ledger)
    frames = source.get("frames")
    if not isinstance(frames, list):
        raise ValueError("M6b full ledger has an unexpected schema")
    for untyped_frame in frames:
        if not isinstance(untyped_frame, Mapping):
            continue
        frame = cast(Mapping[str, object], untyped_frame)
        frame_id = frame.get("frame_id")
        if f"{frame_id}/H10" != condition_id:
            continue
        frame_index = frame.get("frame_index")
        transforms = frame.get("frozen_sweep_transforms")
        if (
            not isinstance(frame_id, str)
            or isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or not isinstance(transforms, list)
        ):
            raise ValueError("selected M6b frame identity is invalid")
        drive_id = frame_id.split("/", maxsplit=1)[0]
        sequence = KittiRawSequence(date_root, date_root / f"{drive_id}_sync")
        current = sequence.frame(frame_index).to_raw_sweep()
        historical = _historical_sweeps(sequence, frame_index, transforms)
        return (
            M8MultiSweepBuilder(MultiSweepBuilderConfig(max_historical_sweeps=10))
            .build(current, historical)
            .points
        )
    raise ValueError(f"frozen full ledger is missing {condition_id}")


def _gpu_identity() -> dict[str, str]:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"telemetry": "unavailable"}
    name, driver, memory = (part.strip() for part in output.split(",", maxsplit=2))
    return {"name": name, "driver": driver, "memory_total_mib": memory}


def run_smoke(args: argparse.Namespace) -> dict[str, object]:
    """Execute the structural gate and return its compact non-semantic record."""

    census = _load_mapping(args.census)
    summary = census.get("summary")
    records = census.get("records")
    if not isinstance(summary, Mapping) or not isinstance(records, list):
        raise ValueError("capacity census has an unexpected schema")
    condition_id = summary.get("max_condition_id")
    if not isinstance(condition_id, str):
        raise ValueError("capacity census lacks its maximum condition")
    selected_record = next(
        (
            record
            for record in records
            if isinstance(record, Mapping) and record.get("condition_id") == condition_id
        ),
        None,
    )
    if not isinstance(selected_record, Mapping):
        raise ValueError("capacity census maximum record is missing")
    points = _reconstruct_selected(
        full_ledger=args.full_ledger,
        date_root=args.date_root,
        condition_id=condition_id,
    )
    feature_sha = _sha256_array(points)
    if feature_sha != selected_record.get("candidate_feature_sha256"):
        raise RuntimeError("selected H10 five-feature identity changed after input-only selection")

    cpu_analytic_xy = candidate_dynamic_pillar_coordinates(points)
    backend = DsvtBackend(
        manifest_path=args.manifest,
        upstream_root=args.upstream_root,
        checkpoint_path=args.checkpoint,
    )
    torch = backend._torch
    expected_xy = candidate_dynamic_pillar_coordinates_cuda(points, torch_module=torch)
    batch, candidate_range_dropped = backend._prepare_batch(points)
    with torch.inference_mode():
        vfe_batch = backend._model.vfe(batch)
        torch.cuda.synchronize(0)
    actual_coords = vfe_batch["voxel_coords"].detach().cpu().contiguous().numpy()
    if not np.all(actual_coords[:, :2] == 0):
        raise RuntimeError("selected DynPillarVFE emitted an unexpected batch/Z coordinate")
    actual_xy = np.ascontiguousarray(actual_coords[:, [3, 2]], dtype=np.int32)
    actual_order = np.lexsort((actual_xy[:, 1], actual_xy[:, 0]))
    actual_xy_sorted = np.ascontiguousarray(actual_xy[actual_order])
    coordinate_set_exact = np.array_equal(actual_xy_sorted, expected_xy)
    coordinate_order_exact = np.array_equal(actual_xy, expected_xy)
    if not coordinate_set_exact:
        expected_merged = expected_xy[:, 0] * 360 + expected_xy[:, 1]
        actual_merged = actual_xy_sorted[:, 0] * 360 + actual_xy_sorted[:, 1]
        missing = np.setdiff1d(expected_merged, actual_merged)
        extra = np.setdiff1d(actual_merged, expected_merged)
        raise RuntimeError(
            "CPU census occupied-coordinate set differs from selected DynPillarVFE: "
            f"CPU={expected_xy.shape[0]}, VFE={actual_xy.shape[0]}, "
            f"missing={missing[:10].tolist()}, extra={extra[:10].tolist()}"
        )
    retained_pillars = int(actual_coords.shape[0])
    candidate_pillars = int(selected_record["candidate_dynamic_pillars"])
    if retained_pillars != candidate_pillars:
        raise RuntimeError("selected DynPillarVFE changed the input-only candidate count")
    del vfe_batch, batch, actual_coords
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize(0)

    started = time.perf_counter()
    output_count, repeated_range_dropped = backend.run_structural_smoke(points)
    wall_seconds = time.perf_counter() - started
    if repeated_range_dropped != candidate_range_dropped:
        raise RuntimeError("candidate-range filtering changed between structural passes")
    peak_allocated = int(torch.cuda.max_memory_allocated(0))
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    host_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    discarded_pillars = candidate_pillars - retained_pillars
    if discarded_pillars != 0:
        raise RuntimeError("selected DSVT unexpectedly truncated candidate pillars")

    return {
        "schema_version": "1.0",
        "status": "m8_phase1e_owner_review_h10_structural_smoke_pass",
        "scientific_measurement": False,
        "accuracy_evaluation_performed": False,
        "ground_truth_loaded": False,
        "semantic_prediction_values_observed": False,
        "prediction_values_serialized": False,
        "condition_id": condition_id,
        "input_point_count": int(points.shape[0]),
        "input_full_feature_sha256": feature_sha,
        "historical_XYZT_projection_sha256": selected_record["historical_XYZT_projection_sha256"],
        "candidate_dynamic_pillars": candidate_pillars,
        "retained_pillars": retained_pillars,
        "discarded_or_truncated_pillars": discarded_pillars,
        "candidate_range_dropped_points": candidate_range_dropped,
        "capacity_semantics": {
            "configured_dynamic_pillar_cap": None,
            "theoretical_xy_cells": backend.capacity_contract.theoretical_xy_cells,
            "occupied_coordinate_set_vs_actual_dynpillar_vfe_exact": coordinate_set_exact,
            "coordinate_order_vs_cpu_sorted_order_exact": coordinate_order_exact,
            "cpu_analytic_pillar_count_noncanonical": int(cpu_analytic_xy.shape[0]),
            "cpu_vs_cuda_boundary_count_difference": int(
                cpu_analytic_xy.shape[0] - expected_xy.shape[0]
            ),
        },
        "complete_model": {
            "output_completed": True,
            "output_count_only": output_count,
            "prediction_values_discarded_immediately": True,
            "postprocess_completed": True,
        },
        "resources_engineering_context_only": {
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
            "host_peak_rss_bytes": host_rss_bytes,
            "wall_seconds": wall_seconds,
        },
        "runtime_identity": {**backend.identity, "gpu": _gpu_identity()},
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--full-ledger", type=Path, required=True)
    parser.add_argument("--date-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    record = run_smoke(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
