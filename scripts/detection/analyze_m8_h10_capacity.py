#!/usr/bin/env python3
"""Census DSVT candidate pillars for the frozen M8 H10 input corpus.

This input-only tool has no detector/backend import and cannot observe ground
truth or predictions. It reconstructs the accepted five-feature inputs, first
requires their frozen ledger identities, then counts occupied candidate XY
pillars using the selected DynPillarVFE coordinate semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np

from laserperception.datasets.kitti_raw import KittiRawSequence
from laserperception.detection.m8_capacity import (
    candidate_dynamic_pillar_count_cuda,
    load_dsvt_capacity_contract,
    require_capacity,
)
from laserperception.detection.m8_input import M8MultiSweepBuilder
from laserperception.detection.multisweep import (
    HistoricalSweep,
    MultiSweepBuilderConfig,
    SweepTransform,
)

INITIAL_SOURCE_BOUNDARY_PILLARS = 3_687
HISTORICAL_M6_MAX_VOXELS_CONTEXT = 43_810


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def build_capacity_census(
    *,
    full_ledger: Path,
    accepted_ledger: Path,
    date_root: Path,
    manifest_path: Path,
    coordinate_device: str,
) -> dict[str, object]:
    """Return the deterministic 428-condition H10 candidate-pillar census."""

    source = _load_mapping(full_ledger)
    accepted = _load_mapping(accepted_ledger)
    manifest = _load_mapping(manifest_path)
    frames = source.get("frames")
    accepted_records = accepted.get("records")
    if not isinstance(frames, list) or not isinstance(accepted_records, list):
        raise ValueError("frozen M6b/M8 ledgers have unexpected schemas")
    accepted_by_id = {
        record["condition_id"]: record
        for record in accepted_records
        if isinstance(record, dict) and isinstance(record.get("condition_id"), str)
    }
    if len(accepted_by_id) != 856:
        raise ValueError("accepted M8 ledger must contain 856 unique conditions")

    contract = load_dsvt_capacity_contract(manifest)
    torch = importlib.import_module("torch")
    if coordinate_device != "cuda:0" or not torch.cuda.is_available():
        raise RuntimeError("canonical M8 census requires selected CUDA device 0 arithmetic")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping) or torch.__version__ != runtime.get("torch"):
        raise RuntimeError("canonical M8 census Torch identity differs from the manifest")
    sequences: dict[str, KittiRawSequence] = {}
    records: list[dict[str, object]] = []
    previous_condition = ""
    for untyped_frame in frames:
        if not isinstance(untyped_frame, Mapping):
            raise ValueError("M6b frame record must be a mapping")
        frame = cast(Mapping[str, object], untyped_frame)
        frame_id = frame.get("frame_id")
        frame_index = frame.get("frame_index")
        transforms = frame.get("frozen_sweep_transforms")
        if (
            not isinstance(frame_id, str)
            or isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or not isinstance(transforms, list)
        ):
            raise ValueError("M6b frame identity or transforms are invalid")
        drive_id = frame_id.split("/", maxsplit=1)[0]
        if drive_id not in sequences:
            sequences[drive_id] = KittiRawSequence(date_root, date_root / f"{drive_id}_sync")
        sequence = sequences[drive_id]
        current = sequence.frame(frame_index).to_raw_sweep()
        historical = _historical_sweeps(sequence, frame_index, transforms)
        m8 = M8MultiSweepBuilder(MultiSweepBuilderConfig(max_historical_sweeps=10)).build(
            current, historical
        )
        condition_id = f"{frame_id}/H10"
        accepted_record = accepted_by_id.get(condition_id)
        if not isinstance(accepted_record, Mapping):
            raise ValueError(f"accepted ledger is missing {condition_id}")
        feature_sha = _sha256_array(m8.points)
        projection_sha = _sha256_array(m8.historical_projection)
        if feature_sha != accepted_record.get("full_M8_XYZIT_sha256"):
            raise RuntimeError(f"{condition_id} full five-feature identity changed")
        if projection_sha != accepted_record.get("M8_XYZT_projection_sha256"):
            raise RuntimeError(f"{condition_id} historical projection identity changed")
        if previous_condition and condition_id <= previous_condition:
            raise RuntimeError("frozen frame order is not strictly increasing")
        previous_condition = condition_id
        pillar_count = candidate_dynamic_pillar_count_cuda(
            m8, torch_module=torch, device=coordinate_device
        )
        require_capacity(pillar_count, contract=contract)
        records.append(
            {
                "condition_id": condition_id,
                "point_count": int(m8.points.shape[0]),
                "candidate_dynamic_pillars": pillar_count,
                "candidate_cap_if_any": contract.dynamic_pillar_cap,
                "would_truncate": False,
                "candidate_feature_sha256": feature_sha,
                "historical_XYZT_projection_sha256": projection_sha,
                "above_initial_source_shape_profile": (
                    pillar_count > INITIAL_SOURCE_BOUNDARY_PILLARS
                ),
            }
        )

    if len(records) != 428:
        raise RuntimeError("H10 census did not reach all 428 frozen conditions")
    counts = [cast(int, record["candidate_dynamic_pillars"]) for record in records]
    max_record = max(records, key=lambda item: cast(int, item["candidate_dynamic_pillars"]))
    return {
        "schema_version": "1.0",
        "status": "m8_phase1e_owner_review_h10_input_capacity_census_pass",
        "scientific_measurement": False,
        "detector_inference_performed": False,
        "ground_truth_loaded": False,
        "condition": "H10 only",
        "condition_count": len(records),
        "accepted_input_ledger": {
            "logical_name": accepted_ledger.name,
            "sha256": _sha256_file(accepted_ledger),
            "rewritten": False,
        },
        "source_m6b_full_ledger": {
            "logical_name": full_ledger.name,
            "sha256": _sha256_file(full_ledger),
        },
        "candidate_contract": {
            "voxel_size": list(contract.voxel_size),
            "grid_size": list(contract.grid_size),
            "theoretical_xy_cell_count": contract.theoretical_xy_cells,
            "dynamic_pillar_count_cap": contract.dynamic_pillar_cap,
            "coordinate_mechanism": "int32 floor XY, merged x*360+y, sorted unique",
            "coordinate_runtime": {
                "device": coordinate_device,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "detector_or_model_loaded": False,
            },
        },
        "historical_m6_context": {
            "maximum_official_pointpillars_voxels": HISTORICAL_M6_MAX_VOXELS_CONTEXT,
            "same_quantity_as_candidate_dynamic_pillars": False,
        },
        "initial_source_shape_deployment_profile": {
            "pillar_count": INITIAL_SOURCE_BOUNDARY_PILLARS,
            "source_shape_only": True,
        },
        "summary": {
            "min": min(counts),
            "median": statistics.median(counts),
            "mean": statistics.fmean(counts),
            "max": max(counts),
            "max_condition_id": max_record["condition_id"],
            "conditions_affected_by_candidate_cap": 0,
            "conditions_above_initial_source_shape_profile": sum(
                cast(bool, record["above_initial_source_shape_profile"]) for record in records
            ),
        },
        "records": records,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-ledger", type=Path, required=True)
    parser.add_argument("--accepted-ledger", type=Path, required=True)
    parser.add_argument("--date-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--coordinate-device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_capacity_census(
        full_ledger=args.full_ledger,
        accepted_ledger=args.accepted_ledger,
        date_root=args.date_root,
        manifest_path=args.manifest,
        coordinate_device=args.coordinate_device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
