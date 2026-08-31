#!/usr/bin/env python3
"""Replay final M8 inputs against, without rewriting, the accepted ledger.

This tool has no detector, ground-truth, or evaluator import. It reconstructs
each frozen H10/H5 input and emits only a compact equality summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np

from laserperception.datasets.kitti_raw import KittiRawSequence
from laserperception.detection.m8_input import M8MultiSweepBuilder
from laserperception.detection.multisweep import (
    HistoricalSweep,
    MultiSweepBuilderConfig,
    SweepTransform,
)

ACCEPTED_LEDGER_SHA256 = "474e87e34c64d669750d4b6f7a64ac46fc9c5c462693fad79ff7c9547a7f1f7c"


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


def revalidate_inputs(
    *,
    full_ledger: Path,
    accepted_ledger: Path,
    date_root: Path,
    implementation_commit: str,
) -> dict[str, object]:
    """Replay all 856 inputs and return compact frozen-identity counts."""

    if len(implementation_commit) != 40 or any(
        character not in "0123456789abcdef" for character in implementation_commit
    ):
        raise ValueError("implementation_commit must be a lowercase full Git SHA")
    accepted_sha = _sha256_file(accepted_ledger)
    if accepted_sha != ACCEPTED_LEDGER_SHA256:
        raise RuntimeError("accepted M8 input ledger identity changed before replay")
    source = _load_mapping(full_ledger)
    accepted = _load_mapping(accepted_ledger)
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

    exact_by_history = {"H10": 0, "H5": 0}
    full_exact = 0
    projection_exact = 0
    intensity_exact = 0
    range_drop_exact = 0
    mismatch_ids: list[str] = []
    sequences: dict[str, KittiRawSequence] = {}

    for frame_number, untyped_frame in enumerate(frames, start=1):
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

        for history, depth in (("H10", 10), ("H5", 5)):
            condition_id = f"{frame_id}/{history}"
            frozen = accepted_by_id.get(condition_id)
            if not isinstance(frozen, Mapping):
                raise ValueError(f"accepted ledger is missing {condition_id}")
            rebuilt = M8MultiSweepBuilder(
                MultiSweepBuilderConfig(max_historical_sweeps=depth)
            ).build(current, historical)
            point_count_exact = rebuilt.points.shape[0] == frozen.get("point_count")
            full_sha = _sha256_array(rebuilt.points)
            full_identity_exact = full_sha == frozen.get("full_M8_XYZIT_sha256")
            projection_sha = _sha256_array(rebuilt.historical_projection)
            projection_identity_exact = projection_sha == frozen.get(
                "M8_XYZT_projection_sha256"
            ) and projection_sha == frozen.get("M6b_XYZT_expected_sha256")
            intensity = np.ascontiguousarray(rebuilt.points[:, 3])
            intensity_sha = _sha256_array(intensity)
            intensity_identity_exact = (
                intensity_sha == frozen.get("raw_intensity_sha256")
                and intensity_sha == frozen.get("candidate_consumed_intensity_sha256")
                and frozen.get("raw_intensity_sha256")
                == frozen.get("candidate_consumed_intensity_sha256")
            )
            candidate_range = np.asarray([-54.0, -54.0, -5.0, 54.0, 54.0, 3.0], dtype=np.float32)
            candidate_inside = np.all(
                rebuilt.points[:, :3] >= candidate_range[:3], axis=1
            ) & np.all(rebuilt.points[:, :3] < candidate_range[3:], axis=1)
            dropped = int(np.count_nonzero(~candidate_inside))
            range_identity_exact = dropped == frozen.get("candidate_range_dropped_points")
            condition_exact = all(
                (
                    point_count_exact,
                    full_identity_exact,
                    projection_identity_exact,
                    intensity_identity_exact,
                    range_identity_exact,
                    frozen.get("condition_id") == condition_id,
                )
            )
            full_exact += int(full_identity_exact)
            projection_exact += int(projection_identity_exact)
            intensity_exact += int(intensity_identity_exact)
            range_drop_exact += int(range_identity_exact)
            exact_by_history[history] += int(condition_exact)
            if not condition_exact:
                mismatch_ids.append(condition_id)
        if frame_number % 25 == 0 or frame_number == len(frames):
            print(
                f"revalidated {frame_number}/{len(frames)} frames "
                f"({frame_number * 2}/856 conditions)",
                flush=True,
            )

    conditions_checked = len(accepted_by_id)
    passed = (
        exact_by_history == {"H10": 428, "H5": 428}
        and full_exact == conditions_checked
        and projection_exact == conditions_checked
        and intensity_exact == conditions_checked
        and range_drop_exact == conditions_checked
        and not mismatch_ids
    )
    record = {
        "schema_version": "1.0",
        "status": "m8_phase1e_final_input_projection_revalidation_pass" if passed else "fail",
        "result": "PASS" if passed else "FAIL",
        "scientific_measurement": False,
        "detector_inference_performed": False,
        "ground_truth_loaded": False,
        "final_implementation_commit": implementation_commit,
        "existing_ledger": {
            "path": accepted_ledger.as_posix(),
            "bytes": accepted_ledger.stat().st_size,
            "sha256": accepted_sha,
            "rewritten": False,
        },
        "source_m6b_full_ledger": {
            "logical_name": full_ledger.name,
            "sha256": _sha256_file(full_ledger),
        },
        "conditions_checked": conditions_checked,
        "H10_exact_count": exact_by_history["H10"],
        "H5_exact_count": exact_by_history["H5"],
        "full_XYZIT_exact_count": full_exact,
        "XYZT_projection_exact_count": projection_exact,
        "intensity_exact_count": intensity_exact,
        "range_drop_exact_count": range_drop_exact,
        "mismatch_count": len(mismatch_ids),
        "mismatch_ids": mismatch_ids,
    }
    if not passed:
        raise RuntimeError(json.dumps(record, indent=2, sort_keys=True))
    return record


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-ledger", type=Path, required=True)
    parser.add_argument("--accepted-ledger", type=Path, required=True)
    parser.add_argument("--date-root", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    record = revalidate_inputs(
        full_ledger=args.full_ledger,
        accepted_ledger=args.accepted_ledger,
        date_root=args.date_root,
        implementation_commit=args.implementation_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
