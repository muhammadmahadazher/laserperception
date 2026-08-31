#!/usr/bin/env python3
"""Build the M8 input-only full-corpus identity ledger.

This tool is deliberately incapable of detector inference. It reads the
frozen M6b transform ledger, reconstructs the accepted physical point corpus,
adds raw KITTI reflectance, and verifies the projected XYZT bytes.
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


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        expected_sha = record.get("lidar2sensor_sha256")
        if _sha256_array(matrix) != expected_sha:
            raise ValueError("frozen transform SHA256 mismatch")
        result.append(
            HistoricalSweep(
                source,
                SweepTransform(matrix, source.source_id, current.source_id),
            )
        )
    return tuple(result)


def build_ledger(*, full_ledger: Path, date_root: Path) -> dict[str, object]:
    """Return a compact 856-condition M8 input identity ledger."""

    source = json.loads(full_ledger.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or not isinstance(source.get("frames"), list):
        raise ValueError("M6b full ledger has an unexpected schema")
    sequences: dict[str, KittiRawSequence] = {}
    records: list[dict[str, object]] = []
    counts = {"H10": 0, "H5": 0}
    exact_counts = {"H10": 0, "H5": 0}

    for untyped_frame in source["frames"]:
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
        ):
            raise ValueError("M6b frame identity is invalid")
        if not isinstance(transforms, list):
            raise ValueError("M6b frozen transforms are missing")
        drive_id = frame_id.split("/", maxsplit=1)[0]
        if drive_id not in sequences:
            sequences[drive_id] = KittiRawSequence(date_root, date_root / f"{drive_id}_sync")
        sequence = sequences[drive_id]
        current = sequence.frame(frame_index).to_raw_sweep()
        historical = _historical_sweeps(sequence, frame_index, transforms)

        for condition, depth in (("H10", 10), ("H5", 5)):
            condition_record = frame.get(condition.lower())
            if not isinstance(condition_record, Mapping):
                raise ValueError(f"M6b {condition} record is missing")
            config = MultiSweepBuilderConfig(max_historical_sweeps=depth)
            m8 = M8MultiSweepBuilder(config).build(current, historical)
            projection = m8.historical_projection
            expected_sha = condition_record.get("model_ready_sha256")
            projection_sha = _sha256_array(projection)
            if projection_sha != expected_sha:
                raise RuntimeError(f"{frame_id} {condition} does not reproduce frozen M6b XYZT")
            exact = True
            intensity = np.ascontiguousarray(m8.points[:, 3])
            candidate_range = np.array([-54.0, -54.0, -5.0, 54.0, 54.0, 3.0], dtype=np.float32)
            candidate_inside = np.all(m8.points[:, :3] >= candidate_range[:3], axis=1) & np.all(
                m8.points[:, :3] < candidate_range[3:], axis=1
            )
            counts[condition] += 1
            exact_counts[condition] += 1
            records.append(
                {
                    "condition_id": f"{frame_id}/{condition}",
                    "point_count": int(m8.points.shape[0]),
                    "M6b_XYZT_expected_sha256": cast(str, expected_sha),
                    "M8_XYZT_projection_sha256": projection_sha,
                    "exact_equal": exact,
                    "raw_intensity_sha256": _sha256_array(intensity),
                    "candidate_consumed_intensity_sha256": _sha256_array(intensity),
                    "intensity_sha256": _sha256_array(intensity),
                    "full_M8_XYZIT_sha256": _sha256_array(m8.points),
                    "candidate_range_dropped_points": int(np.count_nonzero(~candidate_inside)),
                }
            )

    if counts != {"H10": 428, "H5": 428} or exact_counts != counts:
        raise RuntimeError("M8 full-corpus projection gate did not reach 428/428 per condition")
    return {
        "schema_version": "1.0",
        "status": "m8_phase1_input_only_exact_pass",
        "scientific_measurement_authorized": False,
        "detector_inference_performed": False,
        "source_m6b_full_ledger": {
            "logical_name": full_ledger.name,
            "sha256": _sha256_file(full_ledger),
        },
        "feature_contract": ["x", "y", "z", "intensity", "time_lag"],
        "raw_intensity_policy": "KITTI Raw reflectance float32, unchanged",
        "candidate_intensity_transformation": "identity",
        "candidate_range": [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
        "counts": counts,
        "exact_counts": exact_counts,
        "total_conditions": len(records),
        "all_exact": True,
        "records": records,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-ledger", type=Path, required=True)
    parser.add_argument("--date-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_ledger(full_ledger=args.full_ledger, date_root=args.date_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: {result['exact_counts']} exact; no detector inference; wrote {args.output}")


if __name__ == "__main__":
    main()
