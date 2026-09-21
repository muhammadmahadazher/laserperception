"""Detector-free reconstruction of the frozen M8 S1 H10/H5 input corpus."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from laserperception.datasets.kitti_raw import KittiRawSequence
from laserperception.detection.m8_input import M8MultiSweepBuilder
from laserperception.detection.m8_s1_runtime import (
    INPUT_LEDGER_SHA256,
    canonical_frame_ids,
    sha256_file,
)
from laserperception.detection.multisweep import (
    HistoricalSweep,
    MultiSweepBuilderConfig,
    SweepTransform,
)


def _load_mapping(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _historical_sweeps(
    sequence: KittiRawSequence,
    current_index: int,
    records: Sequence[Mapping[str, object]],
) -> tuple[HistoricalSweep, ...]:
    current = sequence.frame(current_index).to_raw_sweep()
    expected = tuple(range(current_index - 1, max(-1, current_index - 11), -1))
    if len(records) != len(expected):
        raise ValueError("frozen transform count does not match available history")
    result = []
    for expected_index, record in zip(expected, records, strict=True):
        if record.get("source_index") != expected_index:
            raise ValueError("frozen transform order mismatch")
        source = sequence.frame(expected_index).to_raw_sweep()
        matrix = np.asarray(record.get("lidar2sensor"), dtype=np.float32)
        if _sha256_array(matrix) != record.get("lidar2sensor_sha256"):
            raise ValueError("frozen transform identity mismatch")
        result.append(
            HistoricalSweep(source, SweepTransform(matrix, source.source_id, current.source_id))
        )
    return tuple(result)


@dataclass(slots=True)
class FrozenInputSource:
    """Reconstruct requested inputs while verifying the accepted frozen ledger."""

    date_root: Path
    frames: Mapping[str, Mapping[str, object]]
    accepted: Mapping[str, Mapping[str, object]]
    sequences: dict[str, KittiRawSequence]

    @classmethod
    def load(
        cls, *, date_root: Path, full_ledger: Path, accepted_ledger: Path
    ) -> FrozenInputSource:
        """Load frozen identities without loading GT, tracklets, or predictions."""

        if sha256_file(accepted_ledger) != INPUT_LEDGER_SHA256:
            raise ValueError("accepted M8 input ledger identity changed")
        source_payload = _load_mapping(full_ledger)
        accepted_payload = _load_mapping(accepted_ledger)
        source_frames = source_payload.get("frames")
        accepted_records = accepted_payload.get("records")
        if not isinstance(source_frames, list) or not isinstance(accepted_records, list):
            raise ValueError("frozen M6b/M8 ledger schema changed")
        frames = {
            str(frame["frame_id"]): frame
            for frame in source_frames
            if isinstance(frame, Mapping) and isinstance(frame.get("frame_id"), str)
        }
        accepted = {
            str(record["condition_id"]): record
            for record in accepted_records
            if isinstance(record, Mapping) and isinstance(record.get("condition_id"), str)
        }
        if tuple(frames) != canonical_frame_ids() or len(accepted) != 856:
            raise ValueError("frozen M8 input order or count changed")
        return cls(date_root, frames, accepted, {})

    def pair(self, frame_id: str) -> tuple[tuple[np.ndarray, dict[str, object]], ...]:
        """Reconstruct and hash one verified H10/H5 pair."""

        frame = self.frames.get(frame_id)
        if frame is None:
            raise ValueError(f"frame is outside the frozen corpus: {frame_id}")
        raw_index = frame.get("frame_index")
        transforms = frame.get("frozen_sweep_transforms")
        if (
            isinstance(raw_index, bool)
            or not isinstance(raw_index, int)
            or not isinstance(transforms, list)
        ):
            raise ValueError("frozen source frame is malformed")
        drive_id = frame_id.split("/", 1)[0]
        if drive_id not in self.sequences:
            self.sequences[drive_id] = KittiRawSequence(
                self.date_root, self.date_root / f"{drive_id}_sync"
            )
        sequence = self.sequences[drive_id]
        current = sequence.frame(raw_index).to_raw_sweep()
        historical = _historical_sweeps(sequence, raw_index, transforms)
        result = []
        for history, depth in (("H10", 10), ("H5", 5)):
            points = (
                M8MultiSweepBuilder(MultiSweepBuilderConfig(max_historical_sweeps=depth))
                .build(current, historical)
                .points
            )
            condition_id = f"{frame_id}/{history}"
            frozen = self.accepted.get(condition_id)
            identity = _sha256_array(points)
            if frozen is None or frozen.get("full_M8_XYZIT_sha256") != identity:
                raise ValueError(f"preflight input identity changed: {condition_id}")
            if frozen.get("point_count") != points.shape[0]:
                raise ValueError(f"preflight input count changed: {condition_id}")
            result.append(
                (
                    points,
                    {
                        "condition_id": condition_id,
                        "history": history,
                        "input_point_count": int(points.shape[0]),
                        "input_sha256": identity,
                    },
                )
            )
        return tuple(result)

    def isolated(self) -> FrozenInputSource:
        """Share immutable ledgers while isolating lazy sequence state for one CPU worker."""

        return FrozenInputSource(self.date_root, self.frames, self.accepted, {})
