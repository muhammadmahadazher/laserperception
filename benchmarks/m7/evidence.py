"""Existing-M6 parsing and future deterministic M7 input-ledger serialization."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from benchmarks.m7.protocol import (
    H5_ORDERED_HASHES_SHA256,
    H10_ORDERED_HASHES_SHA256,
    ORDERED_CORPUS_SHA256,
    PROTOCOL_FREEZE_COMMIT,
    Arm,
    ProtocolViolation,
    canonical_condition_ids,
    condition_id,
)
from benchmarks.m7.provenance import atomic_write_json, canonical_json_bytes, canonical_json_sha256

PoseKey = tuple[str, int, int]


def _pose_dict(key: PoseKey) -> dict[str, object]:
    return {"drive_id": key[0], "frame_index": key[1], "gt_track_id": key[2]}


@dataclass(frozen=True, slots=True)
class PairedSets:
    """Exact H10/H5 matched-state partition for one frozen M6b class."""

    shared: tuple[PoseKey, ...]
    e_only: tuple[PoseKey, ...]
    a_only: tuple[PoseKey, ...]
    neither: tuple[PoseKey, ...]

    def __post_init__(self) -> None:
        groups = (self.shared, self.e_only, self.a_only, self.neither)
        if any(tuple(sorted(group)) != group or len(set(group)) != len(group) for group in groups):
            raise ProtocolViolation(
                "paired-set pose identities must be unique and lexicographically sorted"
            )
        flattened = [key for group in groups for key in group]
        if len(flattened) != len(set(flattened)):
            raise ProtocolViolation("paired-set partitions overlap")

    def to_dict(self) -> dict[str, object]:
        """Return canonical pose lists and per-list SHA256 identities."""

        record: dict[str, object] = {}
        for name, values in (
            ("shared", self.shared),
            ("e_only", self.e_only),
            ("a_only", self.a_only),
            ("neither", self.neither),
        ):
            poses: list[object] = [_pose_dict(value) for value in values]
            record[name] = poses
            record[f"{name}_sha256"] = canonical_json_sha256(poses)
        return record

    def cardinalities(self) -> dict[str, int]:
        """Return the four exact partition counts."""

        return {
            "shared": len(self.shared),
            "e_only": len(self.e_only),
            "a_only": len(self.a_only),
            "neither": len(self.neither),
        }


def _observation_map(frame: Mapping[str, object], class_name: str) -> dict[PoseKey, bool]:
    frame_id = frame.get("frame_id")
    if not isinstance(frame_id, str) or "/" not in frame_id:
        raise ProtocolViolation("M6b result frame identity is malformed")
    drive_id, frame_text = frame_id.split("/", 1)
    classes = frame.get("classes")
    if not isinstance(classes, Mapping):
        raise ProtocolViolation("M6b result classes record is malformed")
    class_record = classes.get(class_name)
    if not isinstance(class_record, Mapping):
        raise ProtocolViolation(f"M6b result lacks class record: {class_name}")
    observations = class_record.get("target_observations")
    if not isinstance(observations, list):
        raise ProtocolViolation("M6b target observations are malformed")
    result: dict[PoseKey, bool] = {}
    for observation in observations:
        if not isinstance(observation, Mapping):
            raise ProtocolViolation("M6b target observation is malformed")
        track_id = observation.get("track_id")
        matched = observation.get("matched")
        if (
            isinstance(track_id, bool)
            or not isinstance(track_id, int)
            or not isinstance(matched, bool)
        ):
            raise ProtocolViolation("M6b target observation identity/match state is malformed")
        key = (drive_id, int(frame_text), track_id)
        if key in result:
            raise ProtocolViolation(f"duplicate M6b target pose identity: {key}")
        result[key] = matched
    return result


def parse_m6b_paired_sets(path: str | Path, class_name: str) -> PairedSets:
    """Parse frozen A/H10 and E/H5 pose identities without evaluating any M7 result."""

    if class_name not in {"car", "pedestrian"}:
        raise ProtocolViolation("M7 paired-set parser supports only car and pedestrian")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    frames = value.get("frame_results")
    if not isinstance(frames, Mapping):
        raise ProtocolViolation("full M6b result lacks frame_results")
    h10 = frames.get("H10")
    h5 = frames.get("H5")
    if not isinstance(h10, list) or not isinstance(h5, list) or len(h10) != len(h5):
        raise ProtocolViolation("full M6b H10/H5 frame results are malformed")
    h5_by_id = {str(frame.get("frame_id")): frame for frame in h5 if isinstance(frame, Mapping)}
    partitions: dict[str, list[PoseKey]] = {
        "shared": [],
        "e_only": [],
        "a_only": [],
        "neither": [],
    }
    for a_frame in h10:
        if not isinstance(a_frame, Mapping):
            raise ProtocolViolation("full M6b H10 frame is malformed")
        frame_id = str(a_frame.get("frame_id"))
        e_frame = h5_by_id.get(frame_id)
        if not isinstance(e_frame, Mapping):
            raise ProtocolViolation(f"full M6b H5 frame is missing: {frame_id}")
        a_observations = _observation_map(a_frame, class_name)
        e_observations = _observation_map(e_frame, class_name)
        if a_observations.keys() != e_observations.keys():
            raise ProtocolViolation(f"M6b A/E GT pose sets differ at {frame_id}")
        for key, a_matched in a_observations.items():
            e_matched = e_observations[key]
            category = {
                (True, True): "shared",
                (False, True): "e_only",
                (True, False): "a_only",
                (False, False): "neither",
            }[(a_matched, e_matched)]
            partitions[category].append(key)
    return PairedSets(
        shared=tuple(sorted(partitions["shared"])),
        e_only=tuple(sorted(partitions["e_only"])),
        a_only=tuple(sorted(partitions["a_only"])),
        neither=tuple(sorted(partitions["neither"])),
    )


@dataclass(frozen=True, slots=True)
class InputLedgerIdentity:
    """Identities binding a future input-only freeze before detector authorization."""

    implementation_commit: str
    m6b_input_asset_sha256: str
    m6b_result_asset_sha256: str

    def to_dict(self) -> dict[str, object]:
        """Return the frozen ledger-level identities."""

        return {
            "protocol_commit": PROTOCOL_FREEZE_COMMIT,
            "implementation_commit": self.implementation_commit,
            "ordered_corpus_sha256": ORDERED_CORPUS_SHA256,
            "h10_ordered_hashes_sha256": H10_ORDERED_HASHES_SHA256,
            "h5_ordered_hashes_sha256": H5_ORDERED_HASHES_SHA256,
            "m6b_input_asset_sha256": self.m6b_input_asset_sha256,
            "m6b_result_asset_sha256": self.m6b_result_asset_sha256,
        }


REQUIRED_CONDITION_FIELDS = frozenset(
    {
        "condition_id",
        "drive_id",
        "frame_index",
        "arm",
        "generation_commit",
        "source_a_sha256",
        "source_e_sha256",
        "point_count",
        "xyz_sha256",
        "model_ready_sha256",
        "selected_row_sha256",
        "lag_bit_patterns",
        "lag_support_count",
        "lag_span_seconds",
        "sweep_ids",
        "per_sweep_point_counts",
        "pillar_structure",
        "lag_scale_provenance",
        "quota_provenance",
        "seed_provenance",
        "f_history_ranks",
        "runtime_versions",
    }
)


def _validate_condition_record(record: Mapping[str, object]) -> str:
    missing = REQUIRED_CONDITION_FIELDS.difference(record)
    if missing:
        raise ProtocolViolation(f"M7 input-ledger condition is missing fields: {sorted(missing)}")
    arm_text = record.get("arm")
    try:
        arm = Arm(str(arm_text))
    except ValueError as error:
        raise ProtocolViolation(f"M7 input-ledger arm is invalid: {arm_text}") from error
    if arm not in (Arm.B, Arm.C, Arm.D, Arm.F):
        raise ProtocolViolation("M7 input ledger may contain only B/C/D/F")
    drive_id = record.get("drive_id")
    frame_index = record.get("frame_index")
    if (
        not isinstance(drive_id, str)
        or isinstance(frame_index, bool)
        or not isinstance(frame_index, int)
    ):
        raise ProtocolViolation("M7 input-ledger drive/frame identity is malformed")
    expected = condition_id(f"{drive_id}/{frame_index:010d}", arm)
    if record.get("condition_id") != expected:
        raise ProtocolViolation(f"M7 input-ledger condition identity mismatch: expected {expected}")
    if arm is Arm.F and record.get("f_history_ranks") != [2, 4, 6, 8, 10]:
        raise ProtocolViolation("M7 Arm-F ledger ranks differ from 2/4/6/8/10")
    return expected


def build_input_ledger(
    identity: InputLedgerIdentity,
    conditions: Sequence[Mapping[str, object]],
    *,
    require_full_corpus: bool = True,
) -> dict[str, object]:
    """Validate and serialize a deterministic input-only ledger without detector fields."""

    records = [dict(record) for record in conditions]
    ids = [_validate_condition_record(record) for record in records]
    if len(ids) != len(set(ids)):
        raise ProtocolViolation("M7 input ledger contains duplicate condition IDs")
    expected = canonical_condition_ids()
    if require_full_corpus:
        if tuple(ids) != expected:
            raise ProtocolViolation(
                "M7 input ledger must contain all 1,712 conditions in canonical B/C/D/F order"
            )
    elif tuple(ids) != tuple(sorted(ids, key=expected.index)):
        raise ProtocolViolation("synthetic M7 input-ledger fixture is not in canonical order")
    return {
        "schema_version": "laserperception.m7.input-ledger.v1",
        "status": "INPUT-ONLY; NO DETECTOR OUTPUT",
        "identity": identity.to_dict(),
        "condition_count": len(records),
        "conditions": records,
    }


def serialize_input_ledger(ledger: Mapping[str, object]) -> bytes:
    """Return deterministic compact ledger bytes."""

    return canonical_json_bytes(dict(ledger))


def write_input_ledger(path: str | Path, ledger: Mapping[str, object]) -> None:
    """Atomically write a previously validated compact input-only ledger."""

    atomic_write_json(path, dict(ledger))
