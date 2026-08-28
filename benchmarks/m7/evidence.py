"""Existing-M6 parsing and future deterministic M7 input-ledger serialization."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType

import ijson.backends.yajl2_c as ijson_backend
from ijson.common import JSONError, ObjectBuilder

from benchmarks.m7.protocol import (
    H5_ORDERED_HASHES_SHA256,
    H10_ORDERED_HASHES_SHA256,
    M6B_INPUT_LEDGER_FULL_SHA256,
    M6B_RESULT_FULL_SHA256,
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
        "provenance_schema",
        "rank_source_identities",
        "rank_to_lag_bit_pattern",
        "pillar_structure",
        "lag_scale_provenance",
        "quota_provenance",
        "seed_provenance",
        "f_history_ranks",
        "runtime_versions",
    }
)

RUNTIME_BINDING_FIELDS = tuple(sorted(REQUIRED_CONDITION_FIELDS))
SEED_PROVENANCE_FIELDS = frozenset(
    {
        "history_rank",
        "seed_text_utf8",
        "sha256",
        "seed_uint64",
        "seed_uint64_hex",
        "selected_ordinals",
    }
)
RUNTIME_SEED_FIELDS = (
    "history_rank",
    "seed_text_utf8",
    "sha256",
    "seed_uint64",
    "seed_uint64_hex",
)
QUOTA_PROVENANCE_FIELDS = frozenset(
    {
        "source_counts_by_rank_1_to_10",
        "h_target",
        "h_total",
        "products",
        "remainders",
        "initial_quotas",
        "incremented_ranks",
        "final_quotas",
        "zero_quota_ranks",
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
    if record.get("provenance_schema") != "laserperception.m7.sweep-provenance.v2":
        raise ProtocolViolation("M7 condition provenance schema is missing or invalid")
    return expected


def _lower_hex(value: object, *, length: int, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProtocolViolation(f"M7 input-ledger {name} must be lowercase {length}-hex")
    return value


def _validate_seed_provenance(record: Mapping[str, object], arm: Arm) -> None:
    seeds = record.get("seed_provenance")
    quota = record.get("quota_provenance")
    if arm not in (Arm.C, Arm.D):
        if seeds is not None or quota is not None:
            raise ProtocolViolation("M7 non-C/D condition unexpectedly contains seed/quota data")
        return
    if not isinstance(seeds, list) or len(seeds) != 10:
        raise ProtocolViolation("M7 C/D seed provenance must contain ranks 1 through 10")
    if not isinstance(quota, Mapping) or set(quota) != QUOTA_PROVENANCE_FIELDS:
        raise ProtocolViolation("M7 C/D quota provenance schema fields differ")
    source_counts = quota.get("source_counts_by_rank_1_to_10")
    final_quotas = quota.get("final_quotas")
    if (
        not isinstance(source_counts, list)
        or not isinstance(final_quotas, list)
        or len(source_counts) != 10
        or len(final_quotas) != 10
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in source_counts
        )
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in final_quotas
        )
    ):
        raise ProtocolViolation("M7 C/D quota counts are malformed")
    drive_id = record.get("drive_id")
    frame_index = record.get("frame_index")
    assert isinstance(drive_id, str)
    assert isinstance(frame_index, int) and not isinstance(frame_index, bool)
    for expected_rank, seed in enumerate(seeds, start=1):
        if not isinstance(seed, Mapping) or set(seed) != SEED_PROVENANCE_FIELDS:
            raise ProtocolViolation("M7 C/D seed provenance schema fields differ")
        text = f"laserperception-m7-c-v1|{drive_id}|{frame_index:010d}|{expected_rank}"
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        uint64 = int.from_bytes(digest[:8], byteorder="big", signed=False)
        if (
            seed.get("history_rank") != expected_rank
            or seed.get("seed_text_utf8") != text
            or seed.get("sha256") != digest.hex()
            or seed.get("seed_uint64") != uint64
            or seed.get("seed_uint64_hex") != f"0x{uint64:016x}"
        ):
            raise ProtocolViolation("M7 C/D seed identity differs from the frozen algorithm")
        ordinals = seed.get("selected_ordinals")
        quota_count = final_quotas[expected_rank - 1]
        source_count = source_counts[expected_rank - 1]
        if not isinstance(ordinals, list) or len(ordinals) != quota_count:
            raise ProtocolViolation("M7 C/D selected ordinal count differs from the frozen quota")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or value >= source_count
            for value in ordinals
        ):
            raise ProtocolViolation("M7 C/D selected ordinal is outside its source sweep")


def _validate_complete_condition_record(
    raw_record: Mapping[str, object], *, expected_implementation_commit: str
) -> str:
    if set(raw_record) != REQUIRED_CONDITION_FIELDS:
        raise ProtocolViolation("M7 input-ledger condition schema fields differ")
    condition = _validate_condition_record(raw_record)
    if raw_record.get("generation_commit") != expected_implementation_commit:
        raise ProtocolViolation("M7 input-ledger generation commit differs")
    for name in (
        "source_a_sha256",
        "source_e_sha256",
        "xyz_sha256",
        "model_ready_sha256",
        "selected_row_sha256",
    ):
        _lower_hex(raw_record.get(name), length=64, name=name)
    point_count = raw_record.get("point_count")
    if isinstance(point_count, bool) or not isinstance(point_count, int) or point_count <= 0:
        raise ProtocolViolation("M7 input-ledger point count must be positive")
    rank_sources = raw_record.get("rank_source_identities")
    rank_lags = raw_record.get("rank_to_lag_bit_pattern")
    sweep_ids = raw_record.get("sweep_ids")
    counts = raw_record.get("per_sweep_point_counts")
    if (
        not isinstance(rank_sources, list)
        or not isinstance(rank_lags, Mapping)
        or not isinstance(sweep_ids, list)
        or not isinstance(counts, Mapping)
    ):
        raise ProtocolViolation("M7 input-ledger provenance mapping is malformed")
    expected_source_fields = {
        "history_rank",
        "source_sweep_id",
        "source_index",
        "timestamp_text",
        "timestamp_nanoseconds",
        "timestamp_microseconds",
        "lag_float32_bits",
    }
    if any(
        not isinstance(value, Mapping) or set(value) != expected_source_fields
        for value in rank_sources
    ):
        raise ProtocolViolation("M7 input-ledger rank-source schema fields differ")
    ranks = [value.get("history_rank") for value in rank_sources]
    if any(
        isinstance(rank, bool) or not isinstance(rank, int) or rank not in range(11)
        for rank in ranks
    ) or ranks != sorted(set(ranks)):
        raise ProtocolViolation("M7 input-ledger rank-source identities are malformed")
    if [value.get("source_sweep_id") for value in rank_sources] != sweep_ids:
        raise ProtocolViolation("M7 input-ledger sweep IDs differ from rank-source identities")
    if set(counts) != {str(rank) for rank in ranks}:
        raise ProtocolViolation("M7 input-ledger sweep counts differ from rank support")
    if set(rank_lags) != set(counts):
        raise ProtocolViolation("M7 input-ledger rank-to-lag mapping differs from rank support")
    if (
        any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for count in counts.values()
        )
        or sum(int(count) for count in counts.values()) != point_count
    ):
        raise ProtocolViolation("M7 input-ledger per-sweep counts differ from point count")
    lag_support = raw_record.get("lag_bit_patterns")
    if not isinstance(lag_support, list) or set(lag_support) != set(rank_lags.values()):
        raise ProtocolViolation("M7 input-ledger lag support differs from rank-to-lag mapping")
    for source in rank_sources:
        assert isinstance(source, Mapping)
        rank = source.get("history_rank")
        source_index = source.get("source_index")
        nanoseconds = source.get("timestamp_nanoseconds")
        microseconds = source.get("timestamp_microseconds")
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank not in range(11)
            or isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index < 0
            or not isinstance(source.get("source_sweep_id"), str)
            or not isinstance(source.get("timestamp_text"), str)
            or isinstance(nanoseconds, bool)
            or not isinstance(nanoseconds, int)
            or isinstance(microseconds, bool)
            or not isinstance(microseconds, int)
            or microseconds != nanoseconds // 1_000
        ):
            raise ProtocolViolation("M7 input-ledger rank-source identity is malformed")
        lag_text = source.get("lag_float32_bits")
        if not isinstance(lag_text, str) or len(lag_text) != 10 or not lag_text.startswith("0x"):
            raise ProtocolViolation("M7 input-ledger source lag bits are malformed")
        _lower_hex(lag_text[2:], length=8, name="source lag bits")
        condition_lag = rank_lags.get(str(rank))
        if (
            not isinstance(condition_lag, str)
            or len(condition_lag) != 10
            or not condition_lag.startswith("0x")
        ):
            raise ProtocolViolation("M7 input-ledger condition lag bits are malformed")
        _lower_hex(condition_lag[2:], length=8, name="condition lag bits")
    arm = Arm(str(raw_record.get("arm")))
    if arm is Arm.F and ranks != [0, 2, 4, 6, 8, 10]:
        raise ProtocolViolation("M7 Arm-F rank-source identities differ")
    _validate_seed_provenance(raw_record, arm)
    return condition


def runtime_binding_projection(record: Mapping[str, object]) -> dict[str, object]:
    """Return the exact compact runtime binding after full archival-record validation."""

    if set(record) != REQUIRED_CONDITION_FIELDS:
        raise ProtocolViolation("M7 runtime projection source fields differ")
    projection = {name: record[name] for name in RUNTIME_BINDING_FIELDS}
    seeds = record.get("seed_provenance")
    if seeds is None:
        projection["seed_provenance"] = None
    else:
        if not isinstance(seeds, list):
            raise ProtocolViolation("M7 runtime projection seed provenance is malformed")
        projected_seeds: list[dict[str, object]] = []
        for seed in seeds:
            if not isinstance(seed, Mapping) or set(seed) != SEED_PROVENANCE_FIELDS:
                raise ProtocolViolation("M7 runtime projection seed schema fields differ")
            projected_seeds.append({name: seed[name] for name in RUNTIME_SEED_FIELDS})
        projection["seed_provenance"] = projected_seeds
    return projection


@dataclass(frozen=True, slots=True)
class RuntimeBindingRecord:
    """Immutable canonical bytes for one fully validated condition's compact binding."""

    condition_id: str
    model_ready_sha256: str
    canonical_projection: bytes

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> RuntimeBindingRecord:
        condition = record.get("condition_id")
        model_ready = record.get("model_ready_sha256")
        if not isinstance(condition, str) or not isinstance(model_ready, str):
            raise ProtocolViolation("M7 runtime projection lacks condition/input identity")
        return cls(condition, model_ready, canonical_json_bytes(runtime_binding_projection(record)))

    def matches(self, record: Mapping[str, object]) -> bool:
        """Return exact canonical projection equality for a regenerated condition."""

        return self.canonical_projection == canonical_json_bytes(runtime_binding_projection(record))


def _runtime_projection_sha256(records: Sequence[RuntimeBindingRecord]) -> str:
    digest = hashlib.sha256()
    digest.update(b"[")
    for index, record in enumerate(records):
        if index:
            digest.update(b",")
        digest.update(record.canonical_projection)
    digest.update(b"]")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class StrictInputLedger:
    """Whole-file small-fixture oracle retaining complete condition dictionaries."""

    identity: Mapping[str, object]
    conditions: tuple[Mapping[str, object], ...]

    def condition(self, condition: str) -> Mapping[str, object]:
        """Return the exact canonical condition record."""

        try:
            index = canonical_condition_ids().index(condition)
        except ValueError as error:
            raise ProtocolViolation(
                f"condition is outside the frozen M7 corpus: {condition}"
            ) from error
        return self.conditions[index]

    def runtime_condition(self, condition: str) -> RuntimeBindingRecord:
        """Project one small-fixture oracle record through the canonical runtime binding."""

        return RuntimeBindingRecord.from_record(self.condition(condition))


@dataclass(frozen=True, slots=True)
class StreamingStrictInputLedger:
    """Bounded-memory index derived only after every full archival record validates."""

    identity: Mapping[str, object]
    conditions: tuple[RuntimeBindingRecord, ...]
    runtime_projection_sha256: str = field(init=False)
    _by_condition: Mapping[str, RuntimeBindingRecord] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        records = tuple(self.conditions)
        by_condition = {record.condition_id: record for record in records}
        if len(by_condition) != len(records):
            raise ProtocolViolation("M7 runtime index contains duplicate condition IDs")
        object.__setattr__(self, "identity", MappingProxyType(dict(self.identity)))
        object.__setattr__(self, "conditions", records)
        object.__setattr__(self, "_by_condition", MappingProxyType(by_condition))
        object.__setattr__(self, "runtime_projection_sha256", _runtime_projection_sha256(records))

    @property
    def condition_ids(self) -> tuple[str, ...]:
        """Return the exact retained canonical condition sequence."""

        return tuple(record.condition_id for record in self.conditions)

    def runtime_condition(self, condition: str) -> RuntimeBindingRecord:
        """Return one compact immutable condition binding by exact identity."""

        try:
            return self._by_condition[condition]
        except KeyError as error:
            raise ProtocolViolation(
                f"condition is outside the frozen M7 corpus: {condition}"
            ) from error


def validate_input_ledger(
    ledger: Mapping[str, object],
    *,
    expected_implementation_commit: str,
    require_full_corpus: bool = True,
) -> StrictInputLedger:
    """Strictly validate schema, identities, order, and every condition field."""

    expected_top = {"schema_version", "status", "identity", "condition_count", "conditions"}
    if set(ledger) != expected_top:
        raise ProtocolViolation("M7 input-ledger top-level schema fields differ")
    if ledger.get("schema_version") != "laserperception.m7.input-ledger.v1":
        raise ProtocolViolation("M7 input-ledger schema is invalid")
    if ledger.get("status") != "INPUT-ONLY; NO DETECTOR OUTPUT":
        raise ProtocolViolation("M7 input-ledger status is invalid")
    identity = ledger.get("identity")
    if not isinstance(identity, Mapping):
        raise ProtocolViolation("M7 input-ledger identity is malformed")
    expected_identity = InputLedgerIdentity(
        expected_implementation_commit,
        M6B_INPUT_LEDGER_FULL_SHA256,
        M6B_RESULT_FULL_SHA256,
    ).to_dict()
    if dict(identity) != expected_identity:
        raise ProtocolViolation("M7 input-ledger frozen identity differs")
    conditions = ledger.get("conditions")
    if not isinstance(conditions, list) or ledger.get("condition_count") != len(conditions):
        raise ProtocolViolation("M7 input-ledger condition count is malformed")
    records: list[Mapping[str, object]] = []
    ids: list[str] = []
    for raw_record in conditions:
        if not isinstance(raw_record, Mapping) or set(raw_record) != REQUIRED_CONDITION_FIELDS:
            raise ProtocolViolation("M7 input-ledger condition schema fields differ")
        condition = _validate_complete_condition_record(
            raw_record,
            expected_implementation_commit=expected_implementation_commit,
        )
        records.append(raw_record)
        ids.append(condition)
    if len(ids) != len(set(ids)):
        raise ProtocolViolation("M7 input ledger contains duplicate condition IDs")
    expected_ids = canonical_condition_ids()
    if require_full_corpus:
        if tuple(ids) != expected_ids:
            raise ProtocolViolation(
                "M7 input ledger must contain all 1,712 conditions in canonical B/C/D/F order"
            )
    elif tuple(ids) != tuple(sorted(ids, key=expected_ids.index)):
        raise ProtocolViolation("synthetic M7 input-ledger fixture is not in canonical order")
    return StrictInputLedger(dict(identity), tuple(records))


def load_strict_input_ledger_whole_for_test(
    path: str | Path,
    *,
    expected_implementation_commit: str,
    require_full_corpus: bool = True,
) -> StrictInputLedger:
    """Retain the legacy whole-file behavior only as a bounded test oracle."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ProtocolViolation("M7 input ledger must be a JSON object")
    return validate_input_ledger(
        value,
        expected_implementation_commit=expected_implementation_commit,
        require_full_corpus=require_full_corpus,
    )


def _normalize_stream_numbers(value: object) -> object:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _normalize_stream_numbers(item)
        return value
    if isinstance(value, dict):
        for name, item in value.items():
            value[name] = _normalize_stream_numbers(item)
        return value
    return value


def _next_event(events: object) -> tuple[str, str, object]:
    try:
        return next(events)  # type: ignore[call-overload]
    except StopIteration as error:
        raise ProtocolViolation("M7 input ledger JSON is truncated") from error


def _build_stream_value(events: object, first_event: str, first_value: object) -> object:
    builder = ObjectBuilder(map_type=dict)
    builder.event(first_event, first_value)
    if first_event not in {"start_map", "start_array"}:
        return _normalize_stream_numbers(builder.value)
    depth = 1
    while depth:
        _, event, value = _next_event(events)
        builder.event(event, value)
        if event in {"start_map", "start_array"}:
            depth += 1
        elif event in {"end_map", "end_array"}:
            depth -= 1
    return _normalize_stream_numbers(builder.value)


def _validate_ledger_header(
    top_values: Mapping[str, object],
    *,
    expected_implementation_commit: str,
    actual_condition_count: int,
) -> Mapping[str, object]:
    expected_top = {"schema_version", "status", "identity", "condition_count", "conditions"}
    if set(top_values) != expected_top:
        raise ProtocolViolation("M7 input-ledger top-level schema fields differ")
    if top_values.get("schema_version") != "laserperception.m7.input-ledger.v1":
        raise ProtocolViolation("M7 input-ledger schema is invalid")
    if top_values.get("status") != "INPUT-ONLY; NO DETECTOR OUTPUT":
        raise ProtocolViolation("M7 input-ledger status is invalid")
    identity = top_values.get("identity")
    if not isinstance(identity, Mapping):
        raise ProtocolViolation("M7 input-ledger identity is malformed")
    expected_identity = InputLedgerIdentity(
        expected_implementation_commit,
        M6B_INPUT_LEDGER_FULL_SHA256,
        M6B_RESULT_FULL_SHA256,
    ).to_dict()
    if dict(identity) != expected_identity:
        raise ProtocolViolation("M7 input-ledger frozen identity differs")
    if top_values.get("condition_count") != actual_condition_count:
        raise ProtocolViolation("M7 input-ledger condition count is malformed")
    return dict(identity)


def load_strict_input_ledger(
    path: str | Path,
    *,
    expected_implementation_commit: str,
    require_full_corpus: bool = True,
) -> StreamingStrictInputLedger:
    """Stream every full record and retain only compact immutable runtime bindings."""

    records: list[RuntimeBindingRecord] = []
    condition_ids: list[str] = []
    top_values: dict[str, object] = {}
    expected_ids = canonical_condition_ids()
    try:
        with Path(path).open("rb") as stream:
            events = iter(ijson_backend.parse(stream, use_float=False, multiple_values=False))
            prefix, event, _ = _next_event(events)
            if prefix != "" or event != "start_map":
                raise ProtocolViolation("M7 input ledger must be a JSON object")
            while True:
                prefix, event, value = _next_event(events)
                if prefix == "" and event == "end_map":
                    break
                if prefix != "" or event != "map_key" or not isinstance(value, str):
                    raise ProtocolViolation("M7 input-ledger top-level structure is malformed")
                name = value
                if name in top_values:
                    raise ProtocolViolation(f"M7 input ledger repeats top-level field: {name}")
                value_prefix, value_event, value = _next_event(events)
                if name != "conditions":
                    top_values[name] = _build_stream_value(events, value_event, value)
                    continue
                if value_prefix != "conditions" or value_event != "start_array":
                    raise ProtocolViolation("M7 input-ledger conditions must be an array")
                top_values[name] = True
                while True:
                    item_prefix, item_event, item_value = _next_event(events)
                    if item_prefix == "conditions" and item_event == "end_array":
                        break
                    if item_prefix != "conditions.item" or item_event != "start_map":
                        raise ProtocolViolation("M7 input-ledger condition is not an object")
                    raw_record = _build_stream_value(events, item_event, item_value)
                    if not isinstance(raw_record, Mapping):
                        raise ProtocolViolation("M7 input-ledger condition is malformed")
                    condition = _validate_complete_condition_record(
                        raw_record,
                        expected_implementation_commit=expected_implementation_commit,
                    )
                    if condition in condition_ids:
                        raise ProtocolViolation("M7 input ledger contains duplicate condition IDs")
                    condition_ids.append(condition)
                    records.append(RuntimeBindingRecord.from_record(raw_record))
            try:
                next(events)
            except StopIteration:
                pass
            else:
                raise ProtocolViolation("M7 input ledger contains trailing JSON values")
    except JSONError as error:
        raise ProtocolViolation("M7 input ledger JSON is truncated or malformed") from error
    identity = _validate_ledger_header(
        top_values,
        expected_implementation_commit=expected_implementation_commit,
        actual_condition_count=len(records),
    )
    if require_full_corpus:
        if tuple(condition_ids) != expected_ids:
            raise ProtocolViolation(
                "M7 input ledger must contain all 1,712 conditions in canonical B/C/D/F order"
            )
    elif tuple(condition_ids) != tuple(sorted(condition_ids, key=expected_ids.index)):
        raise ProtocolViolation("synthetic M7 input-ledger fixture is not in canonical order")
    return StreamingStrictInputLedger(identity, tuple(records))


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
