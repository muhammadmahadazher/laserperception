from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import ijson
import ijson.backends.yajl2_c as ijson_backend
import pytest

from benchmarks.m7.evidence import (
    InputLedgerIdentity,
    RuntimeBindingRecord,
    build_input_ledger,
    load_strict_input_ledger,
    load_strict_input_ledger_whole_for_test,
    runtime_binding_projection,
    serialize_input_ledger,
)
from benchmarks.m7.protocol import (
    M6B_INPUT_LEDGER_FULL_SHA256,
    M6B_RESULT_FULL_SHA256,
    Arm,
    ProtocolViolation,
    canonical_frame_ids,
)
from benchmarks.m7.provenance import canonical_json_bytes

IMPLEMENTATION = "1" * 40


def _quota(selected_count: int, source_count: int) -> dict[str, object]:
    counts = [source_count, *([1] * 9)]
    quotas = [selected_count, *([0] * 9)]
    return {
        "source_counts_by_rank_1_to_10": counts,
        "h_target": selected_count,
        "h_total": sum(counts),
        "products": [selected_count * value for value in counts],
        "remainders": [0] * 10,
        "initial_quotas": quotas,
        "incremented_ranks": [],
        "final_quotas": quotas,
        "zero_quota_ranks": list(range(2, 11)),
    }


def _seeds(
    drive_id: str, frame_index: int, *, selected_count: int = 0, source_count: int = 1
) -> list[dict[str, object]]:
    result = []
    for rank in range(1, 11):
        text = f"laserperception-m7-c-v1|{drive_id}|{frame_index:010d}|{rank}"
        digest = hashlib.sha256(text.encode()).digest()
        seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        result.append(
            {
                "history_rank": rank,
                "seed_text_utf8": text,
                "sha256": digest.hex(),
                "seed_uint64": seed,
                "seed_uint64_hex": f"0x{seed:016x}",
                "selected_ordinals": list(range(selected_count)) if rank == 1 else [],
            }
        )
    assert selected_count <= source_count
    return result


def _condition(frame_id: str, arm: Arm) -> dict[str, object]:
    drive_id, frame_text = frame_id.split("/")
    frame_index = int(frame_text)
    ranks = [0, 2, 4, 6, 8, 10] if arm is Arm.F else [0]
    rank_sources = [
        {
            "history_rank": rank,
            "source_sweep_id": f"sweep-{rank}",
            "source_index": 10 - rank,
            "timestamp_text": f"synthetic-{rank}",
            "timestamp_nanoseconds": (10 - rank) * 100_000_000,
            "timestamp_microseconds": (10 - rank) * 100_000,
            "lag_float32_bits": f"0x{rank:08x}",
        }
        for rank in ranks
    ]
    is_cd = arm in (Arm.C, Arm.D)
    return {
        "condition_id": f"{frame_id}|{arm.value}",
        "drive_id": drive_id,
        "frame_index": frame_index,
        "arm": arm.value,
        "generation_commit": IMPLEMENTATION,
        "source_a_sha256": "a" * 64,
        "source_e_sha256": "e" * 64,
        "point_count": len(ranks),
        "xyz_sha256": "c" * 64,
        "model_ready_sha256": "d" * 64,
        "selected_row_sha256": "f" * 64,
        "lag_bit_patterns": [f"0x{rank:08x}" for rank in ranks],
        "lag_support_count": len(ranks),
        "lag_span_seconds": 0.0,
        "sweep_ids": [f"sweep-{rank}" for rank in ranks],
        "per_sweep_point_counts": {str(rank): 1 for rank in ranks},
        "provenance_schema": "laserperception.m7.sweep-provenance.v2",
        "rank_source_identities": rank_sources,
        "rank_to_lag_bit_pattern": {str(rank): f"0x{rank:08x}" for rank in ranks},
        "pillar_structure": {"candidate_count": 3, "retained_count": 3},
        "lag_scale_provenance": {"scale_binary64_bits": "0x3fe0000000000000"}
        if arm in (Arm.B, Arm.D)
        else None,
        "quota_provenance": _quota(0, 1) if is_cd else None,
        "seed_provenance": _seeds(drive_id, frame_index) if is_cd else None,
        "f_history_ranks": [2, 4, 6, 8, 10] if arm is Arm.F else None,
        "runtime_versions": {"python": "test", "numpy": "test"},
    }


def _ledger() -> dict[str, object]:
    frame = canonical_frame_ids()[0]
    conditions = [_condition(frame, arm) for arm in (Arm.B, Arm.C, Arm.D, Arm.F)]
    identity = InputLedgerIdentity(
        IMPLEMENTATION,
        M6B_INPUT_LEDGER_FULL_SHA256,
        M6B_RESULT_FULL_SHA256,
    )
    return build_input_ledger(identity, conditions, require_full_corpus=False)


def _write(path: Path, ledger: dict[str, object]) -> None:
    path.write_bytes(serialize_input_ledger(ledger))


@pytest.fixture
def four_condition_corpus(monkeypatch: pytest.MonkeyPatch) -> tuple[str, ...]:
    conditions = _ledger()["conditions"]
    assert isinstance(conditions, list)
    ids = tuple(str(record["condition_id"]) for record in conditions)
    monkeypatch.setattr("benchmarks.m7.evidence.canonical_condition_ids", lambda: ids)
    return ids


def test_streaming_parser_selection_is_frozen() -> None:
    assert ijson.__version__ == "3.5.1"
    assert ijson_backend.backend == "yajl2_c"


def test_small_fixture_whole_and_streaming_semantics_are_equivalent(
    tmp_path: Path, four_condition_corpus: tuple[str, ...]
) -> None:
    path = tmp_path / "ledger.json"
    _write(path, _ledger())

    whole = load_strict_input_ledger_whole_for_test(
        path, expected_implementation_commit=IMPLEMENTATION
    )
    streamed = load_strict_input_ledger(path, expected_implementation_commit=IMPLEMENTATION)

    assert dict(streamed.identity) == dict(whole.identity)
    assert streamed.condition_ids == four_condition_corpus
    assert tuple(record.condition_id for record in streamed.conditions) == four_condition_corpus
    for condition in four_condition_corpus:
        assert (
            streamed.runtime_condition(condition).canonical_projection
            == whole.runtime_condition(condition).canonical_projection
        )


Mutation = Callable[[dict[str, object]], None]


def _wrong_top_schema(ledger: dict[str, object]) -> None:
    ledger["unexpected"] = True


def _wrong_identity(ledger: dict[str, object]) -> None:
    identity = ledger["identity"]
    assert isinstance(identity, dict)
    identity["implementation_commit"] = "9" * 40


def _wrong_count(ledger: dict[str, object]) -> None:
    ledger["condition_count"] = 99


def _missing_condition(ledger: dict[str, object]) -> None:
    conditions = ledger["conditions"]
    assert isinstance(conditions, list)
    conditions.pop()
    ledger["condition_count"] = len(conditions)


def _duplicate_condition(ledger: dict[str, object]) -> None:
    conditions = ledger["conditions"]
    assert isinstance(conditions, list)
    conditions[-1] = copy.deepcopy(conditions[0])


def _out_of_order(ledger: dict[str, object]) -> None:
    conditions = ledger["conditions"]
    assert isinstance(conditions, list)
    conditions[0], conditions[1] = conditions[1], conditions[0]


def _condition_at(ledger: dict[str, object], index: int) -> dict[str, object]:
    conditions = ledger["conditions"]
    assert isinstance(conditions, list)
    record = conditions[index]
    assert isinstance(record, dict)
    return record


def _invalid_f_ranks(ledger: dict[str, object]) -> None:
    _condition_at(ledger, 3)["f_history_ranks"] = [1, 2, 3, 4, 5]


def _malformed_source_rank(ledger: dict[str, object]) -> None:
    sources = _condition_at(ledger, 0)["rank_source_identities"]
    assert isinstance(sources, list) and isinstance(sources[0], dict)
    sources[0]["history_rank"] = 11


def _bad_lag_bits(ledger: dict[str, object]) -> None:
    _condition_at(ledger, 0)["lag_bit_patterns"] = ["bad"]


def _incorrect_sweep_counts(ledger: dict[str, object]) -> None:
    _condition_at(ledger, 0)["per_sweep_point_counts"] = {"0": 2}


def _bad_selected_row_sha(ledger: dict[str, object]) -> None:
    _condition_at(ledger, 0)["selected_row_sha256"] = "z" * 64


def _bad_model_ready_sha(ledger: dict[str, object]) -> None:
    _condition_at(ledger, 0)["model_ready_sha256"] = "z" * 64


@pytest.mark.parametrize(
    "mutation",
    [
        _wrong_top_schema,
        _wrong_identity,
        _wrong_count,
        _missing_condition,
        _duplicate_condition,
        _out_of_order,
        _invalid_f_ranks,
        _malformed_source_rank,
        _bad_lag_bits,
        _incorrect_sweep_counts,
        _bad_selected_row_sha,
        _bad_model_ready_sha,
    ],
)
def test_whole_and_streaming_reject_same_bounded_malformed_fixtures(
    tmp_path: Path,
    four_condition_corpus: tuple[str, ...],
    mutation: Mutation,
) -> None:
    del four_condition_corpus
    ledger = copy.deepcopy(_ledger())
    mutation(ledger)
    path = tmp_path / "bad.json"
    _write(path, ledger)

    with pytest.raises(ProtocolViolation):
        load_strict_input_ledger_whole_for_test(path, expected_implementation_commit=IMPLEMENTATION)
    with pytest.raises(ProtocolViolation):
        load_strict_input_ledger(path, expected_implementation_commit=IMPLEMENTATION)


@pytest.mark.parametrize("fraction", [0.0, 0.5, 0.999])
def test_streaming_loader_rejects_truncation(
    tmp_path: Path, four_condition_corpus: tuple[str, ...], fraction: float
) -> None:
    del four_condition_corpus
    payload = serialize_input_ledger(_ledger())
    cut = int(len(payload) * fraction)
    path = tmp_path / "truncated.json"
    path.write_bytes(payload[:cut])

    with pytest.raises(ProtocolViolation, match="truncated|malformed"):
        load_strict_input_ledger(path, expected_implementation_commit=IMPLEMENTATION)


def test_streaming_loader_never_uses_whole_file_json_apis(
    tmp_path: Path,
    four_condition_corpus: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del four_condition_corpus
    path = tmp_path / "ledger.json"
    _write(path, _ledger())

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("whole-file JSON API reached")

    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(json, "load", forbidden)
    monkeypatch.setattr(json, "loads", forbidden)

    streamed = load_strict_input_ledger(path, expected_implementation_commit=IMPLEMENTATION)
    assert len(streamed.conditions) == 4


def test_huge_selected_ordinals_validate_but_are_not_retained(
    tmp_path: Path, four_condition_corpus: tuple[str, ...]
) -> None:
    del four_condition_corpus
    ledger = _ledger()
    condition = _condition_at(ledger, 1)
    condition["quota_provenance"] = _quota(50_000, 50_000)
    condition["seed_provenance"] = _seeds(
        str(condition["drive_id"]),
        int(condition["frame_index"]),
        selected_count=50_000,
        source_count=50_000,
    )
    path = tmp_path / "huge-ordinals.json"
    _write(path, ledger)

    streamed = load_strict_input_ledger(path, expected_implementation_commit=IMPLEMENTATION)
    binding = streamed.conditions[1]

    assert b"selected_ordinals" not in binding.canonical_projection
    assert len(binding.canonical_projection) < 10_000


@pytest.mark.parametrize(
    ("arm", "mutate"),
    [
        (Arm.B, lambda value: value.__setitem__("source_a_sha256", "0" * 64)),
        (Arm.B, lambda value: value.__setitem__("source_e_sha256", "0" * 64)),
        (Arm.B, lambda value: value.__setitem__("point_count", 2)),
        (Arm.B, lambda value: value.__setitem__("xyz_sha256", "0" * 64)),
        (Arm.B, lambda value: value.__setitem__("model_ready_sha256", "0" * 64)),
        (Arm.B, lambda value: value.__setitem__("selected_row_sha256", "0" * 64)),
        (Arm.B, lambda value: value["rank_to_lag_bit_pattern"].__setitem__("0", "0x1")),
        (Arm.B, lambda value: value["rank_source_identities"][0].__setitem__("source_index", 8)),
        (Arm.B, lambda value: value["pillar_structure"].__setitem__("candidate_count", 4)),
        (Arm.B, lambda value: value["lag_scale_provenance"].__setitem__("scale", 0.4)),
        (Arm.C, lambda value: value["quota_provenance"].__setitem__("h_target", 1)),
        (Arm.C, lambda value: value["seed_provenance"][0].__setitem__("seed_uint64", 1)),
        (Arm.F, lambda value: value.__setitem__("f_history_ranks", [1, 2, 3, 4, 5])),
    ],
)
def test_projection_rejects_changes_to_every_retained_binding(
    arm: Arm, mutate: Callable[[dict[str, object]], None]
) -> None:
    record = _condition(canonical_frame_ids()[0], arm)
    binding = RuntimeBindingRecord.from_record(record)
    changed = copy.deepcopy(record)
    mutate(changed)
    assert not binding.matches(changed)


def test_selected_ordinal_residency_is_distinct_from_archival_authorization() -> None:
    first = _condition(canonical_frame_ids()[0], Arm.C)
    first["quota_provenance"] = _quota(1, 2)
    first["seed_provenance"] = _seeds(
        str(first["drive_id"]), int(first["frame_index"]), selected_count=1, source_count=2
    )
    second = copy.deepcopy(first)
    seeds = second["seed_provenance"]
    assert isinstance(seeds, list) and isinstance(seeds[0], dict)
    seeds[0]["selected_ordinals"] = [1]

    assert canonical_json_bytes(first) != canonical_json_bytes(second)
    assert runtime_binding_projection(first) == runtime_binding_projection(second)
    assert RuntimeBindingRecord.from_record(first) == RuntimeBindingRecord.from_record(second)
