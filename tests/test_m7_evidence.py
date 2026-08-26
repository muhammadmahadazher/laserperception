from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.m7.evidence import (
    InputLedgerIdentity,
    build_input_ledger,
    parse_m6b_paired_sets,
    serialize_input_ledger,
    write_input_ledger,
)
from benchmarks.m7.protocol import Arm, ProtocolViolation, canonical_frame_ids


def _observation(track_id: int, matched: bool) -> dict[str, object]:
    return {"track_id": track_id, "matched": matched}


def _frame(
    frame_id: str, car: list[dict[str, object]], ped: list[dict[str, object]]
) -> dict[str, object]:
    return {
        "frame_id": frame_id,
        "classes": {
            "car": {"target_observations": car},
            "pedestrian": {"target_observations": ped},
        },
    }


def test_paired_set_parser_uses_exact_sorted_pose_identities(tmp_path: Path) -> None:
    frame1 = "2011_09_26_drive_0001/0000000010"
    frame2 = "2011_09_26_drive_0091/0000000011"
    value = {
        "frame_results": {
            "H10": [
                _frame(
                    frame2,
                    [_observation(9, False), _observation(8, True)],
                    [_observation(5, True), _observation(6, False)],
                ),
                _frame(
                    frame1,
                    [_observation(2, False), _observation(1, True)],
                    [_observation(3, True), _observation(4, False)],
                ),
            ],
            "H5": [
                _frame(
                    frame1,
                    [_observation(2, False), _observation(1, True)],
                    [_observation(3, False), _observation(4, True)],
                ),
                _frame(
                    frame2,
                    [_observation(9, True), _observation(8, False)],
                    [_observation(5, True), _observation(6, False)],
                ),
            ],
        }
    }
    path = tmp_path / "m6b.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    car = parse_m6b_paired_sets(path, "car")
    ped = parse_m6b_paired_sets(path, "pedestrian")

    assert car.cardinalities() == {"shared": 1, "e_only": 1, "a_only": 1, "neither": 1}
    assert car.shared == (("2011_09_26_drive_0001", 10, 1),)
    assert ped.cardinalities() == {"shared": 1, "e_only": 1, "a_only": 1, "neither": 1}
    assert len(car.to_dict()["shared_sha256"]) == 64


def _ledger_condition(frame_id: str, arm: Arm) -> dict[str, object]:
    drive_id, frame_text = frame_id.split("/")
    return {
        "condition_id": f"{frame_id}|{arm.value}",
        "drive_id": drive_id,
        "frame_index": int(frame_text),
        "arm": arm.value,
        "generation_commit": "1" * 40,
        "source_a_sha256": "a" * 64,
        "source_e_sha256": "e" * 64,
        "point_count": 12,
        "xyz_sha256": "x" * 64,
        "model_ready_sha256": "m" * 64,
        "selected_row_sha256": "s" * 64,
        "lag_bit_patterns": ["0x00000000"],
        "lag_support_count": 1,
        "lag_span_seconds": 0.0,
        "sweep_ids": ["current"],
        "per_sweep_point_counts": {"0": 12},
        "pillar_structure": {"candidate_count": 3},
        "lag_scale_provenance": None,
        "quota_provenance": None,
        "seed_provenance": None,
        "f_history_ranks": [2, 4, 6, 8, 10] if arm is Arm.F else None,
        "runtime_versions": {"python": "test", "numpy": "test"},
    }


def test_input_ledger_is_deterministic_atomic_and_fail_closed(tmp_path: Path) -> None:
    frame = canonical_frame_ids()[0]
    conditions = [_ledger_condition(frame, arm) for arm in (Arm.B, Arm.C, Arm.D, Arm.F)]
    identity = InputLedgerIdentity("1" * 40, "2" * 64, "3" * 64)
    ledger = build_input_ledger(identity, conditions, require_full_corpus=False)
    first = serialize_input_ledger(ledger)
    second = serialize_input_ledger(ledger)
    output = tmp_path / "ledger.json"

    write_input_ledger(output, ledger)

    assert first == second == output.read_bytes()
    assert not output.with_suffix(".json.tmp").exists()
    with pytest.raises(ProtocolViolation, match="duplicate"):
        build_input_ledger(identity, conditions + [conditions[0]], require_full_corpus=False)
    with pytest.raises(ProtocolViolation, match="missing fields"):
        build_input_ledger(identity, [{"condition_id": "bad"}], require_full_corpus=False)
    with pytest.raises(ProtocolViolation, match="canonical order"):
        build_input_ledger(identity, list(reversed(conditions)), require_full_corpus=False)
    bad_f = dict(conditions[-1])
    bad_f["f_history_ranks"] = [1, 2, 3, 4, 5]
    with pytest.raises(ProtocolViolation, match="2/4/6/8/10"):
        build_input_ledger(identity, [bad_f], require_full_corpus=False)
