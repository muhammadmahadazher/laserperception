"""CPU-only checks of the frozen prospective S2 protocol and paired partitions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from benchmarks.m8.derive_s2_partitions_draft import canonical_json_bytes

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "benchmarks/m8/preregistration"
PROTOCOL = BASE / "m8_s2_protocol.json"
PARTITIONS = BASE / "m8_s2_partitions.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_partitions_preserve_approved_s1_identities() -> None:
    frozen = _load(PARTITIONS)
    draft = _load(BASE / "m8_s2_partitions_draft.json")
    assert frozen["source"] == draft["source"]
    assert frozen["classes"] == draft["classes"]
    assert frozen["canonical_pose_key"] == ["drive_id", "frame_index", "gt_track_id"]
    expected = {
        "car": (
            [19, 24, 0, 23, 0],
            [
                "bb010b66448bd735389a1888549205c80a6ffaaf573fabbbf7665639d7fc8ecb",
                "408285adfbf9df5a6f1b58c21bae3b82cc3952a6ce95c7289fc95cce59d2b0f0",
                "37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570",
                "e10777605f4421d004149e499b0d9ce0b3480a0c396e78cc6a560b272c28a67f",
                "37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570",
            ],
        ),
        "pedestrian": (
            [0, 1, 0, 395, 0],
            [
                "37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570",
                "5dfd836fd80fa09bb252521fe8cabca02ffd897d6f3695e4115744f60141d144",
                "37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570",
                "8f377e3a24494ae907917f0762d74c92734d04bc0746a207aa7730733a529184",
                "37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570",
            ],
        ),
    }
    names = ("shared", "e2_only", "a2_only", "neither", "unstable")
    for class_name, (counts, hashes) in expected.items():
        buckets = frozen["classes"][class_name]
        assert [buckets[name]["count"] for name in names] == counts
        assert [buckets[name]["canonical_json_sha256"] for name in names] == hashes
        for name in names:
            bucket = buckets[name]
            assert len(bucket["identities"]) == bucket["count"]
            assert (
                hashlib.sha256(canonical_json_bytes(bucket["identities"])).hexdigest()
                == (bucket["canonical_json_sha256"])
            )
    assert frozen["car_denominator_rule"]["per_process_e2_minus_a2_tp"] == [24, 24, 24]
    assert frozen["car_denominator_rule"]["minimum_gap"] == 20
    assert frozen["car_denominator_rule"]["denominator_eligible"] is True
    assert frozen["car_denominator_rule"]["partition_eligible"] is True
    assert frozen["authorization"] == {
        "S2_PROTOCOL_FROZEN": True,
        "S2_IMPLEMENTATION_FROZEN": False,
        "S2_INPUT_LEDGER_FROZEN": False,
        "S2_READY_TO_EXECUTE": False,
        "S2_INFERENCE_AUTHORIZED": False,
        "B2_C2_D2_F2_CALLS": 0,
    }


def test_protocol_freeze_preserves_execution_and_authorization_barriers() -> None:
    protocol = _load(PROTOCOL)
    scope = protocol["scientific_scope"]
    assert scope["feature_order"] == ["x", "y", "z", "intensity", "time_lag"]
    assert scope["zero_intensity_is_an_s2_arm"] is False
    ref = protocol["frozen_s1_reference"]
    assert ref["a2_car_tp_by_primary_process"] == [19, 19, 19]
    assert ref["e2_car_tp_by_primary_process"] == [43, 43, 43]
    assert ref["car_gap_by_primary_process"] == [24, 24, 24]
    assert (ref["a_ref"], ref["e_ref"], ref["d_ref"]) == (19, 43, 24)
    assert ref["minimum_positive_exactly_stable_gap"] == 20
    assert ref["denominator_eligible"] is True
    assert ref["partition_gate_passes"] is True
    assert ref["normalized_recovery_formula"] == "G_car(X_i)=(TP_X_i-19)/24"
    assert ref["clamp"] is False
    assert protocol["paired_recovery_if_eligible"]["r_aonly"] is None

    arms = protocol["arms"]
    assert all(arm in arms for arm in ("B2", "C2", "D2", "F2"))
    assert "change only time_lag" in arms["B2"]
    assert "exact E2 total point count" in arms["C2"]
    assert "without resampling" in arms["D2"]
    assert "2,4,6,8,10" in arms["F2"]
    assert protocol["m7_mechanical_identity"]["f2_ranks"] == [2, 4, 6, 8, 10]
    identity = protocol["m7_mechanical_identity"]
    assert identity["required_exact_m7_xyzt_hash_matches_total"] == 1712
    assert identity["xyzt_identity_proven_now"] is False

    execution = protocol["prospective_execution_after_separate_authorization"]
    repeat = execution["repeatability"]
    full = execution["full_corpus"]
    assert (
        repeat["fresh_processes"],
        repeat["conditions_per_process"],
        repeat["accepted_calls"],
    ) == (
        10,
        28,
        280,
    )
    assert (full["fresh_processes"], full["conditions_per_process"], full["accepted_calls"]) == (
        3,
        1712,
        5136,
    )
    assert execution["total_proposed_accepted_scientific_calls"] == 5416
    assert execution["actual_s2_calls_now"] == 0
    gate = protocol["prospective_descriptive_interpretation_gate"]
    assert (gate["per_process_g_car_gte"], gate["per_process_r_gain_gte"]) == (0.5, 0.5)
    assert gate["per_process_max_frozen_shared_losses"] == 1
    assert gate["required_complete_processes_passing"] == 3
    assert protocol["authorization"]["PROTOCOL_FROZEN"] is True
    assert protocol["authorization"]["IMPLEMENTATION_FROZEN"] is False
    assert protocol["authorization"]["INPUT_LEDGER_FROZEN"] is False
    assert protocol["authorization"]["RUNTIME_BOUND"] is False
    assert protocol["authorization"]["INFERENCE_AUTHORIZED"] is False
    assert protocol["authorization"]["B2_C2_D2_F2_CALLS"] == 0
    assert "a deterministically specified intervention study" in (
        ROOT / "docs/m8/M8_S2_PROTOCOL.md"
    ).read_text(encoding="utf-8")

    def keys(value: Any) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value))
        return set()

    assert {"p_value", "confidence_interval", "standard_error"}.isdisjoint(keys(protocol))
