"""CPU-only validation of the prospective S2 design and S1-derived pose lists."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from benchmarks.m8 import derive_s2_partitions_draft as derivation

ROOT = Path(__file__).resolve().parents[1]
PARTITIONS = ROOT / "benchmarks/m8/preregistration/m8_s2_partitions_draft.json"
PROTOCOL = ROOT / "benchmarks/m8/preregistration/m8_s2_protocol_draft.json"
DOC = ROOT / "docs/m8/M8_S2_PROTOCOL_DRAFT.md"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_sha(relative: str) -> str:
    data = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    return hashlib.sha256(data).hexdigest()


def test_three_process_classification_keeps_instability_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(derivation, "CLASSES", {"car": 3, "pedestrian": 1})
    pose = ("2011_09_26_drive_0001", 10, 1)
    states = {
        "car": {
            pose: {"a2": [1, 1, 1], "e2": [1, 1, 1]},
            (pose[0], 11, 2): {"a2": [0, 0, 0], "e2": [1, 1, 1]},
            (pose[0], 12, 3): {"a2": [0, 1, 0], "e2": [0, 0, 0]},
        },
        "pedestrian": {(pose[0], 13, 4): {"a2": [0, 0, 0], "e2": [0, 0, 0]}},
    }
    result = derivation.classify(states)
    assert [result["car"][name]["count"] for name in derivation.CATEGORIES] == [1, 1, 0, 0, 1]
    assert result["pedestrian"]["neither"]["count"] == 1
    assert result["car"]["unstable_process_states"][0]["a2_process_states"] == [0, 1, 0]
    assert result["car"]["a2_only"]["canonical_json_sha256"] == hashlib.sha256(b"[]\n").hexdigest()
    states["car"][(pose[0], 11, 2)]["a2"][1] = None
    with pytest.raises(ValueError, match="missing or invalid"):
        derivation.classify(states)


def test_partition_draft_binds_accepted_primary_evidence_and_exact_lists() -> None:
    draft = _json(PARTITIONS)
    source = draft["source"]
    assert source["primary_execution_commit"] == derivation.PRIMARY_COMMIT
    assert source["primary_measurement_manifest_sha256"] == _git_sha(
        "benchmarks/m8/results/m8_s1_measurement_manifest.json"
    )
    assert source["primary_compact_raw_sha256"] == _git_sha(
        "benchmarks/m8/results/m8_s1_primary_raw.json"
    )
    assert source["frozen_s1_protocol_sha256"] == _git_sha(
        "benchmarks/m8/preregistration/m8_s1_protocol.json"
    )
    accepted = _json(ROOT / "benchmarks/m8/results/m8_s1_measurement_manifest.json")[
        "accepted_passes"
    ]
    assert len(source["accepted_primary_passes"]) == 3
    for actual, expected in zip(source["accepted_primary_passes"], accepted, strict=True):
        assert actual["process_uuid"] == expected["process_uuid"]
        assert actual["archive_sha256"] == expected["archive_sha256"]
        assert actual["raw_pass_file_sha256"] == expected["raw_pass_file_sha256"]
        assert actual["condition_file_hashes_verified"] == 856

    expected_counts = {
        "car": (66, [19, 24, 0, 23, 0]),
        "pedestrian": (396, [0, 1, 0, 395, 0]),
    }
    for class_name, (total, counts) in expected_counts.items():
        class_draft = draft["classes"][class_name]
        assert class_draft["eligible_gt_count"] == total
        assert [class_draft[name]["count"] for name in derivation.CATEGORIES] == counts
        all_keys: set[tuple[str, int, int]] = set()
        for name in derivation.CATEGORIES:
            bucket = class_draft[name]
            identities = bucket["identities"]
            assert len(identities) == bucket["count"]
            assert bucket["canonical_json_sha256"] == derivation.sha256(
                derivation.canonical_json_bytes(identities)
            )
            keys = [(row["drive_id"], row["frame_index"], row["gt_track_id"]) for row in identities]
            assert keys == sorted(keys)
            assert not all_keys.intersection(keys)
            all_keys.update(keys)
        assert len(all_keys) == total
    car = draft["classes"]["car"]
    assert car["shared"]["count"] + car["a2_only"]["count"] == 19
    assert car["shared"]["count"] + car["e2_only"]["count"] == 43
    assert car["e2_only"]["count"] - car["a2_only"]["count"] == 24
    gap = draft["car_denominator_proposal"]
    assert gap["per_process_e2_minus_a2_tp"] == [24, 24, 24]
    assert gap["proposed_minimum_gap"] == 20
    assert gap["denominator_eligible"] is True
    assert gap["partition_eligible"] is True
    assert draft["authorization"]["s2_authorized"] is False
    assert draft["authorization"]["b2_c2_d2_f2_calls"] == 0


def test_protocol_remains_draft_and_requires_input_identity_before_inference() -> None:
    protocol = _json(PROTOCOL)
    assert protocol["m7_mechanical_identity"]["required_exact_m7_xyzt_hash_matches_total"] == 1712
    assert protocol["m7_mechanical_identity"]["xyzt_identity_proven_now"] is False
    assert protocol["scientific_scope"]["zero_intensity_is_an_s2_arm"] is False
    assert (
        protocol["prospective_execution_after_separate_authorization"]["repeatability"][
            "accepted_calls"
        ]
        == 280
    )
    assert (
        protocol["prospective_execution_after_separate_authorization"]["full_corpus"][
            "accepted_calls"
        ]
        == 5136
    )
    assert (
        protocol["prospective_execution_after_separate_authorization"][
            "total_proposed_accepted_scientific_calls"
        ]
        == 5416
    )
    assert protocol["authorization"]["protocol_frozen"] is False
    assert protocol["authorization"]["s2_authorized"] is False
    assert protocol["authorization"]["b2_c2_d2_f2_calls"] == 0
    document = DOC.read_text(encoding="utf-8")
    assert "**DRAFT. NOT FROZEN. NO S2 INFERENCE AUTHORIZED.**" in document
    assert "1,712/1,712" in document
