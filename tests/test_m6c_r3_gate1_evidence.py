from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "benchmarks/m6c/results/r3_projected_ros_input_exactness.json"


def _load() -> dict[str, object]:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def test_r3_gate1_passes_every_frozen_membership_and_unique_condition() -> None:
    result = _load()
    assert result["status"] == "PASS"
    assert result["scientific_classification"] == "GATE_1_PROJECTED_ROS_INPUT_EXACTNESS_PASS"
    assert result["protocol_commit"] == "07c4ba293c3d0efbf01c7efb18d389a67828c3fc"
    assert result["measurement_implementation_commit"] == "28d81f3f9d4a5ce92d2dde7b3a6635c5079d1f4b"
    assert result["projected_manifest_sha256"] == (
        "c06cddc6884fef87de99d1c68ec2b5c1f1945f7f9e5ecae6fcb3e4275dd952a2"
    )
    assert result["gate_1a"] == {
        "exact": True,
        "required": 24,
        "totals": {"fail": 0, "pass": 24, "pending": 0},
    }
    assert result["gate_1b"] == {
        "exact": True,
        "required": 856,
        "totals": {"fail": 0, "pass": 856, "pending": 0},
    }
    assert result["unique_conditions"] == {
        "exact": True,
        "required": 860,
        "totals": {"fail": 0, "pass": 860, "pending": 0},
    }
    assert result["failure"] is None


def test_r3_gate1_condition_ledger_is_exact_and_nonduplicated() -> None:
    result = _load()
    conditions = result["conditions"]
    assert len(conditions) == 860
    assert len({record["key"] for record in conditions}) == 860
    assert all(record["status"] == "PASS" for record in conditions)
    assert all(record["expected_sha256"] == record["observed_sha256"] for record in conditions)
    assert sum("Gate1A" in record["gate_membership"] for record in conditions) == 24
    assert sum("Gate1B" in record["gate_membership"] for record in conditions) == 856
    assert sum(record["gate_membership"] == ["Gate1A", "Gate1B"] for record in conditions) == 20
    assert result["overlap"] == {
        "gate_1a_gate_1b_shared_conditions": 20,
        "redundantly_replayed": False,
    }


def test_r3_gate1_ros_counters_are_complete_and_clean() -> None:
    result = _load()
    assert result["ros_counters_this_invocation"] == {
        "history_resets": 0,
        "invalid_points_filtered": 0,
        "model_ready_outputs": 886,
        "raw_frames_received": 886,
        "rejected_frames": 0,
        "tf_failures": 0,
        "valid_raw_frames": 886,
    }
    sessions = result["sessions_this_invocation"]
    assert len(sessions) == 4
    assert sum(session["newly_verified_conditions"] for session in sessions) == 860
    assert sum(session["published_raw_frames"] for session in sessions) == 886


def test_r3_gate1_evidence_is_compact_sanitized_and_byte_frozen() -> None:
    assert EVIDENCE.stat().st_size == 379_364
    assert hashlib.sha256(EVIDENCE.read_bytes()).hexdigest() == (
        "a84a501fd7c5a48fc5421c8c507102b2b8a02aea2266b8fe0cbf18a3f3874549"
    )
    text = EVIDENCE.read_text(encoding="utf-8").lower()
    for forbidden in (
        "j:\\",
        "c:\\users",
        "/root/",
        ".local/",
        '"points_xyzt"',
        '"raw_tensor_values"',
    ):
        assert forbidden not in text
