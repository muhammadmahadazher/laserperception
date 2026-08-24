from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from benchmarks.m6c.generate_projected_reference_manifest import (
    _condition_plan,
    _population_statistics,
    _validate_sha_strings,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks/m6c/preregistration/projected_reference_manifest.json"
CHARACTERIZATION = (
    ROOT / "benchmarks/m6c/diagnostics/r3_projected_vs_original_characterization.json"
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _load(relative: str) -> dict[str, object]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_projected_manifest_plan_has_frozen_membership_cardinality() -> None:
    m6a = _load("benchmarks/m6a/results/kitti_raw_offline_reconstruction.json")
    m6b = _load("benchmarks/m6b/diagnostics/pre_inference_input_ledger.json")
    plan, original = _condition_plan(m6a, m6b)
    assert len(plan) == 860
    assert len(original) == 856
    assert sum("Gate1A" in record["gate_membership"] for record in plan) == 24
    assert sum("Gate1B" in record["gate_membership"] for record in plan) == 856
    assert sum(record["gate_membership"] == ["Gate1A", "Gate1B"] for record in plan) == 20
    assert len({str(record["key"]) for record in plan}) == 860


def test_projected_original_characterization_uses_signed_projected_delta() -> None:
    records = [
        {
            "key": "drive/0000000010|H10",
            "condition": "H10",
            "gate_membership": ["Gate1B"],
            "point_count": 11,
            "model_ready_sha256": "a" * 64,
        },
        {
            "key": "drive/0000000011|H10",
            "condition": "H10",
            "gate_membership": ["Gate1B"],
            "point_count": 8,
            "model_ready_sha256": "b" * 64,
        },
    ]
    original = {
        "drive/0000000010|H10": {
            "point_count": 10,
            "model_ready_input_sha256": "a" * 64,
        },
        "drive/0000000011|H10": {
            "point_count": 10,
            "model_ready_input_sha256": "c" * 64,
        },
    }
    result = _population_statistics(records, original, condition="H10")
    assert result == {
        "conditions_compared": 2,
        "model_ready_sha_identical": 1,
        "model_ready_sha_different": 1,
        "point_count_identical": 0,
        "point_count_different": 2,
        "signed_point_count_delta_min": -2,
        "signed_point_count_delta_max": 1,
        "maximum_absolute_point_count_delta": 2,
        "median_absolute_nonzero_point_count_delta": 1.5,
        "total_projected_points": 19,
        "total_original_points": 20,
    }


def test_sha_validation_rejects_malformed_protocol_identity() -> None:
    _validate_sha_strings({"model_ready_sha256": "a" * 64})
    with pytest.raises(RuntimeError, match="invalid SHA256"):
        _validate_sha_strings({"model_ready_sha256": "a" * 63})


def test_frozen_projected_manifest_contains_860_unique_offline_identities() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert MANIFEST.stat().st_size < 1_000_000
    assert manifest["status"] == "FROZEN_PROJECTED_REFERENCE_IDENTITIES_BEFORE_LIVE_R3"
    assert manifest["population"] == {
        "gate_1a_memberships": 24,
        "gate_1b_memberships": 856,
        "overlap_memberships": 20,
        "unique_conditions": 860,
    }
    conditions = manifest["conditions"]
    assert len(conditions) == 860
    assert len({record["key"] for record in conditions}) == 860
    for record in conditions:
        frame = int(record["frame"])
        requested = int(record["requested_history_depth"])
        assert record["expected_actual_history_depth"] == min(frame, requested)
        assert record["shape"] == [record["point_count"], 4]
        assert record["dtype"] == "float32"
        assert SHA256_PATTERN.fullmatch(record["model_ready_sha256"])
        assert SHA256_PATTERN.fullmatch(record["projected_transforms_sha256"])
    assert manifest["independence"] == {
        "detector_executed": False,
        "gpu_initialized": False,
        "live_builder_node_used": False,
        "ros_initialized": False,
        "ros_messages_used": False,
        "tf2_used": False,
    }


def test_original_projected_characterization_is_complete_and_descriptive() -> None:
    evidence = json.loads(CHARACTERIZATION.read_text(encoding="utf-8"))
    assert evidence["status"] == "DESCRIPTIVE_ONLY_NOT_A_GATE"
    assert evidence["scope"] == {
        "detector_executed": False,
        "gpu_initialized": False,
        "point_count_equality_is_gate": False,
        "ros_outputs_observed": False,
    }
    expected_counts = {"H10": 428, "H5": 428, "total": 856}
    expected_totals = {
        "H10": 569_520_061,
        "H5": 310_967_933,
        "total": 880_487_994,
    }
    for name, count in expected_counts.items():
        population = evidence["populations"][name]
        assert population["conditions_compared"] == count
        assert population["model_ready_sha_identical"] == 0
        assert population["model_ready_sha_different"] == count
        assert population["point_count_identical"] == count
        assert population["point_count_different"] == 0
        assert population["signed_point_count_delta_min"] == 0
        assert population["signed_point_count_delta_max"] == 0
        assert population["maximum_absolute_point_count_delta"] == 0
        assert population["median_absolute_nonzero_point_count_delta"] is None
        assert population["total_original_points"] == expected_totals[name]
        assert population["total_projected_points"] == expected_totals[name]


def test_projected_preregistration_has_no_private_paths_or_raw_arrays() -> None:
    for path in (MANIFEST, CHARACTERIZATION):
        text = path.read_text(encoding="utf-8").lower()
        for forbidden in ("j:\\", "c:\\users", "/root/", ".local/", '"points_xyzt"'):
            assert forbidden not in text
