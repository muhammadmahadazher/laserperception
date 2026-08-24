from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.m6c.generate_projected_reference_manifest import (
    _condition_plan,
    _population_statistics,
    _validate_sha_strings,
)

ROOT = Path(__file__).resolve().parents[1]


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
