"""Regression checks for the immutable M6a-R1 diagnostic chronology."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "benchmarks/m6a/diagnostics/pose_oracle_failure_ec9e341.json"
DIAGNOSIS = ROOT / "benchmarks/m6a/diagnostics/pose_oracle_diagnosis_ec9e341.json"
ORIGINAL_SHA256 = "894960c34b96aac02db2a8a10f13c66e39e3ac99b3ef34ee76c0ef3e88eabaf3"
DIAGNOSIS_SHA256 = "44509f4c28fafbdd848c2627c99cde4615bd8e6011520c2a371b1ee3ce6853d8"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record() -> dict[str, Any]:
    return json.loads(DIAGNOSIS.read_text(encoding="utf-8"))


def test_pose_oracle_diagnosis_preserves_original_failure() -> None:
    record = _record()

    assert _sha256(ORIGINAL) == ORIGINAL_SHA256
    assert _sha256(DIAGNOSIS) == DIAGNOSIS_SHA256
    assert record["status"] == "post_failure_diagnosis"
    assert record["canonical"] is False
    assert record["designed_after_original_failure"] is True
    assert record["original_failure"]["sha256"] == ORIGINAL_SHA256
    assert record["original_failure"]["status"] == "failed_gate"
    assert record["original_failure"]["unchanged_fail"] is True


def test_pose_oracle_diagnosis_records_data_product_timing_cause() -> None:
    record = _record()

    assert record["root_cause_classification"] == "DATA-PRODUCT / TIMING"
    assert record["sequence_mapping"]["mapped_count"] == 271
    assert record["sequence_mapping"]["index_offset"] == 0
    assert record["official_raw_devkit_oracle"]["pass"] is True
    assert record["official_raw_devkit_oracle"]["matrix_max_abs"] == 0.0
    assert record["composed_camera_frame_oracle"]["pass"] is True
    assert record["calibration_comparison"]["pass"] is True
    assert record["protocol_revision_recommendation"]["recommended"] is True
    assert record["protocol_revision_recommendation"]["original_tier_a_failure_preserved"] is True
    assert record["scope"] == {
        "canonical_m6a_evidence_generated": False,
        "detector_run": False,
        "m6b_started": False,
        "original_tolerances_relaxed": False,
        "production_adapter_modified": False,
        "ros_run": False,
        "tensorrt_initialized": False,
    }
