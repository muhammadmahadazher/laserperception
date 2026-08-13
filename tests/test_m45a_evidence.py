from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "benchmarks/m45a/results/nuscenes_mini_multisweep_parity.json"


def test_m45a_evidence_records_exact_complete_gates() -> None:
    record = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert record["schema_version"] == 1
    assert record["status"] == "M4.5a EXACT PARITY PASS"
    assert record["candidate_commit"] == "cc0f20b16412d98939c9544002d02029b35a5971"

    tier_a = record["tier_a"]
    assert tier_a["required_samples"] == 81
    assert tier_a["completed_samples"] == 81
    assert tier_a["exact_samples"] == 81
    assert tier_a["scene_start_samples"] == 2
    assert tier_a["full_history_samples"] == 79
    assert tier_a["passed"] is True
    assert len(tier_a["samples"]) == 81
    assert all(sample["exact"] is True for sample in tier_a["samples"])
    assert all(
        sample["official"]["sha256"] == sample["candidate"]["sha256"]
        for sample in tier_a["samples"]
    )
    assert record["first_failure"] is None
    assert record["tier_b"]["status"] == "not_run_tier_a_passed"

    detector = record["detector_verification"]
    assert detector["sample_count"] == 20
    assert detector["passed"] is True
    assert detector["all_voxel_tensors_exact"] is True
    assert detector["all_raw_tensorrt_outputs_exact"] is True
    assert detector["all_detection_frames_exact"] is True
    assert len(detector["samples"]) == 20


def test_m45a_evidence_is_sanitized_and_preserves_known_workload_hashes() -> None:
    text = EVIDENCE.read_text(encoding="utf-8")
    assert "C:\\Users" not in text
    assert "J:\\" not in text
    assert "/root/" not in text
    assert "My Drive" not in text

    record = json.loads(text)
    assert record["tier_a"]["samples"][0]["candidate"]["sha256"] == (
        "4da6843d2f4fcca676705ecd440047e0d0371efa53ee8d4bed305c72d8e1def4"
    )
    assert record["tier_a"]["samples"][42]["candidate"]["sha256"] == (
        "5c1f5590ce7925d9312fe9ef755d0d849f579ff3d7ae842af75d41a0252cb29a"
    )


def test_m45a_scope_guards_and_artifacts_remain_frozen() -> None:
    record = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    guards = record["scope_guards"]
    assert guards == {
        "engine_changed": False,
        "exact_fast_changed": False,
        "model_changed": False,
        "onnx_changed": False,
        "production_builder_calls_mmdetection3d": False,
        "ros_implemented": False,
        "thresholds_changed": False,
        "voxel_geometry_changed": False,
    }
    assert record["artifacts"]["checkpoint"]["sha256"] == (
        "f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0"
    )
    assert record["artifacts"]["onnx"]["sha256"] == (
        "61ce22a8ca31498675c32576bfb94f0093d31dc95d2762f7254bf915a59ecc16"
    )
    assert record["artifacts"]["tensorrt_engine"]["sha256"] == (
        "a005f75852097cd9b193750560b214cc3d5237ae9b6c106c7fca3d4fc348714b"
    )
