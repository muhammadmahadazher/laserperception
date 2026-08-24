from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "benchmarks/m6c/results/r3_detector_parity_and_ros_contract.json"


def _load() -> dict[str, object]:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def test_r3_detector_passes_unchanged_parity_v2_stage_one() -> None:
    result = _load()
    assert result["status"] == "PASS"
    parity = result["parity_v2"]
    assert parity["status"] == "PASS"
    stage_1 = parity["stage_1"]
    assert stage_1["overall_pass"] is True
    assert stage_1["failed_checks"] == []
    assert all(stage_1["checks"].values())
    assert stage_1["exported_counts"] == {
        "absolute_difference": 0,
        "per_sample_disagreements": [],
        "pytorch_total": 113,
        "relative_difference_from_pytorch": 0.0,
        "tensorrt_total": 113,
    }
    assert stage_1["high_confidence_match_denominator"] == 81
    assert stage_1["high_confidence_coverage"] == {
        "pytorch_matched": 81,
        "pytorch_to_tensorrt": 1.0,
        "pytorch_total": 81,
        "tensorrt_matched": 81,
        "tensorrt_to_pytorch": 1.0,
        "tensorrt_total": 81,
    }
    assert all(
        metric["accepted"] and metric["pass_count"] == 81 and metric["pass_fraction"] == 1.0
        for metric in stage_1["continuous_metrics"].values()
    )
    assert stage_1["full_heading_diagnostics"]["agreement_fraction"] == 1.0
    assert stage_1["class_name_mismatches"] == 0
    assert stage_1["distinct_high_confidence_continuous_outliers"]["count"] == 0
    assert parity["stage_2"] == {"required": False}


def test_r3_detector_inputs_and_ros_output_contract_are_exact() -> None:
    result = _load()
    records = result["conditions"]
    assert len(records) == 10
    assert all(record["projected_model_ready_exact"] is True for record in records)
    assert all(record["detection3darray_contract"]["status"] == "PASS" for record in records)
    assert result["detection3darray_contract"] == {
        "passed": 10,
        "required": 10,
        "status": "PASS",
        "velocity_exposed": False,
    }
    assert result["detector_node_counters"] == {
        "accepted": 10,
        "published": 10,
        "received": 10,
        "rejected": 0,
    }


def test_r3_detector_artifacts_and_parity_implementations_are_frozen() -> None:
    result = _load()
    assert result["frozen_artifacts"] == {
        "checkpoint_sha256": ("f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0"),
        "engine_sha256": ("2e790b1cdbdc1b88c2aafdc81b5921ebee152edd8408158f88437ae4dd1f3e7f"),
        "onnx_sha256": ("61ce22a8ca31498675c32576bfb94f0093d31dc95d2762f7254bf915a59ecc16"),
    }
    assert result["parity_v2"]["identities"] == {
        "config": "91e7cde19076c6452d9ff8e0fefc893a6d429622ed30c2da88127d29d4418df0",
        "matcher": "1be52b850ba5f41e1abf96e83923c1f4dbe65a5a2c592a4e6bb4185dc7e83c00",
        "sample_analyzer": ("37652e464a785174170240e99d593cd9d00a8362008537e182ad0e2b0a83d7f0"),
        "stage_1_evaluator": ("24fd8c7bcf8ee74049682ecd7d93989f4d62736eaeb35033155c0115281c38b4"),
    }


def test_r3_detector_evidence_is_compact_sanitized_and_byte_frozen() -> None:
    assert EVIDENCE.stat().st_size == 1_457_343
    assert hashlib.sha256(EVIDENCE.read_bytes()).hexdigest() == (
        "e415b3067b12a0c501ef4854e2fa7df9cbf809456ffd3dbd7979d2ce7b50177b"
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
    assert '"performance_campaign": false' in text
