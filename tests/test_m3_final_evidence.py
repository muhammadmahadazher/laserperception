"""Validate the canonical final M3 evidence without GPU dependencies."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

EXPECTED_COMMIT = "a129b3507597b25f44ab1a833562f68883ebe8ce"
EXPECTED_ARTIFACTS = {
    "checkpoint": "f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0",
    "onnx": "61ce22a8ca31498675c32576bfb94f0093d31dc95d2762f7254bf915a59ecc16",
    "engine": "a005f75852097cd9b193750560b214cc3d5237ae9b6c106c7fca3d4fc348714b",
}
RESULT = (
    Path(__file__).parents[1] / "benchmarks/m3/results/rtx4060_ros2_humble_exact_tensorrt_fp16.json"
)


def _record() -> dict[str, Any]:
    return dict(json.loads(RESULT.read_text(encoding="utf-8")))


def test_final_m3_result_is_canonical_exact_commit_evidence() -> None:
    record = _record()

    assert record["schema_version"] == "1.0"
    assert record["milestone"] == "M3"
    assert record["canonical"] is True
    assert record["status"] == "complete_representative_20_hz_not_sustained"
    assert record["measurement_commit"] == EXPECTED_COMMIT
    assert record["conclusion"] == {
        "m3_complete": True,
        "representative_20_hz_operation_demonstrated": False,
        "highest_tested_sustainable_rate_hz": 10.0,
        "summary": (
            "The exact production path preserved all correctness gates. W1 did not sustain "
            "20 Hz; bounded characterization sustained 10 Hz and did not sustain 15 Hz."
        ),
    }
    assert record["artifacts"] == EXPECTED_ARTIFACTS
    assert record["production_policy"] == {
        "voxelization_mode": "exact_fast",
        "provenance_mode": "live",
        "fallback_allowed": False,
    }


def test_final_m3_correctness_gates_are_exact() -> None:
    correctness = _record()["correctness"]

    assert correctness["status"] == "pass"
    assert correctness["implementation_commit"] == EXPECTED_COMMIT
    assert correctness["voxelization_mode"] == "exact_fast"
    assert correctness["provenance_mode"] == "live"
    assert correctness["source_external_record"]["sha256"] == (
        "000ba4bd15bc4349a0df29a2252819e00326c406e5b1dc0e787c0c060359d388"
    )
    assert correctness["official_vs_exact_fast_voxel_gate"] == {
        "required_sample_count": 81,
        "completed_sample_count": 81,
        "passed": True,
        "all_samples_exact": True,
        "tensors": ["voxels", "num_points", "coors"],
    }
    detector = correctness["frozen_detector_and_ros_gate"]
    assert detector["sample_count"] == 20
    assert detector["passed"] is True
    assert detector["point_contract"] == ["x", "y", "z", "time_lag"]
    for key, value in detector.items():
        if key.startswith("all_"):
            assert value is True

    assert correctness["low_rate_w1_ros_smoke"] == {
        "source_external_record": {
            "logical_name": "ros_w1_smoke_a129b35.json",
            "sha256": "a0eb5516144529e0bafc398e361dd0d94ad90d8e802e7c34a3bc3ef74077121c",
        },
        "status": "pass",
        "point_count": 354182,
        "published_input": 1,
        "accepted_callbacks": 1,
        "published_detections": 1,
        "sink_received": 1,
        "rejected": 0,
    }


def test_final_m3_twenty_hz_failure_is_reported_honestly() -> None:
    result = _record()["representative_20_hz"]

    assert result["source_external_record"]["sha256"] == (
        "1bc77d7cbbdc6151b2a9c17815528d7bbedd10cee49c458f60336778f23b3046"
    )
    assert result["status"] == "representative_20_hz_not_sustained"
    assert result["callback_processing_latency"]["count"] == 200
    assert math.isclose(result["callback_processing_latency"]["median_ms"], 75.70065)
    assert result["same_host_ros_loopback_latency"]["count"] == 200
    assert math.isclose(result["same_host_ros_loopback_latency"]["median_ms"], 134.2503965)
    rate = result["sustained_rate"]
    assert math.isclose(rate["requested_offered_hz"], 20.0)
    assert math.isclose(rate["effective_detector_output_hz"], 10.825441119694597)
    assert rate["measured_input_drops"] == 159
    assert rate["first_half_input_drops"] == 77
    assert rate["second_half_input_drops"] == 82
    assert rate["falling_behind_between_halves"] is True
    assert rate["sustainable_at_offered_rate"] is False
    assert result["gate"]["twenty_hz_operation_demonstrated"] is False
    assert result["measurement_session"]["eligibility"]["eligible"] is True


def test_final_m3_bounded_characterization_stops_after_two_rates() -> None:
    characterization = _record()["bounded_sustainable_rate_characterization"]

    assert characterization["tested_rate_count"] == 2
    assert characterization["highest_tested_sustainable_rate_hz"] == 10.0
    ten = characterization["ten_hz"]
    assert ten["source_external_record"]["sha256"] == (
        "39e260bc67405346b2e252ad8b15d8e8e93c861c4884cbf78b19d64095e3c30d"
    )
    assert ten["status"] == "bounded_characterization_rate_sustained"
    assert ten["sustained_rate"]["measured_input_drops"] == 0
    assert ten["sustained_rate"]["falling_behind_between_halves"] is False
    assert ten["sustained_rate"]["sustainable_at_offered_rate"] is True

    fifteen = characterization["fifteen_hz"]
    assert fifteen["source_external_record"]["sha256"] == (
        "92d59b01d0888c713080dfa461ed9548c423196e693fb28b1b55eb30ff0a6c52"
    )
    assert fifteen["status"] == "bounded_characterization_rate_not_sustained"
    assert fifteen["sustained_rate"]["measured_input_drops"] == 21
    assert fifteen["sustained_rate"]["sustainable_at_offered_rate"] is False


def test_final_m3_result_is_sanitized() -> None:
    serialized = RESULT.read_text(encoding="utf-8").lower()

    for marker in ("/root/", "/home/", "\\users\\", "my drive", "\\wsl"):
        assert marker not in serialized
