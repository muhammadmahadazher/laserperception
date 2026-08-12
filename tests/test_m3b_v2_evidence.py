"""Validate the committed M3B-V2 diagnostic evidence without GPU dependencies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EXPECTED_COMMIT = "85b6488c92eda266f049ff142fc06bdab658d7ed"
EXPECTED_ARTIFACT_HASHES = {
    "checkpoint": "f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0",
    "onnx": "61ce22a8ca31498675c32576bfb94f0093d31dc95d2762f7254bf915a59ecc16",
    "engine": "a005f75852097cd9b193750560b214cc3d5237ae9b6c106c7fca3d4fc348714b",
}
EVIDENCE = (
    Path(__file__).parents[1]
    / "benchmarks/m3/diagnostics/deterministic_voxelization_v2_85b6488.json"
)


def _record() -> dict[str, Any]:
    return dict(json.loads(EVIDENCE.read_text(encoding="utf-8")))


def test_v2_evidence_is_exact_commit_diagnostic_only() -> None:
    record = _record()

    assert record["status"] == "diagnostic_measurement_not_production"
    assert record["publication_role"] == "diagnostic_evidence_not_canonical_performance"
    assert record["measurement_commit"] == EXPECTED_COMMIT
    assert record["protocol"]["status"] == "protocol_frozen_before_measurement"
    assert {
        name: record["artifacts"][name]["sha256"] for name in EXPECTED_ARTIFACT_HASHES
    } == EXPECTED_ARTIFACT_HASHES
    assert all(value is False for value in record["scope_guards"].values())


def test_v2_all_81_voxel_outputs_are_bit_exact() -> None:
    exact = _record()["exact_voxel_fidelity"]

    assert exact["passed"] is True
    assert exact["required_sample_count"] == 81
    assert exact["completed_sample_count"] == 81
    assert exact["first_mismatch"] is None
    assert [sample["sample_index"] for sample in exact["samples"]] == list(range(81))
    for sample in exact["samples"]:
        comparison = sample["comparison"]
        assert comparison["exact"] is True
        assert comparison["voxel_count_exact"] is True
        assert comparison["first_mismatch"] is None
        assert comparison["zero_filled_unused_slots"] == {
            "reference": True,
            "candidate": True,
        }
        for tensor in comparison["tensors"].values():
            assert tensor["exact"] is True
            assert tensor["reference"] == tensor["candidate"]


def test_v2_repeatability_and_detector_gates_pass_exactly() -> None:
    record = _record()
    repeatability = record["repeatability"]

    assert repeatability["passed"] is True
    assert set(repeatability["samples"]) == {"42", "49"}
    for sample in repeatability["samples"].values():
        assert sample["runs"] == 30
        assert sample["candidate_inputs_exact"] is True
        assert sample["first_input_mismatch"] is None
        assert sample["raw_tensorrt_outputs_repeatable"] is True
        assert len(sample["candidate_input_hashes"]) == 30
        assert len(sample["raw_tensorrt_output_hashes"]) == 30
        assert all(
            hashes == sample["reference_input_hashes"]
            for hashes in sample["candidate_input_hashes"]
        )

    detector = record["detector_fidelity"]
    assert detector["passed"] is True
    assert detector["all_raw_outputs_exact"] is True
    assert detector["all_final_detection_frames_exact"] is True
    assert len(detector["samples"]) == 20
    assert all(sample["raw_tensorrt_outputs_exact"] is True for sample in detector["samples"])
    assert all(sample["final_detection_frame_exact"] is True for sample in detector["samples"])


def test_v2_performance_has_passing_prerequisites_and_eligible_telemetry() -> None:
    performance = _record()["performance"]

    assert performance["status"] == "diagnostic_measurement_not_production"
    assert performance["correctness_prerequisites"] == {
        "exact_voxel_fidelity_passed": True,
        "repeatability_passed": True,
        "detector_fidelity_passed": True,
        "measurement_commit": EXPECTED_COMMIT,
    }
    assert performance["sustained_gpu_warmup"]["passed"] is True
    eligibility = performance["measurement_session"]["eligibility"]
    assert eligibility["eligible"] is True
    assert eligibility["rejection_reasons"] == []
    assert len(eligibility["pairs"]) == 12
    assert all(pair["assessable"] is True for pair in eligibility["pairs"])
    assert all(pair["obvious_material_state_mismatch"] is False for pair in eligibility["pairs"])
    telemetry = performance["measurement_session"]["telemetry"]["summary"]
    assert telemetry["available_sample_count"] == telemetry["sample_count"]
    assert telemetry["sample_count"] > 0


def test_v2_timings_and_live_provenance_are_complete() -> None:
    workloads = _record()["performance"]["workloads"]

    assert set(workloads) == {"W0", "W1", "W2"}
    for name, workload in workloads.items():
        for boundary in (
            workload["hard_voxel_layer_synchronized_wall_ms"],
            workload["complete_preprocessing_synchronized_wall_ms"],
        ):
            assert boundary["reference"]["count"] == 100
            assert boundary["candidate"]["count"] == 100
            assert boundary["candidate_speedup"] > 1.0
        for mode in ("full", "live"):
            direct = workload["direct_tensorrt_e2e_synchronized_wall_ms"][mode]
            assert direct["reference"]["count"] == 100
            assert direct["candidate"]["count"] == 100
            assert direct["reference_candidate_detection_values_exact"] is True
        assert workload["candidate_full_live_detection_values_exact"] is True
        assert workload["reference_full_live_detection_values_exact"] is True
        if name in {"W1", "W2"}:
            assert set(workload["candidate_component_ledger"]) == {"full", "live"}
            for ledger in workload["candidate_component_ledger"].values():
                assert ledger["total_e2e_synchronized_wall_ms"]["count"] == 100
