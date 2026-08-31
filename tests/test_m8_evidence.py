from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_LEDGER = ROOT / "benchmarks/m8/diagnostics/m8_input_projection_ledger.json"
SOURCE_SMOKE = ROOT / "benchmarks/m8/diagnostics/dsvt_source_domain_smoke.json"
DEPLOYMENT_SMOKE = ROOT / "benchmarks/m8/diagnostics/dsvt_deployment_smoke.json"
H10_CENSUS = ROOT / "benchmarks/m8/diagnostics/m8_h10_capacity_census.json"
H10_CAPACITY_SMOKE = ROOT / "benchmarks/m8/diagnostics/dsvt_h10_capacity_smoke.json"
H10_DEPLOYMENT_SMOKE = ROOT / "benchmarks/m8/diagnostics/dsvt_h10_deployment_smoke.json"


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_m8_input_ledger_is_complete_exact_and_input_only() -> None:
    payload = _load(INPUT_LEDGER)

    assert _sha256(INPUT_LEDGER) == (
        "474e87e34c64d669750d4b6f7a64ac46fc9c5c462693fad79ff7c9547a7f1f7c"
    )
    assert payload["status"] == "m8_phase1_input_only_exact_pass"
    assert payload["detector_inference_performed"] is False
    assert payload["scientific_measurement_authorized"] is False
    assert payload["counts"] == {"H10": 428, "H5": 428}
    assert payload["exact_counts"] == {"H10": 428, "H5": 428}
    assert payload["total_conditions"] == 856
    assert payload["all_exact"] is True
    records = payload["records"]
    assert isinstance(records, list)
    assert len(records) == 856
    assert all(record["exact_equal"] for record in records)
    assert all(record["candidate_range_dropped_points"] == 0 for record in records)
    assert all(
        record["raw_intensity_sha256"] == record["candidate_consumed_intensity_sha256"]
        for record in records
    )


def test_source_smoke_retains_repeatability_and_resource_findings() -> None:
    payload = _load(SOURCE_SMOKE)

    assert _sha256(SOURCE_SMOKE) == (
        "aa3fcd58568efc5d428521b16c5aaad4bd662b0f27836682b375612160b1f983"
    )
    assert payload["status"] == "m8_phase1_source_domain_engineering_smoke_pass"
    assert payload["scientific_accuracy_measurement"] is False
    assert payload["source_domain_accuracy_rebenchmarked"] is False
    prediction = payload["prediction"]
    repeatability = payload["repeatability"]
    assert prediction["boxes_finite"] is True
    assert prediction["scores_finite"] is True
    assert prediction["class_ids_valid"] is True
    assert repeatability["repeats"] == 10
    assert repeatability["raw_tensors"]["pred_labels"]["exact_10_of_10"] is True
    assert repeatability["raw_tensors"]["pred_boxes"]["exact_10_of_10"] is False
    assert repeatability["detection_frame_exact_10_of_10"] is False
    assert payload["resources"]["peak_cuda_reserved_bytes"] < 8_188 * 1024 * 1024


def test_deployment_smoke_is_partial_external_and_has_no_latency_claim() -> None:
    payload = _load(DEPLOYMENT_SMOKE)

    assert _sha256(DEPLOYMENT_SMOKE) == (
        "15bc69b231a814f4f794f6f720d55cb1da669aa21fa9280db6c261929b34a8db"
    )
    assert payload["status"] == "m8_phase1_selected_config_partial_deployment_smoke_pass"
    assert payload["scientific_performance_measurement"] is False
    assert payload["engine"]["deserialized"] is True
    assert payload["engine"]["committed"] is False
    assert payload["onnx"]["committed"] is False
    assert payload["boundary"]["start"] == "after DynPillarVFE and DSVT InputLayer"
    assert "postprocess" in payload["boundary"]["excluded"]
    assert payload["reported_upstream_context_only"]["laserperception_measurement"] is False
    assert len(payload["builder_warnings_retained"]) == 4


def test_h10_capacity_census_is_complete_ordered_and_uncapped() -> None:
    payload = _load(H10_CENSUS)

    assert _sha256(H10_CENSUS) == (
        "c7d5da5a1b5162613cdc45dff420ae2d811ead2a6ba753b3202008bcb0ac86a6"
    )
    assert payload["detector_inference_performed"] is False
    assert payload["ground_truth_loaded"] is False
    assert payload["condition_count"] == 428
    assert payload["accepted_input_ledger"]["sha256"] == (
        "474e87e34c64d669750d4b6f7a64ac46fc9c5c462693fad79ff7c9547a7f1f7c"
    )
    assert payload["accepted_input_ledger"]["rewritten"] is False
    assert payload["candidate_contract"]["dynamic_pillar_count_cap"] is None
    assert payload["candidate_contract"]["coordinate_runtime"]["detector_or_model_loaded"] is False
    assert payload["summary"] == {
        "conditions_above_initial_source_shape_profile": 428,
        "conditions_affected_by_candidate_cap": 0,
        "max": 32774,
        "max_condition_id": "2011_09_26_drive_0091/0000000069/H10",
        "mean": 23291.03504672897,
        "median": 22858.5,
        "min": 14163,
    }
    records = payload["records"]
    assert len(records) == 428
    condition_ids = [record["condition_id"] for record in records]
    assert condition_ids == sorted(condition_ids)
    assert all(record["would_truncate"] is False for record in records)


def test_h10_full_model_smoke_retains_only_structural_output() -> None:
    payload = _load(H10_CAPACITY_SMOKE)

    assert _sha256(H10_CAPACITY_SMOKE) == (
        "fd8751778c7124ad66ee5e55e73de6df0b25174bf55daf89c5c18f5298a9c88b"
    )
    assert payload["status"].endswith("structural_smoke_pass")
    assert payload["ground_truth_loaded"] is False
    assert payload["accuracy_evaluation_performed"] is False
    assert payload["semantic_prediction_values_observed"] is False
    assert payload["prediction_values_serialized"] is False
    assert payload["candidate_dynamic_pillars"] == 32774
    assert payload["retained_pillars"] == 32774
    assert payload["discarded_or_truncated_pillars"] == 0
    assert (
        payload["capacity_semantics"]["occupied_coordinate_set_vs_actual_dynpillar_vfe_exact"]
        is True
    )
    assert payload["complete_model"]["output_completed"] is True
    assert payload["complete_model"]["prediction_values_discarded_immediately"] is True


def test_h10_partial_deployment_profile_executes_without_claiming_parity() -> None:
    payload = _load(H10_DEPLOYMENT_SMOKE)

    assert _sha256(H10_DEPLOYMENT_SMOKE) == (
        "ff737e72f668ff5b4ad9815df112d1d8714bc80ac38a8796eb3708d8c2a6771b"
    )
    assert payload["ground_truth_loaded"] is False
    assert payload["latency_claim"] is False
    assert payload["detector_parity_claim"] is False
    assert payload["existing_source_shape_engine"]["accepts_h10_shapes"] is False
    assert payload["h10_input_shapes"]["src"] == [32774, 128]
    assert payload["new_external_profile"]["src"] == {
        "min": [3687, 128],
        "opt": [32774, 128],
        "max": [32774, 128],
    }
    assert payload["engine"]["committed"] is False
    assert payload["engine"]["deserialized"] is True
    assert payload["h10_execution"] == {
        "attempted": True,
        "finite": True,
        "output_shape": [32774, 128],
        "passed": True,
    }
