from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_LEDGER = ROOT / "benchmarks/m8/diagnostics/m8_input_projection_ledger.json"
SOURCE_SMOKE = ROOT / "benchmarks/m8/diagnostics/dsvt_source_domain_smoke.json"
DEPLOYMENT_SMOKE = ROOT / "benchmarks/m8/diagnostics/dsvt_deployment_smoke.json"


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
