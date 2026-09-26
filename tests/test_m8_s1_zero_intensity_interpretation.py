from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "benchmarks/m8/results/m8_s1_zero_intensity_interpretation.json"
DOCUMENT = ROOT / "docs/m8/M8_S1_ZERO_INTENSITY_INTERPRETATION.md"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _tracked_sha256(relative: str) -> str:
    # Git blob identity avoids platform-specific checkout line endings in Markdown.
    data = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    return hashlib.sha256(data).hexdigest()


def test_evidence_binds_exact_frozen_git_blobs() -> None:
    evidence = _record()["evidence"]
    expected = {
        "protocol_sha256": "benchmarks/m8/preregistration/m8_s1_protocol.json",
        "primary_raw_sha256": "benchmarks/m8/results/m8_s1_primary_raw.json",
        "primary_secondary_sha256": "benchmarks/m8/results/m8_s1_secondary_raw.json",
        "primary_manifest_sha256": "benchmarks/m8/results/m8_s1_measurement_manifest.json",
        "zeroi_raw_sha256": "benchmarks/m8/results/m8_s1_zero_intensity_raw.json",
        "zeroi_secondary_sha256": "benchmarks/m8/results/m8_s1_zero_intensity_secondary.json",
        "zeroi_comparison_sha256": "benchmarks/m8/results/m8_s1_zero_intensity_comparison.json",
        "zeroi_manifest_sha256": "benchmarks/m8/results/m8_s1_zero_intensity_manifest.json",
        "zeroi_raw_narrative_sha256": "docs/m8/M8_S1_ZERO_INTENSITY_RAW.md",
    }
    for key, path in expected.items():
        assert evidence[key] == _tracked_sha256(path)
    assert evidence["primary_execution_commit"] == "6994d72c3e7691a86116d1417ac3ae08256d163f"
    assert evidence["zeroi_execution_commit"] == "95fb66ac1f57c41f06f05bd9ef5dac27b1e3ea54"
    assert evidence["primary_accepted_processes"] == evidence["zeroi_accepted_processes"] == 3
    assert evidence["primary_accepted_calls"] == evidence["zeroi_accepted_calls"] == 2568


def test_observations_and_nonpaired_comparison() -> None:
    record = _record()
    observations = record["observations"]
    car = observations["car"]
    ped = observations["pedestrian"]
    assert (car["primary_h10"]["tp"], car["zeroi_h10"]["tp"]) == (19, 17)
    assert (car["primary_h5"]["tp"], car["zeroi_h5"]["tp"]) == (43, 43)
    assert all(car[arm]["gt"] == 66 for arm in car)
    assert (ped["primary_h10"]["tp"], ped["zeroi_h10"]["tp"]) == (0, 0)
    assert (ped["primary_h5"]["tp"], ped["zeroi_h5"]["tp"]) == (1, 0)
    assert all(ped[arm]["gt"] == 396 for arm in ped)
    within = record["within_history"]
    assert within["primary_h5_minus_h10_car_tp"] == 24
    assert within["zeroi_h5_minus_h10_car_tp"] == 26
    assert math.isclose(within["primary_h5_minus_h10_car_recall"], 24 / 66)
    assert math.isclose(within["zeroi_h5_minus_h10_car_recall"], 26 / 66)
    assert within["h10_h5_paired_within_each_process"] is True
    comparison = record["primary_vs_zeroi"]
    assert comparison["process_indices_paired"] is False
    assert math.isclose(
        comparison["median_recall_difference_zeroi_minus_primary"]["h10_car"], -2 / 66
    )
    assert comparison["median_recall_difference_zeroi_minus_primary"]["h5_car"] == 0
    assert comparison["median_recall_difference_zeroi_minus_primary"]["h10_pedestrian"] == 0
    assert math.isclose(
        comparison["median_recall_difference_zeroi_minus_primary"]["h5_pedestrian"], -1 / 396
    )


def test_range_ap_and_claim_boundaries() -> None:
    record = _record()
    observations = record["observations"]
    bands = observations["range_tp"]
    assert (bands["0_20_m"]["primary_h10_car"], bands["0_20_m"]["zeroi_h10_car"]) == (5, 3)
    assert (bands["20_35_m"]["primary_h10_car"], bands["20_35_m"]["zeroi_h10_car"]) == (13, 13)
    assert (bands["35_50_m"]["primary_h10_car"], bands["35_50_m"]["zeroi_h10_car"]) == (1, 1)
    for band in bands.values():
        if isinstance(band, dict):
            assert band["primary_h5_car"] == band["zeroi_h5_car"]
    assert bands["zeroi_pedestrian_recall_zero_in_all_bands"] is True
    ap = observations["annotation_conditioned_ap_iou_0_50"]
    assert all(len(values) == 3 for values in ap.values())
    assert all(
        max(ap[f"zeroi_{arm}_pedestrian"]) < min(ap[f"primary_{arm}_pedestrian"])
        for arm in ("h10", "h5")
    )
    assert record["scientific_interpretation"]["pedestrian_recovery_observed"] is False
    assert record["scientific_interpretation"]["car_h5_direction_persisted_under_zeroi"] is True
    assert (
        record["scientific_interpretation"]["raw_intensity_sufficient_explanation_supported"]
        is False
    )
    assert not any(record["claim_boundaries"].values())
    assert record["reproducibility"]["ap_numerical_spread_present"] is True
    next_science = record["next_science"]
    assert next_science["zeroi_interpretation_complete"] is True
    assert next_science["s1_interpretation_complete"] is True
    assert next_science["s2_design_can_begin"] is True
    assert next_science["s2_ready_to_execute"] is False
    assert next_science["s2_authorized"] is False
    assert next_science["b2_c2_d2_f2_calls"] == 0
    record_text = RECORD.read_text(encoding="utf-8").lower()
    assert not any(
        term in record_text for term in ("p_value", "confidence_interval", "standard_error")
    )
    document = DOCUMENT.read_text(encoding="utf-8")
    assert "Primary and zero-intensity process indices are not paired" in document
    assert "**not ready or authorized to execute**" in document
    assert "**does not rule out**" in document
    assert "all-zero replacement can itself remain outside the training distribution" in document
