from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _json(relative: str) -> dict[str, object]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    tracked_bytes = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    return hashlib.sha256(tracked_bytes).hexdigest()


def test_m8_s1_interpretation_is_bound_to_raw_evidence() -> None:
    interpretation = _json("benchmarks/m8/results/m8_s1_interpretation.json")
    evidence = interpretation["evidence"]
    assert isinstance(evidence, dict)
    expected = {
        "primary_raw_sha256": "benchmarks/m8/results/m8_s1_primary_raw.json",
        "secondary_raw_sha256": "benchmarks/m8/results/m8_s1_secondary_raw.json",
        "measurement_manifest_sha256": "benchmarks/m8/results/m8_s1_measurement_manifest.json",
        "stage_r_sha256": "benchmarks/m8/results/m8_s1_stage_r.json",
        "raw_narrative_sha256": "docs/m8/M8_S1_MEASUREMENT_RAW.md",
        "protocol_sha256": "benchmarks/m8/preregistration/m8_s1_protocol.json",
        "historical_pointpillars_sha256": (
            "benchmarks/m6b/results/kitti_raw_cross_domain_characterization.json"
        ),
    }
    for key, relative in expected.items():
        assert evidence[key] == _sha256(relative)
    assert evidence["execution_commit"] == "6994d72c3e7691a86116d1417ac3ae08256d163f"
    assert evidence["accepted_processes"] == 3
    assert evidence["accepted_canonical_calls"] == 2568


def test_m8_s1_interpretation_preserves_observations_and_boundaries() -> None:
    value = _json("benchmarks/m8/results/m8_s1_interpretation.json")
    observations = value["observations"]
    assert isinstance(observations, dict)
    car = observations["car"]
    pedestrian = observations["pedestrian"]
    assert isinstance(car, dict) and isinstance(pedestrian, dict)
    assert car["A2_H10"]["true_positives"] == 19
    assert car["E2_H5"]["true_positives"] == 43
    assert math.isclose(car["paired_recall_delta"], 24 / 66)
    assert car["direction_criterion_satisfied"] is True
    assert car["range_tp"]["0_20_m"] == {"A2_H10": 5, "E2_H5": 22, "ground_truth": 27}
    assert car["range_tp"]["20_35_m"] == {"A2_H10": 13, "E2_H5": 20, "ground_truth": 30}
    assert car["range_tp"]["35_50_m"] == {"A2_H10": 1, "E2_H5": 1, "ground_truth": 9}
    assert pedestrian["A2_H10"]["true_positives"] == 0
    assert pedestrian["E2_H5"]["true_positives"] == 1

    reproducibility = value["reproducibility"]
    assert isinstance(reproducibility, dict)
    assert reproducibility["discrete_three_process_results_identical"] is True
    assert reproducibility["ap_numerical_spread_present"] is True

    boundaries = value["claim_boundaries"]
    assert isinstance(boundaries, dict)
    assert not any(boundaries.values())
    assert value["zero_intensity"]["recommended_for_owner_authorization"] is True
    assert value["zero_intensity"]["causality_established"] is False
    assert value["s2"]["ready_now"] is False
    assert value["safety"] == {
        "local_gpu_probe_commands_executed": 0,
        "local_gpu_touched": False,
        "new_detector_calls": 0,
        "runpod_actions": 0,
        "runpod_resources_created": False,
        "s2_calls": 0,
        "training_runs": 0,
        "zero_intensity_calls": 0,
    }


def test_m8_s1_interpretation_document_keeps_required_scientific_language() -> None:
    text = (ROOT / "docs/m8/M8_S1_INTERPRETATION.md").read_text(encoding="utf-8")
    assert "frozen detector-stack comparison" in text
    assert "ZERO-INTENSITY INTERVENTION RECOMMENDED FOR OWNER AUTHORIZATION" in text
    assert "S2 is not ready now" in text
    assert "not independent dataset replicates" in text
    assert "neither whole-world physical false-positive precision nor" in text
    assert "No p-values or confidence intervals" in text
