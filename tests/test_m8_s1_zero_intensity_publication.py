"""CPU-only guards for the zero-intensity publication boundary."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.m8 import aggregate_s1_zero_intensity as zeroi

ROOT = Path(__file__).resolve().parents[1]


def _evidence():
    mapping = []
    for frame in range(428):
        for history in ("H10", "H5"):
            condition_id = f"drive/{frame:010d}/{history}"
            mapping.append(
                {
                    "condition_id": condition_id,
                    "primary_input_sha256": "a" * 64,
                    "zeroI_input_sha256": "b" * 64,
                }
            )
    passes = []
    for number in range(1, 4):
        conditions = [
            {
                "frame_id": row["condition_id"].rsplit("/", 1)[0],
                "history": row["condition_id"].rsplit("/", 1)[1],
                "primary_input_sha256": row["primary_input_sha256"],
                "input_sha256": row["zeroI_input_sha256"],
                "intervention": zeroi.INTERVENTION,
            }
            for row in mapping
        ]
        raw = {
            "status": "COMPLETE",
            "runtime_commit": zeroi.ZEROI_COMMIT,
            "logical_pass_id": f"zero-intensity-pass-{number}",
            "attempt_id": f"attempt-{number}",
            "process_uuid": f"00000000-0000-0000-0000-{number:012d}",
            "conditions": conditions,
        }
        raw["result_sha256"] = zeroi.canonical_sha(raw)
        passes.append(raw)
    return passes, {"zeroi_execution_commit": zeroi.ZEROI_COMMIT, "conditions": mapping}


def _reseal(raw):
    raw["result_sha256"] = zeroi.canonical_sha(
        {key: value for key, value in raw.items() if key != "result_sha256"}
    )


def test_zeroi_requires_three_distinct_complete_856_condition_processes():
    passes, ledger = _evidence()
    assert len(zeroi.validate_passes(passes, ledger)) == 856
    with pytest.raises(ValueError, match="exactly three"):
        zeroi.validate_passes(passes[:2], ledger)
    duplicate = deepcopy(passes)
    duplicate[2]["process_uuid"] = duplicate[1]["process_uuid"]
    _reseal(duplicate[2])
    with pytest.raises(ValueError, match="distinct"):
        zeroi.validate_passes(duplicate, ledger)
    incomplete = deepcopy(passes)
    incomplete[0]["status"] = "INCOMPLETE"
    _reseal(incomplete[0])
    with pytest.raises(ValueError, match="status"):
        zeroi.validate_passes(incomplete, ledger)
    short = deepcopy(passes)
    short[0]["conditions"].pop()
    _reseal(short[0])
    with pytest.raises(ValueError, match="856"):
        zeroi.validate_passes(short, ledger)
    wrong_history = deepcopy(passes)
    wrong_history[0]["conditions"][0]["history"] = "H5"
    _reseal(wrong_history[0])
    with pytest.raises(ValueError, match="428"):
        zeroi.validate_passes(wrong_history, ledger)


@pytest.mark.parametrize("field", ["primary_input_sha256", "input_sha256", "intervention"])
def test_zeroi_rejects_missing_or_changed_provenance(field):
    passes, ledger = _evidence()
    del passes[1]["conditions"][200][field]
    _reseal(passes[1])
    with pytest.raises(ValueError, match="provenance"):
        zeroi.validate_passes(passes, ledger)


def test_primary_comparison_uses_separate_distributions_without_pairing():
    def record(arm, car_recall, pedestrian_recall):
        return {
            "arms": {
                arm: {
                    "classes": {
                        "car": {"thresholds": {"0.50": {"recall": car_recall}}},
                        "pedestrian": {"thresholds": {"0.50": {"recall": pedestrian_recall}}},
                    }
                }
            }
        }

    primary = {
        "schema_version": "laserperception.m8.s1.primary-raw.v1",
        "passes": [
            {
                "process_uuid": uuid,
                "arms": {**record("A2", x, 0)["arms"], **record("E2", x + 0.1, 0.1)["arms"]},
            }
            for x, uuid in zip(
                (0.1, 0.2, 0.3),
                (
                    "3256b511-921e-4456-92c6-1fd5d5a8c380",
                    "21f32e37-58c3-42e2-b9c1-2011b20e47b7",
                    "b9c1ab4b-9ea2-428a-a41e-e70fb8528d87",
                ),
                strict=True,
            )
        ],
    }
    zero = {
        "passes": [
            {
                "arms": {
                    **record("A2_zeroI", x, 0)["arms"],
                    **record("E2_zeroI", x + 0.1, 0.1)["arms"],
                }
            }
            for x in (0.4, 0.5, 0.6)
        ]
    }
    result = zeroi.build_descriptive_comparison(primary, zero)
    car = result["arms"]["A2_zeroI"]["classes"]["car"]
    assert car["primary_recall"]["pass_values"] == [0.1, 0.2, 0.3]
    assert car["zeroi_recall"]["pass_values"] == [0.4, 0.5, 0.6]
    assert car["descriptive_difference_of_observed_medians"] == pytest.approx(0.3)
    assert result["process_indices_paired"] is False
    assert "PointPillars" not in str(result)
    assert all(
        word not in str(result) for word in ("p_value", "confidence_interval", "standard_error")
    )
    wrong_primary = deepcopy(primary)
    wrong_primary["passes"][0]["process_uuid"] = "wrong"
    with pytest.raises(ValueError, match="process identities"):
        zeroi.build_descriptive_comparison(wrong_primary, zero)


def test_zeroi_raw_labels_and_offline_boundary(monkeypatch):
    passes, ledger = _evidence()

    def class_record():
        return {
            "thresholds": {
                key: {**{field: 0 for field in zeroi.METRIC_FIELDS}, "matched_gt_identity_set": []}
                for key in zeroi.THRESHOLDS
            },
            "annotation_conditioned_AP": {
                "method": "synthetic",
                "ground_truth_count": 0,
                "prediction_count": 0,
                "average_precision": 0,
                "interpretation": "annotation-conditioned descriptive AP",
            },
        }

    aggregate_pass = {
        history: {
            "condition_count": 428,
            "classes": {class_name: class_record() for class_name in zeroi.CLASSES},
        }
        for history in zeroi.HISTORIES
    }
    monkeypatch.setattr(
        zeroi,
        "aggregate_three_passes",
        lambda _passes: {
            "passes": [aggregate_pass] * 3,
            "paired_history_contrast": {
                class_name: {"pass_values": [0, 0, 0], "minimum": 0, "median": 0, "maximum": 0}
                for class_name in zeroi.CLASSES
            },
        },
    )
    result = zeroi.build_zeroi_raw(passes, ledger)
    assert set(result["passes"][0]["arms"]) == {"A2_zeroI", "E2_zeroI"}
    assert result["detector_calls_added"] == 0
    assert result["accepted_canonical_calls"] == 2568
    assert "PointPillars" not in str(result)
    assert all(
        word not in str(result) for word in ("p_value", "confidence_interval", "standard_error")
    )


def test_published_zeroi_manifest_binds_compact_artifacts():
    manifest = json.loads(
        (ROOT / "benchmarks/m8/results/m8_s1_zero_intensity_manifest.json").read_text()
    )
    assert manifest["accepted_processes"] == 3
    assert manifest["accepted_canonical_calls"] == 2568
    assert manifest["external_runtime"]["stage_r_accepted_processes"] == 10
    assert manifest["external_runtime"]["stage_r_accepted_calls"] == 140
    assert manifest["claim_boundary"]["primary_zeroi_process_indices_paired"] is False
    assert manifest["claim_boundary"]["scientific_interpretation_pending"] is True
    assert manifest["offline_safety"]["new_detector_calls"] == 0
    assert manifest["offline_safety"]["local_gpu_probe_commands_executed"] == 0
    assert manifest["offline_safety"]["runpod_actions_during_offline_publication"] == 0
    assert len({row["process_uuid"] for row in manifest["accepted_passes"]}) == 3
    assert (
        hashlib.sha256(
            (ROOT / "benchmarks/m8/results/m8_s1_primary_raw.json").read_bytes()
        ).hexdigest()
        == manifest["frozen_bindings"]["historical_primary_raw_file_sha256"]
        == zeroi.PRIMARY_RAW_SHA256
    )
    for artifact in manifest["tracked_artifacts"].values():
        path = ROOT / artifact["path"]
        assert len(path.read_bytes()) == artifact["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
    reducer = ROOT / manifest["offline_aggregation"]["reducer_path"]
    assert (
        hashlib.sha256(reducer.read_bytes()).hexdigest()
        == manifest["offline_aggregation"]["reducer_file_sha256"]
    )
    raw = json.loads((ROOT / "benchmarks/m8/results/m8_s1_zero_intensity_raw.json").read_text())
    assert set(raw["passes"][0]["arms"]) == {"A2_zeroI", "E2_zeroI"}
    assert len(raw["process_descriptors"]) == 3
    assert raw["frozen_protocol_sha256"] == manifest["frozen_bindings"]["protocol_sha256"]
    assert (
        "matched_gt_identity_set"
        in raw["passes"][0]["arms"]["A2_zeroI"]["classes"]["car"]["thresholds"]["0.50"]
    )
