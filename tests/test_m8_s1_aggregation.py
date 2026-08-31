from __future__ import annotations

from typing import Any

import pytest

from laserperception.detection.m8_s1_runtime import paired_history_delta
from laserperception.evaluation.m6b_metrics import all_points_average_precision
from laserperception.evaluation.m8_s1_aggregation import (
    aggregate_ranked_ap,
    aggregate_three_passes,
)


def _class_record(tp: int, fn: int) -> dict[str, object]:
    thresholds = {
        threshold: {
            "true_positives": tp,
            "false_positives": 1,
            "false_negatives": fn,
            "ignored_predictions": 0,
            "matched_gt_identity_set": [f"track-{index}" for index in range(tp)],
        }
        for threshold in ("0.30", "0.50", "0.70")
    }
    return {
        "thresholds": thresholds,
        "ranked_dispositions": [
            {
                "score": 0.9,
                "frame_id": "frame",
                "prediction_index": 0,
                "true_positive": tp > 0,
            }
        ],
    }


def _pass(index: int, h10_tp: int, h5_tp: int) -> dict[str, Any]:
    conditions = []
    for frame in range(428):
        for history, total_tp in (("H10", h10_tp), ("H5", h5_tp)):
            tp = total_tp if frame == 0 else 0
            fn = 10 - total_tp if frame == 0 else 0
            conditions.append(
                {
                    "history": history,
                    "classes": {
                        "car": _class_record(tp, fn),
                        "pedestrian": _class_record(tp, fn),
                    },
                }
            )
    return {
        "status": "COMPLETE",
        "mode": "primary-pass",
        "process_uuid": f"process-{index}",
        "conditions": conditions,
    }


def test_paired_history_delta_is_per_pass_not_marginal_medians() -> None:
    result = paired_history_delta([0.0, 0.9, 0.9], [0.8, 0.8, 1.0])
    assert result["pass_values"] == pytest.approx([0.8, -0.1, 0.1])
    assert result["median"] == pytest.approx(0.1)
    prohibited = 0.8 - 0.9
    assert result["median"] != pytest.approx(prohibited)
    assert result["prohibited_formula_used"] is False


def test_aggregation_keeps_three_passes_and_has_no_inferential_fields() -> None:
    result = aggregate_three_passes([_pass(1, 0, 8), _pass(2, 9, 8), _pass(3, 9, 10)])
    assert len(result["passes"]) == 3
    assert result["boxes_averaged"] is False
    assert "confidence_intervals" not in result
    assert "p_values" not in result
    assert "standard_error" not in result
    car = result["paired_history_contrast"]["car"]
    assert car["pass_values"] == pytest.approx([0.8, -0.1, 0.1])
    assert car["minimum"] == pytest.approx(-0.1)
    assert car["median"] == pytest.approx(0.1)
    assert car["maximum"] == pytest.approx(0.8)


def test_annotation_conditioned_ap_reuses_frozen_m6b_implementation() -> None:
    records = [
        {"score": 0.9, "frame_id": "a", "prediction_index": 0, "true_positive": True},
        {"score": 0.8, "frame_id": "b", "prediction_index": 0, "true_positive": False},
        {"score": 0.7, "frame_id": "c", "prediction_index": 0, "true_positive": True},
    ]
    m8 = aggregate_ranked_ap(records, ground_truth_count=2)
    from laserperception.evaluation.m6b_metrics import RankedDisposition

    m6b = all_points_average_precision(
        [RankedDisposition(**record) for record in records], ground_truth_count=2
    )
    assert m8["average_precision"] == m6b["average_precision"]
    assert m8["method"] == m6b["method"]
