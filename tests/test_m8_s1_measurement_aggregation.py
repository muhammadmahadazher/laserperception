from __future__ import annotations

import pytest

from benchmarks.m8 import aggregate_s1_measurement as publish


def _class_record(*, matched: bool, identity: str, frame: int, band: str) -> dict[str, object]:
    threshold = {
        "true_positives": int(matched),
        "false_positives": 0,
        "false_negatives": int(not matched),
        "precision": 1.0 if matched else 0.0,
        "recall": 1.0 if matched else 0.0,
        "f1": 1.0 if matched else 0.0,
        "ignored_predictions": 0,
        "matched_gt_identity_set": [identity] if matched else [],
    }
    return {
        "eligible_GT_count": 1,
        "neighbour_ignore_GT_count": 2,
        "thresholded_prediction_count": int(matched),
        "thresholds": {key: dict(threshold) for key in ("0.30", "0.50", "0.70")},
        "target_observations": [
            {
                "frame_index": frame,
                "gt_identity": identity,
                "matched": matched,
                "matched_iou": 0.8 if matched else None,
                "range_band_metres": band,
                "range_forward_m": 10.0,
                "track_id": 1,
            }
        ],
        "ranked_dispositions": [],
    }


def _condition(history: str, frame: int, matched: bool) -> dict[str, object]:
    identity = "drive/track_1"
    car = _class_record(matched=matched, identity=identity, frame=frame, band="0_20")
    pedestrian = _class_record(matched=False, identity="drive/track_2", frame=frame, band="20_35")
    return {
        "history": history,
        "frame_id": f"drive/{frame:010d}",
        "outside_annotation_fov_prediction_count": 1,
        "predictions": [
            {"class_name": "car", "score": 0.25, "inside_annotation_fov": True},
            {"class_name": "car", "score": 0.9, "inside_annotation_fov": False},
        ],
        "classes": {"car": car, "pedestrian": pedestrian},
    }


def test_secondary_reducer_preserves_range_boundaries_and_track_gaps() -> None:
    conditions = [
        _condition("H10", 1, True),
        _condition("H10", 2, True),
        _condition("H10", 4, True),
    ]
    result = publish.reduce_history(conditions, "H10")
    car = result["classes"]["car"]
    assert car["range_by_iou"]["0.50"][0] == {
        "range_m": [0.0, 20.0],
        "eligible_GT": 3,
        "true_positives": 3,
        "false_negatives": 0,
        "recall": 1.0,
    }
    assert car["track_continuity_by_iou"]["0.50"] == [
        {
            "object_key": "drive/track_1",
            "eligible_eval_frame_count": 3,
            "detected_frames": 3,
            "detection_continuity_fraction": 1.0,
            "longest_consecutive_detected_run": 2,
            "longest_consecutive_miss_run": 0,
            "forward_range_span_m": [10.0, 10.0],
        }
    ]


def test_secondary_reducer_uses_score_threshold_and_fov() -> None:
    result = publish.reduce_history([_condition("H10", 1, True)], "H10")
    population = result["classes"]["car"]["prediction_population"]
    assert population == {
        "total_postprocessed_prediction_population_all_scores_all_classes": 2,
        "inside_FOV_prediction_count_score_0_25": 1,
        "outside_annotation_fov_predictions_score_0_25": 1,
        "outside_annotation_fov_predictions_all_scores_all_classes": 1,
    }


def test_secondary_reducer_fails_closed_on_primary_count_mismatch() -> None:
    condition = _condition("H10", 1, True)
    condition["classes"]["car"]["thresholds"]["0.50"]["true_positives"] = 0
    with pytest.raises(ValueError, match="matched identity count"):
        publish.reduce_history([condition], "H10")


def test_summarize_three_requires_exactly_three_values() -> None:
    assert publish.summarize_three([3, 1, 2]) == {
        "pass_values": [3, 1, 2],
        "minimum": 1,
        "median": 2,
        "maximum": 3,
    }
    with pytest.raises(ValueError, match="exactly three"):
        publish.summarize_three([1, 2])
