from __future__ import annotations

from collections.abc import Mapping

import pytest

from benchmarks.m7 import aggregate_secondary as secondary
from benchmarks.m7.execution import factorial_contrasts
from benchmarks.m7.protocol import ProtocolViolation
from laserperception.detection.types import Detection3D
from laserperception.evaluation.kitti_m6b import M6bGroundTruthBox
from laserperception.evaluation.m6b_metrics import RankedDisposition, count_metrics


def _detection(class_name: str, score: float) -> Detection3D:
    return Detection3D(
        center_xyz=(0.0, 10.0, 0.0),
        size_lwh=(4.0, 2.0, 1.5),
        yaw_rad=0.0,
        score=score,
        class_id=0,
        class_name=class_name,
    )


def _observation(
    key: str,
    frame: int,
    matched: bool,
    range_forward: float,
    score: float = 0.5,
) -> dict[str, object]:
    value: dict[str, object] = {
        "object_key": key,
        "frame_index": frame,
        "track_labelled_frame_count": 8,
        "range_forward_m": range_forward,
        "matched": matched,
    }
    if matched:
        value["prediction_score"] = score
    return value


def _threshold(tp: int, fp: int, fn: int, ignored: int) -> dict[str, float | int]:
    return {**count_metrics(tp, fp, fn), "ignored_predictions": ignored}


def test_secondary_operating_points_are_frozen() -> None:
    assert secondary.IOU_THRESHOLDS == (0.30, 0.50, 0.70)
    assert secondary.OPERATING_SCORE == 0.25


def test_range_slices_preserve_m6b_boundary_convention() -> None:
    observations = [
        _observation("a", 1, True, 0.0),
        _observation("a", 2, False, 19.999),
        _observation("a", 3, True, 20.0),
        _observation("a", 4, False, 34.999),
        _observation("a", 5, True, 35.0),
        _observation("a", 6, False, 50.0),
        _observation("a", 7, True, 50.001),
    ]

    result = secondary._range_slices(observations)

    assert [item["range_m"] for item in result] == [
        [0.0, 20.0],
        [20.0, 35.0],
        [35.0, 50.0],
    ]
    assert [item["eligible_GT"] for item in result] == [2, 2, 2]
    assert [item["recall"] for item in result] == [0.5, 0.5, 0.5]


def test_prediction_population_counts_score_and_fov_disposition() -> None:
    inside = (
        _detection("car", 0.25),
        _detection("car", 0.249),
        _detection("pedestrian", 0.9),
    )
    outside = (_detection("car", 0.8), _detection("pedestrian", 0.7))

    assert secondary._prediction_population(inside, outside, class_name="car") == (1, 1)


def test_neighbour_ignore_reason_uses_m6b_source_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    prediction = _detection("car", 0.8)
    ignored = M6bGroundTruthBox(
        track_id=1,
        frame_index=2,
        source_type="Van",
        evaluation_role="neighbour_ignore",
        class_name="car",
        center_xyz=(0.0, 10.0, 0.0),
        size_lwh=(4.0, 2.0, 1.5),
        yaw_rad=0.0,
    )
    monkeypatch.setattr(secondary, "bev_iou", lambda _prediction, _box: 0.6)

    assert secondary._source_ignore_reason(prediction, (ignored,), 0.5) == "ignored_van"


def test_secondary_aggregate_sums_ignores_and_preserves_track_order_and_runs() -> None:
    first = {
        "thresholds": {key: _threshold(1, 2, 1, 1) for key in ("0.30", "0.50", "0.70")},
        "inside_FOV_prediction_count_score_0_25": 4,
        "outside_annotation_fov_predictions_score_0_25": 3,
        "neighbour_ignore_GT_count": 2,
        "ignored_predictions_by_reason": {"ignored_van": 1},
        "target_observations": [
            _observation("drive/track_b", 10, False, 15.0),
            _observation("drive/track_a", 1, True, 25.0, 0.9),
        ],
        "ranked_dispositions": [RankedDisposition(0.9, "frame1", 0, True)],
    }
    second = {
        "thresholds": {key: _threshold(1, 1, 1, 2) for key in ("0.30", "0.50", "0.70")},
        "inside_FOV_prediction_count_score_0_25": 5,
        "outside_annotation_fov_predictions_score_0_25": 6,
        "neighbour_ignore_GT_count": 1,
        "ignored_predictions_by_reason": {
            "ignored_person_sitting": 1,
            "ignored_van": 1,
        },
        "target_observations": [
            _observation("drive/track_a", 2, True, 26.0, 0.7),
            _observation("drive/track_a", 4, False, 30.0),
        ],
        "ranked_dispositions": [RankedDisposition(0.7, "frame2", 0, True)],
    }

    result = secondary._aggregate_class_frames((first, second), total_postprocessed_predictions=20)

    assert result["thresholds"]["0.50"]["ignored_predictions"] == 3  # type: ignore[index]
    assert result["neighbour_ignore"] == {
        "ignored_predictions": 3,
        "ignored_by_reason": {"ignored_person_sitting": 1, "ignored_van": 2},
    }
    assert result["prediction_population"] == {
        "total_postprocessed_prediction_population_all_scores_all_classes": 20,
        "inside_FOV_prediction_count_score_0_25": 9,
        "outside_annotation_fov_predictions_score_0_25": 9,
        "neighbour_ignore_GT_count": 3,
    }
    tracks = result["track_level"]
    assert [item["object_key"] for item in tracks] == [  # type: ignore[union-attr]
        "drive/track_a",
        "drive/track_b",
    ]
    assert tracks[0]["longest_consecutive_detected_run"] == 2  # type: ignore[index]
    assert tracks[0]["longest_consecutive_miss_run"] == 1  # type: ignore[index]


def test_primary_consistency_is_exact_and_fail_closed() -> None:
    metrics = _threshold(2, 1, 3, 4)
    secondary_record = {
        "thresholds": {"0.50": metrics},
        "score_ranked_PR_summary": {"average_precision": 0.125},
    }
    primary = {**metrics, "average_precision": 0.125}

    secondary._validate_primary_consistency(secondary_record, primary)
    mismatched = dict(primary)
    mismatched["recall"] = float(primary["recall"]) + 1e-16
    with pytest.raises(ProtocolViolation, match="recall"):
        secondary._validate_primary_consistency(secondary_record, mismatched)


def test_pedestrian_factorial_is_descriptive_and_has_no_gate_field() -> None:
    recalls = {
        "A": 219 / 396,
        "B": 199 / 396,
        "C": 224 / 396,
        "D": 212 / 396,
    }
    arms: dict[str, Mapping[str, object]] = {
        name: {"thresholds": {"0.50": {"recall": value}}} for name, value in recalls.items()
    }

    result = secondary._pedestrian_factorial(arms)

    assert result["contrasts"] == pytest.approx(
        factorial_contrasts(
            a=recalls["A"],
            b=recalls["B"],
            c=recalls["C"],
            d=recalls["D"],
        )
    )
    assert result["contrasts"] == pytest.approx(  # type: ignore[comparison-overlap]
        {
            "L": -0.04040404040404039,
            "P": 0.022727272727272763,
            "I": 0.02020202020202022,
        }
    )
    assert not any("gate" in str(key).lower() for key in result)
