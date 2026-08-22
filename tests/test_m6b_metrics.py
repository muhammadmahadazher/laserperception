import pytest

from laserperception.evaluation.m6b_metrics import (
    RankedDisposition,
    all_points_average_precision,
    count_metrics,
    drop_homogeneity_test,
    longest_consecutive_runs,
    wrapped_absolute_yaw_error,
)


def test_count_metrics_handles_normal_and_empty_denominators() -> None:
    assert count_metrics(2, 1, 2) == {
        "true_positives": 2,
        "false_positives": 1,
        "false_negatives": 2,
        "precision": pytest.approx(2 / 3),
        "recall": 0.5,
        "f1": pytest.approx(4 / 7),
    }
    assert count_metrics(0, 0, 0)["f1"] == 0.0


def test_all_points_ap_uses_global_frozen_tie_order_and_envelope() -> None:
    records = (
        RankedDisposition(0.8, "b", 0, False),
        RankedDisposition(0.9, "a", 0, True),
        RankedDisposition(0.8, "a", 1, True),
    )

    result = all_points_average_precision(records, ground_truth_count=2)

    assert result["average_precision"] == pytest.approx(1.0)
    assert [item["frame_id"] for item in result["curve"]] == ["a", "a", "b"]


def test_longest_runs_reset_at_missing_gt_frame() -> None:
    assert longest_consecutive_runs([10, 11, 13, 14, 15], [True, True, True, False, False]) == (
        2,
        2,
    )


def test_drop_homogeneity_requires_expected_counts_and_detects_concentration() -> None:
    inadequate = drop_homogeneity_test([100] * 12, [1] * 12)
    concentrated = drop_homogeneity_test([100] * 12, [80] + [20] * 11)

    assert not inadequate["adequate_expected_counts"]
    assert concentrated["adequate_expected_counts"]
    assert concentrated["reject_homogeneity"]


def test_wrapped_yaw_error_handles_pi_boundary() -> None:
    assert wrapped_absolute_yaw_error(3.13, -3.13) == pytest.approx(0.023185307179586445)
