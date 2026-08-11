from math import pi
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from laserperception.detection.m2_backend import M2Backend
from laserperception.detection.parity_v2 import (
    aggregate_acceptance_v2,
    axis_yaw_difference_degrees,
    classify_discrete_divergence,
    direction_flip_classification,
    direction_population_summary,
    distribution_statistics,
    full_heading_difference_degrees,
    is_direction_flip,
    official_nms_pre_union,
    raw_tensor_difference_statistics,
    tolerance_statistics,
)


def _match(
    *,
    xy: float = 0.0,
    z: float = 0.0,
    dimensions: tuple[float, float, float] = (0.0, 0.0, 0.0),
    score: float = 0.0,
    reference_yaw: float = 0.0,
    candidate_yaw: float = 0.0,
) -> dict[str, object]:
    detection = {
        "center_xyz": [0.0, 0.0, 0.0],
        "size_lwh": [4.0, 2.0, 1.5],
        "yaw_rad": reference_yaw,
        "score": 0.9,
        "class_id": 0,
        "class_name": "car",
        "velocity_xy": None,
    }
    candidate = {**detection, "yaw_rad": candidate_yaw}
    return {
        "reference": detection,
        "candidate": candidate,
        "bev_iou": 1.0,
        "center_displacement_3d_m": float((xy**2 + z**2) ** 0.5),
        "center_displacement_xy_m": xy,
        "center_displacement_z_absolute_m": z,
        "dimension_relative_error_lwh": list(dimensions),
        "circular_yaw_difference_degrees": full_heading_difference_degrees(
            reference_yaw, candidate_yaw
        ),
        "confidence_score_absolute_difference": score,
        "class_equal": True,
        "reference_high_confidence": True,
        "candidate_high_confidence": True,
        "high_confidence": True,
    }


def _report(matches: list[dict[str, object]]) -> dict[str, object]:
    count = len(matches)
    return {
        "sample_index": 0,
        "sample_id": "sample",
        "counts": {
            "pytorch_raw_postprocess": count,
            "tensorrt_raw_postprocess": count,
            "pytorch_exported": count,
            "tensorrt_exported": count,
            "absolute_exported_difference": 0,
            "allowed_exported_difference": max(1, int(0.05 * count)),
            "pytorch_high_confidence": count,
            "tensorrt_high_confidence": count,
        },
        "matches": matches,
        "threshold_edge_disagreements": [],
    }


def test_axis_yaw_is_modulo_pi_while_full_heading_remains_visible() -> None:
    candidate_yaw = pi - 0.01

    assert axis_yaw_difference_degrees(0.0, candidate_yaw) == pytest.approx(np.degrees(0.01))
    assert full_heading_difference_degrees(0.0, candidate_yaw) == pytest.approx(
        180.0 - np.degrees(0.01)
    )
    assert is_direction_flip(0.0, candidate_yaw)
    assert (
        direction_flip_classification(0.0, candidate_yaw)
        == "geometrically_axis_equivalent_but_heading_divergent"
    )


def test_fraction_acceptance_allows_exactly_one_percent_failures() -> None:
    one_failure = tolerance_statistics([0.0] * 99 + [1.0], tolerance=0.25)
    two_failures = tolerance_statistics([0.0] * 98 + [1.0, 1.0], tolerance=0.25)

    assert one_failure["pass_count"] == 99
    assert one_failure["failure_count"] == 1
    assert one_failure["pass_fraction"] == pytest.approx(0.99)
    assert one_failure["accepted"] is True
    assert two_failures["pass_fraction"] == pytest.approx(0.98)
    assert two_failures["accepted"] is False


def test_dimensions_are_evaluated_once_per_detection() -> None:
    matches = [_match() for _ in range(99)]
    matches.append(_match(dimensions=(0.06, 0.07, 0.08)))

    summary = aggregate_acceptance_v2((_report(matches),))
    dimensions = summary["continuous_metrics"]["maximum_dimension_relative_error_per_detection"]

    assert dimensions["count"] == 100
    assert dimensions["pass_count"] == 99
    assert dimensions["failure_count"] == 1
    assert dimensions["pass_fraction"] == pytest.approx(0.99)
    assert dimensions["accepted"] is True


def test_distinct_outlier_is_retained_once_with_every_failed_metric() -> None:
    matches = [_match() for _ in range(99)]
    matches.append(
        _match(
            xy=0.3,
            z=0.3,
            dimensions=(0.06, 0.01, 0.01),
            score=0.06,
            candidate_yaw=np.radians(6.0),
        )
    )

    summary = aggregate_acceptance_v2((_report(matches),))
    outliers = summary["distinct_high_confidence_continuous_outliers"]

    assert outliers["count"] == 1
    assert outliers["denominator"] == 100
    assert outliers["fraction"] == pytest.approx(0.01)
    assert outliers["detections"][0]["failed_metrics"] == [
        "xy",
        "z",
        "dimensions",
        "score",
        "axis_yaw",
    ]


def test_nms_swap_classification_requires_complete_evidence() -> None:
    incomplete = classify_discrete_divergence(
        same_class=True,
        competing_candidates_overlap=True,
        different_survivors_selected=True,
    )
    complete = classify_discrete_divergence(
        same_class=True,
        competing_candidates_overlap=True,
        different_survivors_selected=True,
        candidate_ordering_changed=True,
    )

    assert incomplete == "unexplained_outlier"
    assert complete == "confirmed_nms_survivor_swap"


def test_other_and_unexplained_divergences_remain_distinct() -> None:
    assert (
        classify_discrete_divergence(other_discrete_decision_evidence=True)
        == "other_discrete_output_divergence"
    )
    assert classify_discrete_divergence() == "unexplained_outlier"


def test_raw_tensor_statistics_report_percentiles_shape_and_dtype() -> None:
    reference = np.asarray([[[[0.0, 1.0, 2.0, 3.0]]]], dtype=np.float32)
    candidate = np.asarray([[[[0.0, 1.5, 3.0, 5.0]]]], dtype=np.float32)

    record, differences = raw_tensor_difference_statistics(reference, candidate)
    statistics = record["absolute_difference"]

    assert record["shape"] == [1, 1, 1, 4]
    assert record["shape_consistent"] is True
    assert record["dtype_consistent"] is True
    assert statistics == distribution_statistics(differences)
    assert statistics["count"] == 4
    assert statistics["median"] == pytest.approx(0.75)
    assert statistics["p95"] == pytest.approx(1.85)
    assert statistics["p99"] == pytest.approx(1.97)
    assert statistics["maximum"] == pytest.approx(2.0)
    assert statistics["mean"] == pytest.approx(0.875)


def test_direction_population_reports_flips_and_margin_partitions() -> None:
    reference = np.asarray([[2.0, 1.0], [1.01, 1.0], [0.0, 3.0]])
    candidate = np.asarray([[1.5, 1.0], [0.99, 1.0], [0.0, 2.0]])

    summary = direction_population_summary(reference, candidate)

    assert summary["count"] == 3
    assert summary["direction_argmax_disagreement_count"] == 1
    assert summary["direction_argmax_disagreement_fraction"] == pytest.approx(1.0 / 3.0)
    assert summary["winning_margins"]["agreeing_anchors"]["pytorch"]["count"] == 2
    assert summary["winning_margins"]["disagreeing_anchors"]["pytorch"]["count"] == 1


def test_decision_relevant_population_is_union_of_runtime_nms_pre_pools() -> None:
    reference = np.asarray([[9.0], [8.0], [1.0], [0.0]])
    candidate = np.asarray([[0.0], [8.0], [9.0], [1.0]])

    indices = official_nms_pre_union(reference, candidate, nms_pre=2)

    assert indices.tolist() == [0, 1, 2]


def test_v1_and_v2_manifests_are_separate_and_samples_are_unchanged() -> None:
    root = Path(__file__).resolve().parents[1]
    v1_path = root / "configs" / "detection" / "m2_parity_v1.yaml"
    v2_path = root / "configs" / "detection" / "m2_parity_v2.yaml"
    v1 = v1_path.read_text(encoding="utf-8")
    v2 = v2_path.read_text(encoding="utf-8")
    sample_line = (
        "sample_indices: [0, 4, 8, 12, 16, 21, 25, 29, 33, 37, 42, 46, "
        "50, 54, 58, 63, 67, 71, 75, 80]"
    )

    assert not (root / "configs" / "detection" / "m2_parity.yaml").exists()
    assert "status: protocol_frozen" in v1
    assert "maximum_circular_yaw_difference_degrees: 5.0" in v1
    assert "protocol_version: 2" in v2
    assert "v1_status: failed" in v2
    assert "minimum_per_detection_pass_fraction: 0.99" in v2
    assert sample_line in v1
    assert sample_line in v2


def test_official_nms_pre_uses_pointpillars_pts_test_config() -> None:
    class BackendWithPinnedConfig(M2Backend):
        def __init__(self) -> None:
            self._model = SimpleNamespace(test_cfg={"pts": {"nms_pre": 1000}})

        def initialize(self) -> None:
            return

    backend = BackendWithPinnedConfig()
    assert backend.official_nms_pre == 1000
