from math import pi

import pytest

from laserperception.detection.parity import match_detections, oriented_bev_iou
from laserperception.detection.types import Detection3D


def _box(
    *,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    size: tuple[float, float, float] = (4.0, 2.0, 1.5),
    yaw: float = 0.0,
    score: float = 0.9,
    class_id: int = 0,
) -> Detection3D:
    return Detection3D(
        center_xyz=center,
        size_lwh=size,
        yaw_rad=yaw,
        score=score,
        class_id=class_id,
        class_name="car" if class_id == 0 else "pedestrian",
    )


def test_oriented_bev_iou_handles_identical_disjoint_and_partial_boxes() -> None:
    reference = _box()

    assert oriented_bev_iou(reference, _box()) == pytest.approx(1.0)
    assert oriented_bev_iou(reference, _box(center=(10.0, 0.0, 0.0))) == 0.0
    assert oriented_bev_iou(reference, _box(center=(2.0, 0.0, 0.0))) == pytest.approx(1.0 / 3.0)


def test_matching_is_class_wise_one_to_one_and_uses_best_iou() -> None:
    reference_high = _box(score=0.9)
    reference_low = _box(center=(0.8, 0.0, 0.0), score=0.8)
    best_for_high = _box(center=(0.1, 0.0, 0.0), score=0.7)
    other = _box(center=(0.9, 0.0, 0.0), score=0.95)
    wrong_class = _box(class_id=1)

    result = match_detections(
        (reference_low, reference_high),
        (wrong_class, other, best_for_high),
        minimum_bev_iou=0.5,
    )

    assert len(result.matches) == 2
    assert result.matches[0].reference is reference_high
    assert result.matches[0].candidate is best_for_high
    assert result.matches[1].candidate is other
    assert result.unmatched_reference == ()
    assert result.unmatched_candidate == (wrong_class,)


def test_match_metrics_use_circular_yaw_and_relative_dimensions() -> None:
    reference = _box(size=(4.0, 2.0, 1.0), yaw=pi - 0.1)
    candidate = _box(
        center=(0.1, -0.2, 0.25),
        size=(4.2, 1.9, 1.05),
        yaw=-pi + 0.1,
        score=0.86,
    )

    match = match_detections((reference,), (candidate,), minimum_bev_iou=0.5).matches[0]

    assert match.center_displacement_xy_m == pytest.approx(5**0.5 / 10.0)
    assert match.center_displacement_z_absolute_m == pytest.approx(0.25)
    assert match.dimension_relative_error_lwh == pytest.approx((0.05, 0.05, 0.05))
    assert match.circular_yaw_difference_degrees == pytest.approx(180.0 * 0.2 / pi)
    assert match.confidence_score_absolute_difference == pytest.approx(0.04)
    assert match.class_equal
