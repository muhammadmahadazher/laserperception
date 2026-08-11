from laserperception.detection.parity_validation import (
    aggregate_acceptance,
    analyze_sample,
    is_threshold_edge_score,
)
from laserperception.detection.types import Detection3D, DetectionFrame


def _box(*, score: float, center_x: float = 0.0) -> Detection3D:
    return Detection3D(
        center_xyz=(center_x, 0.0, 0.0),
        size_lwh=(4.0, 2.0, 1.5),
        yaw_rad=0.0,
        score=score,
        class_id=0,
        class_name="car",
    )


def _frame(*detections: Detection3D) -> DetectionFrame:
    return DetectionFrame(
        detections=detections,
        sample_id="sample",
        coordinate_frame="nuscenes_lidar_top",
    )


def test_threshold_edge_band_is_inclusive() -> None:
    assert is_threshold_edge_score(0.20)
    assert is_threshold_edge_score(0.30)
    assert not is_threshold_edge_score(0.199)
    assert not is_threshold_edge_score(0.301)


def test_sample_records_matched_threshold_crossing() -> None:
    report = analyze_sample(
        _frame(_box(score=0.26)),
        _frame(_box(score=0.24)),
        sample_index=0,
    )

    assert report["counts"]["pytorch_exported"] == 1
    assert report["counts"]["tensorrt_exported"] == 0
    assert report["threshold_edge_disagreements"][0]["kind"] == "matched_threshold_crossing"


def test_acceptance_passes_locked_tolerances_for_close_boxes() -> None:
    report = analyze_sample(
        _frame(_box(score=0.90)),
        _frame(_box(score=0.87, center_x=0.05)),
        sample_index=0,
    )

    summary = aggregate_acceptance((report,))

    assert summary["overall_pass"] is True
    assert all(summary["checks"].values())


def test_acceptance_fails_high_confidence_coverage_and_counts() -> None:
    report = analyze_sample(
        _frame(_box(score=0.90), _box(score=0.80, center_x=10.0)),
        _frame(),
        sample_index=0,
    )

    summary = aggregate_acceptance((report,))

    assert summary["overall_pass"] is False
    assert summary["checks"]["pytorch_to_tensorrt_high_confidence_coverage"] is False
    assert summary["checks"]["per_sample_exported_counts"] is False
