import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from laserperception.detection.voxel_fidelity import (
    compare_canonical_voxels,
    saturation_statistics,
    valid_point_count,
)


def test_valid_point_count_uses_half_open_range() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.999, 0.999, 0.999, 2.0],
            [1.0, 0.5, 0.5, 3.0],
            [-0.001, 0.5, 0.5, 4.0],
        ],
        dtype=np.float32,
    )

    assert valid_point_count(points, [0, 0, 0, 1, 1, 1]) == 2


def test_saturation_statistics_attributes_discards_when_max_voxels_not_reached() -> None:
    points = np.zeros((7, 4), dtype=np.float32)
    counts = np.asarray([4, 2], dtype=np.int32)

    result = saturation_statistics(
        points,
        counts,
        point_cloud_range=[-1, -1, -1, 1, 1, 1],
        max_num_points=4,
        max_voxels=10,
    )

    assert result == {
        "total_voxels": 2,
        "saturated_voxels": 1,
        "saturated_voxel_fraction": 0.5,
        "total_valid_input_points": 7,
        "points_retained": 6,
        "points_discarded_after_voxel_limits": 1,
        "points_discarded_due_max_num_points": 1,
        "max_voxels": 10,
        "max_voxels_reached": False,
    }


def test_saturation_statistics_refuses_to_attribute_discards_at_max_voxels() -> None:
    result = saturation_statistics(
        np.zeros((3, 4), dtype=np.float32),
        np.asarray([1, 1], dtype=np.int32),
        point_cloud_range=[-1, -1, -1, 1, 1, 1],
        max_num_points=4,
        max_voxels=2,
    )

    assert result["max_voxels_reached"] is True
    assert result["points_discarded_due_max_num_points"] is None


def test_compare_canonical_voxels_ignores_coordinate_and_point_order() -> None:
    reference_voxels = np.zeros((2, 4, 2), dtype=np.float32)
    reference_voxels[0, :2] = [[1, 2], [3, 4]]
    reference_voxels[1, :2] = [[5, 6], [7, 8]]
    reference_counts = np.asarray([2, 2], dtype=np.int32)
    reference_coors = np.asarray([[0, 0, 0, 0], [0, 0, 0, 1]], dtype=np.int32)

    candidate_voxels = np.zeros_like(reference_voxels)
    candidate_voxels[0, :2] = [[7, 8], [5, 6]]
    candidate_voxels[1, :2] = [[1, 2], [3, 4]]
    candidate_counts = np.asarray([2, 2], dtype=np.int32)
    candidate_coors = np.asarray([[0, 0, 0, 1], [0, 0, 0, 0]], dtype=np.int32)

    result = compare_canonical_voxels(
        reference_voxels,
        reference_counts,
        reference_coors,
        candidate_voxels,
        candidate_counts,
        candidate_coors,
        max_num_points=4,
    )

    assert result["coordinate_order_exact"] is False
    assert result["coordinate_jaccard"] == 1.0
    assert result["common_point_multisets_equal_fraction"] == 1.0
    assert result["non_saturated_point_multisets_equal_fraction"] == 1.0
    assert result["within_voxel_order_difference_count"] == 1


def test_compare_canonical_voxels_exposes_saturated_subset_and_coordinate_changes() -> None:
    reference_voxels = np.asarray([[[1.0], [2.0]]], dtype=np.float32)
    candidate_voxels = np.asarray([[[1.0], [3.0]]], dtype=np.float32)
    counts = np.asarray([2], dtype=np.int32)

    subset_result = compare_canonical_voxels(
        reference_voxels,
        counts,
        np.asarray([[0, 0, 0, 0]], dtype=np.int32),
        candidate_voxels,
        counts,
        np.asarray([[0, 0, 0, 0]], dtype=np.int32),
        max_num_points=2,
    )
    coordinate_result = compare_canonical_voxels(
        reference_voxels,
        counts,
        np.asarray([[0, 0, 0, 0]], dtype=np.int32),
        reference_voxels,
        counts,
        np.asarray([[0, 0, 0, 1]], dtype=np.int32),
        max_num_points=2,
    )

    assert subset_result["saturated_retained_subset_difference_count"] == 1
    assert subset_result["saturated_retained_subset_difference_fraction"] == 1.0
    assert coordinate_result["coordinate_jaccard"] == 0.0
    assert coordinate_result["reference_only_coordinate_count"] == 1
    assert coordinate_result["candidate_only_coordinate_count"] == 1


def test_voxel_validation_rejects_duplicate_coordinates() -> None:
    voxels = np.zeros((2, 2, 1), dtype=np.float32)
    counts = np.ones(2, dtype=np.int32)
    coors = np.zeros((2, 4), dtype=np.int32)

    with pytest.raises(ValueError, match="duplicate voxel coordinates"):
        compare_canonical_voxels(
            voxels,
            counts,
            coors,
            voxels,
            counts,
            coors,
            max_num_points=2,
        )


def test_m3b_protocol_freezes_candidate_samples_and_non_adoption() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = (root / "configs/detection/m3b_voxelization_fidelity_v1.yaml").read_text(
        encoding="utf-8"
    )

    assert "status: protocol_frozen_before_measurement" in protocol
    assert "reference_deterministic: true" in protocol
    assert "candidate_deterministic: false" in protocol
    assert "max_num_points: 64" in protocol
    assert "max_voxels_test: 40000" in protocol
    assert (
        "sample_indices: [0, 4, 8, 12, 16, 21, 25, 29, 33, 37, 42, 46, "
        "50, 54, 58, 63, 67, 71, 75, 80]" in protocol
    )
    assert "runs_per_sample: 30" in protocol
    assert "default_voxelizer_change_allowed: false" in protocol


def test_m3b_diagnostic_record_is_exact_commit_and_fail_closed() -> None:
    root = Path(__file__).resolve().parents[1]
    record_path = root / "benchmarks/m3/diagnostics/voxelization_v1_ad0d38b.json"
    record_bytes = record_path.read_bytes()
    raw_record = record_bytes.decode("utf-8")
    record = json.loads(raw_record)

    assert hashlib.sha256(record_bytes).hexdigest() == (
        "98315416fa148a52ed14f734f923661cb70a70c8c032549fe54bd0dbf4354423"
    )
    assert record["status"] == "diagnostic_measurement_not_production"
    assert record["publication_role"] == "diagnostic_evidence_not_canonical_performance"
    assert record["measurement_commit"] == "ad0d38b6e926f3a03b471c192d3e815cd07d34d1"
    assert record["protocol"] == {
        "logical_name": "configs/detection/m3b_voxelization_fidelity_v1.yaml",
        "sha256": "03422de2d59e7bffae75eac73dd7c9dd47e40bd1e9daba6588762cdfdbf80019",
        "status": "protocol_frozen_before_measurement",
    }
    assert record["runtime_voxelizer_settings"]["official_deterministic"] is True
    assert record["runtime_voxelizer_settings"]["experimental_deterministic"] is False
    assert record["saturation_voxel_and_detector_fidelity"]["sample_count"] == 20
    assert record["repeatability"]["42"]["runs"] == 30
    assert record["repeatability"]["49"]["runs"] == 30

    w2_deterministic = record["repeatability"]["49"]["detections_each_run_vs_deterministic"]
    axis_yaw = w2_deterministic["continuous_metrics"]["axis_yaw_difference_modulo_pi_degrees"]
    assert w2_deterministic["overall_pass"] is False
    assert axis_yaw["accepted"] is False
    assert axis_yaw["failure_count"] == 19
    assert axis_yaw["pass_fraction"] == pytest.approx(0.9898341359015517)
    assert all(value is False for value in record["scope_guards"].values())
    assert "/root/" not in raw_record
    assert "/mnt/" not in raw_record
    assert "J:\\" not in raw_record
