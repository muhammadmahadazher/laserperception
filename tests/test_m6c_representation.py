from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from laserperception.detection.mmdet3d_backend import sha256_file
from laserperception.evaluation.m6c_representation import (
    builder_matrix_from_ros_transform,
    compare_float32_arrays,
    compare_voxel_structures,
    float32_ulp_distances,
    quaternion_to_rotation_matrix_xyzw,
    rotation_summary,
    voxel_structure,
)

ROOT = Path(__file__).resolve().parents[1]


def test_float32_transform_comparison_reports_one_ulp() -> None:
    first = np.eye(4, dtype=np.float32)
    second = first.copy()
    second[0, 3] = np.nextafter(np.float32(0.0), np.float32(1.0))
    comparison = compare_float32_arrays(first, second)
    assert comparison["exact"] is False
    assert comparison["differing_elements"] == 1
    assert comparison["ulp_distance"]["maximum"] == 1.0  # type: ignore[index]
    assert float32_ulp_distances(first, second)[0, 3] == 1


def test_independent_quaternion_reconstruction_is_sign_invariant_and_proper() -> None:
    angle = 0.37
    quaternion = np.array([0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)])
    positive = quaternion_to_rotation_matrix_xyzw(quaternion)
    negative = quaternion_to_rotation_matrix_xyzw(-quaternion)
    assert np.array_equal(positive, negative)
    summary = rotation_summary(positive)
    assert np.isclose(summary["determinant"], 1.0)
    assert summary["maximum_orthonormality_residual"] < 1e-15


def test_builder_mapping_preserves_non_identity_rotation_and_translation() -> None:
    rotation = quaternion_to_rotation_matrix_xyzw(np.array([0.0, 0.0, 0.5, 0.5]))
    translation = np.array([1.0, -2.0, 3.0])
    matrix = builder_matrix_from_ros_transform(rotation, translation)
    assert np.array_equal(matrix[:3, :3], rotation.T)
    assert np.array_equal(matrix[:3, 3], -rotation.T @ translation)


def test_voxel_structure_separates_coordinates_membership_and_values() -> None:
    expected_points = np.array(
        [
            [-49.9, -49.9, -4.9, 0.0],
            [-49.8, -49.8, -4.8, 0.1],
            [0.1, 0.1, 0.0, 0.2],
        ],
        dtype=np.float32,
    )
    observed_points = expected_points.copy()
    observed_points[1, 2] = np.nextafter(observed_points[1, 2], np.float32(0.0))
    comparison = compare_voxel_structures(
        voxel_structure(expected_points),
        voxel_structure(observed_points),
    )
    assert comparison["range_mask"]["exact"] is True  # type: ignore[index]
    assert comparison["discrete_point_coordinates"]["exact"] is True  # type: ignore[index]
    assert comparison["retained_pillars"]["ordering_exact"] is True  # type: ignore[index]
    assert comparison["coors"]["exact"] is True  # type: ignore[index]
    assert comparison["num_points"]["exact"] is True  # type: ignore[index]
    assert comparison["retained_point_membership"]["exact"] is True  # type: ignore[index]
    assert comparison["voxel_feature_values"]["exact"] is False  # type: ignore[index]


def test_frozen_m6_sources_and_r2_failure_are_unchanged() -> None:
    expected = {
        "benchmarks/m6a/results/kitti_raw_offline_reconstruction.json": (
            "a62da9808079994d971c4d47bbdf04f2c50d44ddcfd0c7958f7534512552155b"
        ),
        "benchmarks/m6b/results/kitti_raw_cross_domain_characterization.json": (
            "b9d47120aabb38733d79f987f16277ebab626d2b7c13159c40c30a68b76c1d26"
        ),
        "benchmarks/m6b/diagnostics/pre_inference_input_ledger.json": (
            "2413224808b0140856a0e00f884c53bc9b49ec6b545172f5e7fa57d00802dc15"
        ),
        "benchmarks/m6c/diagnostics/gate_a_failure_frame_0000000001.json": (
            "fd319dfcb13da570c62d83b4475531bbbd735845808225ebffee4afe03f1cea4"
        ),
    }
    for relative, expected_sha256 in expected.items():
        assert sha256_file(ROOT / relative) == expected_sha256
    failure = json.loads(
        (ROOT / "benchmarks/m6c/diagnostics/gate_a_failure_frame_0000000001.json").read_text(
            encoding="utf-8"
        )
    )
    assert failure["scientific_decision"] == "M6c NOT READY — M6A ROS INPUT EXACTNESS FAILED"
