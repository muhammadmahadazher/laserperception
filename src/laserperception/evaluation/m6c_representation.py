"""CPU-only numerical diagnostics for the failed M6c ROS exactness gate."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import cast

import numpy as np

DEFAULT_VOXEL_SIZE = (0.25, 0.25, 8.0)
DEFAULT_POINT_CLOUD_RANGE = (-50.0, -50.0, -5.0, 50.0, 50.0, 3.0)
DEFAULT_MAX_POINTS = 64
DEFAULT_MAX_VOXELS = 40_000


def array_sha256(array: np.ndarray) -> str:
    """Hash one array using its contiguous C-order bytes."""

    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def float32_ulp_distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return representable-value distances for equally shaped finite float32 arrays."""

    left = np.asarray(first)
    right = np.asarray(second)
    if left.dtype != np.dtype(np.float32) or right.dtype != np.dtype(np.float32):
        raise TypeError("ULP comparison requires float32 arrays")
    if left.shape != right.shape:
        raise ValueError("ULP comparison arrays must have equal shapes")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("ULP comparison arrays must be finite")

    def ordered(values: np.ndarray) -> np.ndarray:
        bits = values.view(np.uint32).astype(np.uint64)
        negative = (bits & np.uint64(0x80000000)) != 0
        return np.where(
            negative,
            np.uint64(0xFFFFFFFF) - bits,
            bits + np.uint64(0x80000000),
        )

    left_ordered = ordered(left)
    right_ordered = ordered(right)
    return cast(
        np.ndarray,
        np.maximum(left_ordered, right_ordered) - np.minimum(left_ordered, right_ordered),
    )


def compare_float32_arrays(first: np.ndarray, second: np.ndarray) -> dict[str, object]:
    """Return an exact and descriptive comparison without defining a tolerance."""

    left = np.asarray(first)
    right = np.asarray(second)
    if left.dtype != np.dtype(np.float32) or right.dtype != np.dtype(np.float32):
        raise TypeError("comparison requires float32 arrays")
    if left.shape != right.shape:
        return {
            "shape_exact": False,
            "first_shape": list(left.shape),
            "second_shape": list(right.shape),
            "exact": False,
        }
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("comparison arrays must be finite")

    differing = left != right
    different_count = int(np.count_nonzero(differing))
    result: dict[str, object] = {
        "shape_exact": True,
        "shape": list(left.shape),
        "exact": different_count == 0,
        "differing_elements": different_count,
        "first_sha256": array_sha256(left),
        "second_sha256": array_sha256(right),
    }
    if different_count == 0:
        result.update(
            {
                "maximum_absolute_difference": 0.0,
                "nonzero_absolute_difference": _distribution(np.empty(0, dtype=np.float64)),
                "ulp_distance": _distribution(np.empty(0, dtype=np.float64)),
                "differing_positions": [],
            }
        )
        return result

    absolute = np.abs(left.astype(np.float64) - right.astype(np.float64))[differing]
    ulps = float32_ulp_distances(left, right)[differing]
    result.update(
        {
            "maximum_absolute_difference": float(np.max(absolute)),
            "nonzero_absolute_difference": _distribution(absolute),
            "ulp_distance": _distribution(ulps.astype(np.float64)),
        }
    )
    if left.size <= 64:
        positions: list[dict[str, object]] = []
        for index in np.argwhere(differing):
            item = tuple(int(value) for value in index)
            positions.append(
                {
                    "index": list(item),
                    "first": float(left[item]),
                    "second": float(right[item]),
                    "absolute_difference": float(abs(float(left[item]) - float(right[item]))),
                    "ulp_distance": int(float32_ulp_distances(left, right)[item]),
                }
            )
        result["differing_positions"] = positions
    return result


def quaternion_to_rotation_matrix_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """Independently reconstruct a proper rotation from one finite quaternion."""

    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("quaternion must contain four finite XYZW values")
    norm = float(np.linalg.norm(value))
    if norm == 0.0:
        raise ValueError("quaternion must have non-zero norm")
    x, y, z, w = (float(item) for item in value / norm)
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def builder_matrix_from_ros_transform(
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Apply the accepted ROS column-vector to builder-storage mapping in float64."""

    matrix = np.asarray(rotation, dtype=np.float64)
    vector = np.asarray(translation, dtype=np.float64)
    if matrix.shape != (3, 3) or vector.shape != (3,):
        raise ValueError("ROS transform must contain a 3x3 rotation and 3-vector translation")
    if not np.isfinite(matrix).all() or not np.isfinite(vector).all():
        raise ValueError("ROS transform must be finite")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = matrix.T
    result[:3, 3] = -matrix.T @ vector
    return result


def rotation_summary(rotation: np.ndarray) -> dict[str, float]:
    """Summarize proper-rotation diagnostics without applying an acceptance tolerance."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    residual = matrix.T @ matrix - np.eye(3, dtype=np.float64)
    return {
        "determinant": float(np.linalg.det(matrix)),
        "maximum_orthonormality_residual": float(np.max(np.abs(residual))),
    }


@dataclass(frozen=True, slots=True)
class VoxelStructure:
    """Deterministic CPU mirror of exact-fast grouping and retention identities."""

    range_mask: np.ndarray
    point_coordinates_zyx: np.ndarray
    candidate_keys: np.ndarray
    retained_keys: np.ndarray
    coors_zyx: np.ndarray
    num_points: np.ndarray
    retained_membership: np.ndarray
    voxels: np.ndarray | None


def voxel_structure(
    points_xyzt: np.ndarray,
    *,
    voxel_size: tuple[float, float, float] = DEFAULT_VOXEL_SIZE,
    point_cloud_range: tuple[float, float, float, float, float, float] = (
        DEFAULT_POINT_CLOUD_RANGE
    ),
    max_points: int = DEFAULT_MAX_POINTS,
    max_voxels: int = DEFAULT_MAX_VOXELS,
    include_values: bool = True,
) -> VoxelStructure:
    """Mirror dynamic coordinates, first-touch pillar capacity, and point retention."""

    points = np.asarray(points_xyzt)
    if points.dtype != np.dtype(np.float32):
        raise TypeError("model-ready points must have dtype float32")
    if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
        raise ValueError("model-ready points must be a finite (N, 4) array")
    if max_points <= 0 or max_voxels <= 0:
        raise ValueError("voxel capacities must be positive")

    minimum = np.asarray(point_cloud_range[:3], dtype=np.float64)
    maximum = np.asarray(point_cloud_range[3:], dtype=np.float64)
    size = np.asarray(voxel_size, dtype=np.float64)
    grid = np.rint((maximum - minimum) / size).astype(np.int64)
    xyz = points[:, :3].astype(np.float64)
    coordinates_xyz = np.floor((xyz - minimum) / size).astype(np.int64)
    valid: np.ndarray = np.asarray(
        np.all((coordinates_xyz >= 0) & (coordinates_xyz < grid), axis=1),
        dtype=np.bool_,
    )
    coordinates_zyx_all = np.full((len(points), 3), -1, dtype=np.int32)
    coordinates_zyx_all[valid] = coordinates_xyz[valid, ::-1].astype(np.int32)
    valid_indices = np.flatnonzero(valid).astype(np.int64)
    valid_coordinates = coordinates_zyx_all[valid]
    if len(valid_indices) == 0:
        empty_i32 = np.empty((0,), dtype=np.int32)
        empty_coors = np.empty((0, 3), dtype=np.int32)
        empty_membership = np.empty((0, max_points), dtype=np.int64)
        empty_voxels = np.empty((0, max_points, 4), dtype=np.float32) if include_values else None
        return VoxelStructure(
            valid,
            coordinates_zyx_all,
            empty_i32,
            empty_i32,
            empty_coors,
            empty_i32,
            empty_membership,
            empty_voxels,
        )

    grid_x, grid_y, _ = (int(value) for value in grid)
    keys = (
        valid_coordinates[:, 0].astype(np.int64) * (grid_y * grid_x)
        + valid_coordinates[:, 1].astype(np.int64) * grid_x
        + valid_coordinates[:, 2].astype(np.int64)
    )
    sorted_positions = np.lexsort((valid_indices, keys))
    sorted_keys = keys[sorted_positions]
    sorted_indices = valid_indices[sorted_positions]
    sorted_coordinates = valid_coordinates[sorted_positions]
    group_starts_mask = np.ones(len(sorted_keys), dtype=np.bool_)
    group_starts_mask[1:] = sorted_keys[1:] != sorted_keys[:-1]
    group_starts = np.flatnonzero(group_starts_mask)
    group_ends = np.concatenate([group_starts[1:], np.array([len(sorted_keys)])])
    group_counts = group_ends - group_starts
    group_ids = np.cumsum(group_starts_mask, dtype=np.int64) - 1
    positions = np.arange(len(sorted_keys), dtype=np.int64)
    positions_in_group = positions - group_starts[group_ids]
    first_original_indices = sorted_indices[group_starts]
    first_touch_order = np.argsort(first_original_indices, kind="stable")
    accepted_groups = first_touch_order[:max_voxels]

    group_to_voxel = np.full(len(group_starts), -1, dtype=np.int64)
    group_to_voxel[accepted_groups] = np.arange(len(accepted_groups), dtype=np.int64)
    destinations = group_to_voxel[group_ids]
    retained = (destinations >= 0) & (positions_in_group < max_points)
    membership = np.full((len(accepted_groups), max_points), -1, dtype=np.int64)
    membership[destinations[retained], positions_in_group[retained]] = sorted_indices[retained]
    coors = np.ascontiguousarray(sorted_coordinates[group_starts[accepted_groups]], dtype=np.int32)
    counts = np.minimum(group_counts[accepted_groups], max_points).astype(np.int32)
    voxels: np.ndarray | None = None
    if include_values:
        voxels = np.zeros((len(accepted_groups), max_points, 4), dtype=np.float32)
        voxels[destinations[retained], positions_in_group[retained]] = points[
            sorted_indices[retained]
        ]
    return VoxelStructure(
        range_mask=np.ascontiguousarray(valid),
        point_coordinates_zyx=coordinates_zyx_all,
        candidate_keys=np.ascontiguousarray(sorted_keys[group_starts], dtype=np.int64),
        retained_keys=np.ascontiguousarray(
            sorted_keys[group_starts[accepted_groups]], dtype=np.int64
        ),
        coors_zyx=coors,
        num_points=np.ascontiguousarray(counts),
        retained_membership=membership,
        voxels=voxels,
    )


def compare_voxel_structures(
    expected: VoxelStructure,
    observed: VoxelStructure,
) -> dict[str, object]:
    """Compare voxel membership and feature values as independent properties."""

    mask_shape_exact = expected.range_mask.shape == observed.range_mask.shape
    mask_changes = (
        int(np.count_nonzero(expected.range_mask != observed.range_mask))
        if mask_shape_exact
        else None
    )
    coordinate_shape_exact = (
        expected.point_coordinates_zyx.shape == observed.point_coordinates_zyx.shape
    )
    coordinate_changes = (
        int(
            np.count_nonzero(
                np.any(
                    expected.point_coordinates_zyx != observed.point_coordinates_zyx,
                    axis=1,
                )
            )
        )
        if coordinate_shape_exact
        else None
    )
    added = np.setdiff1d(observed.candidate_keys, expected.candidate_keys)
    removed = np.setdiff1d(expected.candidate_keys, observed.candidate_keys)
    values_comparison: dict[str, object] | None = None
    if expected.voxels is not None and observed.voxels is not None:
        values_comparison = compare_float32_arrays(expected.voxels, observed.voxels)
    return {
        "range_mask": {
            "exact": mask_shape_exact and mask_changes == 0,
            "expected_retained": int(np.count_nonzero(expected.range_mask)),
            "observed_retained": int(np.count_nonzero(observed.range_mask)),
            "points_changing_membership": mask_changes,
        },
        "discrete_point_coordinates": {
            "exact": coordinate_shape_exact and coordinate_changes == 0,
            "points_changed": coordinate_changes,
        },
        "candidate_pillars": {
            "expected_count": int(len(expected.candidate_keys)),
            "observed_count": int(len(observed.candidate_keys)),
            "key_set_exact": len(added) == 0 and len(removed) == 0,
            "keys_added": int(len(added)),
            "keys_removed": int(len(removed)),
        },
        "retained_pillars": {
            "expected_count": int(len(expected.retained_keys)),
            "observed_count": int(len(observed.retained_keys)),
            "key_set_exact": np.array_equal(
                np.sort(expected.retained_keys), np.sort(observed.retained_keys)
            ),
            "ordering_exact": np.array_equal(expected.retained_keys, observed.retained_keys),
        },
        "coors": {
            "exact": np.array_equal(expected.coors_zyx, observed.coors_zyx),
            "expected_sha256": array_sha256(expected.coors_zyx),
            "observed_sha256": array_sha256(observed.coors_zyx),
        },
        "num_points": {
            "exact": np.array_equal(expected.num_points, observed.num_points),
            "expected_sha256": array_sha256(expected.num_points),
            "observed_sha256": array_sha256(observed.num_points),
        },
        "retained_point_membership": {
            "exact": np.array_equal(expected.retained_membership, observed.retained_membership),
            "ordering_exact": np.array_equal(
                expected.retained_membership, observed.retained_membership
            ),
            "expected_sha256": array_sha256(expected.retained_membership),
            "observed_sha256": array_sha256(observed.retained_membership),
        },
        "voxel_feature_values": values_comparison,
    }


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    data = np.asarray(values, dtype=np.float64)
    if len(data) == 0:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "p95": None,
            "p99": None,
            "maximum": None,
        }
    return {
        "count": int(len(data)),
        "minimum": float(np.min(data)),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "maximum": float(np.max(data)),
    }
