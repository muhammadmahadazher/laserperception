"""CPU-only hard-voxel saturation and coordinate-canonical fidelity helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np


def first_exact_array_mismatch(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    name: str,
) -> dict[str, Any] | None:
    """Describe the first shape, dtype, or value mismatch between two arrays."""

    reference_array = np.asarray(reference)
    candidate_array = np.asarray(candidate)
    if reference_array.shape != candidate_array.shape:
        return {
            "tensor": name,
            "kind": "shape",
            "reference_shape": list(reference_array.shape),
            "candidate_shape": list(candidate_array.shape),
        }
    if reference_array.dtype != candidate_array.dtype:
        return {
            "tensor": name,
            "kind": "dtype",
            "reference_dtype": str(reference_array.dtype),
            "candidate_dtype": str(candidate_array.dtype),
        }
    unequal = np.flatnonzero(reference_array.reshape(-1) != candidate_array.reshape(-1))
    if unequal.size == 0:
        return None
    flat_index = int(unequal[0])
    index = tuple(int(value) for value in np.unravel_index(flat_index, reference_array.shape))
    return {
        "tensor": name,
        "kind": "value",
        "index": list(index),
        "reference_value": reference_array[index].item(),
        "candidate_value": candidate_array[index].item(),
    }


def valid_point_count(points: np.ndarray, point_cloud_range: Sequence[float]) -> int:
    """Count points inside MMCV's half-open hard-voxelization range."""

    array = np.asarray(points)
    bounds = np.asarray(tuple(point_cloud_range), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError("points must have shape (N, 3+)")
    if bounds.shape != (6,) or not np.isfinite(bounds).all():
        raise ValueError("point_cloud_range must contain six finite values")
    if not bool(np.all(bounds[3:] > bounds[:3])):
        raise ValueError("point_cloud_range maxima must exceed minima")
    xyz = array[:, :3]
    if not np.isfinite(xyz).all():
        raise ValueError("point coordinates must be finite")
    mask = np.logical_and(xyz >= bounds[:3], xyz < bounds[3:]).all(axis=1)
    return int(np.count_nonzero(mask))


def saturation_statistics(
    points: np.ndarray,
    num_points: np.ndarray,
    *,
    point_cloud_range: Sequence[float],
    max_num_points: int,
    max_voxels: int,
) -> dict[str, int | float | bool | None]:
    """Summarize point retention and hard-voxel capacity saturation."""

    counts = np.asarray(num_points)
    if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("num_points must be a one-dimensional integer array")
    if max_num_points <= 0 or max_voxels <= 0:
        raise ValueError("max_num_points and max_voxels must be positive")
    if bool(np.any(counts <= 0)) or bool(np.any(counts > max_num_points)):
        raise ValueError("num_points values must lie within hard-voxel capacity")
    voxel_count = int(counts.size)
    retained = int(np.sum(counts, dtype=np.int64))
    valid = valid_point_count(points, point_cloud_range)
    if retained > valid:
        raise ValueError("retained point count cannot exceed valid input point count")
    saturated = int(np.count_nonzero(counts == max_num_points))
    max_voxels_reached = voxel_count >= max_voxels
    discarded = valid - retained
    return {
        "total_voxels": voxel_count,
        "saturated_voxels": saturated,
        "saturated_voxel_fraction": saturated / voxel_count if voxel_count else 0.0,
        "total_valid_input_points": valid,
        "points_retained": retained,
        "points_discarded_after_voxel_limits": discarded,
        "points_discarded_due_max_num_points": (None if max_voxels_reached else discarded),
        "max_voxels": int(max_voxels),
        "max_voxels_reached": max_voxels_reached,
    }


def compare_canonical_voxels(
    reference_voxels: np.ndarray,
    reference_num_points: np.ndarray,
    reference_coors: np.ndarray,
    candidate_voxels: np.ndarray,
    candidate_num_points: np.ndarray,
    candidate_coors: np.ndarray,
    *,
    max_num_points: int,
) -> dict[str, int | float | bool]:
    """Compare hard voxels by coordinate while ignoring legitimate output ordering."""

    reference = _validate_voxels(
        reference_voxels,
        reference_num_points,
        reference_coors,
        max_num_points=max_num_points,
        name="reference",
    )
    candidate = _validate_voxels(
        candidate_voxels,
        candidate_num_points,
        candidate_coors,
        max_num_points=max_num_points,
        name="candidate",
    )
    reference_map = _coordinate_index(reference[2], name="reference")
    candidate_map = _coordinate_index(candidate[2], name="candidate")
    reference_coordinates = set(reference_map)
    candidate_coordinates = set(candidate_map)
    common = sorted(reference_coordinates & candidate_coordinates)
    union_count = len(reference_coordinates | candidate_coordinates)

    num_points_equal = 0
    point_multisets_equal = 0
    within_voxel_order_differences = 0
    non_saturated = 0
    non_saturated_equal = 0
    saturated = 0
    saturated_equal = 0
    saturated_subset_differences = 0

    for coordinate in common:
        reference_index = reference_map[coordinate]
        candidate_index = candidate_map[coordinate]
        reference_count = int(reference[1][reference_index])
        candidate_count = int(candidate[1][candidate_index])
        counts_equal = reference_count == candidate_count
        num_points_equal += int(counts_equal)
        reference_points = reference[0][reference_index, :reference_count]
        candidate_points = candidate[0][candidate_index, :candidate_count]
        exact_order = counts_equal and bool(np.array_equal(reference_points, candidate_points))
        multiset_equal = counts_equal and _point_multiset_equal(reference_points, candidate_points)
        point_multisets_equal += int(multiset_equal)
        within_voxel_order_differences += int(multiset_equal and not exact_order)
        is_saturated = reference_count == max_num_points or candidate_count == max_num_points
        if is_saturated:
            saturated += 1
            saturated_equal += int(multiset_equal)
            saturated_subset_differences += int(not multiset_equal)
        else:
            non_saturated += 1
            non_saturated_equal += int(multiset_equal)

    common_count = len(common)
    return {
        "reference_voxel_count": len(reference_coordinates),
        "candidate_voxel_count": len(candidate_coordinates),
        "voxel_count_equal": len(reference_coordinates) == len(candidate_coordinates),
        "coordinate_order_exact": bool(np.array_equal(reference[2], candidate[2])),
        "common_coordinate_count": common_count,
        "reference_only_coordinate_count": len(reference_coordinates - candidate_coordinates),
        "candidate_only_coordinate_count": len(candidate_coordinates - reference_coordinates),
        "coordinate_union_count": union_count,
        "reference_to_candidate_coordinate_coverage": _fraction(
            common_count, len(reference_coordinates)
        ),
        "candidate_to_reference_coordinate_coverage": _fraction(
            common_count, len(candidate_coordinates)
        ),
        "coordinate_jaccard": _fraction(common_count, union_count),
        "common_num_points_equal_count": num_points_equal,
        "common_num_points_equal_fraction": _fraction(num_points_equal, common_count),
        "common_point_multisets_equal_count": point_multisets_equal,
        "common_point_multisets_equal_fraction": _fraction(point_multisets_equal, common_count),
        "within_voxel_order_difference_count": within_voxel_order_differences,
        "non_saturated_common_voxel_count": non_saturated,
        "non_saturated_point_multisets_equal_count": non_saturated_equal,
        "non_saturated_point_multisets_equal_fraction": _fraction(
            non_saturated_equal, non_saturated
        ),
        "saturated_common_voxel_count": saturated,
        "saturated_retained_point_multisets_equal_count": saturated_equal,
        "saturated_retained_point_multisets_equal_fraction": _fraction(saturated_equal, saturated),
        "saturated_retained_subset_difference_count": saturated_subset_differences,
        "saturated_retained_subset_difference_fraction": _fraction(
            saturated_subset_differences, saturated
        ),
    }


def _validate_voxels(
    voxels: np.ndarray,
    num_points: np.ndarray,
    coors: np.ndarray,
    *,
    max_num_points: int,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    voxel_array = np.asarray(voxels)
    count_array = np.asarray(num_points)
    coordinate_array = np.asarray(coors)
    if max_num_points <= 0:
        raise ValueError("max_num_points must be positive")
    if voxel_array.ndim != 3 or voxel_array.shape[1] != max_num_points:
        raise ValueError(f"{name} voxels must have shape (N, max_num_points, C)")
    if count_array.shape != (voxel_array.shape[0],):
        raise ValueError(f"{name} num_points must match the voxel count")
    if coordinate_array.ndim != 2 or coordinate_array.shape[0] != voxel_array.shape[0]:
        raise ValueError(f"{name} coors must have shape (N, D)")
    if not np.issubdtype(count_array.dtype, np.integer):
        raise ValueError(f"{name} num_points must be integer-valued")
    if not np.issubdtype(coordinate_array.dtype, np.integer):
        raise ValueError(f"{name} coors must be integer-valued")
    if bool(np.any(count_array <= 0)) or bool(np.any(count_array > max_num_points)):
        raise ValueError(f"{name} num_points lie outside hard-voxel capacity")
    return voxel_array, count_array, coordinate_array


def _coordinate_index(coors: np.ndarray, *, name: str) -> dict[tuple[int, ...], int]:
    result: dict[tuple[int, ...], int] = {}
    for index, row in enumerate(coors):
        coordinate = tuple(int(value) for value in row)
        if coordinate in result:
            raise ValueError(f"{name} contains duplicate voxel coordinates")
        result[coordinate] = index
    return result


def _point_multiset_equal(reference: np.ndarray, candidate: np.ndarray) -> bool:
    if reference.shape != candidate.shape:
        return False
    if reference.size == 0:
        return True
    return bool(
        np.array_equal(
            reference[_lexicographic_order(reference)],
            candidate[_lexicographic_order(candidate)],
        )
    )


def _lexicographic_order(points: np.ndarray) -> np.ndarray:
    keys = tuple(points[:, column] for column in reversed(range(points.shape[1])))
    return cast(np.ndarray, np.lexsort(keys))


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0
