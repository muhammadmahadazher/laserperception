"""CPU-only pillar-cap diagnostics for the frozen M6b voxel contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VOXEL_SIZE_XY = (0.25, 0.25)
POINT_CLOUD_MIN_XY = (-50.0, -50.0)
GRID_SIZE_XY = (400, 400)
MAX_VOXELS = 40_000


@dataclass(frozen=True, slots=True)
class PillarAudit:
    """Candidate/retained pillar geometry in deterministic first-touch order."""

    in_range_points: int
    candidate_xy_indices: np.ndarray
    retained_xy_indices: np.ndarray
    discarded_xy_indices: np.ndarray
    candidate_first_touch_sweep: np.ndarray
    retained_first_touch_sweep: np.ndarray
    discarded_first_touch_sweep: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "candidate_xy_indices",
            "retained_xy_indices",
            "discarded_xy_indices",
        ):
            array = np.asarray(getattr(self, name))
            if array.dtype != np.dtype(np.int32) or array.ndim != 2 or array.shape[1] != 2:
                raise TypeError(f"{name} must be an int32 (N, 2) array")
            object.__setattr__(self, name, np.ascontiguousarray(array).copy())
        for name in (
            "candidate_first_touch_sweep",
            "retained_first_touch_sweep",
            "discarded_first_touch_sweep",
        ):
            array = np.asarray(getattr(self, name))
            if array.dtype != np.dtype(np.int16) or array.ndim != 1:
                raise TypeError(f"{name} must be an int16 vector")
            object.__setattr__(self, name, np.ascontiguousarray(array).copy())
        if len(self.candidate_xy_indices) != len(self.candidate_first_touch_sweep):
            raise ValueError("candidate coordinate and provenance counts must match")
        if len(self.retained_xy_indices) != len(self.retained_first_touch_sweep):
            raise ValueError("retained coordinate and provenance counts must match")
        if len(self.discarded_xy_indices) != len(self.discarded_first_touch_sweep):
            raise ValueError("discarded coordinate and provenance counts must match")

    @property
    def candidate_count(self) -> int:
        return int(len(self.candidate_xy_indices))

    @property
    def retained_count(self) -> int:
        return int(len(self.retained_xy_indices))

    @property
    def discarded_count(self) -> int:
        return int(len(self.discarded_xy_indices))

    @property
    def overflow(self) -> bool:
        return self.discarded_count > 0

    @property
    def overflow_fraction(self) -> float:
        return self.discarded_count / self.candidate_count if self.candidate_count else 0.0

    def summary(self) -> dict[str, object]:
        """Return the compact per-frame evidence record."""

        return {
            "in_range_points": self.in_range_points,
            "candidate_occupied_pillars": self.candidate_count,
            "retained_pillars": self.retained_count,
            "discarded_pillars": self.discarded_count,
            "overflow_count": self.discarded_count,
            "overflow_fraction": self.overflow_fraction,
            "overflow": self.overflow,
            "first_touch_sweep_histogram": {
                "candidate": _histogram(self.candidate_first_touch_sweep),
                "retained": _histogram(self.retained_first_touch_sweep),
                "discarded": _histogram(self.discarded_first_touch_sweep),
            },
        }


def analyze_pillars(points_xyzt: np.ndarray, *, max_voxels: int = MAX_VOXELS) -> PillarAudit:
    """Reproduce exact-fast candidate grouping and first-occurrence capacity order."""

    points = np.asarray(points_xyzt)
    if points.dtype != np.dtype(np.float32):
        raise TypeError("model-ready points must have dtype float32")
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError("model-ready points must have shape (N, 4)")
    if not np.isfinite(points).all():
        raise ValueError("model-ready points must contain only finite values")
    if isinstance(max_voxels, bool) or not isinstance(max_voxels, int) or max_voxels <= 0:
        raise ValueError("max_voxels must be a positive integer")

    x = np.floor((points[:, 0] - POINT_CLOUD_MIN_XY[0]) / VOXEL_SIZE_XY[0]).astype(np.int32)
    y = np.floor((points[:, 1] - POINT_CLOUD_MIN_XY[1]) / VOXEL_SIZE_XY[1]).astype(np.int32)
    valid = (x >= 0) & (x < GRID_SIZE_XY[0]) & (y >= 0) & (y < GRID_SIZE_XY[1])
    valid_indices = np.flatnonzero(valid)
    keys = y[valid] * GRID_SIZE_XY[0] + x[valid]
    if len(keys) == 0:
        empty_xy = np.empty((0, 2), dtype=np.int32)
        empty_touch = np.empty((0,), dtype=np.int16)
        return PillarAudit(
            len(points), empty_xy, empty_xy, empty_xy, empty_touch, empty_touch, empty_touch
        )

    _, unique_positions = np.unique(keys, return_index=True)
    first_rows = valid_indices[unique_positions]
    order = np.argsort(first_rows, kind="stable")
    first_rows = first_rows[order]
    coordinates = np.column_stack([x[first_rows], y[first_rows]]).astype(np.int32, copy=False)
    sweep_indices = _sweep_indices(points[:, 3])[first_rows]
    split = min(max_voxels, len(coordinates))
    return PillarAudit(
        in_range_points=int(valid.sum()),
        candidate_xy_indices=coordinates,
        retained_xy_indices=coordinates[:split],
        discarded_xy_indices=coordinates[split:],
        candidate_first_touch_sweep=sweep_indices,
        retained_first_touch_sweep=sweep_indices[:split],
        discarded_first_touch_sweep=sweep_indices[split:],
    )


def pillar_centres(xy_indices: np.ndarray) -> np.ndarray:
    """Convert integer X/Y cells into metric model-frame pillar centres."""

    coordinates = np.asarray(xy_indices)
    if (
        coordinates.dtype != np.dtype(np.int32)
        or coordinates.ndim != 2
        or coordinates.shape[1] != 2
    ):
        raise TypeError("pillar indices must be an int32 (N, 2) array")
    return np.column_stack(
        [
            POINT_CLOUD_MIN_XY[0] + (coordinates[:, 0] + 0.5) * VOXEL_SIZE_XY[0],
            POINT_CLOUD_MIN_XY[1] + (coordinates[:, 1] + 0.5) * VOXEL_SIZE_XY[1],
        ]
    )


def pillar_box_overlap_mask(
    xy_indices: np.ndarray,
    *,
    center_xy: tuple[float, float],
    size_lw: tuple[float, float],
    yaw_rad: float,
) -> np.ndarray:
    """Return cells whose area intersects one oriented BEV box footprint."""

    centres = pillar_centres(xy_indices)
    if not np.isfinite(np.asarray((*center_xy, *size_lw, yaw_rad), dtype=np.float64)).all():
        raise ValueError("box geometry must be finite")
    if min(size_lw) <= 0.0:
        raise ValueError("box dimensions must be positive")
    cosine, sine = np.cos(yaw_rad), np.sin(yaw_rad)
    delta = centres - np.asarray(center_xy, dtype=np.float64)
    local_x = cosine * delta[:, 0] + sine * delta[:, 1]
    local_y = -sine * delta[:, 0] + cosine * delta[:, 1]
    half_cell = math_sqrt_two() * VOXEL_SIZE_XY[0] / 2.0
    broad = (np.abs(local_x) <= size_lw[0] / 2.0 + half_cell) & (
        np.abs(local_y) <= size_lw[1] / 2.0 + half_cell
    )
    result = np.zeros(len(centres), dtype=np.bool_)
    for index in np.flatnonzero(broad):
        result[index] = _cell_intersects_oriented_box(centres[index], center_xy, size_lw, yaw_rad)
    return result


def spatial_regions(xy_indices: np.ndarray) -> dict[str, np.ndarray]:
    """Assign the frozen 12 sectors, quadrants, and range bins."""

    centres = pillar_centres(xy_indices)
    angles = (np.degrees(np.arctan2(centres[:, 1], centres[:, 0])) + 360.0) % 360.0
    radii = np.linalg.norm(centres, axis=1)
    quadrants = (centres[:, 0] < 0).astype(np.int16) * 2 + (centres[:, 1] < 0).astype(np.int16)
    radial = np.where(radii < 20.0, 0, np.where(radii < 35.0, 1, 2)).astype(np.int16)
    return {
        "azimuth_sector": np.floor(angles / 30.0).astype(np.int16),
        "cartesian_quadrant": quadrants,
        "radial_bin": radial,
    }


def math_sqrt_two() -> float:
    """Return sqrt(2) without adding another heavy dependency."""

    return float(np.sqrt(2.0))


def _sweep_indices(time_lags: np.ndarray) -> np.ndarray:
    result = np.empty(len(time_lags), dtype=np.int16)
    mapping: dict[int, int] = {}
    next_index = 0
    for row, lag in enumerate(np.asarray(time_lags, dtype=np.float32)):
        bits = int(lag.view(np.uint32))
        if bits not in mapping:
            mapping[bits] = next_index
            next_index += 1
        result[row] = mapping[bits]
    return result


def _histogram(values: np.ndarray) -> dict[str, int]:
    if len(values) == 0:
        return {}
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(key)): int(count) for key, count in zip(unique, counts, strict=True)}


def _cell_intersects_oriented_box(
    cell_center: np.ndarray,
    box_center: tuple[float, float],
    size_lw: tuple[float, float],
    yaw: float,
) -> bool:
    half = VOXEL_SIZE_XY[0] / 2.0
    cell = np.array(
        [
            cell_center + [half, half],
            cell_center + [-half, half],
            cell_center + [-half, -half],
            cell_center + [half, -half],
        ],
        dtype=np.float64,
    )
    local = np.array(
        [
            [size_lw[0] / 2.0, size_lw[1] / 2.0],
            [-size_lw[0] / 2.0, size_lw[1] / 2.0],
            [-size_lw[0] / 2.0, -size_lw[1] / 2.0],
            [size_lw[0] / 2.0, -size_lw[1] / 2.0],
        ]
    )
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    box = local @ rotation.T + np.asarray(box_center)
    return _polygons_intersect(cell, box)


def _polygons_intersect(first: np.ndarray, second: np.ndarray) -> bool:
    for polygon in (first, second):
        for index, start in enumerate(polygon):
            edge = polygon[(index + 1) % len(polygon)] - start
            axis = np.array([-edge[1], edge[0]])
            first_projection = first @ axis
            second_projection = second @ axis
            if (
                first_projection.max() < second_projection.min()
                or second_projection.max() < first_projection.min()
            ):
                return False
    return True
