"""Pure CPU structural-capacity contract for the selected M8 DSVT candidate."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from laserperception.detection.m8_input import M8PointCloud

DSVT_POINT_CLOUD_RANGE = (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0)
DSVT_VOXEL_SIZE = (0.3, 0.3, 8.0)
DSVT_GRID_SIZE = (360, 360, 1)
DSVT_THEORETICAL_XY_CELLS = 129_600
DSVT_SET_SIZE = 90
DSVT_BLOCK_COUNT = 4
DSVT_WINDOW_SHAPE = (30, 30, 1)


@dataclass(frozen=True, slots=True)
class DsvtCapacityContract:
    """Static structural limits derived from the selected source/config."""

    point_cloud_range: tuple[float, float, float, float, float, float]
    voxel_size: tuple[float, float, float]
    grid_size: tuple[int, int, int]
    theoretical_xy_cells: int
    dynamic_pillar_cap: int | None
    set_size: int
    block_count: int
    window_shape: tuple[int, int, int]


def load_dsvt_capacity_contract(manifest: Mapping[str, object]) -> DsvtCapacityContract:
    """Parse and verify the manifest's selected-source capacity identity."""

    capacity = _mapping(manifest, "structural_capacity_contract")
    contract = DsvtCapacityContract(
        point_cloud_range=cast(
            tuple[float, float, float, float, float, float],
            _float_tuple(capacity, "point_cloud_range", 6),
        ),
        voxel_size=cast(
            tuple[float, float, float],
            _float_tuple(capacity, "voxel_size", 3),
        ),
        grid_size=cast(
            tuple[int, int, int],
            _int_tuple(capacity, "grid_size", 3),
        ),
        theoretical_xy_cells=_integer(capacity, "theoretical_xy_cell_count"),
        dynamic_pillar_cap=_optional_integer(capacity, "dynamic_pillar_count_cap"),
        set_size=_integer(capacity, "set_size"),
        block_count=_integer(capacity, "block_count"),
        window_shape=cast(
            tuple[int, int, int],
            _int_tuple(capacity, "window_shape", 3),
        ),
    )
    expected = DsvtCapacityContract(
        point_cloud_range=DSVT_POINT_CLOUD_RANGE,
        voxel_size=DSVT_VOXEL_SIZE,
        grid_size=DSVT_GRID_SIZE,
        theoretical_xy_cells=DSVT_THEORETICAL_XY_CELLS,
        dynamic_pillar_cap=None,
        set_size=DSVT_SET_SIZE,
        block_count=DSVT_BLOCK_COUNT,
        window_shape=DSVT_WINDOW_SHAPE,
    )
    if contract != expected:
        raise ValueError(
            "M8 manifest structural-capacity identity is not the selected DSVT contract"
        )
    return contract


def candidate_dynamic_pillar_coordinates(points_xyzit: M8PointCloud | np.ndarray) -> np.ndarray:
    """Return a CPU analytic replica of occupied candidate XY coordinates.

    This is exact for ordinary analytic fixtures. CUDA float32 division can
    round points lying extremely near a cell boundary differently, so the
    canonical corpus census uses :func:`candidate_dynamic_pillar_coordinates_cuda`.
    """

    points = (
        points_xyzit.points
        if isinstance(points_xyzit, M8PointCloud)
        else M8PointCloud(points_xyzit).points
    )
    point_range = np.asarray(DSVT_POINT_CLOUD_RANGE, dtype=np.float32)
    inside = np.all(points[:, :3] >= point_range[:3], axis=1) & np.all(
        points[:, :3] < point_range[3:], axis=1
    )
    selected = points[inside]
    if selected.shape[0] == 0:
        raise ValueError("candidate range removed every input point")
    xy = np.floor((selected[:, :2] - point_range[:2]) / np.asarray(DSVT_VOXEL_SIZE[:2])).astype(
        np.int32
    )
    grid_xy = np.asarray(DSVT_GRID_SIZE[:2], dtype=np.int32)
    if np.any(xy < 0) or np.any(xy >= grid_xy):
        raise RuntimeError("candidate coordinate generation escaped the selected DSVT grid")
    merged = xy[:, 0] * np.int32(DSVT_GRID_SIZE[1]) + xy[:, 1]
    unique = np.unique(merged)
    coordinates = np.column_stack((unique // DSVT_GRID_SIZE[1], unique % DSVT_GRID_SIZE[1])).astype(
        np.int32
    )
    return np.ascontiguousarray(coordinates)


def candidate_dynamic_pillar_count(points_xyzit: M8PointCloud | np.ndarray) -> int:
    """Return the CPU analytic candidate count without running a detector."""

    return int(candidate_dynamic_pillar_coordinates(points_xyzit).shape[0])


def candidate_dynamic_pillar_coordinates_cuda(
    points_xyzit: M8PointCloud | np.ndarray,
    *,
    torch_module: Any,
    device: str = "cuda:0",
) -> np.ndarray:
    """Execute the selected upstream CUDA coordinate arithmetic exactly.

    The caller supplies Torch so importing the lightweight core package never
    imports an optional GPU dependency. This function runs coordinate
    generation and ``unique`` only; it does not load model weights or execute
    a detector layer.
    """

    points = (
        points_xyzit.points
        if isinstance(points_xyzit, M8PointCloud)
        else M8PointCloud(points_xyzit).points
    )
    point_range = np.asarray(DSVT_POINT_CLOUD_RANGE, dtype=np.float32)
    inside = np.all(points[:, :3] >= point_range[:3], axis=1) & np.all(
        points[:, :3] < point_range[3:], axis=1
    )
    selected = np.ascontiguousarray(points[inside])
    if selected.shape[0] == 0:
        raise ValueError("candidate range removed every input point")
    batch_column = np.zeros((selected.shape[0], 1), dtype=np.float32)
    candidate = np.concatenate((batch_column, selected), axis=1)
    tensor = torch_module.from_numpy(candidate).to(device=device)
    cuda_range = torch_module.tensor(
        DSVT_POINT_CLOUD_RANGE, dtype=torch_module.float32, device=device
    )
    cuda_voxel = torch_module.tensor(DSVT_VOXEL_SIZE, dtype=torch_module.float32, device=device)
    cuda_grid = torch_module.tensor(DSVT_GRID_SIZE, dtype=torch_module.int64, device=device)
    coordinates = torch_module.floor(
        (tensor[:, [1, 2]] - cuda_range[[0, 1]]) / cuda_voxel[[0, 1]]
    ).int()
    mask = ((coordinates >= 0) & (coordinates < cuda_grid[[0, 1]])).all(dim=1)
    coordinates = coordinates[mask]
    merged = (
        tensor[mask, 0].int() * DSVT_THEORETICAL_XY_CELLS
        + coordinates[:, 0] * DSVT_GRID_SIZE[1]
        + coordinates[:, 1]
    )
    unique = torch_module.unique(merged, dim=0)
    xy = torch_module.stack((unique // DSVT_GRID_SIZE[1], unique % DSVT_GRID_SIZE[1]), dim=1)
    return np.ascontiguousarray(xy.detach().cpu().numpy(), dtype=np.int32)


def candidate_dynamic_pillar_count_cuda(
    points_xyzit: M8PointCloud | np.ndarray,
    *,
    torch_module: Any,
    device: str = "cuda:0",
) -> int:
    """Count candidate pillars with exact selected CUDA arithmetic only."""

    return int(
        candidate_dynamic_pillar_coordinates_cuda(
            points_xyzit, torch_module=torch_module, device=device
        ).shape[0]
    )


def require_capacity(
    pillar_count: int,
    *,
    contract: DsvtCapacityContract,
    deployment_profile_max: int | None = None,
) -> None:
    """Fail closed on a candidate or explicitly requested deployment limit."""

    if isinstance(pillar_count, bool) or not isinstance(pillar_count, int) or pillar_count <= 0:
        raise ValueError("pillar_count must be a positive integer")
    if pillar_count > contract.theoretical_xy_cells:
        raise RuntimeError("pillar count exceeds the selected spatial grid")
    if contract.dynamic_pillar_cap is not None and pillar_count > contract.dynamic_pillar_cap:
        raise RuntimeError("pillar count exceeds the selected runtime cap")
    if deployment_profile_max is not None and pillar_count > deployment_profile_max:
        raise RuntimeError("pillar count exceeds the selected TensorRT profile")


def _mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"M8 manifest {key} must be a mapping")
    return value


def _integer(parent: Mapping[str, object], key: str) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"M8 manifest {key} must be an integer")
    return value


def _optional_integer(parent: Mapping[str, object], key: str) -> int | None:
    value = parent.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"M8 manifest {key} must be an integer or null")
    return value


def _float_tuple(parent: Mapping[str, object], key: str, length: int) -> tuple[float, ...]:
    value = parent.get(key)
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"M8 manifest {key} must have {length} values")
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        raise ValueError(f"M8 manifest {key} values must be numeric")
    return tuple(float(item) for item in value)


def _int_tuple(parent: Mapping[str, object], key: str, length: int) -> tuple[int, ...]:
    value = parent.get(key)
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"M8 manifest {key} must have {length} values")
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        raise ValueError(f"M8 manifest {key} values must be integers")
    return tuple(value)
