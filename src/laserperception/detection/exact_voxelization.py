"""Experimental exact-semantics hard voxelization using lazy CUDA dependencies."""

from __future__ import annotations

import importlib
from typing import Any


class ExactDeterministicVoxelizer:
    """Reproduce pinned MMCV deterministic hard-voxel selection with tensor ops.

    The class is diagnostic-only. Importing this module remains CPU-safe; PyTorch and
    MMCV are loaded lazily when an instance is constructed.
    """

    def __init__(self, official_layer: Any) -> None:
        self._torch = importlib.import_module("torch")
        voxelization = importlib.import_module("mmcv.ops").Voxelization
        self.voxel_size = tuple(float(value) for value in official_layer.voxel_size)
        self.point_cloud_range = tuple(float(value) for value in official_layer.point_cloud_range)
        self.max_num_points = int(official_layer.max_num_points)
        self.max_voxels = tuple(int(value) for value in official_layer.max_voxels)
        grid_value = getattr(official_layer, "grid_size", None)
        if grid_value is None:
            grid_value = getattr(official_layer, "grid_shape", None)
        if grid_value is None:
            raise ValueError("official voxel layer does not expose grid_size or grid_shape")
        values = grid_value.tolist() if hasattr(grid_value, "tolist") else grid_value
        grid = tuple(int(value) for value in values)
        if len(grid) != 3 or any(value <= 0 for value in grid):
            raise ValueError("official voxel grid must contain three positive dimensions")
        if self.max_num_points <= 0 or any(value <= 0 for value in self.max_voxels):
            raise ValueError("official hard-voxel capacities must be positive")
        self.grid_size_xyz = grid
        self.dynamic_coordinate_layer = voxelization(
            voxel_size=list(self.voxel_size),
            point_cloud_range=list(self.point_cloud_range),
            max_num_points=-1,
            max_voxels=-1,
            deterministic=True,
        )
        self.dynamic_coordinate_layer.eval()

    @property
    def training(self) -> bool:
        """Expose the effective capacity mode used by the candidate."""

        return bool(self.dynamic_coordinate_layer.training)

    def eval(self) -> ExactDeterministicVoxelizer:
        """Use the official test-time ``max_voxels`` capacity."""

        self.dynamic_coordinate_layer.eval()
        return self

    def __call__(self, points: Any) -> tuple[Any, Any, Any]:
        """Return exact hard voxels, Z/Y/X coordinates, and point counts."""

        torch = self._torch
        if getattr(points, "device", None) != torch.device("cuda:0"):
            raise ValueError("exact candidate requires points on cuda:0")
        if points.dtype != torch.float32:
            raise ValueError("exact candidate requires float32 points")
        if points.ndim != 2 or int(points.shape[1]) < 3:
            raise ValueError("points must have shape (N, C) with at least XYZ")

        source = points.contiguous()
        coordinates = self.dynamic_coordinate_layer(source)
        if coordinates.dtype != torch.int32 or tuple(coordinates.shape) != (
            int(source.shape[0]),
            3,
        ):
            raise RuntimeError("MMCV dynamic coordinates violated the pinned int32 (N, 3) contract")

        valid_mask = coordinates[:, 0] >= 0
        original_indices = torch.arange(
            int(source.shape[0]), device=source.device, dtype=torch.int64
        )[valid_mask]
        valid_coordinates = coordinates[valid_mask]
        if int(original_indices.numel()) == 0:
            return (
                source.new_zeros((0, self.max_num_points, int(source.shape[1]))),
                coordinates.new_zeros((0, 3)),
                coordinates.new_zeros((0,)),
            )

        grid_x, grid_y, _ = self.grid_size_xyz
        coordinate_keys = (
            valid_coordinates[:, 0].to(torch.int64) * (grid_y * grid_x)
            + valid_coordinates[:, 1].to(torch.int64) * grid_x
            + valid_coordinates[:, 2].to(torch.int64)
        )
        point_stride = int(source.shape[0]) + 1
        composite_keys = coordinate_keys * point_stride + original_indices
        sorted_positions = torch.argsort(composite_keys)
        sorted_keys = coordinate_keys[sorted_positions]
        sorted_original_indices = original_indices[sorted_positions]
        sorted_coordinates = valid_coordinates[sorted_positions]

        group_starts_mask = torch.ones_like(sorted_keys, dtype=torch.bool)
        group_starts_mask[1:] = sorted_keys[1:] != sorted_keys[:-1]
        group_starts = torch.nonzero(group_starts_mask, as_tuple=False).flatten()
        group_ids = torch.cumsum(group_starts_mask.to(torch.int64), dim=0) - 1
        positions = torch.arange(int(sorted_keys.numel()), device=source.device, dtype=torch.int64)
        positions_in_group = positions - group_starts[group_ids]
        group_ends = torch.cat(
            [
                group_starts[1:],
                group_starts.new_tensor([int(sorted_keys.numel())]),
            ]
        )
        group_counts = group_ends - group_starts

        first_original_indices = sorted_original_indices[group_starts]
        first_occurrence_order = torch.argsort(first_original_indices)
        capacity_index = 0 if self.training else 1
        accepted_groups = first_occurrence_order[: self.max_voxels[capacity_index]]
        voxel_count = int(accepted_groups.numel())

        group_to_voxel = torch.full(
            (int(group_starts.numel()),),
            -1,
            device=source.device,
            dtype=torch.int64,
        )
        group_to_voxel[accepted_groups] = torch.arange(
            voxel_count, device=source.device, dtype=torch.int64
        )
        destination_voxels = group_to_voxel[group_ids]
        retained = (destination_voxels >= 0) & (positions_in_group < self.max_num_points)

        voxels = source.new_zeros((voxel_count, self.max_num_points, int(source.shape[1])))
        voxels[
            destination_voxels[retained],
            positions_in_group[retained],
        ] = source[sorted_original_indices[retained]]
        output_coordinates = sorted_coordinates[group_starts[accepted_groups]].contiguous()
        num_points = group_counts[accepted_groups].clamp(max=self.max_num_points).to(torch.int32)
        return voxels, output_coordinates, num_points.contiguous()
