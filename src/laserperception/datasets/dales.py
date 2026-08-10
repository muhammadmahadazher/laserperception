"""Memory-conscious DALES tile discovery and deterministic grid patching.

DALES format and split-level facts come from the dataset paper:
https://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Varney_DALES_A_Large-Scale_Aerial_LiDAR_Data_Set_for_Semantic_Segmentation_CVPRW_2020_paper.html

Large tiles are read with laspy's documented ``LasReader.chunk_iterator`` API. Only scaled XYZ and
classification are retained. Patches are raw, non-overlapping, and are not normalized or mapped.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import laspy
import numpy as np

from laserperception.core import PointCloud

DALES_ADAPTER_VERSION: Final = "dales-chunked-grid-v1"
DALES_EXPECTED_TILE_COUNTS = MappingProxyType({"train": 29, "test": 11})
_REQUIRED_DIMENSIONS = ("x", "y", "z", "classification")


@dataclass(frozen=True)
class PatchBounds:
    """Horizontal half-open bounds ``[xmin, xmax) × [ymin, ymax)`` in meters."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def __post_init__(self) -> None:
        values = np.asarray((self.xmin, self.ymin, self.xmax, self.ymax), dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("patch bounds must be finite")
        if self.xmax <= self.xmin or self.ymax <= self.ymin:
            raise ValueError("patch maximum bounds must exceed minimum bounds")

    def contains_xy(self, xy: np.ndarray) -> np.ndarray:
        """Return a mask using the documented half-open boundary policy."""
        points = np.asarray(xy)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"xy must have shape (N, 2); received {points.shape}")
        return (
            (points[:, 0] >= self.xmin)
            & (points[:, 0] < self.xmax)
            & (points[:, 1] >= self.ymin)
            & (points[:, 1] < self.ymax)
        )


@dataclass(frozen=True)
class DalesTileInfo:
    """Stable provenance for one discovered DALES tile."""

    dataset: str
    split: str
    tile_id: str
    tile_path: Path
    relative_path: str


@dataclass(frozen=True)
class DalesPatchInfo:
    """Stable provenance and grid location for one non-empty DALES patch."""

    dataset: str
    split: str
    tile_id: str
    row: int
    column: int
    bounds: PatchBounds


@dataclass(frozen=True)
class DalesPatchSample:
    """One raw, non-empty spatial patch and its provenance."""

    info: DalesPatchInfo
    cloud: PointCloud


@dataclass(frozen=True)
class DalesTilePartition:
    """Result of one streamed tile partitioning pass."""

    tile: DalesTileInfo
    patches: tuple[DalesPatchSample, ...]
    total_point_count: int
    finite_point_count: int
    non_finite_point_count: int
    raw_xyz_min: tuple[float, float, float] | None
    raw_xyz_max: tuple[float, float, float] | None
    grid_cell_count: int
    empty_patch_count: int


def _patch_size_xy(patch_size_m: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(patch_size_m, tuple):
        if len(patch_size_m) != 2:
            raise ValueError("patch_size_m tuple must contain x and y sizes")
        size_x, size_y = float(patch_size_m[0]), float(patch_size_m[1])
    else:
        size_x = size_y = float(patch_size_m)
    if not np.isfinite(size_x) or not np.isfinite(size_y) or size_x <= 0 or size_y <= 0:
        raise ValueError("patch sizes must be positive finite values")
    return size_x, size_y


def _dimension_names(header: laspy.LasHeader) -> tuple[str, ...]:
    return tuple(str(name) for name in header.point_format.dimension_names)


class DalesDataset:
    """Discover DALES LAS/LAZ tiles and partition them without loading optional dimensions.

    The official paper defines 29 training and 11 test tiles but does not provide a stable filename
    manifest. This adapter therefore requires an explicit ``train`` or ``test`` directory and
    discovers its LAS/LAZ files deterministically. Expected counts are recorded, not enforced, so
    subsets can be audited safely.
    """

    def __init__(self, root: str | Path, *, split: str) -> None:
        self.root = Path(root).expanduser()
        if not self.root.is_dir():
            raise FileNotFoundError(f"DALES root does not exist: {self.root}")
        if split not in DALES_EXPECTED_TILE_COUNTS:
            supported = ", ".join(DALES_EXPECTED_TILE_COUNTS)
            raise ValueError(f"unsupported DALES split {split!r}; choose {supported}")
        self.split = split
        self._split_root = self._resolve_split_root()
        self._tiles = self._discover_tiles()

    def _resolve_split_root(self) -> Path:
        candidates: tuple[Path, ...] = (
            self.root / self.split,
            self.root / "dales_las" / self.split,
        )
        if self.root.name.lower() == self.split:
            candidates = (self.root, *candidates)
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        raise FileNotFoundError(
            f"DALES root must contain an explicit {self.split!r} split directory"
        )

    def _discover_tiles(self) -> tuple[DalesTileInfo, ...]:
        paths = sorted(
            (
                path
                for path in self._split_root.rglob("*")
                if path.is_file() and path.suffix.lower() in {".las", ".laz"}
            ),
            key=lambda path: path.relative_to(self._split_root).as_posix().casefold(),
        )
        if not paths:
            raise FileNotFoundError(f"no DALES LAS/LAZ tiles found in {self._split_root}")
        return tuple(
            DalesTileInfo(
                dataset="dales",
                split=self.split,
                tile_id=path.stem,
                tile_path=path,
                relative_path=path.relative_to(self._split_root).as_posix(),
            )
            for path in paths
        )

    def __len__(self) -> int:
        """Return the number of deterministically discovered tiles."""
        return len(self._tiles)

    def tile_info(self, index: int) -> DalesTileInfo:
        """Return tile provenance without reading point records."""
        return self._tiles[index]

    def iter_tile_chunks(self, index: int, *, chunk_size: int = 1_000_000) -> Iterator[PointCloud]:
        """Yield raw XYZ and classification chunks without retaining other LAS dimensions."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        tile = self.tile_info(index)
        with laspy.open(tile.tile_path) as reader:
            names = _dimension_names(reader.header)
            if "classification" not in names:
                raise ValueError(f"DALES tile lacks classification dimension: {tile.relative_path}")
            for chunk_index, points in enumerate(reader.chunk_iterator(chunk_size)):
                xyz = np.column_stack(
                    (
                        np.asarray(points.x, dtype=np.float64),
                        np.asarray(points.y, dtype=np.float64),
                        np.asarray(points.z, dtype=np.float64),
                    )
                )
                labels = np.asarray(points.classification)
                yield PointCloud(
                    xyz=xyz,
                    labels=labels,
                    metadata={
                        "format": tile.tile_path.suffix.lower().lstrip("."),
                        "source_file": tile.tile_path.name,
                        "dataset": "dales",
                        "split": self.split,
                        "tile_id": tile.tile_id,
                        "chunk_index": chunk_index,
                        "reader": "laspy_chunk_iterator",
                        "retained_dimensions": _REQUIRED_DIMENSIONS,
                        "available_dimensions": names,
                        "coordinates_normalized": False,
                    },
                )

    def partition_tile(
        self,
        index: int,
        *,
        patch_size_m: float | tuple[float, float] = 50.0,
        chunk_size: int = 1_000_000,
    ) -> DalesTilePartition:
        """Partition one tile in a single chunked pass and return only non-empty patches.

        Grid origin is the tile header's minimum X/Y. Boundaries are half-open. Points exactly on
        an internal maximum boundary belong to the adjacent patch. Non-finite points are counted
        but cannot be assigned to a spatial cell. Normalization and ontology mapping are not
        performed.
        """
        size_x, size_y = _patch_size_xy(patch_size_m)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        tile = self.tile_info(index)

        xyz_buckets: dict[int, list[np.ndarray]] = {}
        label_buckets: dict[int, list[np.ndarray]] = {}
        total_points = 0
        non_finite_points = 0
        raw_min: np.ndarray | None = None
        raw_max: np.ndarray | None = None

        with laspy.open(tile.tile_path) as reader:
            names = _dimension_names(reader.header)
            if "classification" not in names:
                raise ValueError(f"DALES tile lacks classification dimension: {tile.relative_path}")
            if int(reader.header.point_count) == 0:
                return DalesTilePartition(tile, (), 0, 0, 0, None, None, 0, 0)

            origin_x = float(reader.header.x_min)
            origin_y = float(reader.header.y_min)
            max_x = float(reader.header.x_max)
            max_y = float(reader.header.y_max)
            columns = max(1, int(np.floor((max_x - origin_x) / size_x)) + 1)
            rows = max(1, int(np.floor((max_y - origin_y) / size_y)) + 1)
            grid_cell_count = rows * columns

            for points in reader.chunk_iterator(chunk_size):
                xyz = np.column_stack(
                    (
                        np.asarray(points.x, dtype=np.float64),
                        np.asarray(points.y, dtype=np.float64),
                        np.asarray(points.z, dtype=np.float64),
                    )
                )
                labels = np.asarray(points.classification)
                total_points += int(xyz.shape[0])
                finite_mask = np.all(np.isfinite(xyz), axis=1)
                non_finite_points += int(np.count_nonzero(~finite_mask))
                if not np.any(finite_mask):
                    continue
                finite_xyz = xyz[finite_mask]
                finite_labels = labels[finite_mask]
                chunk_min = finite_xyz.min(axis=0)
                chunk_max = finite_xyz.max(axis=0)
                raw_min = chunk_min if raw_min is None else np.minimum(raw_min, chunk_min)
                raw_max = chunk_max if raw_max is None else np.maximum(raw_max, chunk_max)

                column_ids = np.floor((finite_xyz[:, 0] - origin_x) / size_x).astype(np.int64)
                row_ids = np.floor((finite_xyz[:, 1] - origin_y) / size_y).astype(np.int64)
                if (
                    np.any(column_ids < 0)
                    or np.any(column_ids >= columns)
                    or np.any(row_ids < 0)
                    or np.any(row_ids >= rows)
                ):
                    raise ValueError(
                        f"DALES tile points fall outside LAS header bounds: {tile.relative_path}"
                    )
                grid_ids = row_ids * columns + column_ids
                for grid_id in np.unique(grid_ids):
                    mask = grid_ids == grid_id
                    key = int(grid_id)
                    xyz_buckets.setdefault(key, []).append(
                        finite_xyz[mask].astype(np.float32, copy=True)
                    )
                    label_buckets.setdefault(key, []).append(
                        np.array(finite_labels[mask], copy=True)
                    )

        patches: list[DalesPatchSample] = []
        assigned_points = 0
        for grid_id in sorted(xyz_buckets):
            row, column = divmod(grid_id, columns)
            xyz = np.concatenate(xyz_buckets.pop(grid_id), axis=0)
            labels = np.concatenate(label_buckets.pop(grid_id), axis=0)
            assigned_points += int(xyz.shape[0])
            bounds = PatchBounds(
                xmin=origin_x + column * size_x,
                ymin=origin_y + row * size_y,
                xmax=origin_x + (column + 1) * size_x,
                ymax=origin_y + (row + 1) * size_y,
            )
            info = DalesPatchInfo(
                dataset="dales",
                split=self.split,
                tile_id=tile.tile_id,
                row=row,
                column=column,
                bounds=bounds,
            )
            patches.append(
                DalesPatchSample(
                    info=info,
                    cloud=PointCloud(
                        xyz=xyz,
                        labels=labels,
                        metadata={
                            "format": tile.tile_path.suffix.lower().lstrip("."),
                            "source_file": tile.tile_path.name,
                            "dataset": "dales",
                            "split": self.split,
                            "tile_id": tile.tile_id,
                            "patch_row": row,
                            "patch_column": column,
                            "patch_bounds_xy": (
                                bounds.xmin,
                                bounds.ymin,
                                bounds.xmax,
                                bounds.ymax,
                            ),
                            "patch_boundary_policy": "half_open_xy",
                            "reader": "laspy_chunk_iterator",
                            "retained_dimensions": _REQUIRED_DIMENSIONS,
                            "available_dimensions": names,
                            "coordinates_normalized": False,
                        },
                    ),
                )
            )

        finite_points = total_points - non_finite_points
        if assigned_points != finite_points:
            raise RuntimeError(
                f"internal patch conservation error: assigned {assigned_points} of {finite_points}"
            )
        raw_min_tuple = (
            None if raw_min is None else (float(raw_min[0]), float(raw_min[1]), float(raw_min[2]))
        )
        raw_max_tuple = (
            None if raw_max is None else (float(raw_max[0]), float(raw_max[1]), float(raw_max[2]))
        )
        return DalesTilePartition(
            tile=tile,
            patches=tuple(patches),
            total_point_count=total_points,
            finite_point_count=finite_points,
            non_finite_point_count=non_finite_points,
            raw_xyz_min=raw_min_tuple,
            raw_xyz_max=raw_max_tuple,
            grid_cell_count=grid_cell_count,
            empty_patch_count=grid_cell_count - len(patches),
        )
