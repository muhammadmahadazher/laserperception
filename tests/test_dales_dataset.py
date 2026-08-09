from pathlib import Path

import laspy
import numpy as np
import pytest

from laserperception.datasets import DalesDataset, PatchBounds
from laserperception.ontology import IGNORE_ID, map_dales_labels, mapping_coverage
from laserperception.transforms import normalize_coordinates


def _write_tile(
    path: Path,
    xyz: np.ndarray,
    labels: np.ndarray,
    *,
    with_extra_dimensions: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.01, 0.01, 0.01])
    las = laspy.LasData(header)
    if with_extra_dimensions:
        las.add_extra_dim(laspy.ExtraBytesParams(name="temperature", type=np.float32))
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    las.classification = labels
    las.intensity = np.arange(len(xyz), dtype=np.uint16) + 100
    if with_extra_dimensions:
        las.temperature = np.arange(len(xyz), dtype=np.float32) + 20.0
    las.write(path)


def _boundary_tile(root: Path) -> Path:
    xyz = np.array(
        [
            [0.0, 0.0, 1.0],
            [49.99, 49.99, 2.0],
            [50.0, 0.0, 3.0],
            [99.99, 50.0, 4.0],
            [100.0, 100.0, 5.0],
        ],
        dtype=np.float64,
    )
    labels = np.array([1, 2, 3, 5, 8], dtype=np.uint8)
    path = root / "train" / "boundary.las"
    _write_tile(path, xyz, labels, with_extra_dimensions=True)
    return path


def test_tile_discovery_is_split_explicit_and_deterministic(tmp_path: Path) -> None:
    _write_tile(
        tmp_path / "train" / "z.las",
        np.array([[0.0, 0.0, 0.0]]),
        np.array([1], dtype=np.uint8),
    )
    _write_tile(
        tmp_path / "train" / "nested" / "a.las",
        np.array([[1.0, 1.0, 1.0]]),
        np.array([2], dtype=np.uint8),
    )

    dataset = DalesDataset(tmp_path, split="train")

    assert len(dataset) == 2
    assert [dataset.tile_info(index).relative_path for index in range(2)] == [
        "nested/a.las",
        "z.las",
    ]


def test_split_validation_and_missing_tiles_fail_cleanly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="root does not exist"):
        DalesDataset(tmp_path / "missing", split="train")
    with pytest.raises(ValueError, match="unsupported DALES split"):
        DalesDataset(tmp_path, split="valid")
    with pytest.raises(FileNotFoundError, match="explicit 'train' split"):
        DalesDataset(tmp_path, split="train")
    (tmp_path / "train").mkdir()
    with pytest.raises(FileNotFoundError, match="no DALES LAS/LAZ tiles"):
        DalesDataset(tmp_path, split="train")


def test_chunked_reader_retains_only_experiment_dimensions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _boundary_tile(tmp_path)
    dataset = DalesDataset(tmp_path, split="train")

    def fail_full_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("dataset fast path must not call laspy.read")

    monkeypatch.setattr(laspy, "read", fail_full_read)
    chunks = list(dataset.iter_tile_chunks(0, chunk_size=2))

    assert [len(chunk) for chunk in chunks] == [2, 2, 1]
    assert all(chunk.attributes == {} for chunk in chunks)
    assert all(chunk.labels is not None for chunk in chunks)
    assert chunks[0].metadata["retained_dimensions"] == (
        "x",
        "y",
        "z",
        "classification",
    )
    assert "intensity" in chunks[0].metadata["available_dimensions"]
    assert "temperature" in chunks[0].metadata["available_dimensions"]


def test_partition_uses_half_open_non_overlapping_grid_and_conserves_points(
    tmp_path: Path,
) -> None:
    _boundary_tile(tmp_path)
    dataset = DalesDataset(tmp_path, split="train")

    partition = dataset.partition_tile(0, patch_size_m=(50.0, 50.0), chunk_size=2)

    assert partition.total_point_count == 5
    assert partition.finite_point_count == 5
    assert partition.non_finite_point_count == 0
    assert partition.grid_cell_count == 9
    assert partition.empty_patch_count == 5
    assert sum(len(patch.cloud) for patch in partition.patches) == 5
    assert [(patch.info.row, patch.info.column) for patch in partition.patches] == [
        (0, 0),
        (0, 1),
        (1, 1),
        (2, 2),
    ]

    patch_points = {
        (patch.info.row, patch.info.column): patch.cloud.xyz for patch in partition.patches
    }
    assert patch_points[(0, 0)].shape[0] == 2
    assert np.any(np.all(np.isclose(patch_points[(0, 1)], [50.0, 0.0, 3.0]), axis=1))
    assert np.any(np.all(np.isclose(patch_points[(1, 1)], [99.99, 50.0, 4.0]), axis=1))
    assert np.any(np.all(np.isclose(patch_points[(2, 2)], [100.0, 100.0, 5.0]), axis=1))

    all_points = np.concatenate([patch.cloud.xyz for patch in partition.patches])
    assert len({tuple(point) for point in all_points}) == 5
    assert all(patch.cloud.attributes == {} for patch in partition.patches)


def test_patch_then_normalize_then_map_is_explicit_and_preserves_raw_patch(tmp_path: Path) -> None:
    _boundary_tile(tmp_path)
    partition = DalesDataset(tmp_path, split="train").partition_tile(0, patch_size_m=50.0)
    patch = partition.patches[0].cloud
    raw_xyz = patch.xyz.copy()
    raw_labels = patch.labels.copy() if patch.labels is not None else None

    normalized = normalize_coordinates(patch)
    assert np.array_equal(patch.xyz, raw_xyz)
    assert np.allclose(normalized.xyz.min(axis=0), np.zeros(3))
    assert raw_labels is not None and np.array_equal(normalized.labels, raw_labels)

    all_labels = np.concatenate(
        [sample.cloud.labels for sample in partition.patches if sample.cloud.labels is not None]
    )
    mapped = map_dales_labels(all_labels)
    coverage = mapping_coverage(all_labels, mapped)
    assert coverage.total_count == 5
    assert coverage.ignored_count == 1
    assert coverage.mapped_histogram[0] == 1
    assert mapped.tolist().count(IGNORE_ID) == 1


def test_patch_bounds_boundary_policy() -> None:
    bounds = PatchBounds(0.0, 0.0, 50.0, 50.0)
    xy = np.array([[0.0, 0.0], [49.99, 49.99], [50.0, 1.0], [1.0, 50.0]])
    assert bounds.contains_xy(xy).tolist() == [True, True, False, False]
    with pytest.raises(ValueError, match="maximum"):
        PatchBounds(0.0, 0.0, 0.0, 1.0)


def test_empty_tile_produces_no_patches(tmp_path: Path) -> None:
    empty_path = tmp_path / "train" / "empty.las"
    _write_tile(
        empty_path,
        np.empty((0, 3), dtype=np.float64),
        np.empty((0,), dtype=np.uint8),
    )
    partition = DalesDataset(tmp_path, split="train").partition_tile(0)
    assert partition.patches == ()
    assert partition.total_point_count == 0
    assert partition.empty_patch_count == 0
