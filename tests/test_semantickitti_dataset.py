from pathlib import Path

import numpy as np
import pytest

from laserperception.datasets import SEMANTICKITTI_SPLITS, SemanticKITTIDataset


def _write_scan(root: Path, sequence: str, frame: str, semantic_ids: list[int]) -> None:
    scan_dir = root / "sequences" / sequence / "velodyne"
    label_dir = root / "sequences" / sequence / "labels"
    scan_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    point_count = len(semantic_ids)
    records = np.arange(point_count * 4, dtype="<f4").reshape(point_count, 4)
    records.tofile(scan_dir / f"{frame}.bin")
    semantic = np.asarray(semantic_ids, dtype=np.uint32)
    instance = np.arange(point_count, dtype=np.uint32)
    ((instance << np.uint32(16)) | semantic).astype("<u4").tofile(label_dir / f"{frame}.label")


def test_official_split_manifest_is_pinned_and_deterministic() -> None:
    assert SEMANTICKITTI_SPLITS["train"] == (
        "00",
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "09",
        "10",
    )
    assert SEMANTICKITTI_SPLITS["valid"] == ("08",)
    assert SEMANTICKITTI_SPLITS["test"] == tuple(f"{value:02d}" for value in range(11, 22))


def test_discovery_orders_official_sequences_and_numeric_frames(tmp_path: Path) -> None:
    _write_scan(tmp_path, "01", "000010", [70])
    _write_scan(tmp_path, "00", "000002", [40, 50])
    _write_scan(tmp_path, "00", "000001", [10])

    dataset = SemanticKITTIDataset(tmp_path, split="train", sequences=["01", "00"])

    assert len(dataset) == 3
    assert [(dataset.sample_info(i).sequence, dataset.sample_info(i).frame) for i in range(3)] == [
        ("00", "000001"),
        ("00", "000002"),
        ("01", "000010"),
    ]
    cloud = dataset.load(1)
    assert len(cloud) == 2
    assert cloud.labels is not None and cloud.labels.tolist() == [40, 50]
    assert "remission" in cloud.attributes
    assert cloud.metadata["coordinates_normalized"] is False


def test_test_split_allows_missing_labels_by_default(tmp_path: Path) -> None:
    scan_dir = tmp_path / "sequences" / "11" / "velodyne"
    scan_dir.mkdir(parents=True)
    np.zeros((2, 4), dtype="<f4").tofile(scan_dir / "000000.bin")

    dataset = SemanticKITTIDataset(tmp_path, split="test", sequences=[11])

    assert dataset.sample_info(0).label_path is None
    assert dataset.load(0).labels is None


def test_missing_root_sequence_and_scan_fail_cleanly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="root does not exist"):
        SemanticKITTIDataset(tmp_path / "missing", split="train")

    (tmp_path / "sequences").mkdir()
    with pytest.raises(FileNotFoundError, match="sequence directory: 00"):
        SemanticKITTIDataset(tmp_path, split="train", sequences=["00"])

    (tmp_path / "sequences" / "00" / "velodyne").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no SemanticKITTI .bin scans"):
        SemanticKITTIDataset(tmp_path, split="train", sequences=["00"])


def test_broken_scan_label_pairs_are_rejected(tmp_path: Path) -> None:
    _write_scan(tmp_path, "00", "000000", [40])
    label_dir = tmp_path / "sequences" / "00" / "labels"
    (label_dir / "000000.label").unlink()
    np.zeros(1, dtype="<u4").tofile(label_dir / "000001.label")

    with pytest.raises(ValueError, match="labels without scans"):
        SemanticKITTIDataset(tmp_path, split="train", sequences=["00"])

    (label_dir / "000001.label").unlink()
    with pytest.raises(ValueError, match="scans without labels"):
        SemanticKITTIDataset(tmp_path, split="train", sequences=["00"])


def test_label_point_count_mismatch_is_rejected_on_load(tmp_path: Path) -> None:
    _write_scan(tmp_path, "00", "000000", [40, 50])
    label_path = tmp_path / "sequences" / "00" / "labels" / "000000.label"
    np.zeros(1, dtype="<u4").tofile(label_path)
    dataset = SemanticKITTIDataset(tmp_path, split="train", sequences=["00"])

    with pytest.raises(ValueError, match="label count"):
        dataset.load(0)


def test_split_and_subset_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported SemanticKITTI split"):
        SemanticKITTIDataset(tmp_path, split="validation")
    with pytest.raises(ValueError, match="do not belong"):
        SemanticKITTIDataset(tmp_path, split="train", sequences=["08"])
