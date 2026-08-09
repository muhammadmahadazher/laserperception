import json
from pathlib import Path

import laspy
import numpy as np
import pytest

from laserperception.audit import audit_dales, audit_semantickitti, main, write_json_report
from laserperception.datasets import DalesDataset, SemanticKITTIDataset


def _write_semantic_scan(
    root: Path, sequence: str, frame: str, xyz: np.ndarray, semantic: np.ndarray
) -> None:
    scan_dir = root / "sequences" / sequence / "velodyne"
    label_dir = root / "sequences" / sequence / "labels"
    scan_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    remission = np.linspace(0.0, 1.0, len(xyz), dtype=np.float32)
    np.column_stack((xyz, remission)).astype("<f4").tofile(scan_dir / f"{frame}.bin")
    semantic.astype("<u4").tofile(label_dir / f"{frame}.label")


def _write_dales_tile(path: Path, offset: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.01, 0.01, 0.01])
    las = laspy.LasData(header)
    las.x = np.array([0.0, 49.99, 50.0, 100.0]) + offset
    las.y = np.array([0.0, 49.99, 0.0, 100.0]) + offset
    las.z = np.array([1.0, 2.0, 3.0, 4.0])
    las.classification = np.array([1, 2, 5, 8], dtype=np.uint8)
    las.intensity = np.array([100, 200, 300, 400], dtype=np.uint16)
    las.write(path)


def test_semantickitti_audit_counts_labels_coordinates_and_limit(tmp_path: Path) -> None:
    _write_semantic_scan(
        tmp_path,
        "00",
        "000000",
        np.array([[1.0, 2.0, -1.0], [3.0, 4.0, 5.0]]),
        np.array([40, 30], dtype=np.uint32),
    )
    _write_semantic_scan(
        tmp_path,
        "00",
        "000001",
        np.array([[10.0, 20.0, 30.0]]),
        np.array([50], dtype=np.uint32),
    )
    dataset = SemanticKITTIDataset(tmp_path, split="train", sequences=["00"])

    report = audit_semantickitti(dataset, max_samples=1, normalization="min_xyz")

    assert report["schema_version"] == "1.0"
    assert report["dataset"] == "semantickitti"
    assert report["counts"] == {
        "scans_available": 2,
        "scans_inspected": 1,
        "total_points": 2,
        "labelled_points": 2,
        "labelled_scans": 1,
    }
    assert report["source_semantic_label_histogram"] == {"30": 1, "40": 1}
    assert report["shared_ontology_histogram"] == {"Ground": 1}
    assert report["ignored_point_count"] == 1
    assert report["ignored_fraction"] == pytest.approx(0.5)
    assert report["raw_coordinate_statistics"]["z_range"] == [-1.0, 5.0]
    assert report["normalized_coordinate_statistics"]["xyz_min"] == [0.0, 0.0, 0.0]
    assert report["samples"] == [{"sequence": "00", "frame": "000000"}]


def test_dales_audit_reports_patch_coverage_and_normalized_ranges(tmp_path: Path) -> None:
    _write_dales_tile(tmp_path / "test" / "tile.las")
    dataset = DalesDataset(tmp_path, split="test")

    report = audit_dales(
        dataset,
        patch_size_m=(50.0, 50.0),
        chunk_size=2,
        normalization="min_xyz",
    )

    assert report["counts"] == {
        "tiles_available": 1,
        "tiles_inspected": 1,
        "grid_cells": 9,
        "patches_produced": 3,
        "empty_patches": 6,
        "total_points": 4,
        "points_in_patches": 4,
    }
    assert report["source_classification_histogram"] == {"1": 1, "2": 1, "5": 1, "8": 1}
    assert report["shared_ontology_histogram"] == {
        "Ground": 1,
        "Natural": 1,
        "Building": 1,
    }
    assert report["ignored_point_count"] == 1
    assert report["ignored_fraction"] == pytest.approx(0.25)
    assert report["patch_policy"]["boundary"] == "[xmin, xmax) and [ymin, ymax)"
    assert all(
        patch_range["xyz_min"] == [0.0, 0.0, 0.0]
        for patch_range in report["normalized_patch_ranges"]
    )


def test_audit_json_is_redacted_and_deterministic_except_runtime_metadata(tmp_path: Path) -> None:
    root = tmp_path / "private" / "dataset"
    _write_dales_tile(root / "train" / "tile.las")
    dataset = DalesDataset(root, split="train")
    first = audit_dales(dataset, max_tiles=1)
    second = audit_dales(dataset, max_tiles=1)
    for report in (first, second):
        report.pop("timestamp")
        report.pop("git_commit")
    assert first == second

    output = tmp_path / "audit-reports" / "report.json"
    write_json_report(first, output)
    raw = output.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    assert parsed["tiles"] == ["tile"]
    assert str(root) not in raw
    assert "tile.las" not in raw


def test_dales_audit_preserves_float64_scaled_raw_bounds(tmp_path: Path) -> None:
    _write_dales_tile(tmp_path / "train" / "utm.las", offset=500_000.01)

    report = audit_dales(DalesDataset(tmp_path, split="train"), max_tiles=1)

    assert report["raw_coordinate_statistics"]["xyz_min"][0] == pytest.approx(500_000.01, abs=1e-9)


def test_dales_max_tiles_limits_work(tmp_path: Path) -> None:
    _write_dales_tile(tmp_path / "train" / "a.las")
    _write_dales_tile(tmp_path / "train" / "b.las", offset=200.0)
    report = audit_dales(DalesDataset(tmp_path, split="train"), max_tiles=1)
    assert report["counts"]["tiles_available"] == 2
    assert report["counts"]["tiles_inspected"] == 1
    assert report["tiles"] == ["a"]


def test_cli_uses_environment_and_writes_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_semantic_scan(
        tmp_path,
        "00",
        "000000",
        np.array([[1.0, 2.0, 3.0]]),
        np.array([40], dtype=np.uint32),
    )
    monkeypatch.setenv("LASERPERCEPTION_SEMANTICKITTI_ROOT", str(tmp_path))
    output = tmp_path / "audit-reports" / "semantic.json"

    result = main(
        [
            "semantickitti",
            "--split",
            "train",
            "--sequences",
            "00",
            "--max-samples",
            "1",
            "--json",
            str(output),
        ]
    )

    assert result == 0
    assert output.is_file()
    assert "Dataset: semantickitti" in capsys.readouterr().out


def test_cli_fails_cleanly_without_dataset_root(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("LASERPERCEPTION_DALES_ROOT", raising=False)
    with pytest.raises(SystemExit) as error:
        main(["dales", "--split", "train"])
    assert error.value.code == 2
    assert "LASERPERCEPTION_DALES_ROOT" in capsys.readouterr().err


def test_audit_limits_must_be_positive(tmp_path: Path) -> None:
    _write_dales_tile(tmp_path / "train" / "tile.las")
    with pytest.raises(ValueError, match="max_tiles must be positive"):
        audit_dales(DalesDataset(tmp_path, split="train"), max_tiles=0)
