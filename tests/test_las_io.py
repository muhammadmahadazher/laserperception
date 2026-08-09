from pathlib import Path

import laspy
import numpy as np
import pytest

from laserperception.io import load_las


def _synthetic_las() -> laspy.LasData:
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.array([100.0, 200.0, -20.0])
    las = laspy.LasData(header)
    las.x = np.array([101.25, 102.50])
    las.y = np.array([205.00, 207.75])
    las.z = np.array([-10.00, -9.25])
    las.classification = np.array([1, 8], dtype=np.uint8)
    las.intensity = np.array([100, 200], dtype=np.uint16)
    return las


def test_load_synthetic_las_preserves_coordinates_metadata_and_attributes(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.las"
    _synthetic_las().write(path)

    cloud = load_las(path)

    expected = np.array([[101.25, 205.0, -10.0], [102.5, 207.75, -9.25]], dtype=np.float32)
    assert np.allclose(cloud.xyz, expected, atol=0.006)
    assert not np.allclose(cloud.xyz.min(axis=0), np.zeros(3))
    assert cloud.labels is not None
    assert np.array_equal(cloud.labels, np.array([1, 8], dtype=np.uint8))
    assert np.array_equal(cloud.attributes["intensity"], np.array([100, 200]))
    assert cloud.metadata["las_version"] == "1.2"
    assert cloud.metadata["point_format_id"] == 3
    assert cloud.metadata["scales"] == pytest.approx((0.01, 0.01, 0.01))
    assert cloud.metadata["coordinates_normalized"] is False
    assert "classification" in cloud.metadata["available_dimensions"]


def test_load_laz_when_backend_is_available(tmp_path: Path) -> None:
    if not laspy.LazBackend.detect_available():
        pytest.skip("optional LAZ backend is not installed")
    path = tmp_path / "synthetic.laz"
    _synthetic_las().write(path)
    cloud = load_las(path)
    assert len(cloud) == 2
    assert cloud.metadata["format"] == "laz"


def test_las_loader_rejects_other_suffixes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=".las or .laz"):
        load_las(tmp_path / "cloud.bin")
