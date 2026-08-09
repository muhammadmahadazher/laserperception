import numpy as np
import pytest

from laserperception import PointCloud
from laserperception.transforms import normalize_coordinates


def test_min_xyz_normalization_is_non_mutating_and_preserves_data() -> None:
    cloud = PointCloud(
        xyz=np.array([[10.0, -2.0, 4.0], [12.0, 3.0, 9.0]]),
        labels=np.array([1, 2]),
        attributes={"intensity": np.array([10, 20])},
        metadata={"source": "synthetic"},
    )
    original_xyz = cloud.xyz.copy()

    normalized = normalize_coordinates(cloud)

    assert np.array_equal(cloud.xyz, original_xyz)
    assert np.allclose(normalized.xyz.min(axis=0), np.zeros(3))
    assert np.array_equal(normalized.labels, cloud.labels)
    assert np.array_equal(normalized.attributes["intensity"], cloud.attributes["intensity"])
    assert normalized.metadata["source"] == "synthetic"
    assert normalized.metadata["coordinate_normalization"]["mode"] == "min_xyz"
    assert normalized.metadata["coordinate_normalization"]["source_origin_xyz"] == pytest.approx(
        [10.0, -2.0, 4.0]
    )


def test_unknown_mode_raises() -> None:
    cloud = PointCloud(xyz=np.ones((1, 3)))
    with pytest.raises(ValueError, match="unsupported"):
        normalize_coordinates(cloud, mode="centroid_xyz")


def test_empty_cloud_cannot_be_normalized() -> None:
    cloud = PointCloud(xyz=np.empty((0, 3)))
    with pytest.raises(ValueError, match="empty"):
        normalize_coordinates(cloud)
