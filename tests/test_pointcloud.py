import numpy as np
import pytest

from laserperception import PointCloud


def test_valid_cloud_is_canonical_and_defensively_copied() -> None:
    xyz = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    labels = np.array([2, 3], dtype=np.uint16)
    remission = np.array([0.1, 0.2], dtype=np.float32)

    cloud = PointCloud(
        xyz=xyz,
        labels=labels,
        attributes={"remission": remission},
        metadata={"nested": {"source": "synthetic"}},
    )
    xyz[0, 0] = 99
    labels[0] = 99
    remission[0] = 99

    assert len(cloud) == 2
    assert cloud.xyz.dtype == np.float32
    assert cloud.xyz[0, 0] == 1
    assert cloud.labels is not None and cloud.labels[0] == 2
    assert cloud.attributes["remission"][0] == pytest.approx(0.1)


@pytest.mark.parametrize(
    "xyz",
    [
        np.zeros(3),
        np.zeros((2, 2)),
        np.zeros((2, 3, 1)),
    ],
)
def test_invalid_xyz_shapes_raise(xyz: np.ndarray) -> None:
    with pytest.raises(ValueError, match="shape"):
        PointCloud(xyz=xyz)


def test_non_numeric_xyz_raises() -> None:
    with pytest.raises(TypeError, match="numeric"):
        PointCloud(xyz=np.array([["x", "y", "z"]]))


def test_labels_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="labels length"):
        PointCloud(xyz=np.zeros((2, 3)), labels=np.array([1]))


def test_attribute_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="attribute 'intensity' length"):
        PointCloud(xyz=np.zeros((2, 3)), attributes={"intensity": np.array([1])})


def test_empty_cloud_is_supported() -> None:
    cloud = PointCloud(
        xyz=np.empty((0, 3), dtype=np.float64),
        labels=np.empty((0,), dtype=np.uint8),
        attributes={"intensity": np.empty((0,), dtype=np.uint16)},
    )
    assert len(cloud) == 0
    assert cloud.xyz.shape == (0, 3)
    assert cloud.xyz.dtype == np.float32
