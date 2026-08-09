import numpy as np
import pytest

from laserperception.ontology import (
    IGNORE_ID,
    SharedClass,
    map_dales_labels,
    map_semantickitti_labels,
)


def test_semantickitti_mapping_groups_verified_ids() -> None:
    source = np.array([40, 50, 70, 10, 80, 51, 30, 999], dtype=np.uint16)
    mapped = map_semantickitti_labels(source)
    expected = np.array(
        [
            SharedClass.GROUND,
            SharedClass.BUILDING,
            SharedClass.NATURAL,
            SharedClass.VEHICLE,
            SharedClass.POLE,
            SharedClass.FENCE,
            IGNORE_ID,
            IGNORE_ID,
        ],
        dtype=np.int16,
    )
    assert np.array_equal(mapped, expected)


def test_dales_mapping_ignores_unknown_and_power_lines() -> None:
    source = np.arange(9, dtype=np.uint8)
    mapped = map_dales_labels(source)
    assert mapped.tolist() == [-1, 0, 2, 3, 3, -1, 5, 4, 1]


def test_mapping_requires_integer_vector() -> None:
    with pytest.raises(TypeError, match="integer"):
        map_dales_labels(np.array([1.0]))
    with pytest.raises(ValueError, match="shape"):
        map_dales_labels(np.array([[1]], dtype=np.uint8))
