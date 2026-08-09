import numpy as np
import pytest

from laserperception.ontology import IGNORE_ID, label_histogram, mapping_coverage


def test_mapping_coverage_counts_source_mapped_and_ignored_labels() -> None:
    source = np.array([1, 1, 2, 5, 99], dtype=np.uint16)
    mapped = np.array([0, 0, 2, IGNORE_ID, IGNORE_ID], dtype=np.int16)

    coverage = mapping_coverage(source, mapped)

    assert coverage.total_count == 5
    assert coverage.source_histogram == {1: 2, 2: 1, 5: 1, 99: 1}
    assert coverage.mapped_histogram == {0: 2, 2: 1}
    assert coverage.ignored_count == 2
    assert coverage.ignored_fraction == pytest.approx(0.4)


def test_empty_mapping_coverage_is_defined() -> None:
    labels = np.empty((0,), dtype=np.uint8)
    coverage = mapping_coverage(labels, labels.astype(np.int16))
    assert coverage.total_count == 0
    assert coverage.ignored_fraction == 0.0


def test_histogram_and_coverage_validate_vectors() -> None:
    with pytest.raises(TypeError, match="integer"):
        label_histogram(np.array([1.0]))
    with pytest.raises(ValueError, match="shape"):
        label_histogram(np.array([[1]], dtype=np.uint8))
    with pytest.raises(ValueError, match="lengths differ"):
        mapping_coverage(np.array([1, 2]), np.array([0]))
