"""Quantitative coverage for explicit source-to-shared ontology mappings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from laserperception.ontology.mappings import IGNORE_ID


@dataclass(frozen=True)
class MappingCoverage:
    """Counts of source labels, mapped labels, and ignored points."""

    total_count: int
    source_histogram: dict[int, int]
    mapped_histogram: dict[int, int]
    ignored_count: int
    ignored_fraction: float


def label_histogram(labels: np.ndarray) -> dict[int, int]:
    """Return a deterministic integer histogram for a one-dimensional label vector."""
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError(f"labels must have shape (N,); received {values.shape}")
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError(f"labels must have an integer dtype; received {values.dtype}")
    unique, counts = np.unique(values, return_counts=True)
    return {int(label): int(count) for label, count in zip(unique, counts, strict=True)}


def mapping_coverage(
    source_labels: np.ndarray,
    mapped_labels: np.ndarray,
    *,
    ignore_id: int = IGNORE_ID,
) -> MappingCoverage:
    """Summarize an already explicit mapping without changing either label vector."""
    source = np.asarray(source_labels)
    mapped = np.asarray(mapped_labels)
    if source.ndim != 1 or mapped.ndim != 1:
        raise ValueError("source_labels and mapped_labels must both have shape (N,)")
    if source.shape != mapped.shape:
        raise ValueError(
            f"source and mapped label lengths differ: {source.shape[0]} != {mapped.shape[0]}"
        )
    source_histogram = label_histogram(source)
    mapped_all = label_histogram(mapped)
    ignored_count = mapped_all.pop(int(ignore_id), 0)
    total_count = int(source.shape[0])
    ignored_fraction = ignored_count / total_count if total_count else 0.0
    return MappingCoverage(
        total_count=total_count,
        source_histogram=source_histogram,
        mapped_histogram=mapped_all,
        ignored_count=ignored_count,
        ignored_fraction=ignored_fraction,
    )
