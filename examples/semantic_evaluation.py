"""Synthetic CPU evaluation example; no model, detector or benchmark claim."""

import numpy as np

from laserperception.perception.contracts import CoordinateContract
from laserperception.semantic import (
    EXPERIMENT_001_TAXONOMY,
    SemanticPointFrame,
    SemanticProvenance,
    evaluate_semantics,
)

coordinates = CoordinateContract(
    "synthetic",
    "right",
    ("forward", "left", "up"),
    "m",
    "not_applicable",
    ("not_applicable",) * 3,
    "not_applicable",
    "synthetic example",
)


def result(labels: list[int]) -> SemanticPointFrame:
    return SemanticPointFrame(
        "synthetic-001",
        "synthetic",
        coordinates,
        len(labels),
        np.asarray(labels, dtype=np.int64),
        EXPERIMENT_001_TAXONOMY,
        SemanticProvenance("synthetic", "example", "inline fixture"),
    )


print(evaluate_semantics(result([0, 1, 3, -2]), result([0, 1, 3, 2])).to_json())
