"""Immutable semantic predictions with explicit source-row identity."""

from dataclasses import dataclass

import numpy as np

from laserperception.ontology.taxonomy import SemanticTaxonomy
from laserperception.perception.contracts import CoordinateContract
from laserperception.perception.serialization import JsonRecord


@dataclass(frozen=True)
class SemanticProvenance(JsonRecord):
    producer: str
    operation: str
    source: str
    model_id: str | None = None


def integer_vector(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if array.dtype.kind == "u" and np.any(array > np.iinfo(np.int64).max):
        raise ValueError(f"{name} exceeds signed int64")
    return np.frombuffer(array.astype(np.int64).tobytes(), dtype=np.int64)


@dataclass(frozen=True, eq=False)
class SemanticPointFrame:
    sample_id: str
    frame_id: str
    coordinates: CoordinateContract
    source_point_count: int
    class_ids: np.ndarray
    taxonomy: SemanticTaxonomy
    provenance: SemanticProvenance
    source_rows: np.ndarray | None = None
    confidence: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not all(isinstance(v, str) and v.strip() for v in (self.sample_id, self.frame_id)):
            raise ValueError("sample and frame identity must be non-empty")
        if type(self.source_point_count) is not int or not 0 <= self.source_point_count < 2**63:
            raise ValueError("source_point_count must be a non-negative signed int64 integer")
        if not isinstance(self.coordinates, CoordinateContract):
            raise TypeError("coordinates must be a CoordinateContract")
        if not isinstance(self.taxonomy, SemanticTaxonomy):
            raise TypeError("taxonomy must be a SemanticTaxonomy")
        if not isinstance(self.provenance, SemanticProvenance):
            raise TypeError("provenance must be SemanticProvenance")
        labels = integer_vector(self.class_ids, "class_ids")
        if self.source_rows is None:
            if len(labels) != self.source_point_count:
                raise ValueError("filtered results require explicit source_rows")
            rows = integer_vector(np.arange(len(labels), dtype=np.int64), "source_rows")
        else:
            rows = integer_vector(self.source_rows, "source_rows")
        if len(rows) != len(labels):
            raise ValueError("source_rows length differs from class_ids")
        if (
            np.any(rows < 0)
            or np.any(rows >= self.source_point_count)
            or np.any(np.diff(rows) <= 0)
        ):
            raise ValueError("source_rows must be bounded, unique and strictly increasing")
        object.__setattr__(self, "class_ids", labels)
        object.__setattr__(self, "source_rows", rows)
        if self.confidence is not None:
            values = np.asarray(self.confidence)
            if values.dtype.kind != "f" or values.shape != labels.shape:
                raise ValueError("confidence must be a row-aligned floating array")
            if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
                raise ValueError("confidence must be finite and in [0, 1]")
            object.__setattr__(
                self,
                "confidence",
                np.frombuffer(values.astype(np.float64).tobytes(), dtype=np.float64),
            )

    def to_json(self) -> str:
        from .serialization import semantic_frame_to_json

        return semantic_frame_to_json(self)

    @classmethod
    def from_json(cls, value: str) -> "SemanticPointFrame":
        from .serialization import semantic_frame_from_json

        return semantic_frame_from_json(value)
