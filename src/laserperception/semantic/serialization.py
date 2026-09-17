"""Bounded inline JSON and identity-checked NPY sidecars; no pickle or network."""

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from laserperception.ontology.taxonomy import SemanticTaxonomy
from laserperception.perception.contracts import CoordinateContract
from laserperception.perception.serialization import JsonRecord
from laserperception.worker.paths import resolve_artifact_path, validate_relative_path

from .types import SemanticPointFrame, SemanticProvenance

INLINE_POINT_LIMIT = 4096


@dataclass(frozen=True)
class ArrayPayload(JsonRecord):
    dtype: Literal["int64", "float64"]
    shape: tuple[int]
    values: tuple[int | float, ...] | None
    path: str | None
    size_bytes: int | None
    sha256: str | None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.shape[0] < 0:
            raise ValueError("array shape must be non-negative")
        if self.values is not None:
            if (
                any(v is not None for v in (self.path, self.size_bytes, self.sha256))
                or len(self.values) != self.shape[0]
            ):
                raise ValueError("inline array metadata differs")
            if len(self.values) > INLINE_POINT_LIMIT:
                raise ValueError("large semantic arrays require NPY sidecars")
            if self.dtype == "int64" and any(type(v) is not int for v in self.values):
                raise ValueError("int64 inline arrays require integer values")
            if self.dtype == "int64" and any(not -(2**63) <= v < 2**63 for v in self.values):
                raise ValueError("int64 inline value exceeds signed range")
        else:
            if self.path is None or self.size_bytes is None or self.sha256 is None:
                raise ValueError("sidecar requires path, size and SHA256")
            validate_relative_path(self.path)
            if (
                self.size_bytes <= 0
                or len(self.sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.sha256)
            ):
                raise ValueError("invalid sidecar identity")


@dataclass(frozen=True)
class SemanticEnvelope(JsonRecord):
    schema_version: Literal["laserperception.semantic-point-frame.v1"]
    sample_id: str
    frame_id: str
    coordinates: CoordinateContract
    source_point_count: int
    taxonomy: SemanticTaxonomy
    provenance: SemanticProvenance
    source_row_semantics: Literal["strictly-increasing-source-row-indices"]
    class_ids: ArrayPayload
    source_rows: ArrayPayload
    confidence: ArrayPayload | None


def _payload(array: np.ndarray, path: Path | None = None) -> ArrayPayload:
    dtype: Literal["int64", "float64"] = "int64" if array.dtype == np.dtype("int64") else "float64"
    if path is None:
        values: tuple[int | float, ...] = (
            tuple(int(v) for v in array) if dtype == "int64" else tuple(float(v) for v in array)
        )
        return ArrayPayload(dtype, (len(array),), values, None, None, None)
    validate_relative_path(path.name)
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
    data = path.read_bytes()
    return ArrayPayload(
        dtype, (len(array),), None, path.name, len(data), hashlib.sha256(data).hexdigest()
    )


def _envelope(frame: SemanticPointFrame, path: Path | None = None) -> SemanticEnvelope:
    def payload(array: np.ndarray, suffix: str) -> ArrayPayload:
        sidecar = path.with_name(path.stem + "." + suffix + ".npy") if path else None
        return _payload(array, sidecar)

    assert frame.source_rows is not None
    return SemanticEnvelope(
        "laserperception.semantic-point-frame.v1",
        frame.sample_id,
        frame.frame_id,
        frame.coordinates,
        frame.source_point_count,
        frame.taxonomy,
        frame.provenance,
        "strictly-increasing-source-row-indices",
        payload(frame.class_ids, "labels"),
        payload(frame.source_rows, "rows"),
        payload(frame.confidence, "confidence") if frame.confidence is not None else None,
    )


def _array(payload: ArrayPayload, root: Path | None) -> np.ndarray:
    if payload.values is not None:
        # Conversion must preserve integer identities, including signed bounds.
        return np.asarray(payload.values, dtype=payload.dtype)
    if root is None or payload.path is None:
        raise ValueError("sidecar envelope requires an explicit file root")
    path = resolve_artifact_path(root, payload.path)
    if path.stat().st_size != payload.size_bytes:
        raise ValueError("semantic sidecar size mismatch")
    data = path.read_bytes()
    if len(data) != payload.size_bytes or hashlib.sha256(data).hexdigest() != payload.sha256:
        raise ValueError("semantic sidecar size/SHA256 mismatch")
    with io.BytesIO(data) as stream:
        version = np.lib.format.read_magic(stream)  # type: ignore[no-untyped-call]
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)  # type: ignore[no-untyped-call]
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)  # type: ignore[no-untyped-call]
        else:
            raise ValueError("unsupported semantic NPY version")
        if shape != payload.shape or dtype != np.dtype(payload.dtype) or fortran:
            raise ValueError("semantic sidecar dtype/shape/order mismatch")
        if stream.tell() + payload.shape[0] * dtype.itemsize != len(data):
            raise ValueError("semantic NPY data length/trailing bytes mismatch")
        stream.seek(0)
        array = np.load(stream, allow_pickle=False)
    if not isinstance(array, np.ndarray):
        raise ValueError("semantic sidecar must be one NPY array")
    return array


def _frame(envelope: SemanticEnvelope, root: Path | None = None) -> SemanticPointFrame:
    if (
        envelope.class_ids.dtype != "int64"
        or envelope.source_rows.dtype != "int64"
        or (envelope.confidence and envelope.confidence.dtype != "float64")
    ):
        raise ValueError("semantic array role/dtype mismatch")
    return SemanticPointFrame(
        envelope.sample_id,
        envelope.frame_id,
        envelope.coordinates,
        envelope.source_point_count,
        _array(envelope.class_ids, root),
        envelope.taxonomy,
        envelope.provenance,
        _array(envelope.source_rows, root),
        _array(envelope.confidence, root) if envelope.confidence else None,
    )


def semantic_frame_to_json(frame: SemanticPointFrame) -> str:
    if len(frame.class_ids) > INLINE_POINT_LIMIT:
        raise ValueError("large semantic results require save_semantic_frame with sidecars")
    return _envelope(frame).to_json()


def semantic_frame_from_json(value: str) -> SemanticPointFrame:
    return _frame(SemanticEnvelope.from_json(value))


def save_semantic_frame(
    frame: SemanticPointFrame, path: str | Path, *, inline: bool = False
) -> None:
    """Create new files, arrays first and metadata last; never overwrite other state."""
    target = Path(path)
    validate_relative_path(target.name)
    if target.exists():
        raise FileExistsError(target)
    envelope = _envelope(frame, None if inline else target)
    with target.open("x", encoding="utf-8") as stream:
        stream.write(envelope.to_json())


def load_semantic_frame(path: str | Path) -> SemanticPointFrame:
    target = Path(path)
    return _frame(SemanticEnvelope.from_json(target.read_text(encoding="utf-8")), target.parent)
