"""Canonical bytes, hashes, and sweep provenance for M7."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmarks.m7.protocol import ProtocolViolation

FLOAT32_LE = np.dtype("<f4")
UINT64_LE = np.dtype("<u8")
UINT32_LE = np.dtype("<u4")


def canonical_model_ready(points: np.ndarray) -> np.ndarray:
    """Return C-contiguous little-endian float32 ``(N, 4)`` model input."""

    array = np.asarray(points)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ProtocolViolation("model-ready input must have shape (N, 4)")
    if array.dtype.kind != "f" or array.dtype.itemsize != 4:
        raise ProtocolViolation("model-ready input must use IEEE-754 float32 values")
    canonical = np.ascontiguousarray(array, dtype=FLOAT32_LE)
    if not np.isfinite(canonical).all():
        raise ProtocolViolation("model-ready input must contain only finite values")
    return canonical


def model_ready_sha256(points: np.ndarray) -> str:
    """Hash canonical model-ready bytes independent of view/contiguity state."""

    return hashlib.sha256(canonical_model_ready(points).tobytes(order="C")).hexdigest()


def xyz_sha256(points: np.ndarray) -> str:
    """Hash canonical XYZ bytes without the lag feature."""

    return hashlib.sha256(canonical_model_ready(points)[:, :3].tobytes(order="C")).hexdigest()


def selected_rows_bytes(rows: np.ndarray) -> bytes:
    """Return selected global A row identities as C-contiguous little-endian uint64."""

    array = np.asarray(rows)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ProtocolViolation("selected-row identity must be a one-dimensional integer vector")
    if array.dtype.kind == "i" and np.any(array < 0):
        raise ProtocolViolation("selected-row identity cannot contain negative indices")
    return np.ascontiguousarray(array, dtype=UINT64_LE).tobytes(order="C")


def selected_rows_sha256(rows: np.ndarray) -> str:
    """Hash the frozen selected-row uint64 representation."""

    return hashlib.sha256(selected_rows_bytes(rows)).hexdigest()


def canonical_array_sha256(array: np.ndarray, *, dtype: np.dtype[object]) -> str:
    """Hash an explicitly typed C-contiguous array."""

    value = np.ascontiguousarray(np.asarray(array), dtype=dtype)
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def canonical_json_bytes(value: Mapping[str, object] | list[object]) -> bytes:
    """Return the frozen compact JSON representation used for identities."""

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: Mapping[str, object] | list[object]) -> str:
    """Hash canonical UTF-8 JSON bytes."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def atomic_write_json(path: str | Path, value: Mapping[str, object] | list[object]) -> None:
    """Write canonical JSON through an adjacent atomic replacement."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    temporary.replace(target)


def verify_external_asset(path: str | Path, *, expected_bytes: int, expected_sha256: str) -> str:
    """Fail closed unless one external release asset has the frozen size and SHA256."""

    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"required external M6 asset is missing: {artifact}")
    if artifact.stat().st_size != expected_bytes:
        raise ProtocolViolation(
            f"external M6 asset byte-size mismatch: expected {expected_bytes}, "
            f"found {artifact.stat().st_size}"
        )
    hasher = hashlib.sha256()
    with artifact.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    digest = hasher.hexdigest()
    if digest != expected_sha256:
        raise ProtocolViolation(
            f"external M6 asset SHA256 mismatch: expected {expected_sha256}, found {digest}"
        )
    return digest


@dataclass(frozen=True, slots=True)
class SweepProvenance:
    """Per-row frozen source identity for an A-derived model-ready array."""

    history_rank: np.ndarray
    source_sweep_id: tuple[str, ...]
    within_sweep_ordinal: np.ndarray
    global_a_row_index: np.ndarray

    def __post_init__(self) -> None:
        ranks = np.ascontiguousarray(self.history_rank, dtype=np.int16)
        ordinals = np.ascontiguousarray(self.within_sweep_ordinal, dtype=UINT64_LE)
        global_rows = np.ascontiguousarray(self.global_a_row_index, dtype=UINT64_LE)
        lengths = {len(ranks), len(self.source_sweep_id), len(ordinals), len(global_rows)}
        if len(lengths) != 1:
            raise ProtocolViolation("sweep provenance columns must have equal row counts")
        if np.any((ranks < 0) | (ranks > 10)):
            raise ProtocolViolation("history ranks must be current=0 or history=1..10")
        if len(set(int(value) for value in global_rows)) != len(global_rows):
            raise ProtocolViolation("global A row identities must be unique")
        for rank in np.unique(ranks):
            positions = np.flatnonzero(ranks == rank)
            rank_ids = {self.source_sweep_id[int(position)] for position in positions}
            if len(rank_ids) != 1:
                raise ProtocolViolation(f"history rank {int(rank)} has mismatched source sweep IDs")
            rank_ordinals = ordinals[positions]
            if len(np.unique(rank_ordinals)) != len(rank_ordinals) or np.any(
                np.diff(rank_ordinals.astype(np.int64)) <= 0
            ):
                raise ProtocolViolation(
                    f"history rank {int(rank)} has duplicated or reordered source ordinals"
                )
        object.__setattr__(self, "history_rank", ranks)
        object.__setattr__(self, "within_sweep_ordinal", ordinals)
        object.__setattr__(self, "global_a_row_index", global_rows)

    def validate_lags(self, points: np.ndarray, *, expected_ranks: tuple[int, ...]) -> None:
        """Require exact rank support and positive-zero/current versus historical lag semantics."""

        array = canonical_model_ready(points)
        if len(array) != len(self.history_rank):
            raise ProtocolViolation("point and sweep-provenance row counts differ")
        actual_ranks = tuple(int(value) for value in np.unique(self.history_rank))
        if actual_ranks != expected_ranks:
            raise ProtocolViolation(
                f"unexpected history-rank support: expected {expected_ranks}, found {actual_ranks}"
            )
        lag_bits = array[:, 3].view(UINT32_LE)
        current = self.history_rank == 0
        if np.any(lag_bits[current] != np.uint32(0)):
            raise ProtocolViolation("current rows must carry float32 positive zero lag bits")
        if np.any((lag_bits[~current] & np.uint32(0x7FFFFFFF)) == np.uint32(0)):
            raise ProtocolViolation("historical rows must carry nonzero lag bits")
        for rank in actual_ranks:
            bits = np.unique(lag_bits[self.history_rank == rank])
            if len(bits) != 1:
                raise ProtocolViolation(f"history rank {rank} has multiple lag bit patterns")

    def subset(self, positions: np.ndarray) -> SweepProvenance:
        """Return provenance for retained A row positions without renumbering global identities."""

        selected = np.asarray(positions, dtype=np.int64)
        return SweepProvenance(
            history_rank=self.history_rank[selected],
            source_sweep_id=tuple(self.source_sweep_id[int(index)] for index in selected),
            within_sweep_ordinal=self.within_sweep_ordinal[selected],
            global_a_row_index=self.global_a_row_index[selected],
        )


def canonical_a_provenance(provenance: SweepProvenance, row_count: int) -> None:
    """Require the A provenance vector to identify every row in original order."""

    expected = np.arange(row_count, dtype=UINT64_LE)
    if not np.array_equal(provenance.global_a_row_index, expected):
        raise ProtocolViolation("A global row identities must equal original row positions")
    for rank in np.unique(provenance.history_rank):
        positions = np.flatnonzero(provenance.history_rank == rank)
        expected_ordinals = np.arange(len(positions), dtype=UINT64_LE)
        if not np.array_equal(provenance.within_sweep_ordinal[positions], expected_ordinals):
            raise ProtocolViolation("A source ordinals must enumerate each range-filtered sweep")
