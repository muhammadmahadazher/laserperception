"""Pure deterministic transformations for frozen M7 arms B, C, D, and F."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import numpy as np

from benchmarks.m7.protocol import F_HISTORY_RANKS, ProtocolViolation
from benchmarks.m7.provenance import (
    FLOAT32_LE,
    UINT32_LE,
    UINT64_LE,
    SweepProvenance,
    canonical_a_provenance,
    canonical_model_ready,
    selected_rows_sha256,
)

MASK64 = 0xFFFFFFFFFFFFFFFF
SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
SPLITMIX_MUL1 = 0xBF58476D1CE4E5B9
SPLITMIX_MUL2 = 0x94D049BB133111EB
SEED_PREFIX = "laserperception-m7-c-v1"


def _immutable_copy(array: np.ndarray, *, dtype: np.dtype[object]) -> np.ndarray:
    result = np.ascontiguousarray(array, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class LagScale:
    """Exact Arm-B binary64 scale and frozen representations."""

    t10_f32: float
    t5_f32: float
    t10_f32_bits: int
    t5_f32_bits: int
    scale: float
    scale_bits: int
    scale_hex: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe scale provenance."""

        return {
            "t10_f32": self.t10_f32,
            "t5_f32": self.t5_f32,
            "t10_f32_bits": f"0x{self.t10_f32_bits:08x}",
            "t5_f32_bits": f"0x{self.t5_f32_bits:08x}",
            "t10_binary64_hex": float(self.t10_f32).hex(),
            "t5_binary64_hex": float(self.t5_f32).hex(),
            "scale_binary64_bits": f"0x{self.scale_bits:016x}",
            "scale_binary64_hex": self.scale_hex,
        }


@dataclass(frozen=True, slots=True)
class InterventionResult:
    """One immutable A-derived intervention result."""

    points: np.ndarray
    provenance: SweepProvenance
    selected_global_rows: np.ndarray
    selected_row_sha256: str
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        points = _immutable_copy(canonical_model_ready(self.points), dtype=FLOAT32_LE)
        rows = _immutable_copy(self.selected_global_rows, dtype=UINT64_LE)
        if len(points) != len(rows) or len(points) != len(self.provenance.history_rank):
            raise ProtocolViolation("intervention result columns have inconsistent row counts")
        if selected_rows_sha256(rows) != self.selected_row_sha256:
            raise ProtocolViolation("intervention selected-row SHA256 is inconsistent")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "selected_global_rows", rows)


@dataclass(frozen=True, slots=True)
class QuotaRecord:
    """Exact integer largest-remainder apportionment provenance."""

    source_counts: tuple[int, ...]
    h_target: int
    h_total: int
    products: tuple[int, ...]
    remainders: tuple[int, ...]
    initial_quotas: tuple[int, ...]
    incremented_ranks: tuple[int, ...]
    final_quotas: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        """Return complete JSON-safe quota provenance."""

        return {
            "source_counts_by_rank_1_to_10": list(self.source_counts),
            "h_target": self.h_target,
            "h_total": self.h_total,
            "products": list(self.products),
            "remainders": list(self.remainders),
            "initial_quotas": list(self.initial_quotas),
            "incremented_ranks": list(self.incremented_ranks),
            "final_quotas": list(self.final_quotas),
            "zero_quota_ranks": [
                rank for rank, quota in enumerate(self.final_quotas, start=1) if quota == 0
            ],
        }


@dataclass(frozen=True, slots=True)
class CResult:
    """Arm-C result plus the selection provenance Arm D must reuse."""

    intervention: InterventionResult
    quota: QuotaRecord
    seed_identities: tuple[dict[str, object], ...]


def _float32_bits(value: np.float32) -> int:
    return int(np.asarray([value], dtype=FLOAT32_LE).view(UINT32_LE)[0])


def derive_lag_scale(
    a_points: np.ndarray,
    e_points: np.ndarray,
    a_provenance: SweepProvenance,
    e_provenance: SweepProvenance,
) -> LagScale:
    """Derive the exact frozen B scale from existing float32 lag values."""

    a = canonical_model_ready(a_points)
    e = canonical_model_ready(e_points)
    canonical_a_provenance(a_provenance, len(a))
    a_provenance.validate_lags(a, expected_ranks=tuple(range(11)))
    e_provenance.validate_lags(e, expected_ranks=tuple(range(6)))
    a_historical = a[a_provenance.history_rank != 0, 3]
    e_historical = e[e_provenance.history_rank != 0, 3]
    t10_f32 = np.max(np.abs(a_historical)).astype(FLOAT32_LE)
    t5_f32 = np.max(np.abs(e_historical)).astype(FLOAT32_LE)
    t10 = float(t10_f32)
    t5 = float(t5_f32)
    if not np.isfinite((t10, t5)).all() or t10 <= 0.0 or t5 <= 0.0:
        raise ProtocolViolation("T10 and T5 must be finite and strictly positive")
    scale = t5 / t10
    if not np.isfinite(scale) or scale <= 0.0:
        raise ProtocolViolation("Arm-B binary64 scale must be finite and strictly positive")
    return LagScale(
        t10_f32=t10,
        t5_f32=t5,
        t10_f32_bits=_float32_bits(t10_f32),
        t5_f32_bits=_float32_bits(t5_f32),
        scale=scale,
        scale_bits=struct.unpack(">Q", struct.pack(">d", scale))[0],
        scale_hex=scale.hex(),
    )


def _apply_lag_scale(
    points: np.ndarray, provenance: SweepProvenance, scale: LagScale
) -> np.ndarray:
    source = canonical_model_ready(points)
    result = source.copy(order="C")
    scaled64 = source[:, 3].astype(np.float64) * np.float64(scale.scale)
    result[:, 3] = scaled64.astype(FLOAT32_LE)
    current = provenance.history_rank == 0
    result[current, 3] = np.float32(0.0)
    before = np.unique(source[~current, 3].view(UINT32_LE))
    after = np.unique(result[~current, 3].view(UINT32_LE))
    if len(before) != len(after):
        raise ProtocolViolation("Arm-B float32 cast collapsed distinct historical lag supports")
    if np.any(np.signbit(source[~current, 3]) != np.signbit(result[~current, 3])):
        raise ProtocolViolation("Arm-B scaling changed historical lag sign")
    return result


def construct_b(
    a_points: np.ndarray,
    e_points: np.ndarray,
    a_provenance: SweepProvenance,
    e_provenance: SweepProvenance,
) -> tuple[InterventionResult, LagScale]:
    """Construct Arm B with one binary64 multiply and one final float32 cast."""

    a = canonical_model_ready(a_points)
    scale = derive_lag_scale(a, e_points, a_provenance, e_provenance)
    result = _apply_lag_scale(a, a_provenance, scale)
    if not np.array_equal(result[:, :3].view(UINT32_LE), a[:, :3].view(UINT32_LE)):
        raise ProtocolViolation("Arm B changed frozen A XYZ bytes")
    rows = np.arange(len(a), dtype=UINT64_LE)
    return (
        InterventionResult(
            points=result,
            provenance=a_provenance,
            selected_global_rows=rows,
            selected_row_sha256=selected_rows_sha256(rows),
            metadata={"arm": "B", "lag_scale": scale.to_dict()},
        ),
        scale,
    )


def allocate_quotas(source_counts: tuple[int, ...], h_target: int) -> QuotaRecord:
    """Use exact integer largest-remainder apportionment over history ranks 1..10."""

    if len(source_counts) != 10 or any(
        isinstance(value, bool) or value < 0 for value in source_counts
    ):
        raise ProtocolViolation("Arm-C source counts must be ten nonnegative integers")
    if isinstance(h_target, bool) or not isinstance(h_target, int):
        raise ProtocolViolation("Arm-C H_target must be an integer")
    h_total = sum(source_counts)
    if h_total <= 0 or not 0 <= h_target <= h_total:
        raise ProtocolViolation("Arm-C requires 0 <= H_target <= positive H_total")
    products = tuple(h_target * count for count in source_counts)
    initial = tuple(product // h_total for product in products)
    remainders = tuple(product % h_total for product in products)
    remaining = h_target - sum(initial)
    order = sorted(range(10), key=lambda index: (-remainders[index], index + 1))
    incremented = tuple(index + 1 for index in order[:remaining])
    quotas = list(initial)
    for index in order[:remaining]:
        quotas[index] += 1
    final = tuple(quotas)
    if sum(final) != h_target or any(
        quota < 0 or quota > count for quota, count in zip(final, source_counts, strict=True)
    ):
        raise ProtocolViolation("Arm-C quota invariants failed")
    return QuotaRecord(
        source_counts=source_counts,
        h_target=h_target,
        h_total=h_total,
        products=products,
        remainders=remainders,
        initial_quotas=initial,
        incremented_ranks=incremented,
        final_quotas=final,
    )


def seed_text(drive_id: str, frame_index: int, history_rank: int) -> bytes:
    """Return exact UTF-8 seed bytes without a terminator."""

    if not drive_id or "|" in drive_id:
        raise ProtocolViolation("drive identity is empty or contains the seed delimiter")
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise ProtocolViolation("frame index must be a nonnegative integer")
    if isinstance(history_rank, bool) or history_rank not in range(1, 11):
        raise ProtocolViolation("history rank must be decimal 1 through 10")
    return f"{SEED_PREFIX}|{drive_id}|{frame_index:010d}|{history_rank}".encode()


def seed_identity(drive_id: str, frame_index: int, history_rank: int) -> tuple[bytes, str, int]:
    """Return exact seed bytes, SHA256 hex digest, and first-eight-byte big-endian seed."""

    text = seed_text(drive_id, frame_index, history_rank)
    digest = hashlib.sha256(text).digest()
    return text, digest.hex(), int.from_bytes(digest[:8], byteorder="big", signed=False)


def splitmix64_key(seed: int, ordinal: int) -> int:
    """Return the frozen SplitMix64 key for one within-sweep ordinal."""

    if not 0 <= seed <= MASK64 or isinstance(ordinal, bool) or ordinal < 0:
        raise ProtocolViolation("SplitMix64 seed/ordinal is outside the frozen unsigned domain")
    x = (seed ^ ordinal) & MASK64
    z = (x + SPLITMIX_GAMMA) & MASK64
    z = ((z ^ (z >> 30)) * SPLITMIX_MUL1) & MASK64
    z = ((z ^ (z >> 27)) * SPLITMIX_MUL2) & MASK64
    return (z ^ (z >> 31)) & MASK64


def select_lowest_ordinals(keys: tuple[int, ...], quota: int) -> tuple[int, ...]:
    """Select lowest ``(key, ordinal)`` pairs; ordinal is the exact tie-break."""

    if isinstance(quota, bool) or not 0 <= quota <= len(keys):
        raise ProtocolViolation("selection quota is outside the source sweep")
    return tuple(sorted(range(len(keys)), key=lambda ordinal: (keys[ordinal], ordinal))[:quota])


def construct_c(
    a_points: np.ndarray,
    e_points: np.ndarray,
    a_provenance: SweepProvenance,
    e_provenance: SweepProvenance,
    *,
    drive_id: str,
    frame_index: int,
) -> CResult:
    """Construct exact Arm C and retain all selection identities for Arm D."""

    a = canonical_model_ready(a_points)
    e = canonical_model_ready(e_points)
    canonical_a_provenance(a_provenance, len(a))
    a_provenance.validate_chronology(a, expected_ranks=tuple(range(11)))
    e_provenance.validate_lags(e, expected_ranks=tuple(range(6)))
    n0 = int(np.sum(a_provenance.history_rank == 0))
    source_counts = tuple(int(np.sum(a_provenance.history_rank == rank)) for rank in range(1, 11))
    quota = allocate_quotas(source_counts, len(e) - n0)
    keep_positions = list(np.flatnonzero(a_provenance.history_rank == 0))
    seed_records: list[dict[str, object]] = []
    for rank, selected_count in enumerate(quota.final_quotas, start=1):
        rank_positions = np.flatnonzero(a_provenance.history_rank == rank)
        text, digest, seed = seed_identity(drive_id, frame_index, rank)
        keys = tuple(splitmix64_key(seed, ordinal) for ordinal in range(len(rank_positions)))
        selected_ordinals = select_lowest_ordinals(keys, selected_count)
        keep_positions.extend(int(rank_positions[ordinal]) for ordinal in selected_ordinals)
        seed_records.append(
            {
                "history_rank": rank,
                "seed_text_utf8": text.decode("utf-8"),
                "sha256": digest,
                "seed_uint64": seed,
                "seed_uint64_hex": f"0x{seed:016x}",
                "selected_ordinals": list(selected_ordinals),
            }
        )
    positions = np.asarray(sorted(keep_positions), dtype=np.int64)
    if len(positions) != len(e) or len(np.unique(positions)) != len(positions):
        raise ProtocolViolation("Arm C did not produce the exact unique E row count")
    selected_global = a_provenance.global_a_row_index[positions]
    if np.any(selected_global >= len(a)) or np.any(np.diff(selected_global.astype(np.int64)) <= 0):
        raise ProtocolViolation("Arm C selected rows are out of range, duplicated, or reordered")
    points = a[positions].copy(order="C")
    result_provenance = a_provenance.subset(positions)
    intervention = InterventionResult(
        points=points,
        provenance=result_provenance,
        selected_global_rows=selected_global,
        selected_row_sha256=selected_rows_sha256(selected_global),
        metadata={
            "arm": "C",
            "n0": n0,
            "n5": len(e),
            "quota": quota.to_dict(),
            "seed_identities": seed_records,
        },
    )
    if not np.array_equal(points, a[selected_global.astype(np.int64)]):
        raise ProtocolViolation("Arm C rows are not exact byte copies from A")
    return CResult(intervention=intervention, quota=quota, seed_identities=tuple(seed_records))


def construct_d(c_result: CResult, scale: LagScale) -> InterventionResult:
    """Construct D from C's frozen result without invoking any selection logic."""

    c = c_result.intervention
    points = _apply_lag_scale(c.points, c.provenance, scale)
    return InterventionResult(
        points=points,
        provenance=c.provenance,
        selected_global_rows=c.selected_global_rows,
        selected_row_sha256=c.selected_row_sha256,
        metadata={
            "arm": "D",
            "c_selected_row_sha256": c.selected_row_sha256,
            "lag_scale": scale.to_dict(),
        },
    )


def construct_f(a_points: np.ndarray, a_provenance: SweepProvenance) -> InterventionResult:
    """Keep complete current and exact H10 history ranks 2/4/6/8/10."""

    a = canonical_model_ready(a_points)
    canonical_a_provenance(a_provenance, len(a))
    a_provenance.validate_chronology(a, expected_ranks=tuple(range(11)))
    selected_ranks = (0, *F_HISTORY_RANKS)
    positions = np.flatnonzero(np.isin(a_provenance.history_rank, selected_ranks))
    if not all(np.any(a_provenance.history_rank[positions] == rank) for rank in selected_ranks):
        raise ProtocolViolation("Arm F is missing a frozen complete-sweep rank")
    provenance = a_provenance.subset(positions)
    rows = a_provenance.global_a_row_index[positions]
    points = a[positions].copy(order="C")
    return InterventionResult(
        points=points,
        provenance=provenance,
        selected_global_rows=rows,
        selected_row_sha256=selected_rows_sha256(rows),
        metadata={"arm": "F", "history_ranks": list(F_HISTORY_RANKS)},
    )
