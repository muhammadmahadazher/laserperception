"""Fail-closed pre-inference structural checks for M7 interventions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from benchmarks.m7.interventions import CResult, InterventionResult
from benchmarks.m7.protocol import F_HISTORY_RANKS, ProtocolViolation
from benchmarks.m7.provenance import (
    FLOAT32_LE,
    UINT32_LE,
    SweepProvenance,
    canonical_model_ready,
)
from laserperception.evaluation.m6b_pillars import PillarAudit, analyze_pillars


def _int32_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array, dtype=np.dtype("<i4"))
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


@dataclass(frozen=True, slots=True)
class PillarStructure:
    """Exact-fast candidate/order and retained-selection identities."""

    coordinate_order_sha256: str
    candidate_key_order_sha256: str
    retained_selection_sha256: str
    discarded_selection_sha256: str
    candidate_count: int
    retained_count: int
    discarded_count: int

    @classmethod
    def from_audit(cls, audit: PillarAudit) -> PillarStructure:
        """Construct identities from the existing M6 exact-fast audit helper."""

        candidate = np.ascontiguousarray(audit.candidate_xy_indices, dtype=np.dtype("<i4"))
        keys = candidate[:, 1].astype(np.int64) * 400 + candidate[:, 0].astype(np.int64)
        return cls(
            coordinate_order_sha256=_int32_hash(candidate),
            candidate_key_order_sha256=hashlib.sha256(
                np.ascontiguousarray(keys, dtype=np.dtype("<i8")).tobytes(order="C")
            ).hexdigest(),
            retained_selection_sha256=_int32_hash(audit.retained_xy_indices),
            discarded_selection_sha256=_int32_hash(audit.discarded_xy_indices),
            candidate_count=audit.candidate_count,
            retained_count=audit.retained_count,
            discarded_count=audit.discarded_count,
        )

    @classmethod
    def from_points(cls, points: np.ndarray) -> PillarStructure:
        """Inspect synthetic/non-M7 arrays with the existing CPU exact-fast structure oracle."""

        return cls.from_audit(analyze_pillars(canonical_model_ready(points)))

    def to_dict(self) -> dict[str, object]:
        """Return the compact ledger representation."""

        return {
            "coordinate_order_sha256": self.coordinate_order_sha256,
            "candidate_key_order_sha256": self.candidate_key_order_sha256,
            "retained_selection_sha256": self.retained_selection_sha256,
            "discarded_selection_sha256": self.discarded_selection_sha256,
            "candidate_count": self.candidate_count,
            "retained_count": self.retained_count,
            "discarded_count": self.discarded_count,
            "overflow": self.discarded_count > 0,
        }


def require_equal_structure(
    first: PillarStructure,
    second: PillarStructure,
    *,
    relation: str,
) -> None:
    """Require every coordinate/order/retention identity to match."""

    if first != second:
        differing = [
            name
            for name in first.__dataclass_fields__
            if getattr(first, name) != getattr(second, name)
        ]
        raise ProtocolViolation(f"{relation} pillar structure differs: {', '.join(differing)}")


def _same_xyz(first: np.ndarray, second: np.ndarray) -> bool:
    first_xyz = canonical_model_ready(first)[:, :3].view(UINT32_LE)
    second_xyz = canonical_model_ready(second)[:, :3].view(UINT32_LE)
    return np.array_equal(first_xyz, second_xyz)


def validate_b_against_a(
    a_points: np.ndarray,
    b: InterventionResult,
    a_structure: PillarStructure,
    b_structure: PillarStructure,
) -> None:
    """Require B/A to differ only in lag under identical row and pillar identities."""

    a = canonical_model_ready(a_points)
    if len(a) != len(b.points) or not _same_xyz(a, b.points):
        raise ProtocolViolation("B/A row count or XYZ bytes differ")
    expected_rows = np.arange(len(a), dtype=np.dtype("<u8"))
    if not np.array_equal(b.selected_global_rows, expected_rows):
        raise ProtocolViolation("B/A row order or provenance differs")
    if np.any(b.points[b.provenance.history_rank == 0, 3].view(UINT32_LE) != 0):
        raise ProtocolViolation("B current lag is not positive zero")
    require_equal_structure(a_structure, b_structure, relation="B/A")


def validate_c_against_a_e(
    a_points: np.ndarray,
    e_points: np.ndarray,
    c: CResult,
) -> None:
    """Require C to be the unique, ordered, native-lag A subset with E row count."""

    a = canonical_model_ready(a_points)
    e = canonical_model_ready(e_points)
    result = c.intervention
    rows = result.selected_global_rows.astype(np.int64)
    if len(result.points) != len(e):
        raise ProtocolViolation("C row count does not equal E")
    if len(np.unique(rows)) != len(rows) or np.any(rows < 0) or np.any(rows >= len(a)):
        raise ProtocolViolation("C selected rows are duplicated or outside A")
    if np.any(np.diff(rows) <= 0):
        raise ProtocolViolation("C selected rows do not preserve original global A order")
    if not np.array_equal(result.points.view(UINT32_LE), a[rows].view(UINT32_LE)):
        raise ProtocolViolation("C contains a mutated or synthetic A row")
    expected_current_rows = np.flatnonzero(a[:, 3].view(UINT32_LE) == 0)
    actual_current_rows = rows[result.provenance.history_rank == 0]
    if not np.array_equal(actual_current_rows, expected_current_rows):
        raise ProtocolViolation("C does not retain every A current row in original order")


def validate_d_against_c(
    c: CResult,
    d: InterventionResult,
    c_structure: PillarStructure,
    d_structure: PillarStructure,
) -> None:
    """Require D to reuse C rows/XYZ/order and change only lag."""

    source = c.intervention
    if source.selected_row_sha256 != d.selected_row_sha256 or not np.array_equal(
        source.selected_global_rows, d.selected_global_rows
    ):
        raise ProtocolViolation("D/C selected-row identity differs")
    if len(source.points) != len(d.points) or not _same_xyz(source.points, d.points):
        raise ProtocolViolation("D/C row population or XYZ bytes differ")
    require_equal_structure(c_structure, d_structure, relation="D/C")


def validate_f_against_a(
    a_points: np.ndarray,
    a_provenance: SweepProvenance,
    f: InterventionResult,
) -> None:
    """Require F to contain complete exact current and 2/4/6/8/10 A sweeps."""

    a = canonical_model_ready(a_points)
    expected_ranks = (0, *F_HISTORY_RANKS)
    actual_ranks = tuple(int(value) for value in np.unique(f.provenance.history_rank))
    if actual_ranks != expected_ranks:
        raise ProtocolViolation(
            f"F rank identity differs: expected {expected_ranks}, found {actual_ranks}"
        )
    rows = f.selected_global_rows.astype(np.int64)
    if not np.array_equal(f.points.view(UINT32_LE), a[rows].view(UINT32_LE)):
        raise ProtocolViolation("F contains a mutated or synthetic A row")
    for rank in expected_ranks:
        expected_global_rows = a_provenance.global_a_row_index[a_provenance.history_rank == rank]
        actual_global_rows = f.selected_global_rows[f.provenance.history_rank == rank]
        actual_ordinals = f.provenance.within_sweep_ordinal[f.provenance.history_rank == rank]
        if not np.array_equal(expected_global_rows, actual_global_rows) or not np.array_equal(
            actual_ordinals, np.arange(len(actual_ordinals), dtype=np.uint64)
        ):
            raise ProtocolViolation(f"F rank {rank} is not a complete sweep")


def validate_frozen_float32(points: np.ndarray) -> None:
    """Explicitly require canonical finite float32 values for structure inspection."""

    if canonical_model_ready(points).dtype != FLOAT32_LE:
        raise ProtocolViolation("structure input is not canonical little-endian float32")
