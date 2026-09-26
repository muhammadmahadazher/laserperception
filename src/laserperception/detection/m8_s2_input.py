"""CPU-only five-feature lift of already-constructed frozen M7 interventions.

M7 owns lag arithmetic, quotas, seeds, and row selection. This module only
copies the selected M8 source rows and substitutes M7's resulting lag column.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

FLOAT32_LE = np.dtype("<f4")
UINT64_LE = np.dtype("<u8")
ARM_ORDER = ("B2", "C2", "D2", "F2")


class M7Result(Protocol):
    """Minimal M7 result boundary, without a core-wheel benchmark dependency."""

    points: np.ndarray
    selected_global_rows: np.ndarray
    selected_row_sha256: str


@dataclass(frozen=True, slots=True)
class S2Input:
    """One verified five-feature input and its selected A2 row identities."""

    arm: str
    points: np.ndarray
    selected_global_rows: np.ndarray
    selected_row_sha256: str


def canonical_xyzit(points: np.ndarray) -> np.ndarray:
    """Require finite C-contiguous little-endian float32 XYZIT without changing values."""

    source = np.asarray(points)
    if source.ndim != 2 or source.shape[1] != 5 or source.shape[0] == 0:
        raise ValueError("S2 input must have nonempty shape (N, 5)")
    if source.dtype.kind != "f" or source.dtype.itemsize != 4:
        raise TypeError("S2 input must be float32")
    result = np.ascontiguousarray(source, dtype=FLOAT32_LE)
    if not np.isfinite(result).all():
        raise ValueError("S2 input must contain finite values")
    return result


def canonical_xyzt(points: np.ndarray) -> np.ndarray:
    """Project XYZIT to canonical XYZT for the frozen M7 byte gate."""

    return np.ascontiguousarray(canonical_xyzit(points)[:, [0, 1, 2, 4]], dtype=FLOAT32_LE)


def array_sha256(array: np.ndarray, *, dtype: np.dtype[Any] = FLOAT32_LE) -> str:
    """Hash explicitly typed C-order bytes, including canonical byte order."""

    return hashlib.sha256(np.ascontiguousarray(array, dtype=dtype).tobytes(order="C")).hexdigest()


def _m7_points(result: M7Result) -> np.ndarray:
    points = np.asarray(result.points)
    if (
        points.ndim != 2
        or points.shape[1] != 4
        or points.dtype.kind != "f"
        or points.dtype.itemsize != 4
    ):
        raise ValueError("M7 intervention must be float32 XYZT")
    return np.ascontiguousarray(points, dtype=FLOAT32_LE)


def _rows(result: M7Result, source_count: int) -> np.ndarray:
    rows = np.asarray(result.selected_global_rows)
    if rows.ndim != 1 or rows.dtype.kind not in "iu" or len(rows) == 0:
        raise ValueError("M7 selected rows must be a nonempty integer vector")
    if rows.dtype.kind == "i" and np.any(rows < 0):
        raise ValueError("M7 selected rows cannot be negative")
    canonical = np.ascontiguousarray(rows, dtype=UINT64_LE)
    if np.any(canonical >= source_count) or np.any(np.diff(canonical.astype(np.int64)) <= 0):
        raise ValueError("M7 selected rows must be unique, in range, and ordered")
    if array_sha256(canonical, dtype=UINT64_LE) != result.selected_row_sha256:
        raise ValueError("M7 selected-row hash changed")
    return canonical


def _lift(arm: str, a2: np.ndarray, m7: M7Result, *, replace_lag: bool) -> S2Input:
    source = canonical_xyzit(a2)
    rows = _rows(m7, len(source))
    m7_points = _m7_points(m7)
    if len(rows) != len(m7_points):
        raise ValueError("M7 rows and points differ in length")
    lifted = source[rows.astype(np.int64)].copy(order="C")
    if replace_lag:
        lifted[:, 4] = m7_points[:, 3]
    if canonical_xyzt(lifted).tobytes(order="C") != m7_points.tobytes(order="C"):
        raise ValueError(f"{arm} projected XYZT differs from M7")
    if lifted[:, 3].tobytes(order="C") != source[rows.astype(np.int64), 3].tobytes(order="C"):
        raise ValueError(f"{arm} changed source intensity")
    return S2Input(arm, lifted, rows, m7.selected_row_sha256)


def lift_b2(a2: np.ndarray, m7_b: M7Result) -> S2Input:
    """Copy every A2 row, changing only the M7 B lag column."""

    result = _lift("B2", a2, m7_b, replace_lag=True)
    if not np.array_equal(result.selected_global_rows, np.arange(len(a2), dtype=UINT64_LE)):
        raise ValueError("B2 must preserve every A2 row in order")
    return result


def lift_c2(a2: np.ndarray, m7_c: M7Result, *, e2_count: int) -> S2Input:
    """Copy the exact M7 C selected A2 rows with their native lag and intensity."""

    result = _lift("C2", a2, m7_c, replace_lag=False)
    if len(result.points) != e2_count:
        raise ValueError("C2 point count must equal E2")
    return result


def lift_d2(c2: S2Input, m7_d: M7Result) -> S2Input:
    """Reuse C2's exact rows, changing only the M7 D lag column."""

    rows = np.ascontiguousarray(m7_d.selected_global_rows, dtype=UINT64_LE)
    if (
        not np.array_equal(rows, c2.selected_global_rows)
        or m7_d.selected_row_sha256 != c2.selected_row_sha256
    ):
        raise ValueError("D2 must reuse C2 selected rows")
    m7_points = _m7_points(m7_d)
    if len(m7_points) != len(c2.points):
        raise ValueError("D2 and C2 point counts differ")
    lifted = c2.points.copy(order="C")
    lifted[:, 4] = m7_points[:, 3]
    if canonical_xyzt(lifted).tobytes(order="C") != m7_points.tobytes(order="C"):
        raise ValueError("D2 projected XYZT differs from M7")
    if lifted[:, :4].tobytes(order="C") != c2.points[:, :4].copy(order="C").tobytes(order="C"):
        raise ValueError("D2 changed C2 XYZ or intensity")
    return S2Input("D2", lifted, rows, c2.selected_row_sha256)


def lift_f2(a2: np.ndarray, m7_f: M7Result) -> S2Input:
    """Copy complete selected A2 sweeps with native lag and intensity."""

    return _lift("F2", a2, m7_f, replace_lag=False)
