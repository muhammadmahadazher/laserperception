"""Synthetic CPU checks for the M7-to-M8 S2 input lift."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from laserperception.detection.m8_s2_input import (
    ARM_ORDER,
    UINT64_LE,
    array_sha256,
    canonical_xyzt,
    lift_b2,
    lift_c2,
    lift_d2,
    lift_f2,
)


def _m7_result(
    a2: np.ndarray, rows: list[int], *, lags: list[float] | None = None
) -> SimpleNamespace:
    selected = np.asarray(rows, dtype=UINT64_LE)
    points = canonical_xyzt(a2[selected.astype(np.int64)])
    if lags is not None:
        points[:, 3] = np.asarray(lags, dtype="<f4")
    return SimpleNamespace(
        points=points,
        selected_global_rows=selected,
        selected_row_sha256=array_sha256(selected, dtype=UINT64_LE),
    )


def _a2() -> np.ndarray:
    return np.asarray(
        [
            [1.0, 2.0, 3.0, 0.31, 0.0],
            [4.0, 5.0, 6.0, 0.57, 0.1],
            [7.0, 8.0, 9.0, 0.73, 0.2],
            [10.0, 11.0, 12.0, 0.91, 0.3],
        ],
        dtype="<f4",
    )


def test_lift_preserves_intensity_rows_and_m7_lag_bytes() -> None:
    a2 = _a2()
    b = _m7_result(a2, [0, 1, 2, 3], lags=[0.0, 0.05, 0.1, 0.15])
    c = _m7_result(a2, [0, 2, 3])
    d = _m7_result(a2, [0, 2, 3], lags=[0.0, 0.1, 0.15])
    f = _m7_result(a2, [0, 1, 3])

    b2 = lift_b2(a2, b)
    c2 = lift_c2(a2, c, e2_count=3)
    d2 = lift_d2(c2, d)
    f2 = lift_f2(a2, f)

    assert ARM_ORDER == ("B2", "C2", "D2", "F2")
    assert np.array_equal(b2.selected_global_rows, np.arange(len(a2), dtype=UINT64_LE))
    assert b2.points[:, :4].tobytes(order="C") == a2[:, :4].tobytes(order="C")
    assert c2.points.tobytes(order="C") == a2[[0, 2, 3]].tobytes(order="C")
    assert d2.selected_row_sha256 == c2.selected_row_sha256
    assert d2.points[:, :4].tobytes(order="C") == c2.points[:, :4].tobytes(order="C")
    assert f2.points.tobytes(order="C") == a2[[0, 1, 3]].tobytes(order="C")
    for lifted, m7 in ((b2, b), (c2, c), (d2, d), (f2, f)):
        assert canonical_xyzt(lifted.points).tobytes(order="C") == m7.points.tobytes(order="C")
        assert lifted.points[:, 3].tobytes(order="C") == a2[
            lifted.selected_global_rows.astype(np.int64), 3
        ].tobytes(order="C")
        assert lifted.points.dtype == np.dtype("<f4")
        assert lifted.points.flags.c_contiguous
        assert int(lifted.points[0, 4].view("<u4")) == 0
    assert np.all(b2.points[:, 3] != 0)


def test_lift_rejects_wrong_selection_mutation_and_count() -> None:
    a2 = _a2()
    c = _m7_result(a2, [0, 2, 3])
    with pytest.raises(ValueError, match="point count"):
        lift_c2(a2, c, e2_count=2)
    wrong_rows = _m7_result(a2, [0, 1, 3])
    with pytest.raises(ValueError, match="reuse C2 selected rows"):
        lift_d2(lift_c2(a2, c, e2_count=3), wrong_rows)
    mutated = _m7_result(a2, [0, 2, 3])
    mutated.points[1, 0] += np.float32(1)
    with pytest.raises(ValueError, match="projected XYZT"):
        lift_f2(a2, mutated)


def test_canonical_hashing_includes_little_endian_byte_order() -> None:
    little = _a2()
    big = little.astype(">f4")
    assert array_sha256(little) == array_sha256(big)
    assert canonical_xyzt(big).dtype == np.dtype("<f4")
    assert canonical_xyzt(big).tobytes(order="C") == canonical_xyzt(little).tobytes(order="C")


def test_s2_input_path_imports_no_accelerator_or_detector() -> None:
    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys; "
        "import laserperception.detection.m8_s2_input; "
        "import benchmarks.m8.prepare_s2_inputs; "
        "assert not any(x == 'torch' or x.startswith('torch.') or x.startswith('pcdet') "
        "or x.endswith('.m8_backend') for x in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True)
