from __future__ import annotations

import numpy as np
import pytest

from laserperception.detection.m8_input import (
    M8MultiSweepBuilder,
    M8PointCloud,
    m8_elapsed_seconds,
)
from laserperception.detection.multisweep import (
    HistoricalSweep,
    MultiSweepBuilder,
    MultiSweepBuilderConfig,
    RawSweep,
    SweepTransform,
)


def _sweep(points: list[list[float]], timestamp: int, source: str) -> RawSweep:
    return RawSweep(np.array(points, dtype=np.float32), timestamp, source)


def test_feature_construction_preserves_rows_intensity_and_historical_projection() -> None:
    current = _sweep(
        [[1.0, 2.0, 0.0, 0.25, 7.0], [3.0, 4.0, 1.0, 0.75, 8.0]],
        2_000_000,
        "current",
    )
    history = _sweep([[5.0, 6.0, 2.0, 0.5, 9.0]], 1_500_000, "history")
    transform = SweepTransform(np.eye(4, dtype=np.float32), "history", "current")
    historical = [HistoricalSweep(history, transform)]

    result = M8MultiSweepBuilder().build(current, historical)
    reference = MultiSweepBuilder().build(current, historical)

    assert result.points.dtype == np.float32
    assert result.points.flags.c_contiguous
    assert np.array_equal(result.historical_projection, reference.points_xyzt)
    assert result.points[:, 3].tolist() == [0.25, 0.75, 0.5]
    assert result.points[:, 4].tolist() == [0.0, 0.0, 0.5]


def test_transform_changes_geometry_but_not_corresponding_intensity() -> None:
    current = _sweep([[0.0, 0.0, 0.0, 0.1, 0.0]], 2_000_000, "current")
    history = _sweep([[2.0, 3.0, 0.0, 0.9, 0.0]], 1_000_000, "history")
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = np.array([1.0, 2.0, 0.0], dtype=np.float32)
    historical = [HistoricalSweep(history, SweepTransform(matrix, "history", "current"))]

    result = M8MultiSweepBuilder().build(current, historical)

    assert result.points[1].tolist() == pytest.approx([1.0, 1.0, 0.0, 0.9, 1.0])


def test_builder_preserves_configured_history_depth_and_range_membership() -> None:
    current = _sweep([[0.0, 0.0, 0.0, 0.1, 0.0]], 3_000_000, "current")
    histories = []
    for index in range(2):
        source = f"history-{index}"
        sweep = _sweep(
            [[1.0 + index, 1.0, 0.0, 0.2 + index, 0.0]],
            2_000_000 - index,
            source,
        )
        histories.append(
            HistoricalSweep(sweep, SweepTransform(np.eye(4, dtype=np.float32), source, "current"))
        )

    result = M8MultiSweepBuilder(MultiSweepBuilderConfig(max_historical_sweeps=1)).build(
        current, histories
    )

    assert result.points.shape == (2, 5)
    assert result.points[:, 3].tolist() == pytest.approx([0.1, 0.2])


def test_m8_point_cloud_rejects_wrong_feature_contract() -> None:
    with pytest.raises(TypeError, match="float32"):
        M8PointCloud(np.ones((2, 5), dtype=np.float64))
    with pytest.raises(ValueError, match="shape"):
        M8PointCloud(np.ones((2, 4), dtype=np.float32))


def test_timestamp_fixture_is_positive_float32_current_minus_history() -> None:
    lag = m8_elapsed_seconds(2_000_000, 1_250_000)
    current = m8_elapsed_seconds(2_000_000, 2_000_000)

    assert lag.dtype == np.float32
    assert lag == np.float32(0.75)
    assert current == np.float32(0.0)
    assert not np.signbit(current)
    with pytest.raises(ValueError, match="must not be newer"):
        m8_elapsed_seconds(1_000_000, 1_000_001)
