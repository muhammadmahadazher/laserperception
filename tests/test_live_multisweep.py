from __future__ import annotations

from math import sqrt

import numpy as np
import pytest

from laserperception.detection.live_multisweep import (
    LiveSweepHistory,
    acquisition_identity,
    live_raw_sweep_from_xyz,
    stamp_microseconds,
    stamp_nanoseconds,
    sweep_transform_from_ros,
)
from laserperception.detection.multisweep import HistoricalSweep, MultiSweepBuilder
from laserperception.detection.ros2_contract import TimeStamp


def _live(index: int, stamp_ns: int | None = None):
    timestamp = index * 1_000_000 if stamp_ns is None else stamp_ns
    stamp = TimeStamp(sec=0, nanosec=timestamp)
    points = np.array([[float(index + 1), 0.0, 0.0]], dtype=np.float32)
    return live_raw_sweep_from_xyz(points, frame_id="lidar", stamp=stamp)


def test_ros_stamp_integer_adaptation_preserves_microseconds_and_identity() -> None:
    exact = TimeStamp(sec=42, nanosec=123_456_000)
    submicrosecond = TimeStamp(sec=42, nanosec=123_456_999)

    assert stamp_nanoseconds(exact) == 42_123_456_000
    assert stamp_microseconds(exact) == 42_123_456
    assert stamp_microseconds(submicrosecond) == 42_123_456
    assert acquisition_identity(" lidar ", exact) == "lidar@42.123456000"


def test_xyz_adapter_uses_zero_placeholders_without_invented_features() -> None:
    points = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 0.5]], dtype=np.float32)
    live = live_raw_sweep_from_xyz(
        points,
        frame_id="lidar",
        stamp=TimeStamp(sec=1, nanosec=500_000_000),
    )

    assert live.sweep.points.dtype == np.float32
    assert np.array_equal(live.sweep.points[:, :3], points)
    assert np.array_equal(live.sweep.points[:, 3:], np.zeros((2, 2), dtype=np.float32))
    assert live.sweep.timestamp_microseconds == 1_500_000
    assert live.sweep.source_id == "lidar@1.500000000"


def test_xyz_adapter_rejects_empty_nonfinite_or_wrong_type() -> None:
    stamp = TimeStamp(sec=1, nanosec=0)
    with pytest.raises(ValueError, match="at least one"):
        live_raw_sweep_from_xyz(np.empty((0, 3), np.float32), frame_id="lidar", stamp=stamp)
    with pytest.raises(ValueError, match="finite"):
        live_raw_sweep_from_xyz(
            np.array([[np.nan, 0.0, 0.0]], np.float32), frame_id="lidar", stamp=stamp
        )
    with pytest.raises(TypeError, match="float32"):
        live_raw_sweep_from_xyz(np.ones((1, 3), np.float64), frame_id="lidar", stamp=stamp)


def test_history_warms_naturally_caps_at_ten_and_orders_nearest_first() -> None:
    history = LiveSweepHistory(max_historical_sweeps=10)
    first = _live(0)
    selection = history.select_for_current(first)
    assert selection.historical == ()
    history.store_current(first)

    second = _live(1)
    selection = history.select_for_current(second)
    assert [item.sweep.source_id for item in selection.historical] == [first.sweep.source_id]
    history.store_current(second)

    for index in range(2, 12):
        current = _live(index)
        history.select_for_current(current)
        history.store_current(current)
    assert history.depth == 10

    current = _live(12)
    selection = history.select_for_current(current)
    expected = [_live(index).sweep.source_id for index in range(11, 1, -1)]
    assert [item.sweep.source_id for item in selection.historical] == expected
    history.store_current(current)


def test_history_resets_on_nonincreasing_time_and_optional_large_gap() -> None:
    history = LiveSweepHistory(max_historical_sweeps=10, reset_gap_sec=0.5)
    initial = _live(1, 100_000_000)
    history.select_for_current(initial)
    history.store_current(initial)

    gap = _live(2, 700_000_001)
    selected = history.select_for_current(gap)
    assert selected.reset_reason == "gap"
    assert selected.historical == ()
    history.store_current(gap)

    regression = _live(3, 700_000_001)
    selected = history.select_for_current(regression)
    assert selected.reset_reason == "time_regression"
    assert selected.historical == ()
    history.store_current(regression)


def test_history_can_retain_a_valid_current_after_failed_build_attempt() -> None:
    history = LiveSweepHistory()
    failed_current = _live(0)
    history.select_for_current(failed_current)
    history.store_current(failed_current)

    next_current = _live(1)
    selected = history.select_for_current(next_current)
    assert selected.historical == (failed_current,)
    history.store_current(next_current)


def test_ros_transform_adapter_identity_and_signed_translations() -> None:
    identity = sweep_transform_from_ros(
        translation_xyz=(0.0, 0.0, 0.0),
        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        source_id="history",
        target_id="current",
    )
    assert np.array_equal(identity.lidar2sensor, np.eye(4, dtype=np.float32))

    positive = sweep_transform_from_ros(
        translation_xyz=(1.25, -2.5, 0.75),
        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        source_id="history",
        target_id="current",
    )
    negative = sweep_transform_from_ros(
        translation_xyz=(-1.25, 2.5, -0.75),
        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        source_id="history",
        target_id="current",
    )
    assert positive.lidar2sensor[:3, 3].tolist() == [-1.25, 2.5, -0.75]
    assert negative.lidar2sensor[:3, 3].tolist() == [1.25, -2.5, 0.75]


def test_ros_transform_adapter_rotation_and_translation_drive_builder_correctly() -> None:
    current = _live(2, 2_000_000)
    history = live_raw_sweep_from_xyz(
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        frame_id="lidar",
        stamp=TimeStamp(sec=0, nanosec=1_000_000),
    )
    transform = sweep_transform_from_ros(
        translation_xyz=(2.0, 3.0, 0.0),
        quaternion_xyzw=(0.0, 0.0, sqrt(0.5), sqrt(0.5)),
        source_id=history.sweep.source_id,
        target_id=current.sweep.source_id,
    )

    output = MultiSweepBuilder().build(
        current.sweep,
        [HistoricalSweep(history.sweep, transform)],
    )
    assert output.points_xyzt[1, :3] == pytest.approx([2.0, 4.0, 0.0], abs=1e-6)


def test_ros_transform_adapter_rejects_malformed_quaternion() -> None:
    with pytest.raises(ValueError, match="normalized"):
        sweep_transform_from_ros(
            translation_xyz=(0.0, 0.0, 0.0),
            quaternion_xyzw=(0.0, 0.0, 0.0, 2.0),
            source_id="history",
            target_id="current",
        )
