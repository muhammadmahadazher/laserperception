from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from laserperception.detection.multisweep import (
    HistoricalSweep,
    LidarPose,
    MultiSweepBuilder,
    MultiSweepBuilderConfig,
    RawSweep,
    SweepTransform,
)


def _raw(source_id: str, timestamp_us: int, rows: list[list[float]]) -> RawSweep:
    return RawSweep(np.asarray(rows, dtype=np.float32), timestamp_us, source_id)


def _identity_transform(source_id: str, target_id: str = "current") -> SweepTransform:
    return SweepTransform(np.eye(4, dtype=np.float32), source_id, target_id)


def _pose(
    *,
    l2e_rotation: np.ndarray | None = None,
    l2e_translation: np.ndarray | None = None,
    e2g_rotation: np.ndarray | None = None,
    e2g_translation: np.ndarray | None = None,
) -> LidarPose:
    return LidarPose(
        np.eye(3, dtype=np.float64) if l2e_rotation is None else l2e_rotation,
        np.zeros(3, dtype=np.float64) if l2e_translation is None else l2e_translation,
        np.eye(3, dtype=np.float64) if e2g_rotation is None else e2g_rotation,
        np.zeros(3, dtype=np.float64) if e2g_translation is None else e2g_translation,
    )


def test_raw_sweep_requires_exact_shape_dtype_and_timestamp() -> None:
    with pytest.raises(TypeError, match="dtype float32"):
        RawSweep(np.ones((2, 5), dtype=np.float64), 1, "source")
    with pytest.raises(ValueError, match=r"shape \(N, 5\)"):
        RawSweep(np.ones((2, 4), dtype=np.float32), 1, "source")
    with pytest.raises(ValueError, match="non-empty"):
        RawSweep(np.empty((0, 5), dtype=np.float32), 1, "source")
    with pytest.raises(TypeError, match="integer"):
        RawSweep(np.ones((2, 5), dtype=np.float32), 1.0, "source")  # type: ignore[arg-type]


def test_raw_sweep_copies_to_contiguous_storage() -> None:
    backing = np.arange(30, dtype=np.float32).reshape(3, 10)
    noncontiguous = backing[:, ::2]
    sweep = RawSweep(noncontiguous, 1_000_000, " source ")
    backing.fill(-1)
    assert sweep.points.flags.c_contiguous
    assert sweep.source_id == "source"
    assert sweep.points[0].tolist() == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert sweep.timestamp_seconds == 1.0


def test_raw_nuscenes_file_load_and_invalid_length(tmp_path: Path) -> None:
    valid = tmp_path / "valid.bin"
    values = np.arange(10, dtype=np.float32)
    values.tofile(valid)
    sweep = RawSweep.from_nuscenes_file(
        valid, timestamp_microseconds=1_234_567, source_id="sample_data_token"
    )
    assert np.array_equal(sweep.points, values.reshape(2, 5))

    invalid = tmp_path / "invalid.bin"
    np.arange(6, dtype=np.float32).tofile(invalid)
    with pytest.raises(ValueError, match="divisible by five"):
        RawSweep.from_nuscenes_file(invalid, timestamp_microseconds=1, source_id="invalid")


def test_transform_from_identity_poses_is_exact_float32_identity() -> None:
    transform = SweepTransform.from_poses(
        source_id="history", target_id="current", sweep_pose=_pose(), current_pose=_pose()
    )
    assert transform.lidar2sensor.dtype == np.float32
    assert np.array_equal(transform.lidar2sensor, np.eye(4, dtype=np.float32))


def test_transform_from_poses_reproduces_translation() -> None:
    transform = SweepTransform.from_poses(
        source_id="history",
        target_id="current",
        sweep_pose=_pose(e2g_translation=np.array([1.25, -2.5, 0.75], dtype=np.float64)),
        current_pose=_pose(),
    )
    expected = np.eye(4, dtype=np.float32)
    expected[:3, 3] = [-1.25, 2.5, -0.75]
    assert np.array_equal(transform.lidar2sensor, expected)


def test_builder_applies_rotation_then_translation_with_upstream_casts() -> None:
    current = _raw("current", 2_000_001, [[20.0, 20.0, 0.0, 1.0, 7.0]])
    history = _raw(
        "history",
        1_000_000,
        [[1.234567, -2.345678, 0.333333, 9.0, 2.0]],
    )
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = np.array(
        [[0.0, -1.0000001, 0.0], [0.99999994, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    matrix[:3, 3] = np.array([0.12345679, -0.2345679, 0.34567893], dtype=np.float32)
    transform = SweepTransform(matrix, "history", "current")

    output = MultiSweepBuilder().build(current, [HistoricalSweep(history, transform)])
    expected_xyz = history.points.copy()
    reloaded = np.array(matrix.tolist())
    expected_xyz[:, :3] = expected_xyz[:, :3] @ reloaded[:3, :3]
    expected_xyz[:, :3] -= reloaded[:3, 3]

    assert np.array_equal(output.points_xyzt[1, :3], expected_xyz[0, :3])
    assert output.points_xyzt[1, 3] == np.float32((2_000_001 / 1e6) - (1_000_000 / 1e6))


def test_builder_identity_transform_and_time_lag_sign_units() -> None:
    current = _raw("current", 1_500_123, [[10.0, 0.0, 0.0, 5.0, 99.0]])
    history = _raw("history", 1_000_000, [[11.0, 0.0, 0.0, 6.0, 88.0]])
    output = MultiSweepBuilder().build(
        current, [HistoricalSweep(history, _identity_transform("history"))]
    )
    assert output.points_xyzt[0].tolist() == [10.0, 0.0, 0.0, 0.0]
    assert output.points_xyzt[1, :3].tolist() == [11.0, 0.0, 0.0]
    expected_lag = np.float32((1_500_123 / 1_000_000) - (1_000_000 / 1_000_000))
    assert output.points_xyzt[1, 3] == expected_lag
    assert output.points_xyzt[1, 3] > 0


def test_remove_close_uses_strict_square_and_preserves_order() -> None:
    current = _raw("current", 2_000_000, [[10.0, 10.0, 0.0, 0.0, 0.0]])
    history = _raw(
        "history",
        1_000_000,
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.999, 1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.25, 0.0, 0.0, 0.0],
            [2.0, 2.0, 0.0, 0.0, 0.0],
        ],
    )
    builder = MultiSweepBuilder(MultiSweepBuilderConfig(remove_close=True))
    output = builder.build(current, [HistoricalSweep(history, _identity_transform("history"))])
    assert output.points_xyzt[:, :2].tolist() == [
        [10.0, 10.0],
        [pytest.approx(0.999), 1.0],
        [-1.0, 0.25],
        [2.0, 2.0],
    ]


def test_scene_start_keeps_current_only_without_padding() -> None:
    current = _raw(
        "current",
        1_000_000,
        [[1.0, 2.0, 2.5, 10.0, 5.0], [2.0, 3.0, 1.0, 20.0, 6.0]],
    )
    output = MultiSweepBuilder().build(current, [])
    assert output.points_xyzt.tolist() == [[1.0, 2.0, 2.5, 0.0], [2.0, 3.0, 1.0, 0.0]]


def test_padding_duplicates_current_after_original_when_explicitly_enabled() -> None:
    current = _raw("current", 1_000_000, [[1.0, 2.0, 0.0, 3.0, 4.0]])
    builder = MultiSweepBuilder(
        MultiSweepBuilderConfig(max_historical_sweeps=2, pad_empty_sweeps=True)
    )
    output = builder.build(current, [])
    assert output.points_xyzt.tolist() == [[1.0, 2.0, 0.0, 0.0]] * 3


def test_concatenation_uses_current_then_history_sequence_and_source_order() -> None:
    current = _raw("current", 3_000_000, [[1.0, 0.0, 0.0, 0.0, 0.0]])
    first = _raw("first", 2_000_000, [[2.0, 0.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0, 0.0]])
    second = _raw("second", 1_000_000, [[4.0, 0.0, 0.0, 0.0, 0.0]])
    output = MultiSweepBuilder().build(
        current,
        [
            HistoricalSweep(first, _identity_transform("first")),
            HistoricalSweep(second, _identity_transform("second")),
        ],
    )
    assert output.points_xyzt[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert output.points_xyzt[:, 3].tolist() == [0.0, 1.0, 1.0, 2.0]


def test_test_mode_selection_takes_only_first_configured_history() -> None:
    current = _raw("current", 4_000_000, [[1.0, 0.0, 0.0, 0.0, 0.0]])
    history = [
        HistoricalSweep(
            _raw(f"h{index}", index * 1_000_000, [[float(index + 2), 0, 0, 0, 0]]),
            _identity_transform(f"h{index}"),
        )
        for index in range(3, 0, -1)
    ]
    output = MultiSweepBuilder(MultiSweepBuilderConfig(max_historical_sweeps=2)).build(
        current, history
    )
    assert output.points_xyzt[:, 0].tolist() == [1.0, 5.0, 4.0]


def test_strict_point_cloud_range_boundaries_are_excluded() -> None:
    current = _raw(
        "current",
        1,
        [
            [-50.0, 0.0, 0.0, 0.0, 0.0],
            [49.999, 0.0, 0.0, 0.0, 0.0],
            [0.0, 50.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -5.0, 0.0, 0.0],
            [0.0, 0.0, 2.999, 0.0, 0.0],
        ],
    )
    output = MultiSweepBuilder().build(current, [])
    assert output.points_xyzt[:, :3].tolist() == [
        [49.999000549316406, 0.0, 0.0],
        [0.0, 0.0, 2.999000072479248],
    ]


def test_builder_output_is_float32_contiguous_repeatable_and_stably_hashed() -> None:
    current = _raw("current", 2_000_000, [[1.0, 2.0, 0.5, 4.0, 9.0]])
    history = _raw("history", 1_500_000, [[3.0, 4.0, -0.5, 5.0, 8.0]])
    item = HistoricalSweep(history, _identity_transform("history"))
    first = MultiSweepBuilder().build(current, [item])
    second = MultiSweepBuilder().build(current, [item])
    assert first.points_xyzt.dtype == np.float32
    assert first.points_xyzt.flags.c_contiguous
    assert first.points_xyzt.shape == (2, 4)
    assert np.array_equal(first.points_xyzt, second.points_xyzt)
    assert first.sha256 == hashlib.sha256(first.points_xyzt.tobytes(order="C")).hexdigest()
    assert first.sha256 == "ea811305ebc2370feaa64d1014b5c69a9e64fc571e4accc2edc880a19b7df8c5"


def test_source_and_target_mismatches_fail_closed() -> None:
    history = _raw("history", 1, [[1.0, 1.0, 1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="source_id"):
        HistoricalSweep(history, _identity_transform("other"))

    current = _raw("current", 2, [[2.0, 2.0, 2.0, 0.0, 0.0]])
    wrong_target = HistoricalSweep(history, _identity_transform("history", "not-current"))
    with pytest.raises(ValueError, match="target_id"):
        MultiSweepBuilder().build(current, [wrong_target])
