from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from laserperception.datasets.kitti_raw import (
    KITTI_TO_MODEL_ROTATION,
    KittiCalibration,
    KittiOxtsRecord,
    KittiRawFrame,
    KittiRawSequence,
    KittiTimestamp,
    load_kitti_odometry_poses,
    official_oxts_poses,
    oxts_pose_in_rectified_camera,
    read_kitti_raw_velodyne,
    rotation_angle_radians,
    select_m6a_reconstruction_frames,
)
from laserperception.detection.multisweep import (
    MultiSweepBuilder,
    MultiSweepBuilderConfig,
    SweepTransform,
)
from laserperception.evaluation.m6b_input_oracle import (
    freeze_sweep_transforms,
    reconstruct_from_frozen_transforms,
)


def _identity_calibration(root: Path, *, imu_translation: str = "0 0 0") -> None:
    root.mkdir(parents=True, exist_ok=True)
    rigid = "calib_time: 25-May-2012 16:47:16\nR: 1 0 0 0 1 0 0 0 1\nT: {translation}\n"
    (root / "calib_imu_to_velo.txt").write_text(
        rigid.format(translation=imu_translation), encoding="utf-8"
    )
    (root / "calib_velo_to_cam.txt").write_text(rigid.format(translation="0 0 0"), encoding="utf-8")
    (root / "calib_cam_to_cam.txt").write_text("R_rect_00: 1 0 0 0 1 0 0 0 1\n", encoding="utf-8")


def _synthetic_sequence(tmp_path: Path, frame_count: int = 12) -> KittiRawSequence:
    date_root = tmp_path / "2011_09_26"
    drive_root = date_root / "2011_09_26_drive_0001_sync"
    point_root = drive_root / "velodyne_points/data"
    oxts_root = drive_root / "oxts/data"
    point_root.mkdir(parents=True)
    oxts_root.mkdir(parents=True)
    _identity_calibration(date_root)
    timestamp_lines: list[str] = []
    for index in range(frame_count):
        values = np.array(
            [
                [float(index + 1), 0.0, 0.0, 0.25],
                [float(index + 1), 1.0, 0.5, 0.75],
            ],
            dtype="<f4",
        )
        values.tofile(point_root / f"{index:010d}.bin")
        (oxts_root / f"{index:010d}.txt").write_text(
            "49.0 8.0 100.0 0.0 0.0 0.0\n", encoding="utf-8"
        )
        seconds, tenth = divmod(index, 10)
        timestamp_lines.append(f"2011-09-26 00:00:{seconds:02d}.{tenth * 100_000_000:09d}")
    (drive_root / "velodyne_points/timestamps.txt").write_text(
        "\n".join(timestamp_lines) + "\n", encoding="utf-8"
    )
    return KittiRawSequence(date_root, drive_root)


def _pose(
    *,
    rotation: np.ndarray | None = None,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    if rotation is not None:
        value[:3, :3] = rotation
    value[:3, 3] = translation
    return value


def test_raw_decode_preserves_exact_xyzi_dtype_bytes_and_order(tmp_path: Path) -> None:
    source = np.array(
        [[1.0, 2.0, 3.0, 0.25], [-4.0, 5.0, -6.0, 0.75]],
        dtype="<f4",
    )
    path = tmp_path / "scan.bin"
    path.write_bytes(source.tobytes(order="C"))
    decoded = read_kitti_raw_velodyne(path)
    assert decoded.dtype == np.float32
    assert decoded.flags.c_contiguous
    assert np.array_equal(decoded, source)
    assert decoded.astype("<f4", copy=False).tobytes(order="C") == path.read_bytes()


def test_raw_decode_rejects_empty_malformed_and_nonfinite(tmp_path: Path) -> None:
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="non-empty"):
        read_kitti_raw_velodyne(empty)

    malformed = tmp_path / "malformed.bin"
    malformed.write_bytes(b"123")
    with pytest.raises(ValueError, match="divisible by 16"):
        read_kitti_raw_velodyne(malformed)

    nonfinite = tmp_path / "nonfinite.bin"
    np.array([[1.0, 2.0, np.nan, 0.5]], dtype="<f4").tofile(nonfinite)
    with pytest.raises(ValueError, match="NaN or infinite"):
        read_kitti_raw_velodyne(nonfinite)


def test_timestamp_parse_is_integer_exact_and_floors_microseconds() -> None:
    timestamp = KittiTimestamp.parse("1970-01-01 00:00:01.000001999")
    assert timestamp.nanoseconds == 1_000_001_999
    assert timestamp.microseconds == 1_000_001
    assert timestamp.discarded_nanoseconds == 999
    boundary = KittiTimestamp.parse("1970-01-01 00:00:01.000002000")
    assert boundary.microseconds == 1_000_002
    assert boundary.discarded_nanoseconds == 0
    with pytest.raises(ValueError, match="invalid calendar date"):
        KittiTimestamp.parse("2011-02-31 00:00:00.000000000")
    with pytest.raises(ValueError, match="must match"):
        KittiTimestamp.parse("2011-09-26 00:00:00.1")


def test_calibration_direction_inverse_and_virtual_basis(tmp_path: Path) -> None:
    _identity_calibration(tmp_path, imu_translation="1 2 3")
    calibration = KittiCalibration.from_date_root(tmp_path)
    assert calibration.imu_to_velodyne[:3, 3].tolist() == [1.0, 2.0, 3.0]
    assert calibration.velodyne_to_imu[:3, 3].tolist() == [-1.0, -2.0, -3.0]
    assert np.allclose(
        calibration.imu_to_velodyne @ calibration.velodyne_to_imu,
        np.eye(4),
        atol=1e-12,
    )
    assert np.array_equal(calibration.model_to_imu[:3, :3], KITTI_TO_MODEL_ROTATION.T)
    assert np.isclose(np.linalg.det(calibration.model_to_imu[:3, :3]), 1.0)


def test_model_frame_basis_and_reflectance_adaptation() -> None:
    record = KittiOxtsRecord.parse("49 8 100 0 0 0")
    frame = KittiRawFrame(
        0,
        "drive/0000000000",
        KittiTimestamp.parse("2011-09-26 00:00:00.000000001"),
        np.array(
            [[1.0, 0.0, 0.0, 0.25], [0.0, 1.0, 0.0, 0.75]],
            dtype=np.float32,
        ),
        record,
    )
    sweep = frame.to_raw_sweep()
    assert sweep.points.tolist() == [
        [0.0, 1.0, 0.0, 0.25, 0.0],
        [-1.0, 0.0, 0.0, 0.75, 0.0],
    ]
    assert (
        frame.source_sha256
        == hashlib.sha256(frame.points_xyzi.astype("<f4", copy=False).tobytes()).hexdigest()
    )


def test_oxts_origin_rotation_order_and_translation_are_analytic() -> None:
    first = KittiOxtsRecord.parse("0 0 0 0 0 0")
    second = KittiOxtsRecord.parse(f"0 0 5 0 0 {math.pi / 2}")
    poses = official_oxts_poses([first, second])
    expected_rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    assert np.array_equal(poses[0], np.eye(4))
    assert np.allclose(poses[1][:3, :3], expected_rotation, atol=1e-15)
    assert poses[1][:3, 3].tolist() == [0.0, 0.0, 5.0]

    combined = KittiOxtsRecord.parse("0 0 0 0.2 -0.3 0.4")
    combined_rotation = official_oxts_poses([first, combined])[1][:3, :3]
    assert np.allclose(combined_rotation.T @ combined_rotation, np.eye(3), atol=1e-15)
    assert np.isclose(np.linalg.det(combined_rotation), 1.0)


def test_odometry_load_and_camera_conjugation(tmp_path: Path) -> None:
    pose_path = tmp_path / "04.txt"
    pose_path.write_text("1 0 0 1 0 1 0 2 0 0 1 3\n", encoding="utf-8")
    loaded = load_kitti_odometry_poses(pose_path)
    assert len(loaded) == 1
    assert loaded[0][:3, 3].tolist() == [1.0, 2.0, 3.0]
    calibration = KittiCalibration(np.eye(4), np.eye(4), np.eye(4))
    assert np.array_equal(oxts_pose_in_rectified_camera(loaded[0], calibration), loaded[0])


def test_virtual_frame_relative_transform_matches_basis_conjugation(tmp_path: Path) -> None:
    _identity_calibration(tmp_path)
    calibration = KittiCalibration.from_date_root(tmp_path)
    yaw = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    history_pose = calibration.model_lidar_pose(_pose(rotation=yaw, translation=(1.0, 2.0, 0.0)))
    current_pose = calibration.model_lidar_pose(np.eye(4, dtype=np.float64))
    transform = SweepTransform.from_poses(
        source_id="history",
        target_id="current",
        sweep_pose=history_pose,
        current_pose=current_pose,
    )
    point_model = np.array([[2.0, 3.0, 0.5]], dtype=np.float32)
    actual = point_model.copy()
    matrix = np.array(transform.lidar2sensor.tolist())
    actual[:] = actual @ matrix[:3, :3] - matrix[:3, 3]

    native_storage = np.eye(4, dtype=np.float64)
    native_storage[:3, :3] = yaw.T
    native_storage[:3, 3] = -yaw.T @ np.array([1.0, 2.0, 0.0])
    expected_storage = np.eye(4, dtype=np.float64)
    expected_storage[:3, :3] = (
        KITTI_TO_MODEL_ROTATION @ native_storage[:3, :3] @ KITTI_TO_MODEL_ROTATION.T
    )
    expected_storage[:3, 3] = native_storage[:3, 3] @ KITTI_TO_MODEL_ROTATION.T
    expected_model = point_model @ expected_storage[:3, :3]
    expected_model -= expected_storage[:3, 3]
    assert np.array_equal(actual, expected_model.astype(np.float32))


def test_rotation_angle_and_selection_are_deterministic() -> None:
    assert rotation_angle_radians(np.eye(3)) == 0.0
    yaw = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    assert rotation_angle_radians(yaw) == pytest.approx(math.pi / 2)
    poses = [_pose(translation=(float(index * index) / 100.0, 0.0, 0.0)) for index in range(108)]
    selected = select_m6a_reconstruction_frames(poses)
    assert len(selected) == 24
    assert selected == tuple(sorted(set(selected)))
    assert {0, 1, 2, 5, 10}.issubset(selected)
    assert sum(index >= 10 for index in selected) >= 10
    assert select_m6a_reconstruction_frames(poses) == selected


def test_sequence_reconstructs_current_shallow_and_full_history_exactly(tmp_path: Path) -> None:
    sequence = _synthetic_sequence(tmp_path)
    current_only = sequence.reconstruct(0)
    assert current_only.selected_indices == (0,)
    assert current_only.source_counts == (2,)
    assert current_only.point_cloud.points_xyzt.tolist() == [
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 1.0, 0.5, 0.0],
    ]

    shallow = sequence.reconstruct(2)
    assert shallow.selected_indices == (2, 1, 0)
    assert shallow.source_counts == (2, 2, 2)
    assert shallow.point_cloud.points_xyzt[:, 1].tolist() == [3.0, 3.0, 2.0, 2.0, 1.0, 1.0]
    lag_1 = np.float32(
        sequence.timestamps[2].microseconds / 1_000_000
        - sequence.timestamps[1].microseconds / 1_000_000
    )
    lag_2 = np.float32(
        sequence.timestamps[2].microseconds / 1_000_000
        - sequence.timestamps[0].microseconds / 1_000_000
    )
    assert shallow.point_cloud.points_xyzt[:, 3].tolist() == [
        0.0,
        0.0,
        lag_1,
        lag_1,
        lag_2,
        lag_2,
    ]

    full = sequence.reconstruct(11)
    assert full.selected_indices == tuple(range(11, 0, -1))
    assert len(full.source_counts) == 11
    assert full.pre_builder_row_count == 22
    assert full.point_cloud.points_xyzt.shape == (22, 4)
    assert np.all(np.diff(full.point_cloud.points_xyzt[::2, 3]) >= 0)
    assert sequence.reconstruct(11).point_cloud.sha256 == full.point_cloud.sha256
    assert full.point_cloud.points_xyzt.dtype == np.float32
    assert full.point_cloud.points_xyzt.flags.c_contiguous


def test_frozen_transform_reconstruction_preserves_builder_bytes(tmp_path: Path) -> None:
    sequence = _synthetic_sequence(tmp_path)
    transforms = freeze_sweep_transforms(sequence, 11)
    builder = MultiSweepBuilder(MultiSweepBuilderConfig(max_historical_sweeps=5))
    expected = sequence.reconstruct(11, builder=builder)
    actual = reconstruct_from_frozen_transforms(
        sequence,
        11,
        transforms,
        builder=builder,
    )
    assert actual.selected_indices == expected.selected_indices
    assert actual.source_counts == expected.source_counts
    assert np.array_equal(actual.point_cloud.points_xyzt, expected.point_cloud.points_xyzt)

    corrupted = [dict(record) for record in transforms]
    corrupted[0]["lidar2sensor_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        reconstruct_from_frozen_transforms(
            sequence,
            11,
            corrupted,
            builder=builder,
        )


def test_zero_motion_changes_only_lag_and_preserves_source_order(tmp_path: Path) -> None:
    sequence = _synthetic_sequence(tmp_path, frame_count=3)
    result = sequence.reconstruct(2).point_cloud.points_xyzt
    assert result[:, :3].tolist() == [
        [0.0, 3.0, 0.0],
        [-1.0, 3.0, 0.5],
        [0.0, 2.0, 0.0],
        [-1.0, 2.0, 0.5],
        [0.0, 1.0, 0.0],
        [-1.0, 1.0, 0.5],
    ]
    lag_1 = np.float32(
        sequence.timestamps[2].microseconds / 1_000_000
        - sequence.timestamps[1].microseconds / 1_000_000
    )
    lag_2 = np.float32(
        sequence.timestamps[2].microseconds / 1_000_000
        - sequence.timestamps[0].microseconds / 1_000_000
    )
    assert result[:, 3].tolist() == [0.0, 0.0, lag_1, lag_1, lag_2, lag_2]
