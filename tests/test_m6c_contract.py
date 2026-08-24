from __future__ import annotations

import hashlib
from math import radians
from pathlib import Path

import numpy as np
import pytest

from laserperception.datasets.kitti_raw import (
    KITTI_TO_MODEL_ROTATION,
    KittiCalibration,
    KittiRawSequence,
)
from laserperception.datasets.kitti_ros_replay import (
    kitti_ros_replay_acquisition,
    model_lidar_pose_to_world_transform,
)
from laserperception.detection.live_multisweep import (
    LiveSweepHistory,
    live_raw_sweep_from_xyz,
)
from laserperception.detection.m6c_contract import (
    M6cInputProgress,
    M6cProgressIdentity,
    require_file_sha256,
)
from laserperception.detection.ros2_contract import TimeStamp


def _write_sequence(root: Path, frame_count: int = 12) -> KittiRawSequence:
    date_root = root / "2011_09_26"
    drive_root = date_root / "2011_09_26_drive_0001_sync"
    point_root = drive_root / "velodyne_points/data"
    oxts_root = drive_root / "oxts/data"
    point_root.mkdir(parents=True)
    oxts_root.mkdir(parents=True)
    rigid = "R: 1 0 0 0 1 0 0 0 1\nT: 0 0 0\n"
    (date_root / "calib_imu_to_velo.txt").write_text(rigid, encoding="utf-8")
    (date_root / "calib_velo_to_cam.txt").write_text(rigid, encoding="utf-8")
    (date_root / "calib_cam_to_cam.txt").write_text(
        "R_rect_00: 1 0 0 0 1 0 0 0 1\n", encoding="utf-8"
    )
    timestamps: list[str] = []
    for index in range(frame_count):
        np.array([[1.0, 2.0, 3.0, 0.5]], dtype="<f4").tofile(point_root / f"{index:010d}.bin")
        yaw = radians(float(index))
        (oxts_root / f"{index:010d}.txt").write_text(
            f"49.0 8.0 {100.0 + index} 0.0 0.0 {yaw}\n", encoding="utf-8"
        )
        timestamps.append(f"2011-09-26 00:00:{index:02d}.123456789")
    (drive_root / "velodyne_points/timestamps.txt").write_text(
        "\n".join(timestamps) + "\n", encoding="utf-8"
    )
    return KittiRawSequence(date_root, drive_root)


def _quaternion_rotation(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = quaternion
    scale = 2.0 / sum(value * value for value in quaternion)
    return np.array(
        [
            [1 - scale * (y * y + z * z), scale * (x * y - z * w), scale * (x * z + y * w)],
            [scale * (x * y + z * w), 1 - scale * (x * x + z * z), scale * (y * z - x * w)],
            [scale * (x * z - y * w), scale * (y * z + x * w), 1 - scale * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _live(index: int):
    return live_raw_sweep_from_xyz(
        np.array([[float(index), 0.0, 0.0]], dtype=np.float32),
        frame_id="kitti_model_aligned_lidar",
        stamp=TimeStamp(sec=index + 1, nanosec=0),
    )


def test_kitti_replay_preserves_frozen_axis_rows_and_official_timestamp(tmp_path: Path) -> None:
    sequence = _write_sequence(tmp_path)
    acquisition = kitti_ros_replay_acquisition(sequence, 0)
    native = sequence.frame(0).points_xyzi[:, :3]
    expected = native @ KITTI_TO_MODEL_ROTATION.astype(np.float32).T
    np.testing.assert_array_equal(acquisition.points_xyz, expected)
    assert acquisition.points_xyz.dtype == np.float32
    assert acquisition.stamp_components == (1316995200, 123456789)
    assert sequence.timestamps[0].microseconds == 1_316_995_200_123_456


def test_pose_to_ros_tf_keeps_nonidentity_rotation_and_translation(tmp_path: Path) -> None:
    sequence = _write_sequence(tmp_path)
    pose = sequence.lidar_pose(2)
    translation, quaternion = model_lidar_pose_to_world_transform(pose)
    expected_rotation = pose.ego_to_global_rotation @ pose.lidar_to_ego_rotation
    expected_translation = (
        pose.ego_to_global_rotation @ pose.lidar_to_ego_translation + pose.ego_to_global_translation
    )
    assert not np.allclose(expected_rotation, np.eye(3))
    assert not np.allclose(expected_translation, np.zeros(3))
    np.testing.assert_allclose(_quaternion_rotation(quaternion), expected_rotation, atol=1e-15)
    np.testing.assert_allclose(translation, expected_translation, atol=0.0)


@pytest.mark.parametrize("depth", [5, 10])
def test_m6c_h5_and_h10_history_depths_are_exact(depth: int) -> None:
    history = LiveSweepHistory(max_historical_sweeps=depth)
    for index in range(12):
        current = _live(index)
        selection = history.select_for_current(current)
        if index == 11:
            assert len(selection.historical) == depth
            assert [item.stamp.sec for item in selection.historical] == list(
                range(11, 11 - depth, -1)
            )
        history.store_current(current)


def test_m6c_engine_identity_fails_closed_on_mismatch(tmp_path: Path) -> None:
    engine = tmp_path / "candidate.engine"
    engine.write_bytes(b"not the frozen engine")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        require_file_sha256(engine, "0" * 64, artifact_name="M6c structural 40k engine")
    actual = hashlib.sha256(engine.read_bytes()).hexdigest()
    assert require_file_sha256(engine, actual, artifact_name="test engine") == actual


def test_m6c_progress_resumes_only_under_identical_frozen_identity(tmp_path: Path) -> None:
    identity = M6cProgressIdentity("a" * 40, "b" * 40, "c" * 64, "d" * 64)
    path = tmp_path / "progress.json"
    progress = M6cInputProgress(path, identity, ["drive/frame|H10"])
    progress.mark(
        "drive/frame|H10",
        status="PASS",
        expected_sha256="e" * 64,
        observed_sha256="e" * 64,
        point_count=1,
        history_depth=10,
        timestamp_nanoseconds=123,
        elapsed_seconds=0.5,
    )
    resumed = M6cInputProgress(path, identity, ["drive/frame|H10"])
    assert resumed.passed("drive/frame|H10")
    assert resumed.totals() == {"pass": 1, "fail": 0, "pending": 0}
    changed = M6cProgressIdentity("f" * 40, "b" * 40, "c" * 64, "d" * 64)
    with pytest.raises(RuntimeError, match="identity differs"):
        M6cInputProgress(path, changed, ["drive/frame|H10"])


def test_pose_helper_rejects_improper_rotation() -> None:
    calibration = KittiCalibration(np.eye(4), np.eye(4), np.eye(4))
    pose = calibration.model_lidar_pose(np.eye(4))
    object.__setattr__(pose, "ego_to_global_rotation", np.diag([1.0, 1.0, -1.0]))
    with pytest.raises(ValueError, match="determinant"):
        model_lidar_pose_to_world_transform(pose)
