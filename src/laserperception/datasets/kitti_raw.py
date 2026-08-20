"""Official KITTI Raw decoding and ROS-independent multi-sweep reconstruction.

The implementation follows the KITTI Raw devkit equations with NumPy only.
Dataset files remain external and no detector or ROS dependency is imported.
"""

from __future__ import annotations

import calendar
import hashlib
import math
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from laserperception.detection.multisweep import (
    HistoricalSweep,
    LidarPose,
    MultiSweepBuilder,
    RawSweep,
    SweepTransform,
)
from laserperception.detection.ros2_contract import ModelReadyPointCloud

KITTI_RAW_FIELDS = 4
KITTI_RAW_RECORD_BYTES = KITTI_RAW_FIELDS * np.dtype("<f4").itemsize
EARTH_RADIUS_METRES = 6_378_137.0
KITTI_TIMESTAMP_PATTERN = re.compile(
    r"^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2}) "
    r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})\."
    r"(?P<nanosecond>\d{9})$"
)

KITTI_TO_MODEL_ROTATION = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
KITTI_TO_MODEL_ROTATION.setflags(write=False)


@dataclass(frozen=True, slots=True)
class KittiTimestamp:
    """One exact official KITTI timestamp and v0.2 clock adaptation."""

    original_text: str
    nanoseconds: int

    def __post_init__(self) -> None:
        text = self.original_text.strip()
        if not text:
            raise ValueError("timestamp text must be non-empty")
        if isinstance(self.nanoseconds, bool) or not isinstance(self.nanoseconds, int):
            raise TypeError("timestamp nanoseconds must be an integer")
        object.__setattr__(self, "original_text", text)

    @classmethod
    def parse(cls, text: str) -> KittiTimestamp:
        """Parse official nanosecond text without floating-point Unix seconds."""

        stripped = text.strip()
        match = KITTI_TIMESTAMP_PATTERN.fullmatch(stripped)
        if match is None:
            raise ValueError("KITTI timestamp must match YYYY-MM-DD HH:MM:SS.nnnnnnnnn")
        fields = {name: int(value) for name, value in match.groupdict().items()}
        if not (1 <= fields["month"] <= 12 and 1 <= fields["day"] <= 31):
            raise ValueError("KITTI timestamp contains an invalid calendar date")
        if not (
            0 <= fields["hour"] <= 23
            and 0 <= fields["minute"] <= 59
            and 0 <= fields["second"] <= 59
        ):
            raise ValueError("KITTI timestamp contains an invalid clock time")
        try:
            seconds = calendar.timegm(
                (
                    fields["year"],
                    fields["month"],
                    fields["day"],
                    fields["hour"],
                    fields["minute"],
                    fields["second"],
                    0,
                    0,
                    0,
                )
            )
        except (OverflowError, ValueError) as exc:
            raise ValueError("KITTI timestamp contains an invalid calendar date") from exc
        expected = (
            fields["year"],
            fields["month"],
            fields["day"],
            fields["hour"],
            fields["minute"],
            fields["second"],
        )
        if tuple(time.gmtime(seconds)[:6]) != expected:
            raise ValueError("KITTI timestamp contains an invalid calendar date")
        return cls(stripped, seconds * 1_000_000_000 + fields["nanosecond"])

    @property
    def microseconds(self) -> int:
        """Return the frozen v0.2 floor-to-microseconds adaptation."""

        return self.nanoseconds // 1_000

    @property
    def discarded_nanoseconds(self) -> int:
        """Return the exact sub-microsecond remainder."""

        return self.nanoseconds % 1_000


@dataclass(frozen=True, slots=True)
class KittiOxtsRecord:
    """The official OXTS fields needed for the KITTI pose equation."""

    latitude_degrees: float
    longitude_degrees: float
    altitude_metres: float
    roll_radians: float
    pitch_radians: float
    yaw_radians: float
    source_values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.source_values) < 6:
            raise ValueError("OXTS record must contain at least six fields")
        if not np.isfinite(np.asarray(self.source_values, dtype=np.float64)).all():
            raise ValueError("OXTS record must contain only finite values")

    @classmethod
    def parse(cls, text: str) -> KittiOxtsRecord:
        """Parse one official space-separated OXTS row."""

        try:
            values = tuple(float(value) for value in text.split())
        except ValueError as exc:
            raise ValueError("OXTS record contains a non-numeric field") from exc
        if len(values) < 6:
            raise ValueError("OXTS record must contain at least six fields")
        return cls(
            latitude_degrees=values[0],
            longitude_degrees=values[1],
            altitude_metres=values[2],
            roll_radians=values[3],
            pitch_radians=values[4],
            yaw_radians=values[5],
            source_values=values,
        )


@dataclass(frozen=True, slots=True)
class KittiCalibration:
    """Validated official KITTI Raw rigid calibration matrices."""

    imu_to_velodyne: np.ndarray
    velodyne_to_camera0: np.ndarray
    camera0_rectification: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "imu_to_velodyne",
            _validated_transform(self.imu_to_velodyne, "imu_to_velodyne"),
        )
        object.__setattr__(
            self,
            "velodyne_to_camera0",
            _validated_transform(self.velodyne_to_camera0, "velodyne_to_camera0"),
        )
        object.__setattr__(
            self,
            "camera0_rectification",
            _validated_transform(self.camera0_rectification, "camera0_rectification"),
        )

    @classmethod
    def from_date_root(cls, date_root: str | Path) -> KittiCalibration:
        """Read the three official date-level calibration files."""

        root = Path(date_root)
        imu_values = _read_keyed_floats(root / "calib_imu_to_velo.txt")
        velo_values = _read_keyed_floats(root / "calib_velo_to_cam.txt")
        camera_values = _read_keyed_floats(root / "calib_cam_to_cam.txt")
        imu_to_velodyne = _transform_from_calibration_fields(imu_values, "R", "T")
        velodyne_to_camera0 = _transform_from_calibration_fields(velo_values, "R", "T")
        if "R_rect_00" not in camera_values or len(camera_values["R_rect_00"]) != 9:
            raise ValueError("calib_cam_to_cam.txt must contain nine R_rect_00 values")
        camera0_rectification = np.eye(4, dtype=np.float64)
        camera0_rectification[:3, :3] = np.asarray(
            camera_values["R_rect_00"], dtype=np.float64
        ).reshape(3, 3)
        return cls(imu_to_velodyne, velodyne_to_camera0, camera0_rectification)

    @property
    def velodyne_to_imu(self) -> np.ndarray:
        """Return the Velodyne-to-IMU transform."""

        return np.linalg.inv(self.imu_to_velodyne)

    @property
    def rectified_camera0_from_imu(self) -> np.ndarray:
        """Return the raw calibration chain used by the odometry oracle."""

        return cast(
            np.ndarray,
            self.camera0_rectification @ self.velodyne_to_camera0 @ self.imu_to_velodyne,
        )

    @property
    def model_to_imu(self) -> np.ndarray:
        """Return virtual model-frame sensor-to-IMU calibration."""

        velodyne_from_model = np.eye(4, dtype=np.float64)
        velodyne_from_model[:3, :3] = KITTI_TO_MODEL_ROTATION.T
        return cast(np.ndarray, self.velodyne_to_imu @ velodyne_from_model)

    def model_lidar_pose(self, ego_to_global: np.ndarray) -> LidarPose:
        """Build the existing lidar pose for the virtual model-axis sensor."""

        pose = _validated_transform(ego_to_global, "ego_to_global")
        model_to_imu = self.model_to_imu
        return LidarPose(
            lidar_to_ego_rotation=model_to_imu[:3, :3].copy(),
            lidar_to_ego_translation=model_to_imu[:3, 3].copy(),
            ego_to_global_rotation=pose[:3, :3].copy(),
            ego_to_global_translation=pose[:3, 3].copy(),
        )


@dataclass(frozen=True, slots=True)
class KittiRawFrame:
    """One exact official KITTI Raw synchronized Velodyne/OXTS frame."""

    index: int
    source_id: str
    timestamp: KittiTimestamp
    points_xyzi: np.ndarray
    oxts: KittiOxtsRecord

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("frame index must be a non-negative integer")
        if not self.source_id.strip():
            raise ValueError("source_id must be non-empty")
        points = np.asarray(self.points_xyzi)
        if points.dtype != np.dtype(np.float32):
            raise TypeError("KITTI source points must have dtype float32")
        if points.ndim != 2 or points.shape[1] != KITTI_RAW_FIELDS:
            raise ValueError("KITTI source points must have shape (N, 4)")
        if points.shape[0] == 0:
            raise ValueError("KITTI source points must be non-empty")
        if not np.isfinite(points).all():
            raise ValueError("KITTI source points must contain only finite values")
        object.__setattr__(self, "source_id", self.source_id.strip())
        object.__setattr__(self, "points_xyzi", np.ascontiguousarray(points).copy())

    @property
    def source_sha256(self) -> str:
        """Hash the exact decoded little-endian float32 record bytes."""

        raw = self.points_xyzi.astype("<f4", copy=False).tobytes(order="C")
        return hashlib.sha256(raw).hexdigest()

    def to_raw_sweep(self) -> RawSweep:
        """Rotate native XYZR into the frozen five-column model-axis contract."""

        points = np.zeros((self.points_xyzi.shape[0], 5), dtype=np.float32)
        alignment = KITTI_TO_MODEL_ROTATION.astype(np.float32)
        points[:, :3] = self.points_xyzi[:, :3] @ alignment.T
        points[:, 3] = self.points_xyzi[:, 3]
        return RawSweep(points, self.timestamp.microseconds, self.source_id)


@dataclass(frozen=True, slots=True)
class KittiReconstructionResult:
    """One production KITTI reconstruction and its selected source contract."""

    current_index: int
    selected_indices: tuple[int, ...]
    source_counts: tuple[int, ...]
    point_cloud: ModelReadyPointCloud

    @property
    def pre_builder_row_count(self) -> int:
        """Return rows presented to the existing builder before its range crop."""

        return sum(self.source_counts)


class KittiRawSequence:
    """Deterministic chronological access to an external KITTI Raw sync drive."""

    def __init__(self, date_root: str | Path, drive_root: str | Path) -> None:
        self.date_root = Path(date_root)
        self.drive_root = Path(drive_root)
        self.calibration = KittiCalibration.from_date_root(self.date_root)
        self._point_files = tuple(sorted((self.drive_root / "velodyne_points/data").glob("*.bin")))
        self._oxts_files = tuple(sorted((self.drive_root / "oxts/data").glob("*.txt")))
        self._timestamps = _read_timestamp_file(self.drive_root / "velodyne_points/timestamps.txt")
        if not self._point_files:
            raise ValueError("KITTI Raw sequence contains no Velodyne .bin files")
        frame_count = len(self._point_files)
        if len(self._oxts_files) != frame_count or len(self._timestamps) != frame_count:
            raise ValueError("KITTI Raw Velodyne, OXTS, and selected timestamp counts must match")
        expected_stems = tuple(f"{index:010d}" for index in range(frame_count))
        if tuple(path.stem for path in self._point_files) != expected_stems:
            raise ValueError("KITTI Raw Velodyne frames must be contiguous from index zero")
        if tuple(path.stem for path in self._oxts_files) != expected_stems:
            raise ValueError("KITTI Raw OXTS frames must be contiguous from index zero")
        self._oxts_records = tuple(
            KittiOxtsRecord.parse(path.read_text(encoding="utf-8")) for path in self._oxts_files
        )
        self._ego_to_global = official_oxts_poses(self._oxts_records)

    def __len__(self) -> int:
        return len(self._point_files)

    @property
    def timestamps(self) -> tuple[KittiTimestamp, ...]:
        """Return exact selected acquisition timestamps in chronological order."""

        return self._timestamps

    @property
    def ego_to_global_poses(self) -> tuple[np.ndarray, ...]:
        """Return copies of official-equation first-frame-normalized OXTS poses."""

        return tuple(pose.copy() for pose in self._ego_to_global)

    def frame(self, index: int) -> KittiRawFrame:
        """Decode one source frame without sorting, filtering, or normalization."""

        self._validate_index(index)
        return KittiRawFrame(
            index=index,
            source_id=f"{self.drive_root.name}/{index:010d}",
            timestamp=self._timestamps[index],
            points_xyzi=read_kitti_raw_velodyne(self._point_files[index]),
            oxts=self._oxts_records[index],
        )

    def lidar_pose(self, index: int) -> LidarPose:
        """Return virtual model-axis lidar calibration and official OXTS pose."""

        self._validate_index(index)
        return self.calibration.model_lidar_pose(self._ego_to_global[index])

    def reconstruct(
        self,
        current_index: int,
        *,
        builder: MultiSweepBuilder | None = None,
    ) -> KittiReconstructionResult:
        """Use the unchanged builder with current and up to ten previous frames."""

        self._validate_index(current_index)
        current_frame = self.frame(current_index)
        current = current_frame.to_raw_sweep()
        current_pose = self.lidar_pose(current_index)
        historical: list[HistoricalSweep] = []
        source_counts = [current_frame.points_xyzi.shape[0]]
        selected_indices = [current_index]
        for index in range(current_index - 1, max(-1, current_index - 11), -1):
            frame = self.frame(index)
            sweep = frame.to_raw_sweep()
            transform = SweepTransform.from_poses(
                source_id=sweep.source_id,
                target_id=current.source_id,
                sweep_pose=self.lidar_pose(index),
                current_pose=current_pose,
            )
            historical.append(HistoricalSweep(sweep, transform))
            selected_indices.append(index)
            source_counts.append(frame.points_xyzi.shape[0])
        point_cloud = (builder or MultiSweepBuilder()).build(current, historical)
        return KittiReconstructionResult(
            current_index=current_index,
            selected_indices=tuple(selected_indices),
            source_counts=tuple(source_counts),
            point_cloud=point_cloud,
        )

    def _validate_index(self, index: int) -> None:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("frame index must be an integer")
        if not 0 <= index < len(self):
            raise IndexError(f"frame index {index} is outside [0, {len(self)})")


def read_kitti_raw_velodyne(path: str | Path) -> np.ndarray:
    """Decode exact official little-endian float32 XYZR records."""

    source = Path(path)
    byte_count = source.stat().st_size
    if byte_count == 0:
        raise ValueError("KITTI Raw Velodyne file must be non-empty")
    if byte_count % KITTI_RAW_RECORD_BYTES != 0:
        raise ValueError("KITTI Raw Velodyne file size must be divisible by 16 bytes")
    points = np.fromfile(source, dtype="<f4").reshape(-1, KITTI_RAW_FIELDS)
    if not np.isfinite(points).all():
        raise ValueError("KITTI Raw Velodyne file contains NaN or infinite values")
    return cast(np.ndarray, np.ascontiguousarray(points, dtype=np.float32))


def official_oxts_poses(records: Sequence[KittiOxtsRecord]) -> tuple[np.ndarray, ...]:
    """Transcribe the official KITTI Raw Mercator and rotation path."""

    if not records:
        raise ValueError("at least one OXTS record is required")
    scale = math.cos(math.radians(records[0].latitude_degrees))
    unnormalized: list[np.ndarray] = []
    for record in records:
        translation = np.array(
            [
                scale * math.radians(record.longitude_degrees) * EARTH_RADIUS_METRES,
                scale
                * EARTH_RADIUS_METRES
                * math.log(math.tan(math.pi * (90.0 + record.latitude_degrees) / 360.0)),
                record.altitude_metres,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(translation).all():
            raise ValueError("OXTS Mercator conversion produced a non-finite value")
        rotation = (
            _rotation_z(record.yaw_radians)
            @ _rotation_y(record.pitch_radians)
            @ _rotation_x(record.roll_radians)
        )
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation
        pose[:3, 3] = translation
        unnormalized.append(pose)
    first_inverse = np.linalg.inv(unnormalized[0])
    return tuple(
        _validated_transform(first_inverse @ pose, "normalized OXTS pose") for pose in unnormalized
    )


def load_kitti_odometry_poses(path: str | Path) -> tuple[np.ndarray, ...]:
    """Load official odometry camera poses from one 12-value-per-row text file."""

    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    if not lines or any(not line for line in lines):
        raise ValueError("KITTI odometry pose file must contain non-empty rows")
    poses: list[np.ndarray] = []
    for line in lines:
        try:
            values = np.asarray([float(value) for value in line.split()], dtype=np.float64)
        except ValueError as exc:
            raise ValueError("KITTI odometry pose row contains a non-numeric value") from exc
        if values.shape != (12,) or not np.isfinite(values).all():
            raise ValueError("KITTI odometry pose row must contain 12 finite values")
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :] = values.reshape(3, 4)
        poses.append(_validated_transform(pose, "odometry pose"))
    return tuple(poses)


def oxts_pose_in_rectified_camera(
    ego_to_global: np.ndarray,
    calibration: KittiCalibration,
) -> np.ndarray:
    """Express a first-IMU-normalized OXTS pose in rectified camera-0 axes."""

    pose = _validated_transform(ego_to_global, "ego_to_global")
    camera_from_imu = calibration.rectified_camera0_from_imu
    return cast(np.ndarray, camera_from_imu @ pose @ np.linalg.inv(camera_from_imu))


def rotation_angle_radians(rotation: np.ndarray) -> float:
    """Return the stable SO(3) angle of a finite 3x3 matrix."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    return math.acos(cosine)


def select_m6a_reconstruction_frames(poses: Sequence[np.ndarray]) -> tuple[int, ...]:
    """Apply the preregistered structural/systematic/motion selection algorithm."""

    if len(poses) < 24:
        raise ValueError("M6a frame selection requires at least 24 poses")
    validated = tuple(_validated_transform(pose, "selection pose") for pose in poses)
    structural = {0, 1, 2, 5, 10}
    final_index = len(validated) - 1
    systematic = {11 + (rank * (final_index - 11)) // 15 for rank in range(16)}
    selected = structural | systematic
    scored: list[tuple[float, int]] = []
    for index in range(11, len(validated)):
        relative = np.linalg.inv(validated[index - 1]) @ validated[index]
        score = float(np.linalg.norm(relative[:3, 3])) + rotation_angle_radians(relative[:3, :3])
        scored.append((score, index))
    ranked = sorted(scored, key=lambda item: (item[0], item[1]))

    def add_first_unused(order: Sequence[tuple[float, int]]) -> None:
        for _, index in order:
            if index not in selected:
                selected.add(index)
                return
        raise ValueError("M6a frame selection could not find an unused motion frame")

    add_first_unused(ranked)
    median_rank = (len(ranked) - 1) / 2.0
    median_order = sorted(
        enumerate(ranked),
        key=lambda item: (abs(item[0] - median_rank), item[1][1]),
    )
    add_first_unused([item for _, item in median_order])
    add_first_unused(tuple(reversed(ranked)))
    if len(selected) != 24:
        raise AssertionError(f"preregistered selection produced {len(selected)} frames, not 24")
    return tuple(sorted(selected))


def _read_timestamp_file(path: Path) -> tuple[KittiTimestamp, ...]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError(f"KITTI timestamp file must contain non-empty rows: {path.name}")
    timestamps = tuple(KittiTimestamp.parse(line) for line in lines)
    if any(
        current.nanoseconds <= previous.nanoseconds
        for previous, current in zip(timestamps, timestamps[1:], strict=False)
    ):
        raise ValueError("KITTI selected timestamps must be strictly increasing")
    return timestamps


def _read_keyed_floats(path: Path) -> dict[str, tuple[float, ...]]:
    values: dict[str, tuple[float, ...]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"calibration row lacks a key separator: {path.name}")
        key, raw_values = line.split(":", 1)
        if key == "calib_time":
            continue
        try:
            parsed = tuple(float(value) for value in raw_values.split())
        except ValueError as exc:
            raise ValueError(f"calibration row contains non-numeric data: {key}") from exc
        if parsed and not np.isfinite(np.asarray(parsed, dtype=np.float64)).all():
            raise ValueError(f"calibration row contains a non-finite value: {key}")
        values[key] = parsed
    return values


def _transform_from_calibration_fields(
    fields: dict[str, tuple[float, ...]],
    rotation_key: str,
    translation_key: str,
) -> np.ndarray:
    if rotation_key not in fields or len(fields[rotation_key]) != 9:
        raise ValueError(f"calibration must contain nine {rotation_key} values")
    if translation_key not in fields or len(fields[translation_key]) != 3:
        raise ValueError(f"calibration must contain three {translation_key} values")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(fields[rotation_key], dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(fields[translation_key], dtype=np.float64)
    return transform


def _validated_transform(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value)
    if matrix.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must have dtype float64")
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4)")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    if not np.array_equal(matrix[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)):
        raise ValueError(f"{name} must have homogeneous final row [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
        raise ValueError(f"{name} rotation determinant must be +1")
    copied = np.ascontiguousarray(matrix).copy()
    copied.setflags(write=False)
    return cast(np.ndarray, copied)


def _rotation_x(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )


def _rotation_y(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def _rotation_z(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
