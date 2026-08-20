"""Generate the post-failure M6a KITTI pose-oracle diagnosis.

This script is diagnostic-only. It never initializes the detector, TensorRT, ROS,
or the reconstruction path, and it never rewrites the original failed artifact.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import subprocess
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from laserperception.datasets.kitti_raw import (  # type: ignore[import-untyped]
    EARTH_RADIUS_METRES,
    KittiCalibration,
    KittiOxtsRecord,
    KittiTimestamp,
    load_kitti_odometry_poses,
    official_oxts_poses,
    oxts_pose_in_rectified_camera,
)

ORIGINAL_FAILURE_RELATIVE = Path("benchmarks/m6a/diagnostics/pose_oracle_failure_ec9e341.json")
ORIGINAL_FAILURE_SHA256 = "894960c34b96aac02db2a8a10f13c66e39e3ac99b3ef34ee76c0ef3e88eabaf3"
MEASUREMENT_IMPLEMENTATION_COMMIT = "ec9e341056807d5549353c8ef362fd109b25f2f2"
CLASSIFICATIONS = ("IMPLEMENTATION", "DATA-PRODUCT / TIMING", "MIXED", "UNKNOWN")

RAW_DEVKIT_LIMITS = {
    "matrix_max_abs": 1e-12,
    "rotation_angle_rad": 1e-10,
    "translation_norm_m": 1e-9,
}
CALIBRATION_LIMITS = {
    "rotation_matrix_max_abs": 1e-9,
    "rotation_angle_rad": 1e-8,
    "translation_norm_m": 1e-6,
}
FRAME_ZERO_LIMITS = {"matrix_max_abs": 1e-12, "translation_norm_m": 1e-12}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp_file(path: Path) -> tuple[int, ...]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError(f"timestamp file contains an empty row: {path.name}")
    return tuple(KittiTimestamp.parse(line).nanoseconds for line in lines)


def _odometry_times_nanoseconds(path: Path) -> tuple[int, ...]:
    values = tuple(Decimal(line.strip()) for line in path.read_text().splitlines())
    return tuple(int(value * Decimal(1_000_000_000)) for value in values)


def _records(path: Path) -> tuple[KittiOxtsRecord, ...]:
    files = tuple(sorted(path.glob("*.txt")))
    if not files:
        raise ValueError(f"no OXTS records found under {path}")
    return tuple(KittiOxtsRecord.parse(file.read_text(encoding="utf-8")) for file in files)


def _keyed_floats(path: Path) -> dict[str, tuple[float, ...]]:
    values: dict[str, tuple[float, ...]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("calib_time"):
            continue
        key, raw = line.split(":", 1)
        values[key] = tuple(float(value) for value in raw.split())
    return values


def _rigid(fields: dict[str, tuple[float, ...]], key: str = "R") -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(fields[key], dtype=np.float64).reshape(3, 3)
    matrix[:3, 3] = np.asarray(fields["T"], dtype=np.float64)
    return matrix


def _odometry_tr(path: Path) -> np.ndarray:
    values = _keyed_floats(path)["Tr"]
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :] = np.asarray(values, dtype=np.float64).reshape(3, 4)
    return matrix


def _independent_raw_calibration(date_root: Path) -> dict[str, np.ndarray]:
    imu_to_velo = _rigid(_keyed_floats(date_root / "calib_imu_to_velo.txt"))
    velo_to_raw_camera = _rigid(_keyed_floats(date_root / "calib_velo_to_cam.txt"))
    camera_fields = _keyed_floats(date_root / "calib_cam_to_cam.txt")
    rectification = np.eye(4, dtype=np.float64)
    rectification[:3, :3] = np.asarray(camera_fields["R_rect_00"], dtype=np.float64).reshape(3, 3)
    return {
        "imu_to_velodyne": imu_to_velo,
        "velodyne_to_raw_camera0": velo_to_raw_camera,
        "raw_to_rectified_camera0": rectification,
        "velodyne_to_rectified_camera0": rectification @ velo_to_raw_camera,
        "imu_to_rectified_camera0": rectification @ velo_to_raw_camera @ imu_to_velo,
    }


def _rotation_x(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )


def _rotation_y(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _direct_devkit_world_poses(
    records: Sequence[KittiOxtsRecord],
    *,
    scale_latitude_degrees: float | None = None,
) -> tuple[np.ndarray, ...]:
    """Directly transcribe official convertOxtsToPose.m before normalization."""

    if not records:
        raise ValueError("at least one OXTS record is required")
    latitude = (
        records[0].latitude_degrees if scale_latitude_degrees is None else scale_latitude_degrees
    )
    scale = math.cos(math.radians(latitude))
    poses: list[np.ndarray] = []
    for record in records:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = np.array(
            [
                scale * math.radians(record.longitude_degrees) * EARTH_RADIUS_METRES,
                scale
                * EARTH_RADIUS_METRES
                * math.log(math.tan(math.pi * (90.0 + record.latitude_degrees) / 360.0)),
                record.altitude_metres,
            ],
            dtype=np.float64,
        )
        pose[:3, :3] = (
            _rotation_z(record.yaw_radians)
            @ _rotation_y(record.pitch_radians)
            @ _rotation_x(record.roll_radians)
        )
        poses.append(pose)
    return tuple(poses)


def _normalize(poses: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
    first_inverse = np.linalg.inv(poses[0])
    return tuple(first_inverse @ pose for pose in poses)


def _to_camera(poses: Sequence[np.ndarray], camera_from_imu: np.ndarray) -> tuple[np.ndarray, ...]:
    imu_from_camera = np.linalg.inv(camera_from_imu)
    return tuple(camera_from_imu @ pose @ imu_from_camera for pose in poses)


def _rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quat = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        elif axis == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quat = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quat = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )
    quat /= np.linalg.norm(quat)
    return quat


def _rotation_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return a stable quaternion geodesic distance between two rotations."""

    if np.array_equal(left, right):
        return 0.0

    left_quaternion = _rotation_to_quaternion(left)
    right_quaternion = _rotation_to_quaternion(right)
    dot = min(1.0, max(-1.0, abs(float(np.dot(left_quaternion, right_quaternion)))))
    return 2.0 * math.acos(dot)


def _quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _slerp(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    left = _rotation_to_quaternion(first)
    right = _rotation_to_quaternion(second)
    dot = float(np.dot(left, right))
    if dot < 0.0:
        right = -right
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        result = left + fraction * (right - left)
        result /= np.linalg.norm(result)
    else:
        angle = math.acos(dot)
        sine = math.sin(angle)
        result = (
            math.sin((1.0 - fraction) * angle) / sine * left
            + math.sin(fraction * angle) / sine * right
        )
    return _quaternion_to_rotation(result)


def _interpolate_world_poses(
    raw_timestamps: Sequence[int],
    raw_world_poses: Sequence[np.ndarray],
    target_timestamps: Sequence[int],
) -> tuple[np.ndarray, ...]:
    interpolated: list[np.ndarray] = []
    for timestamp in target_timestamps:
        after = bisect.bisect_left(raw_timestamps, timestamp)
        if after == 0 or after == len(raw_timestamps):
            raise ValueError("target timestamp lies outside raw OXTS coverage")
        before = after - 1
        span = raw_timestamps[after] - raw_timestamps[before]
        fraction = (timestamp - raw_timestamps[before]) / span
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = (1.0 - fraction) * raw_world_poses[before][
            :3, 3
        ] + fraction * raw_world_poses[after][:3, 3]
        pose[:3, :3] = _slerp(
            raw_world_poses[before][:3, :3], raw_world_poses[after][:3, :3], fraction
        )
        interpolated.append(pose)
    return tuple(interpolated)


def _stats(values: npt.ArrayLike) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "population_std": float(np.std(array)),
    }


def _pearson(left: npt.ArrayLike, right: npt.ArrayLike) -> float | None:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if np.std(left_array) == 0.0 or np.std(right_array) == 0.0:
        return None
    return float(np.corrcoef(left_array, right_array)[0, 1])


def _pose_errors(candidate: Sequence[np.ndarray], oracle: Sequence[np.ndarray]) -> dict[str, Any]:
    if len(candidate) != len(oracle):
        raise ValueError("pose sequences must have equal length")
    signed = np.asarray(
        [left[:3, 3] - right[:3, 3] for left, right in zip(candidate, oracle, strict=True)]
    )
    norms = np.linalg.norm(signed, axis=1)
    rotations = np.asarray(
        [
            _rotation_distance(left[:3, :3], right[:3, :3])
            for left, right in zip(candidate, oracle, strict=True)
        ]
    )
    matrices = np.asarray(
        [
            np.max(np.abs(left[:3, :3] - right[:3, :3]))
            for left, right in zip(candidate, oracle, strict=True)
        ]
    )
    return {
        "frame_count": len(candidate),
        "translation_norm_m": {
            "statistics": _stats(norms),
            "max_frame": int(np.argmax(norms)),
            "values": norms.tolist(),
        },
        "translation_signed_m": {
            axis: {
                "statistics": _stats(signed[:, index]),
                "values": signed[:, index].tolist(),
                "sign_changes": int(np.sum(np.diff(np.signbit(signed[:, index])) != 0)),
            }
            for index, axis in enumerate(("x", "y", "z"))
        },
        "rotation_angle_rad": {
            "statistics": _stats(rotations),
            "max_frame": int(np.argmax(rotations)),
            "values": rotations.tolist(),
        },
        "rotation_matrix_max_abs": {
            "statistics": _stats(matrices),
            "max_frame": int(np.argmax(matrices)),
            "values": matrices.tolist(),
        },
    }


def _relative_errors(
    candidate: Sequence[np.ndarray], oracle: Sequence[np.ndarray], delta: int
) -> dict[str, Any]:
    left_relative = [
        np.linalg.inv(candidate[index - delta]) @ candidate[index]
        for index in range(delta, len(candidate))
    ]
    right_relative = [
        np.linalg.inv(oracle[index - delta]) @ oracle[index] for index in range(delta, len(oracle))
    ]
    result = _pose_errors(left_relative, right_relative)
    result["delta_frames"] = delta
    result["target_frame_indices"] = list(range(delta, len(candidate)))
    return result


def _identity_result(pose: np.ndarray) -> dict[str, Any]:
    identity = np.eye(4, dtype=np.float64)
    matrix = float(np.max(np.abs(pose - identity)))
    translation = float(np.linalg.norm(pose[:3, 3]))
    rotation = _rotation_distance(pose[:3, :3], identity[:3, :3])
    return {
        "matrix": pose.tolist(),
        "matrix_max_abs_from_identity": matrix,
        "rotation_angle_rad_from_identity": rotation,
        "translation_norm_m_from_identity": translation,
        "pass": matrix <= FRAME_ZERO_LIMITS["matrix_max_abs"]
        and translation <= FRAME_ZERO_LIMITS["translation_norm_m"],
    }


def _comparison_gate(
    left: Sequence[np.ndarray], right: Sequence[np.ndarray], limits: dict[str, float]
) -> dict[str, Any]:
    matrix = max(float(np.max(np.abs(a - b))) for a, b in zip(left, right, strict=True))
    rotation = max(
        _rotation_distance(a[:3, :3], b[:3, :3]) for a, b in zip(left, right, strict=True)
    )
    translation = max(
        float(np.linalg.norm(a[:3, 3] - b[:3, 3])) for a, b in zip(left, right, strict=True)
    )
    return {
        "frame_count": len(left),
        "matrix_max_abs": matrix,
        "rotation_angle_rad": rotation,
        "translation_norm_m": translation,
        "limits": limits,
        "pass": matrix <= limits["matrix_max_abs"]
        and rotation <= limits["rotation_angle_rad"]
        and translation <= limits["translation_norm_m"],
    }


def _calibration_comparison(raw: np.ndarray, odometry: np.ndarray) -> dict[str, Any]:
    matrix = float(np.max(np.abs(raw[:3, :3] - odometry[:3, :3])))
    rotation = _rotation_distance(raw[:3, :3], odometry[:3, :3])
    translation = float(np.linalg.norm(raw[:3, 3] - odometry[:3, 3]))
    return {
        "raw_derived_velodyne_to_rectified_camera0": raw.tolist(),
        "odometry_sequence_04_Tr": odometry.tolist(),
        "rotation_matrix_max_abs": matrix,
        "rotation_angle_rad": rotation,
        "translation_norm_m": translation,
        "raw_rotation_determinant": float(np.linalg.det(raw[:3, :3])),
        "odometry_rotation_determinant": float(np.linalg.det(odometry[:3, :3])),
        "raw_orthonormality_max_abs": float(
            np.max(np.abs(raw[:3, :3].T @ raw[:3, :3] - np.eye(3)))
        ),
        "odometry_orthonormality_max_abs": float(
            np.max(np.abs(odometry[:3, :3].T @ odometry[:3, :3] - np.eye(3)))
        ),
        "limits": CALIBRATION_LIMITS,
        "pass": matrix <= CALIBRATION_LIMITS["rotation_matrix_max_abs"]
        and rotation <= CALIBRATION_LIMITS["rotation_angle_rad"]
        and translation <= CALIBRATION_LIMITS["translation_norm_m"],
    }


def _timestamp_ledger(
    drive_root: Path, odometry_root: Path, unsynced_root: Path, mapped_count: int
) -> tuple[dict[str, Any], dict[str, list[int]]]:
    stream_paths = {
        "image_00": drive_root / "image_00/timestamps.txt",
        "image_01": drive_root / "image_01/timestamps.txt",
        "image_02": drive_root / "image_02/timestamps.txt",
        "image_03": drive_root / "image_03/timestamps.txt",
        "oxts_sync": drive_root / "oxts/timestamps.txt",
        "velodyne_selected": drive_root / "velodyne_points/timestamps.txt",
        "velodyne_start": drive_root / "velodyne_points/timestamps_start.txt",
        "velodyne_end": drive_root / "velodyne_points/timestamps_end.txt",
    }
    streams = {name: list(_timestamp_file(path)) for name, path in stream_paths.items()}
    lengths = {name: len(values) for name, values in streams.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"synchronized timestamp counts differ: {lengths}")
    image = streams["image_00"][:mapped_count]
    deltas = {
        f"{name}_minus_image_00_ns": [
            value - image[index] for index, value in enumerate(values[:mapped_count])
        ]
        for name, values in streams.items()
        if name != "image_00"
    }
    odometry_ns = list(
        _odometry_times_nanoseconds(odometry_root / "dataset/sequences/04/times.txt")
    )
    image_elapsed = [value - image[0] for value in image]
    deltas["odometry_times_minus_image_00_elapsed_ns"] = [
        odometry_ns[index] - value for index, value in enumerate(image_elapsed)
    ]

    raw_timestamps = list(_timestamp_file(unsynced_root / "timestamps.txt"))
    raw_lookup = {timestamp: index for index, timestamp in enumerate(raw_timestamps)}
    synced_raw_indices = [raw_lookup[value] for value in streams["oxts_sync"][:mapped_count]]
    previous: list[int] = []
    following: list[int] = []
    nearest: list[int] = []
    for timestamp in image:
        after = bisect.bisect_left(raw_timestamps, timestamp)
        before = after - 1
        if before < 0 or after == len(raw_timestamps):
            raise ValueError("image timestamp lies outside unsynced OXTS coverage")
        previous.append(before)
        following.append(after)
        nearest.append(
            before
            if timestamp - raw_timestamps[before] <= raw_timestamps[after] - timestamp
            else after
        )
    raw_deltas = {
        "previous_to_image_00_ns": [
            image[i] - raw_timestamps[index] for i, index in enumerate(previous)
        ],
        "following_from_image_00_ns": [
            raw_timestamps[index] - image[i] for i, index in enumerate(following)
        ],
        "nearest_minus_image_00_ns": [
            raw_timestamps[index] - image[i] for i, index in enumerate(nearest)
        ],
    }
    summary = {
        "synchronized_stream_counts": lengths,
        "mapped_count": mapped_count,
        "offset_statistics": {name: _stats(values) for name, values in deltas.items()},
        "offset_values_ns": deltas,
        "raw_100hz": {
            "available": True,
            "record_count": len(raw_timestamps),
            "first_timestamp_ns": raw_timestamps[0],
            "last_timestamp_ns": raw_timestamps[-1],
            "mapped_image_coverage": raw_timestamps[0] < image[0]
            and raw_timestamps[-1] > image[-1],
            "synced_oxts_exact_raw_timestamp_matches": len(synced_raw_indices),
            "synced_oxts_matches_nearest_to_image_00": sum(
                left == right for left, right in zip(synced_raw_indices, nearest, strict=True)
            ),
            "first_synced_raw_index": synced_raw_indices[0],
            "last_synced_raw_index": synced_raw_indices[-1],
            "previous_indices": previous,
            "following_indices": following,
            "nearest_indices": nearest,
            "synced_indices": synced_raw_indices,
            "distance_statistics": {name: _stats(values) for name, values in raw_deltas.items()},
            "distance_values_ns": raw_deltas,
        },
    }
    indexes = {
        "previous": previous,
        "nearest": nearest,
        "following": following,
        "synced": synced_raw_indices,
        "raw_timestamps": raw_timestamps,
        "image_timestamps": image,
    }
    return summary, indexes


def _timing_signature(
    records: Sequence[KittiOxtsRecord],
    timestamp_offsets_ns: Sequence[int],
    production_errors: dict[str, Any],
    production_vs_interpolation: dict[str, Any],
) -> dict[str, Any]:
    offset_seconds = np.abs(np.asarray(timestamp_offsets_ns, dtype=np.float64)) / 1e9
    speeds = np.asarray(
        [np.linalg.norm(record.source_values[8:11]) for record in records], dtype=np.float64
    )
    angular_rates = np.asarray(
        [np.linalg.norm(record.source_values[20:23]) for record in records], dtype=np.float64
    )
    predicted_translation = speeds * offset_seconds
    predicted_rotation = angular_rates * offset_seconds
    actual_translation = production_errors["translation_norm_m"]["values"]
    actual_rotation = production_errors["rotation_angle_rad"]["values"]
    interpolation_translation = production_vs_interpolation["translation_norm_m"]["values"]
    interpolation_rotation = production_vs_interpolation["rotation_angle_rad"]["values"]
    return {
        "model": "first-order magnitude only; no correction applied",
        "speed_mps": {"statistics": _stats(speeds), "values": speeds.tolist()},
        "angular_rate_radps": {
            "statistics": _stats(angular_rates),
            "values": angular_rates.tolist(),
        },
        "speed_times_abs_offset_m": {
            "statistics": _stats(predicted_translation),
            "values": predicted_translation.tolist(),
            "pearson_with_observed_translation_error": _pearson(
                predicted_translation, actual_translation
            ),
        },
        "angular_rate_times_abs_offset_rad": {
            "statistics": _stats(predicted_rotation),
            "values": predicted_rotation.tolist(),
            "pearson_with_observed_rotation_error": _pearson(predicted_rotation, actual_rotation),
        },
        "production_vs_interpolated_signature": {
            "translation_error_pearson_with_observed": _pearson(
                interpolation_translation, actual_translation
            ),
            "rotation_error_pearson_with_observed": _pearson(
                interpolation_rotation, actual_rotation
            ),
            "translation_magnitude_difference_m": _stats(
                np.asarray(actual_translation) - np.asarray(interpolation_translation)
            ),
            "rotation_magnitude_difference_rad": _stats(
                np.asarray(actual_rotation) - np.asarray(interpolation_rotation)
            ),
        },
    }


def generate(
    data_root: Path, classification: str, diagnosis_protocol_commit: str
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    original_path = repo_root / ORIGINAL_FAILURE_RELATIVE
    if _sha256(original_path) != ORIGINAL_FAILURE_SHA256:
        raise RuntimeError("original failed artifact hash changed")
    if classification not in CLASSIFICATIONS:
        raise ValueError(f"classification must be one of {CLASSIFICATIONS}")

    extracted = data_root / "extracted"
    date_root = extracted / "2011_09_30"
    drive_root = date_root / "2011_09_30_drive_0016_sync"
    unsynced_root = date_root / "2011_09_30_drive_0016_extract/oxts"
    odometry_root = extracted / "odometry"
    synced_records = _records(drive_root / "oxts/data")
    raw_records = _records(unsynced_root / "data")
    oracle = load_kitti_odometry_poses(odometry_root / "dataset/poses/04.txt")
    mapped_count = len(oracle)
    synced_records = synced_records[:mapped_count]

    production_imu = official_oxts_poses(synced_records)
    production_calibration = KittiCalibration.from_date_root(date_root)
    production_camera = tuple(
        oxts_pose_in_rectified_camera(pose, production_calibration) for pose in production_imu
    )

    independent_calibration = _independent_raw_calibration(date_root)
    direct_world = _direct_devkit_world_poses(synced_records)
    direct_imu = _normalize(direct_world)
    direct_camera = _to_camera(direct_imu, independent_calibration["imu_to_rectified_camera0"])
    raw_devkit_gate = _comparison_gate(production_imu, direct_imu, RAW_DEVKIT_LIMITS)
    composed_gate = _comparison_gate(production_camera, direct_camera, RAW_DEVKIT_LIMITS)

    raw_tr = independent_calibration["velodyne_to_rectified_camera0"]
    odometry_tr = _odometry_tr(odometry_root / "dataset/sequences/04/calib.txt")
    calibration = _calibration_comparison(raw_tr, odometry_tr)

    timestamps, indexes = _timestamp_ledger(drive_root, odometry_root, unsynced_root, mapped_count)
    raw_world = _direct_devkit_world_poses(raw_records)
    previous_imu = _normalize(tuple(raw_world[index] for index in indexes["previous"]))
    nearest_imu = _normalize(tuple(raw_world[index] for index in indexes["nearest"]))
    interpolated_world = _interpolate_world_poses(
        indexes["raw_timestamps"], raw_world, indexes["image_timestamps"]
    )
    interpolated_imu = _normalize(interpolated_world)
    camera_from_imu = independent_calibration["imu_to_rectified_camera0"]
    variants = {
        "synchronized_oxts_production": production_camera,
        "nearest_previous_raw_oxts": _to_camera(previous_imu, camera_from_imu),
        "nearest_raw_oxts": _to_camera(nearest_imu, camera_from_imu),
        "linear_translation_quaternion_slerp_raw_oxts": _to_camera(
            interpolated_imu, camera_from_imu
        ),
    }
    variant_errors = {name: _pose_errors(poses, oracle) for name, poses in variants.items()}
    production_errors = variant_errors["synchronized_oxts_production"]
    relative = {
        str(delta): _relative_errors(production_camera, oracle, delta) for delta in (1, 2, 5, 10)
    }
    production_vs_interpolation = _pose_errors(
        production_camera, variants["linear_translation_quaternion_slerp_raw_oxts"]
    )
    raw_synchronized_count = len(_timestamp_file(drive_root / "image_00/timestamps.txt"))
    timing_signature = _timing_signature(
        synced_records,
        timestamps["offset_values_ns"]["oxts_sync_minus_image_00_ns"],
        production_errors,
        production_vs_interpolation,
    )
    translation_values = production_errors["translation_norm_m"]["values"]
    rotation_values = production_errors["rotation_angle_rad"]["values"]
    distance = [float(np.linalg.norm(pose[:3, 3])) for pose in oracle]
    speed = [float(np.linalg.norm(record.source_values[8:11])) for record in synced_records]
    angular = [float(np.linalg.norm(record.source_values[20:23])) for record in synced_records]

    archives = data_root / "archives"
    archive_paths = {
        "raw_synced_drive": archives / "2011_09_30_drive_0016_sync.zip",
        "raw_date_calibration": archives / "2011_09_30_calib.zip",
        "odometry_poses": archives / "data_odometry_poses.zip",
        "odometry_calibration": archives / "data_odometry_calib.zip",
        "raw_devkit": archives / "devkit_raw_data.zip",
        "odometry_devkit": archives / "devkit_odometry.zip",
    }
    source_hashes = {name: _sha256(path) for name, path in archive_paths.items() if path.exists()}

    run_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    original = json.loads(original_path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "status": "post_failure_diagnosis",
        "canonical": False,
        "designed_after_original_failure": True,
        "original_failure": {
            "path": ORIGINAL_FAILURE_RELATIVE.as_posix(),
            "sha256": ORIGINAL_FAILURE_SHA256,
            "status": original["status"],
            "unchanged_fail": True,
            "frozen_tolerances": {
                name: values["tolerance"]
                for name, values in original["comparison"].items()
                if isinstance(values, dict) and "tolerance" in values
            },
            "observed_maxima": {
                name: values["value"]
                for name, values in original["comparison"].items()
                if isinstance(values, dict) and "value" in values
            },
        },
        "commits": {
            "measurement_implementation": MEASUREMENT_IMPLEMENTATION_COMMIT,
            "diagnosis_protocol": diagnosis_protocol_commit,
            "diagnosis_run": run_commit,
        },
        "source_provenance": {
            "official_raw_page": "https://www.cvlibs.net/datasets/kitti/raw_data.php",
            "official_odometry_page": "https://www.cvlibs.net/datasets/kitti/eval_odometry.php",
            "official_home_changelog": "https://www.cvlibs.net/datasets/kitti/",
            "official_ijrr_paper": "https://www.cvlibs.net/publications/Geiger2013IJRR.pdf",
            "raw_sync_policy": "image_00 reference; closest native 100 Hz OXTS packet",
            "odometry_pose_product": "current post-2013 properly interpolated (subsampled) poses",
            "exact_odometry_interpolation_algorithm_available": False,
            "archive_sha256": source_hashes,
            "unsynced_raw_archive": {
                "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0016/2011_09_30_drive_0016_extract.zip",
                "content_length_bytes": 1_724_245_728,
                "etag": "78d1f40e3341bd7ae246e49d6ed12094-206",
                "retrieval": "HTTP range extraction of OXTS only",
            },
        },
        "sequence_mapping": {
            "odometry_sequence": "04",
            "raw_drive": "2011_09_30_drive_0016",
            "official_raw_frame_range": [0, 270],
            "raw_synchronized_count": raw_synchronized_count,
            "mapped_count": mapped_count,
            "index_offset": 0,
            "unmapped_raw_tail_frames": list(range(mapped_count, raw_synchronized_count)),
        },
        "transform_formula": "C_i = T_Crect_from_I P_i inverse(T_Crect_from_I)",
        "calibration_comparison": calibration,
        "frame_zero": {
            "limits": FRAME_ZERO_LIMITS,
            "candidate": _identity_result(production_camera[0]),
            "odometry": _identity_result(oracle[0]),
            "candidate_vs_odometry": _pose_errors(production_camera[:1], oracle[:1]),
        },
        "official_raw_devkit_oracle": raw_devkit_gate,
        "composed_camera_frame_oracle": composed_gate,
        "absolute_error": production_errors,
        "error_correlations": {
            "translation_norm_vs_frame": _pearson(translation_values, list(range(mapped_count))),
            "translation_norm_vs_odometry_distance": _pearson(translation_values, distance),
            "translation_norm_vs_speed": _pearson(translation_values, speed),
            "rotation_angle_vs_frame": _pearson(rotation_values, list(range(mapped_count))),
            "rotation_angle_vs_angular_rate": _pearson(rotation_values, angular),
        },
        "relative_error": relative,
        "timestamp_ledger": timestamps,
        "timing_variants_vs_odometry": variant_errors,
        "timing_hypothesis": timing_signature,
        "root_cause_classification": classification,
        "protocol_revision_recommendation": {
            "recommended": classification == "DATA-PRODUCT / TIMING",
            "prospective_only": True,
            "correctness_oracle": (
                "official KITTI Raw devkit semantics with strict arithmetic tolerance"
            ),
            "external_trajectory_check": (
                "KITTI Odometry GT reported independently without numerical-equality expectation"
            ),
            "original_tier_a_failure_preserved": True,
        },
        "scope": {
            "original_tolerances_relaxed": False,
            "production_adapter_modified": False,
            "detector_run": False,
            "tensorrt_initialized": False,
            "ros_run": False,
            "m6b_started": False,
            "canonical_m6a_evidence_generated": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classification", choices=CLASSIFICATIONS, default="UNKNOWN")
    parser.add_argument("--diagnosis-protocol-commit", required=True)
    args = parser.parse_args()
    result = generate(args.data_root, args.classification, args.diagnosis_protocol_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"output": str(args.output), "classification": args.classification}, indent=2))


if __name__ == "__main__":
    main()
