"""Diagnose the failed exact W1 raw-ROS hash without relaxing its acceptance gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from laserperception_ros.raw_replay_node import (
    W1_SAMPLE_TOKEN,
    _acquisitions,
    _stamp,
    _transform,
)
from pyquaternion import Quaternion
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import Buffer

from laserperception.detection.live_multisweep import (
    acquisition_identity,
    live_raw_sweep_from_xyz,
    sweep_transform_from_ros,
)
from laserperception.detection.multisweep import (
    HistoricalSweep,
    LidarPose,
    MultiSweepBuilder,
    RawSweep,
    SweepTransform,
)
from laserperception.detection.ros2_contract import TimeStamp

EXPECTED_HASH = "5c1f5590ce7925d9312fe9ef755d0d849f579ff3d7ae842af75d41a0252cb29a"
OBSERVED_HASH = "50205b0992cc23e8cfde265430a51ae65fbf49cadce7b4f9e3b9f7bc0547f467"
FIXED_FRAME = "nuscenes_map"
EGO_FRAME = "nuscenes_ego"
LIDAR_FRAME = "nuscenes_lidar_top"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _time_stamp(timestamp_microseconds: int) -> TimeStamp:
    seconds, microseconds = divmod(timestamp_microseconds, 1_000_000)
    return TimeStamp(seconds, microseconds * 1_000)


def _raw_points(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size == 0 or values.size % 5 != 0:
        raise RuntimeError("raw sweep is empty or malformed")
    return values.reshape(-1, 5)


def _pose(acquisition: Any) -> LidarPose:
    return LidarPose(
        np.asarray(Quaternion(acquisition.calibration["rotation"]).rotation_matrix, np.float64),
        np.asarray(acquisition.calibration["translation"], np.float64),
        np.asarray(Quaternion(acquisition.ego_pose["rotation"]).rotation_matrix, np.float64),
        np.asarray(acquisition.ego_pose["translation"], np.float64),
    )


def _ordered_float32(value: np.float32) -> int:
    signed = int(np.asarray(value, dtype=np.float32).view(np.int32))
    return 0x80000000 - signed if signed < 0 else signed + 0x80000000


def _retained_provenance(
    current: RawSweep,
    history: list[HistoricalSweep],
) -> list[dict[str, Any]]:
    parts: list[np.ndarray] = []
    identities: list[dict[str, Any]] = []
    current_points = current.points.copy()
    current_points[:, 4] = np.float32(0.0)
    parts.append(current_points)
    identities.extend(
        {
            "acquisition_identity": current.source_id,
            "history_index": None,
            "source_point_index": index,
        }
        for index in range(len(current_points))
    )
    for history_index, item in enumerate(history):
        points = item.sweep.points.copy()
        matrix = np.array(item.transform.lidar2sensor.tolist())
        points[:, :3] = points[:, :3] @ matrix[:3, :3]
        points[:, :3] -= matrix[:3, 3]
        points[:, 4] = current.timestamp_seconds - item.sweep.timestamp_seconds
        parts.append(points)
        identities.extend(
            {
                "acquisition_identity": item.sweep.source_id,
                "history_index": history_index,
                "source_point_index": index,
            }
            for index in range(len(points))
        )
    concatenated = np.concatenate(parts, axis=0)
    mask = (
        (concatenated[:, 0] > -50.0)
        & (concatenated[:, 0] < 50.0)
        & (concatenated[:, 1] > -50.0)
        & (concatenated[:, 1] < 50.0)
        & (concatenated[:, 2] > -5.0)
        & (concatenated[:, 2] < 3.0)
    )
    return [record for record, retained in zip(identities, mask, strict=True) if bool(retained)]


def main() -> None:
    args = _arguments()
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version="v1.0-mini", dataroot=str(args.data_root), verbose=False)
    acquisitions = _acquisitions(nusc, W1_SAMPLE_TOKEN, 10)
    current_acquisition = acquisitions[-1]
    historical_acquisitions = list(reversed(acquisitions[:-1]))
    current_stamp = _time_stamp(current_acquisition.timestamp_microseconds)
    current_xyz = np.ascontiguousarray(_raw_points(current_acquisition.path)[:, :3])
    current_live = live_raw_sweep_from_xyz(
        current_xyz,
        frame_id=LIDAR_FRAME,
        stamp=current_stamp,
    )

    buffer = Buffer(cache_time=Duration(seconds=10.0))
    for acquisition in acquisitions:
        stamp = _stamp(acquisition.timestamp_microseconds)
        buffer.set_transform(
            _transform(
                parent=FIXED_FRAME,
                child=EGO_FRAME,
                stamp=stamp,
                record=acquisition.ego_pose,
            ),
            "m45b_diagnostic",
        )
        buffer.set_transform(
            _transform(
                parent=EGO_FRAME,
                child=LIDAR_FRAME,
                stamp=stamp,
                record=acquisition.calibration,
            ),
            "m45b_diagnostic",
        )

    reference_history: list[HistoricalSweep] = []
    live_history: list[HistoricalSweep] = []
    transform_records: list[dict[str, Any]] = []
    current_pose = _pose(current_acquisition)
    for history_index, acquisition in enumerate(historical_acquisitions):
        stamp = _time_stamp(acquisition.timestamp_microseconds)
        identity = acquisition_identity(LIDAR_FRAME, stamp)
        raw_points = _raw_points(acquisition.path)
        reference_raw = RawSweep(
            raw_points,
            timestamp_microseconds=acquisition.timestamp_microseconds,
            source_id=identity,
        )
        live_raw = live_raw_sweep_from_xyz(
            np.ascontiguousarray(raw_points[:, :3]),
            frame_id=LIDAR_FRAME,
            stamp=stamp,
        ).sweep
        reference_transform = SweepTransform.from_poses(
            source_id=identity,
            target_id=current_live.sweep.source_id,
            sweep_pose=_pose(acquisition),
            current_pose=current_pose,
        )
        returned = buffer.lookup_transform_full(
            LIDAR_FRAME,
            Time(nanoseconds=current_acquisition.timestamp_microseconds * 1_000),
            LIDAR_FRAME,
            Time(nanoseconds=acquisition.timestamp_microseconds * 1_000),
            FIXED_FRAME,
        )
        live_transform = sweep_transform_from_ros(
            translation_xyz=(
                returned.transform.translation.x,
                returned.transform.translation.y,
                returned.transform.translation.z,
            ),
            quaternion_xyzw=(
                returned.transform.rotation.x,
                returned.transform.rotation.y,
                returned.transform.rotation.z,
                returned.transform.rotation.w,
            ),
            source_id=identity,
            target_id=current_live.sweep.source_id,
        )
        reference_history.append(HistoricalSweep(reference_raw, reference_transform))
        live_history.append(HistoricalSweep(live_raw, live_transform))
        differing = np.argwhere(reference_transform.lidar2sensor != live_transform.lidar2sensor)
        first_matrix_difference = None
        if differing.size:
            row, column = (int(value) for value in differing[0])
            reference_value = np.float32(reference_transform.lidar2sensor[row, column])
            live_value = np.float32(live_transform.lidar2sensor[row, column])
            first_matrix_difference = {
                "row": row,
                "column": column,
                "reference_value": float(reference_value),
                "live_value": float(live_value),
                "absolute_difference": float(abs(reference_value - live_value)),
                "ulp_difference": abs(
                    _ordered_float32(reference_value) - _ordered_float32(live_value)
                ),
            }
        transform_records.append(
            {
                "history_index": history_index,
                "sample_data_token": acquisition.sample_data_token,
                "acquisition_identity": identity,
                "source_frame": LIDAR_FRAME,
                "source_stamp": {
                    "sec": stamp.sec,
                    "nanosec": stamp.nanosec,
                    "microseconds": acquisition.timestamp_microseconds,
                },
                "target_frame": LIDAR_FRAME,
                "target_stamp": {
                    "sec": current_stamp.sec,
                    "nanosec": current_stamp.nanosec,
                    "microseconds": current_acquisition.timestamp_microseconds,
                },
                "fixed_frame": FIXED_FRAME,
                "returned_transform": {
                    "translation_xyz": [
                        returned.transform.translation.x,
                        returned.transform.translation.y,
                        returned.transform.translation.z,
                    ],
                    "quaternion_xyzw": [
                        returned.transform.rotation.x,
                        returned.transform.rotation.y,
                        returned.transform.rotation.z,
                        returned.transform.rotation.w,
                    ],
                },
                "encoded_sweep_transform": live_transform.lidar2sensor.tolist(),
                "m45a_reference_sweep_transform": reference_transform.lidar2sensor.tolist(),
                "matrix_exact": bool(
                    np.array_equal(
                        reference_transform.lidar2sensor,
                        live_transform.lidar2sensor,
                    )
                ),
                "first_matrix_difference": first_matrix_difference,
            }
        )

    reference = MultiSweepBuilder().build(current_live.sweep, reference_history)
    live = MultiSweepBuilder().build(current_live.sweep, live_history)
    minimum_rows = min(len(reference.points_xyzt), len(live.points_xyzt))
    differing_rows = np.flatnonzero(
        np.any(reference.points_xyzt[:minimum_rows] != live.points_xyzt[:minimum_rows], axis=1)
    )
    first_row = int(differing_rows[0]) if len(differing_rows) else minimum_rows
    if first_row < minimum_rows:
        differing_fields = np.flatnonzero(
            reference.points_xyzt[first_row] != live.points_xyzt[first_row]
        )
        field_index = int(differing_fields[0])
        reference_value = np.float32(reference.points_xyzt[first_row, field_index])
        live_value = np.float32(live.points_xyzt[first_row, field_index])
        field = ("x", "y", "z", "time_lag")[field_index]
        value_difference: dict[str, Any] = {
            "first_differing_row": first_row,
            "field": field,
            "official_value": float(reference_value),
            "live_ros_value": float(live_value),
            "absolute_difference": float(abs(reference_value - live_value)),
            "ulp_difference": abs(_ordered_float32(reference_value) - _ordered_float32(live_value)),
        }
    else:
        value_difference = {
            "first_differing_row": first_row,
            "field": "row_presence",
            "official_value": None,
            "live_ros_value": None,
            "absolute_difference": None,
            "ulp_difference": None,
        }
    provenance = _retained_provenance(current_live.sweep, reference_history)
    source = provenance[first_row] if first_row < len(provenance) else {}
    history_index = source.get("history_index")
    related_transform = transform_records[int(history_index)] if history_index is not None else None
    any_matrix_difference = any(not record["matrix_exact"] for record in transform_records)
    classification = (
        "F. ROS TransformStamped-to-SweepTransform conversion"
        if any_matrix_difference
        else "I. other"
    )
    diagnostic = {
        "schema_version": 1,
        "status": "failed_exact_gate",
        "sample_index": 42,
        "sample_token": W1_SAMPLE_TOKEN,
        "official_m45a": {
            "shape": list(reference.points_xyzt.shape),
            "sha256": reference.sha256,
            "accepted_sha256": EXPECTED_HASH,
        },
        "m45b_live_ros": {
            "shape": list(live.points_xyzt.shape),
            "sha256": live.sha256,
            "observed_acceptance_run_sha256": OBSERVED_HASH,
        },
        "first_output_difference": {**value_difference, **source},
        "related_transform": related_transform,
        "first_differing_transform": next(
            (record for record in transform_records if not record["matrix_exact"]),
            None,
        ),
        "all_transform_records": transform_records,
        "tf_lookup_api": "Buffer.lookup_transform_full",
        "timestamp_conversion": {
            "policy": "floor integer ROS nanoseconds to integer microseconds",
            "current_ros_nanoseconds": current_acquisition.timestamp_microseconds * 1_000,
            "current_microseconds": current_acquisition.timestamp_microseconds,
            "current_seconds_binary64": current_live.sweep.timestamp_seconds,
            "source_microseconds": (
                historical_acquisitions[int(history_index)].timestamp_microseconds
                if history_index is not None
                else None
            ),
            "source_seconds_binary64": (
                historical_acquisitions[int(history_index)].timestamp_microseconds / 1_000_000
                if history_index is not None
                else None
            ),
        },
        "classification": classification,
        "classification_basis": (
            "raw point counts/order and exact microsecond stamps match; tf2-derived float32 "
            "SweepTransform matrices differ from the accepted M4.5a matrices"
            if any_matrix_difference
            else "no transform matrix mismatch was found"
        ),
        "acceptance_decision": "STOP; do not relax the exact W1 gate",
    }
    if reference.sha256 != EXPECTED_HASH:
        raise RuntimeError("diagnostic M4.5a reconstruction did not reproduce the accepted hash")
    if live.sha256 != OBSERVED_HASH:
        raise RuntimeError("diagnostic tf2 reconstruction did not reproduce the failed live hash")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(diagnostic, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(diagnostic, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
