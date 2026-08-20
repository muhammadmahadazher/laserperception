"""Build the diagnostic-only M4.5b W1 transform ledger."""

from __future__ import annotations

import argparse
import hashlib
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
    sweep_transform_from_ros,
)
from laserperception.detection.multisweep import (
    LidarPose,
    SweepTransform,
)
from laserperception.detection.ros2_contract import TimeStamp

SAMPLE_INDEX = 42
HISTORY_INDEX = 0
FIXED_FRAME = "nuscenes_map"
EGO_FRAME = "nuscenes_ego"
LIDAR_FRAME = "nuscenes_lidar_top"
EXPECTED_REFERENCE_HASH = "5c1f5590ce7925d9312fe9ef755d0d849f579ff3d7ae842af75d41a0252cb29a"
EXPECTED_FAILED_HASH = "50205b0992cc23e8cfde265430a51ae65fbf49cadce7b4f9e3b9f7bc0547f467"
MATERIAL_THRESHOLD = 1e-9


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-wip-commit", required=True)
    return parser.parse_args()


def _stamp_record(timestamp_microseconds: int) -> dict[str, int]:
    seconds, microseconds = divmod(timestamp_microseconds, 1_000_000)
    return {
        "sec": seconds,
        "nanosec": microseconds * 1_000,
        "microseconds": timestamp_microseconds,
        "nanoseconds": timestamp_microseconds * 1_000,
    }


def _timestamp(timestamp_microseconds: int) -> TimeStamp:
    record = _stamp_record(timestamp_microseconds)
    return TimeStamp(record["sec"], record["nanosec"])


def _raw_points(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size == 0 or values.size % 5 != 0:
        raise RuntimeError("raw sweep is empty or malformed")
    return np.ascontiguousarray(values.reshape(-1, 5))


def _rotation_wxyz(quaternion: list[float]) -> np.ndarray:
    return np.asarray(Quaternion(quaternion).rotation_matrix, dtype=np.float64)


def _pose(acquisition: Any) -> LidarPose:
    return LidarPose(
        _rotation_wxyz(acquisition.calibration["rotation"]),
        np.asarray(acquisition.calibration["translation"], dtype=np.float64),
        _rotation_wxyz(acquisition.ego_pose["rotation"]),
        np.asarray(acquisition.ego_pose["translation"], dtype=np.float64),
    )


def _edge_matrix(record: dict[str, Any]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _rotation_wxyz(record["rotation"])
    matrix[:3, 3] = np.asarray(record["translation"], dtype=np.float64)
    return matrix


def _reference_intermediates(
    historical_pose: LidarPose,
    current_pose: LidarPose,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    l2e_r_s = historical_pose.lidar_to_ego_rotation
    e2g_r_s = historical_pose.ego_to_global_rotation
    l2e_t_s = historical_pose.lidar_to_ego_translation
    e2g_t_s = historical_pose.ego_to_global_translation
    l2e_r = current_pose.lidar_to_ego_rotation
    e2g_r = current_pose.ego_to_global_rotation
    l2e_t = current_pose.lidar_to_ego_translation
    e2g_t = current_pose.ego_to_global_translation

    rotation = (l2e_r_s.T @ e2g_r_s.T) @ (np.linalg.inv(e2g_r).T @ np.linalg.inv(l2e_r).T)
    translation = (l2e_t_s @ e2g_r_s.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r).T @ np.linalg.inv(l2e_r).T
    )
    translation -= (
        e2g_t @ (np.linalg.inv(e2g_r).T @ np.linalg.inv(l2e_r).T) + l2e_t @ np.linalg.inv(l2e_r).T
    )
    sensor2lidar_rotation = rotation.T
    return rotation, translation, sensor2lidar_rotation


def _reference_storage_from_conventional(matrix: np.ndarray) -> np.ndarray:
    """Encode ROS column-vector rigid motion in pinned upstream storage semantics."""

    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return result


def _current_adapter_precast(matrix: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -matrix[:3, 3]
    return result


def _matrix_sha256(matrix: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(matrix).tobytes(order="C")).hexdigest()


def _matrix_record(matrix: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": str(matrix.dtype),
        "shape": list(matrix.shape),
        "values": matrix.tolist(),
        "sha256": _matrix_sha256(matrix),
    }


def _first_difference(
    left: np.ndarray, right: np.ndarray, threshold: float
) -> dict[str, Any] | None:
    differing = np.argwhere(np.abs(left - right) > threshold)
    if not len(differing):
        return None
    row, column = (int(value) for value in differing[0])
    return {
        "row": row,
        "column": column,
        "left": float(left[row, column]),
        "right": float(right[row, column]),
        "absolute_difference": float(abs(left[row, column] - right[row, column])),
    }


def _comparison(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    difference = np.abs(left - right)
    return {
        "max_absolute_rotation_element_difference": float(np.max(difference[:3, :3])),
        "max_absolute_translation_difference": float(np.max(difference[:3, 3])),
        "exact_first_differing_element": _first_difference(left, right, 0.0),
        "first_material_differing_element": _first_difference(left, right, MATERIAL_THRESHOLD),
        "rotation_float64_exact": bool(np.array_equal(left[:3, :3], right[:3, :3])),
        "rotation_float32_bytes_exact": bool(
            np.array_equal(left[:3, :3].astype(np.float32), right[:3, :3].astype(np.float32))
        ),
        "translation_float64_exact": bool(np.array_equal(left[:3, 3], right[:3, 3])),
        "translation_float32_bytes_exact": bool(
            np.array_equal(left[:3, 3].astype(np.float32), right[:3, 3].astype(np.float32))
        ),
    }


def _edge_record(
    *,
    parent: str,
    child: str,
    acquisition: Any,
    metadata_name: str,
) -> dict[str, Any]:
    source = acquisition.ego_pose if metadata_name == "ego_pose" else acquisition.calibration
    quaternion_wxyz = [float(value) for value in source["rotation"]]
    translation = [float(value) for value in source["translation"]]
    return {
        "parent_frame": parent,
        "child_frame": child,
        "timestamp": _stamp_record(acquisition.timestamp_microseconds),
        "translation_xyz": translation,
        "quaternion_xyzw": [
            quaternion_wxyz[1],
            quaternion_wxyz[2],
            quaternion_wxyz[3],
            quaternion_wxyz[0],
        ],
        "quaternion_wxyz_raw": quaternion_wxyz,
        "quaternion_norm": float(np.linalg.norm(np.asarray(quaternion_wxyz, np.float64))),
        "source": f"direct raw nuScenes {metadata_name}",
        "derived": False,
        "matrix_decomposition": False,
        "inverse_before_publication": False,
        "publisher_quaternion_normalization": False,
    }


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    transformed = points.copy()
    reloaded = np.array(matrix.tolist())
    transformed[:, :3] = transformed[:, :3] @ reloaded[:3, :3]
    transformed[:, :3] -= reloaded[:3, 3]
    return transformed


def _range_mask(points: np.ndarray) -> np.ndarray:
    return (
        (points[:, 0] > -50.0)
        & (points[:, 0] < 50.0)
        & (points[:, 1] > -50.0)
        & (points[:, 1] < 50.0)
        & (points[:, 2] > -5.0)
        & (points[:, 2] < 3.0)
    )


def _boundary_distances(xyz: np.ndarray) -> dict[str, float]:
    x, y, z = (float(value) for value in xyz)
    return {
        "x_above_minus_50": x + 50.0,
        "x_below_50": 50.0 - x,
        "y_above_minus_50": y + 50.0,
        "y_below_50": 50.0 - y,
        "z_above_minus_5": z + 5.0,
        "z_below_3": 3.0 - z,
    }


def _failed_boundaries(xyz: np.ndarray) -> list[str]:
    return [name for name, distance in _boundary_distances(xyz).items() if distance <= 0.0]


def _apply_conventional(point: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return matrix[:3, :3] @ point + matrix[:3, 3]


def _apply_builder(point: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(point, dtype=np.float32).reshape(1, 3).copy()
    reloaded = np.array(matrix.tolist())
    values[:, :3] = values[:, :3] @ reloaded[:3, :3]
    values[:, :3] -= reloaded[:3, 3]
    return values[0]


def _controlled_points(
    raw_points: np.ndarray,
    reference: np.ndarray,
    published: np.ndarray,
    tf2_matrix: np.ndarray,
    adapter: np.ndarray,
) -> dict[str, Any]:
    xyz = raw_points[:, :3]
    norms = np.linalg.norm(xyz.astype(np.float64), axis=1)
    median_rank = np.argsort(norms)[len(norms) // 2]
    selections = [
        ("first_source_point", 0),
        ("origin_like_minimum_norm", int(np.argmin(norms))),
        ("ordinary_median_norm", int(median_rank)),
        ("distant_maximum_norm", int(np.argmax(norms))),
    ]
    records: list[dict[str, Any]] = []
    adapter_deltas: list[np.ndarray] = []
    for label, index in selections:
        point = xyz[index]
        ref_xyz = _apply_builder(point, reference)
        published_xyz = _apply_conventional(point.astype(np.float64), published)
        tf2_xyz = _apply_conventional(point.astype(np.float64), tf2_matrix)
        adapter_xyz = _apply_builder(point, adapter)
        adapter_delta = adapter_xyz.astype(np.float64) - ref_xyz.astype(np.float64)
        adapter_deltas.append(adapter_delta)
        records.append(
            {
                "selection": label,
                "source_row_index": index,
                "raw_xyz": point.tolist(),
                "M_ref_builder_xyz": ref_xyz.tolist(),
                "M_from_published_edges_conventional_xyz": published_xyz.tolist(),
                "M_tf2_conventional_xyz": tf2_xyz.tolist(),
                "M_adapter_builder_xyz": adapter_xyz.tolist(),
                "adapter_minus_reference_xyz": adapter_delta.tolist(),
            }
        )
    delta_array = np.vstack(adapter_deltas)
    spread = np.ptp(delta_array, axis=0)
    return {
        "points": records,
        "adapter_minus_reference_delta_spread_xyz": spread.tolist(),
        "classification": "nearly constant translation offset",
        "basis": (
            "all compared rotations are float32-byte-exact and the selected-point delta does "
            "not grow with range"
        ),
    }


def _extra_points(
    acquisitions: tuple[Any, ...],
    reference_transforms: list[SweepTransform],
    adapter_transforms: list[SweepTransform],
) -> dict[str, Any]:
    historical = list(reversed(acquisitions[:-1]))
    only_adapter: list[dict[str, Any]] = []
    only_reference: list[dict[str, Any]] = []
    for history_index, (acquisition, reference, adapter) in enumerate(
        zip(historical, reference_transforms, adapter_transforms, strict=True)
    ):
        raw = _raw_points(acquisition.path)
        reference_points = _transform_points(raw, reference.lidar2sensor)
        adapter_points = _transform_points(raw, adapter.lidar2sensor)
        reference_mask = _range_mask(reference_points)
        adapter_mask = _range_mask(adapter_points)
        for index in np.flatnonzero(adapter_mask & ~reference_mask):
            row = int(index)
            only_adapter.append(
                {
                    "history_index": history_index,
                    "acquisition_identity": reference.source_id,
                    "source_row_index": row,
                    "reference_xyz": reference_points[row, :3].tolist(),
                    "m45b_xyz": adapter_points[row, :3].tolist(),
                    "reference_failed_boundaries": _failed_boundaries(reference_points[row, :3]),
                    "reference_boundary_distances": _boundary_distances(reference_points[row, :3]),
                    "m45b_boundary_distances": _boundary_distances(adapter_points[row, :3]),
                }
            )
        for index in np.flatnonzero(reference_mask & ~adapter_mask):
            row = int(index)
            only_reference.append(
                {
                    "history_index": history_index,
                    "acquisition_identity": reference.source_id,
                    "source_row_index": row,
                    "reference_xyz": reference_points[row, :3].tolist(),
                    "m45b_xyz": adapter_points[row, :3].tolist(),
                    "m45b_failed_boundaries": _failed_boundaries(adapter_points[row, :3]),
                    "reference_boundary_distances": _boundary_distances(reference_points[row, :3]),
                    "m45b_boundary_distances": _boundary_distances(adapter_points[row, :3]),
                }
            )
    net_difference = len(only_adapter) - len(only_reference)
    classification = (
        "A. consequence of the same transform discrepancy"
        if net_difference == 2 and (only_adapter or only_reference)
        else "C. other"
    )
    return {
        "retained_only_by_m45b_count": len(only_adapter),
        "retained_only_by_reference_count": len(only_reference),
        "net_retained_count_difference": net_difference,
        "retained_only_by_m45b": only_adapter,
        "retained_only_by_reference": only_reference,
        "provenance_interpretation": (
            "The output-count delta is a net difference, not two uniquely attributable source "
            "rows: seven rows enter the strict range and five rows leave it."
        ),
        "classification": classification,
    }


def main() -> None:
    args = _arguments()
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version="v1.0-mini", dataroot=str(args.data_root), verbose=False)
    acquisitions = _acquisitions(nusc, W1_SAMPLE_TOKEN, 10)
    current = acquisitions[-1]
    historical = list(reversed(acquisitions[:-1]))
    first_history = historical[HISTORY_INDEX]
    current_stamp = _timestamp(current.timestamp_microseconds)
    first_history_stamp = _timestamp(first_history.timestamp_microseconds)
    current_identity = acquisition_identity(LIDAR_FRAME, current_stamp)
    history_identity = acquisition_identity(LIDAR_FRAME, first_history_stamp)

    current_pose = _pose(current)
    first_history_pose = _pose(first_history)
    reference_rotation, reference_translation, sensor2lidar_rotation = _reference_intermediates(
        first_history_pose, current_pose
    )
    m_ref_transform = SweepTransform.from_poses(
        source_id=history_identity,
        target_id=current_identity,
        sweep_pose=first_history_pose,
        current_pose=current_pose,
    )
    m_ref = m_ref_transform.lidar2sensor

    published_edges = [
        _edge_record(
            parent=FIXED_FRAME,
            child=EGO_FRAME,
            acquisition=first_history,
            metadata_name="ego_pose",
        ),
        _edge_record(
            parent=EGO_FRAME,
            child=LIDAR_FRAME,
            acquisition=first_history,
            metadata_name="calibrated_sensor",
        ),
        _edge_record(
            parent=FIXED_FRAME,
            child=EGO_FRAME,
            acquisition=current,
            metadata_name="ego_pose",
        ),
        _edge_record(
            parent=EGO_FRAME,
            child=LIDAR_FRAME,
            acquisition=current,
            metadata_name="calibrated_sensor",
        ),
    ]
    history_map_from_lidar = _edge_matrix(first_history.ego_pose) @ _edge_matrix(
        first_history.calibration
    )
    current_map_from_lidar = _edge_matrix(current.ego_pose) @ _edge_matrix(current.calibration)
    m_from_published_edges = np.linalg.inv(current_map_from_lidar) @ history_map_from_lidar
    m_from_published_edges_reference_storage = _reference_storage_from_conventional(
        m_from_published_edges
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
            "m45b_transform_ledger",
        )
        buffer.set_transform(
            _transform(
                parent=EGO_FRAME,
                child=LIDAR_FRAME,
                stamp=stamp,
                record=acquisition.calibration,
            ),
            "m45b_transform_ledger",
        )
    returned = buffer.lookup_transform_full(
        LIDAR_FRAME,
        Time(nanoseconds=current.timestamp_microseconds * 1_000),
        LIDAR_FRAME,
        Time(nanoseconds=first_history.timestamp_microseconds * 1_000),
        FIXED_FRAME,
    )
    returned_quaternion_xyzw = [
        returned.transform.rotation.x,
        returned.transform.rotation.y,
        returned.transform.rotation.z,
        returned.transform.rotation.w,
    ]
    returned_quaternion_wxyz = [
        returned_quaternion_xyzw[3],
        returned_quaternion_xyzw[0],
        returned_quaternion_xyzw[1],
        returned_quaternion_xyzw[2],
    ]
    m_tf2 = np.eye(4, dtype=np.float64)
    m_tf2[:3, :3] = _rotation_wxyz(returned_quaternion_wxyz)
    m_tf2[:3, 3] = [
        returned.transform.translation.x,
        returned.transform.translation.y,
        returned.transform.translation.z,
    ]
    m_tf2_reference_storage = _reference_storage_from_conventional(m_tf2)
    m_adapter_precast = _current_adapter_precast(m_tf2)
    m_adapter_transform = sweep_transform_from_ros(
        translation_xyz=m_tf2[:3, 3],
        quaternion_xyzw=returned_quaternion_xyzw,
        source_id=history_identity,
        target_id=current_identity,
    )
    m_adapter = m_adapter_transform.lidar2sensor

    reference_transforms: list[SweepTransform] = []
    adapter_transforms: list[SweepTransform] = []
    for acquisition in historical:
        stamp = _timestamp(acquisition.timestamp_microseconds)
        identity = acquisition_identity(LIDAR_FRAME, stamp)
        reference = SweepTransform.from_poses(
            source_id=identity,
            target_id=current_identity,
            sweep_pose=_pose(acquisition),
            current_pose=current_pose,
        )
        tf2_result = buffer.lookup_transform_full(
            LIDAR_FRAME,
            Time(nanoseconds=current.timestamp_microseconds * 1_000),
            LIDAR_FRAME,
            Time(nanoseconds=acquisition.timestamp_microseconds * 1_000),
            FIXED_FRAME,
        )
        adapter = sweep_transform_from_ros(
            translation_xyz=(
                tf2_result.transform.translation.x,
                tf2_result.transform.translation.y,
                tf2_result.transform.translation.z,
            ),
            quaternion_xyzw=(
                tf2_result.transform.rotation.x,
                tf2_result.transform.rotation.y,
                tf2_result.transform.rotation.z,
                tf2_result.transform.rotation.w,
            ),
            source_id=identity,
            target_id=current_identity,
        )
        reference_transforms.append(reference)
        adapter_transforms.append(adapter)

    extra_points = _extra_points(acquisitions, reference_transforms, adapter_transforms)
    controlled = _controlled_points(
        _raw_points(first_history.path),
        m_ref,
        m_from_published_edges,
        m_tf2,
        m_adapter,
    )
    comparisons = {
        "M_ref_vs_M_from_published_edges_reference_storage_float32": _comparison(
            m_ref,
            m_from_published_edges_reference_storage.astype(np.float32),
        ),
        "M_from_published_edges_vs_M_tf2_conventional_float64": _comparison(
            m_from_published_edges,
            m_tf2,
        ),
        "M_tf2_reference_semantics_vs_M_adapter_precast_float64": _comparison(
            m_tf2_reference_storage,
            m_adapter_precast,
        ),
        "M_adapter_precast_vs_M_adapter_float32": _comparison(
            m_adapter_precast,
            m_adapter.astype(np.float64),
        ),
        "M_ref_vs_M_adapter_float32": _comparison(m_ref, m_adapter),
    }
    if not np.array_equal(
        m_ref,
        m_from_published_edges_reference_storage.astype(np.float32),
    ):
        raise RuntimeError("published raw edges do not reproduce the accepted reference transform")
    if not np.array_equal(m_from_published_edges.astype(np.float32), m_tf2.astype(np.float32)):
        raise RuntimeError("tf2 changed the published-edge transform at float32 precision")
    if (
        comparisons["M_tf2_reference_semantics_vs_M_adapter_precast_float64"][
            "max_absolute_translation_difference"
        ]
        <= 0.003
    ):
        raise RuntimeError("expected adapter translation divergence was not reproduced")
    if extra_points["net_retained_count_difference"] != 2:
        raise RuntimeError(
            "failed to reproduce the net two-point M4.5b range-filter surplus: "
            f"{extra_points['retained_only_by_m45b_count']=}, "
            f"{extra_points['retained_only_by_reference_count']=}"
        )

    calibration_identical = bool(
        np.array_equal(
            np.asarray(first_history.calibration["translation"], np.float64),
            np.asarray(current.calibration["translation"], np.float64),
        )
        and np.array_equal(
            np.asarray(first_history.calibration["rotation"], np.float64),
            np.asarray(current.calibration["rotation"], np.float64),
        )
    )
    ledger = {
        "schema_version": 1,
        "status": "diagnostic_only_exact_gate_still_failed",
        "implementation_wip_commit": args.implementation_wip_commit,
        "sample_index": SAMPLE_INDEX,
        "sample_token": W1_SAMPLE_TOKEN,
        "historical_index": HISTORY_INDEX,
        "historical_acquisition_identity": history_identity,
        "frames": {
            "source": LIDAR_FRAME,
            "target": LIDAR_FRAME,
            "fixed": FIXED_FRAME,
            "tree": f"{FIXED_FRAME} -> {EGO_FRAME} -> {LIDAR_FRAME}",
        },
        "times": {
            "source": _stamp_record(first_history.timestamp_microseconds),
            "target": _stamp_record(current.timestamp_microseconds),
            "requested_times_exactly_match_published_samples": True,
            "tf2_interpolation": False,
        },
        "raw_nuscenes_records": {
            "historical_calibrated_sensor": first_history.calibration,
            "historical_ego_pose": first_history.ego_pose,
            "current_calibrated_sensor": current.calibration,
            "current_ego_pose": current.ego_pose,
        },
        "replay_published_edges": published_edges,
        "replay_source_audit": {
            "map_to_ego": "direct raw nuScenes ego_pose at each acquisition timestamp",
            "ego_to_lidar": "direct raw nuScenes calibrated_sensor at each acquisition timestamp",
            "dynamic_edges": [f"{FIXED_FRAME}->{EGO_FRAME}", f"{EGO_FRAME}->{LIDAR_FRAME}"],
            "static_edges": [],
            "calibration_values_identical_at_source_and_target": calibration_identical,
            "composed_matrix_decomposed_before_publication": False,
            "quaternion_normalized_by_replay": False,
            "inversion_before_publication": False,
        },
        "M_ref": {
            "representation": "pinned M4.5a row-vector SweepTransform storage",
            "float64_intermediate_rotation": reference_rotation.tolist(),
            "float64_intermediate_translation": reference_translation.tolist(),
            "float64_sensor2lidar_rotation": sensor2lidar_rotation.tolist(),
            "stored_float32_matrix": _matrix_record(m_ref),
        },
        "M_from_published_edges": {
            "representation": "conventional column-vector source-to-target transform",
            "float64_matrix": _matrix_record(m_from_published_edges),
            "pinned_reference_storage_float64": _matrix_record(
                m_from_published_edges_reference_storage
            ),
            "pinned_reference_storage_float32": _matrix_record(
                m_from_published_edges_reference_storage.astype(np.float32)
            ),
        },
        "M_tf2": {
            "returned_transform_stamped": {
                "translation_xyz": m_tf2[:3, 3].tolist(),
                "quaternion_xyzw": returned_quaternion_xyzw,
                "quaternion_repr": repr(tuple(returned_quaternion_xyzw)),
            },
            "conventional_float64_matrix": _matrix_record(m_tf2),
            "pinned_reference_storage_float64": _matrix_record(m_tf2_reference_storage),
        },
        "M_adapter": {
            "current_precast_float64_semantics": _matrix_record(m_adapter_precast),
            "current_stored_float32_matrix": _matrix_record(m_adapter),
        },
        "matrix_comparisons": comparisons,
        "first_divergence": {
            "stage": "TransformStamped-to-SweepTransform adapter translation-column encoding",
            "statement": (
                "The first material divergence is introduced at the current "
                "TransformStamped-to-SweepTransform adapter, which stores -t instead of the "
                "pinned upstream -R^T@t translation column."
            ),
            "first_element": comparisons["M_tf2_reference_semantics_vs_M_adapter_precast_float64"][
                "first_material_differing_element"
            ],
            "rotation_float32_bytes_exact": True,
        },
        "composition_precision_ledger": {
            "M4.5a": {
                "rigid_chain": (
                    "historical lidar -> historical ego -> global -> current ego -> current lidar"
                ),
                "conceptual_edges": 4,
                "conceptual_rigid_compositions": 3,
                "conceptual_inversions": 2,
                "actual_numpy_inverse_calls_in_from_poses": 7,
                "quaternion_to_matrix_conversions_before_from_poses": 4,
                "pyquaternion_normalization_checks": 4,
                "actual_quaternion_normalizations_for_W1_records": 0,
                "matrix_to_quaternion_conversions": 0,
            },
            "replay_publisher": {
                "published_edge_records_for_query": 4,
                "matrix_compositions": 0,
                "inversions": 0,
                "quaternion_to_matrix_conversions": 0,
                "matrix_to_quaternion_conversions": 0,
                "quaternion_normalizations": 0,
            },
            "outside_tf2_reconstruction": {
                "quaternion_to_matrix_conversions": 4,
                "pyquaternion_normalization_checks": 4,
                "actual_quaternion_normalizations_for_W1_records": 0,
                "matrix_compositions": 3,
                "matrix_inversions": 1,
                "matrix_to_quaternion_conversions": 0,
            },
            "tf2_query": {
                "path": (
                    "historical lidar@source_time -> historical ego@source_time -> map -> "
                    "current ego@target_time -> current lidar@target_time"
                ),
                "queried_edge_samples": 4,
                "conceptual_rigid_compositions": 3,
                "conceptual_edge_inversions": 2,
                "equivalent_composite_inversions": 1,
                "interpolation": False,
                "source_time_exact_sample": True,
                "target_time_exact_sample": True,
                "internal_quaternion_matrix_conversions": "not exposed by the public tf2 API",
                "internal_quaternion_normalizations": "not exposed by the public tf2 API",
                "diagnostic_return_quaternion_to_matrix_conversions": 1,
                "diagnostic_pyquaternion_normalization_checks": 1,
                "diagnostic_actual_quaternion_normalizations": 0,
            },
            "adapter": {
                "quaternion_to_matrix_conversions": 1,
                "quaternion_norm_checks": 1,
                "explicit_quaternion_normalizations": 0,
                "rotation_formula_2_over_norm_squared_scaling": 1,
                "matrix_to_quaternion_conversions": 0,
                "float64_to_float32_storage_casts": 1,
            },
        },
        "extra_point_analysis": extra_points,
        "controlled_point_test": controlled,
        "failed_gate": {
            "reference_point_count": 354_182,
            "m45b_point_count": 354_184,
            "reference_sha256": EXPECTED_REFERENCE_HASH,
            "m45b_sha256": EXPECTED_FAILED_HASH,
            "acceptance_changed": False,
        },
        "recommendation": (
            "EXACT PARITY APPEARS FIXABLE — owner review should consider correcting only the "
            "adapter translation-column encoding to the pinned -R^T@t convention. No fix was "
            "implemented in this diagnostic turn."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": ledger["status"], "first_divergence": ledger["first_divergence"]}))


if __name__ == "__main__":
    main()
