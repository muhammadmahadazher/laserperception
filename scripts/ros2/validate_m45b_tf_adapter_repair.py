"""Validate the M4.5b TF adapter repair against exact M4.5a references."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from laserperception_ros.raw_replay_node import _acquisitions, _stamp, _transform
from pyquaternion import Quaternion
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import Buffer
from validate_m45b_raw_ros import _run_case, _same_frame_time_travel, _tf2_version

from laserperception.detection.live_multisweep import (
    acquisition_identity,
    sweep_transform_from_ros,
)
from laserperception.detection.multisweep import LidarPose, SweepTransform
from laserperception.detection.ros2_contract import TimeStamp

BASE_MAIN = "9c0fecbb45ebb1d0c65e61a99f13b72558327527"
FAILURE_WIP_COMMIT = "931978d9faaf192e8d2e71409eb467d90a138a8f"
DIAGNOSTIC_COMMIT = "cb4621602a199fc17ed51c1759015012b74d9848"
W1_INDEX = 42
FIXED_FRAME = "nuscenes_map"
EGO_FRAME = "nuscenes_ego"
LIDAR_FRAME = "nuscenes_lidar_top"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("LASERPERCEPTION_NUSCENES_ROOT", "")),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("benchmarks/m45a/results/nuscenes_mini_multisweep_parity.json"),
    )
    parser.add_argument(
        "--failure-diagnostic",
        type=Path,
        default=Path("benchmarks/m45b/diagnostics/w1_raw_ros_hash_failure.json"),
    )
    parser.add_argument(
        "--transform-ledger",
        type=Path,
        default=Path("benchmarks/m45b/diagnostics/w1_tf_transform_ledger.json"),
    )
    parser.add_argument("--repair-commit", required=True)
    parser.add_argument("--timeout-sec", type=float, default=20.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference_samples(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload["tier_a"]["samples"]
    result = {int(record["sample_index"]): record for record in samples}
    if len(result) != 81 or not all(bool(record["exact"]) for record in result.values()):
        raise RuntimeError("accepted M4.5a reference does not contain 81 exact samples")
    return result


def _expected(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_token": record["sample_token"],
        "point_count": int(record["final_point_count"]),
        "sha256": record["official"]["sha256"],
    }


def _rotation_wxyz(values: list[float]) -> np.ndarray:
    return np.asarray(Quaternion(values).rotation_matrix, dtype=np.float64)


def _pose(acquisition: Any) -> LidarPose:
    return LidarPose(
        _rotation_wxyz(acquisition.calibration["rotation"]),
        np.asarray(acquisition.calibration["translation"], dtype=np.float64),
        _rotation_wxyz(acquisition.ego_pose["rotation"]),
        np.asarray(acquisition.ego_pose["translation"], dtype=np.float64),
    )


def _global_from_lidar_rotation(acquisition: Any) -> np.ndarray:
    return _rotation_wxyz(acquisition.ego_pose["rotation"]) @ _rotation_wxyz(
        acquisition.calibration["rotation"]
    )


def _rotation_angle_degrees(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _max_history_rotation_degrees(nusc: Any, record: dict[str, Any]) -> float:
    acquisitions = _acquisitions(nusc, str(record["sample_token"]), 10)
    if len(acquisitions) != 11:
        raise RuntimeError(f"sample {record['sample_index']} is not full-history")
    current_rotation = _global_from_lidar_rotation(acquisitions[-1])
    return max(
        _rotation_angle_degrees(current_rotation.T @ _global_from_lidar_rotation(history))
        for history in acquisitions[:-1]
    )


def _select_sentinels(
    nusc: Any,
    references: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    full_history = [
        record for record in references.values() if int(record["historical_sweep_count"]) == 10
    ]
    if len(full_history) != 79:
        raise RuntimeError(f"expected 79 full-history samples, found {len(full_history)}")
    candidates: list[dict[str, Any]] = []
    for record in full_history:
        index = int(record["sample_index"])
        metric = _max_history_rotation_degrees(nusc, record)
        candidates.append({"record": record, "sample_index": index, "metric": metric})
    eligible = sorted(
        (item for item in candidates if item["sample_index"] != W1_INDEX),
        key=lambda item: (item["metric"], item["sample_index"]),
    )
    selected = [eligible[0], eligible[len(eligible) // 2], eligible[-1]]
    reasons = (
        "minimum maximum historical-to-current rotation among eligible full-history samples",
        "median maximum historical-to-current rotation among eligible full-history samples",
        "maximum maximum historical-to-current rotation among eligible full-history samples",
    )
    for item, reason in zip(selected, reasons, strict=True):
        item["selection_reason"] = reason
    selection = {
        "procedure": (
            "Among the 79 accepted M4.5a samples with ten historical sweeps, exclude W1 index 42, "
            "compute each sample's maximum historical-to-current lidar rotation angle, then select "
            "the minimum, median, and maximum ranked candidates."
        ),
        "full_history_pool_count": len(full_history),
        "eligible_after_excluding_w1_count": len(eligible),
        "metric": "maximum historical-to-current lidar rotation angle in degrees",
    }
    return selected, selection


def _yaw_rotation(angle_radians: float) -> np.ndarray:
    cosine = math.cos(angle_radians)
    sine = math.sin(angle_radians)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _rotation_from_xyzw(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = quaternion
    scale = 2.0 / (x * x + y * y + z * z + w * w)
    return np.array(
        [
            [1.0 - scale * (y * y + z * z), scale * (x * y - z * w), scale * (x * z + y * w)],
            [scale * (x * y + z * w), 1.0 - scale * (x * x + z * z), scale * (y * z - x * w)],
            [scale * (x * z - y * w), scale * (y * z + x * w), 1.0 - scale * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _unit_regression() -> dict[str, Any]:
    yaw_degrees = 30.0
    yaw = math.radians(yaw_degrees)
    rotation = _yaw_rotation(yaw)
    translation = np.array([1.25, -2.5, 0.75], dtype=np.float64)
    correct = -rotation.T @ translation
    old = -translation
    encoded = sweep_transform_from_ros(
        translation_xyz=translation,
        quaternion_xyzw=(0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)),
        source_id="history",
        target_id="current",
    )
    expected = np.eye(4, dtype=np.float64)
    expected[:3, :3] = rotation.T
    expected[:3, 3] = correct
    expected_float32 = expected.astype(np.float32)
    exact = bool(np.array_equal(encoded.lidar2sensor, expected_float32))
    old_distinct = bool(np.linalg.norm(correct - old) > 0.5)
    return {
        "yaw_degrees": yaw_degrees,
        "translation_xyz": translation.tolist(),
        "rotation_source_to_target": rotation.tolist(),
        "correct_translation_storage": correct.tolist(),
        "old_incorrect_translation_storage": old.tolist(),
        "formula_separation_l2_m": float(np.linalg.norm(correct - old)),
        "expected_storage_float32": expected_float32.tolist(),
        "observed_storage_float32": encoded.lidar2sensor.tolist(),
        "old_behavior_materially_distinct": old_distinct,
        "exact": exact,
    }


def _map_to_lidar_pose(
    seconds: int,
    translation: np.ndarray,
    yaw_degrees: float,
) -> TransformStamped:
    yaw = math.radians(yaw_degrees)
    transform = TransformStamped()
    transform.header.frame_id = "map"
    transform.header.stamp = Time(seconds=seconds).to_msg()
    transform.child_frame_id = "lidar"
    transform.transform.translation.x = float(translation[0])
    transform.transform.translation.y = float(translation[1])
    transform.transform.translation.z = float(translation[2])
    transform.transform.rotation.z = math.sin(yaw / 2.0)
    transform.transform.rotation.w = math.cos(yaw / 2.0)
    return transform


def _rotation_time_travel() -> dict[str, Any]:
    source_translation = np.array([0.5, -1.0, 0.25], dtype=np.float64)
    target_translation = np.array([2.0, 1.5, -0.5], dtype=np.float64)
    source_yaw_degrees = 10.0
    target_yaw_degrees = 40.0
    source_rotation = _yaw_rotation(math.radians(source_yaw_degrees))
    target_rotation = _yaw_rotation(math.radians(target_yaw_degrees))
    buffer = Buffer()
    buffer.set_transform(
        _map_to_lidar_pose(1, source_translation, source_yaw_degrees), "m45b_repair"
    )
    buffer.set_transform(
        _map_to_lidar_pose(2, target_translation, target_yaw_degrees), "m45b_repair"
    )
    returned = buffer.lookup_transform_full(
        "lidar", Time(seconds=2), "lidar", Time(seconds=1), "map"
    )
    quaternion = (
        returned.transform.rotation.x,
        returned.transform.rotation.y,
        returned.transform.rotation.z,
        returned.transform.rotation.w,
    )
    returned_rotation = _rotation_from_xyzw(quaternion)
    returned_translation = np.array(
        [
            returned.transform.translation.x,
            returned.transform.translation.y,
            returned.transform.translation.z,
        ],
        dtype=np.float64,
    )
    expected_rotation = target_rotation.T @ source_rotation
    expected_translation = target_rotation.T @ (source_translation - target_translation)
    expected_storage = np.eye(4, dtype=np.float64)
    expected_storage[:3, :3] = expected_rotation.T
    expected_storage[:3, 3] = -expected_rotation.T @ expected_translation
    encoded = sweep_transform_from_ros(
        translation_xyz=returned_translation,
        quaternion_xyzw=quaternion,
        source_id="lidar@1",
        target_id="lidar@2",
    )
    source_point = np.array([1.0, 2.0, 0.25], dtype=np.float32)
    observed_point = source_point @ encoded.lidar2sensor[:3, :3]
    observed_point -= encoded.lidar2sensor[:3, 3]
    expected_point = source_point @ expected_storage[:3, :3].astype(np.float32)
    expected_point -= expected_storage[:3, 3].astype(np.float32)
    old = -expected_translation
    checks = {
        "cross_time_nonidentity": not bool(np.allclose(returned_rotation, np.eye(3))),
        "returned_rotation_matches_known_poses": bool(
            np.allclose(returned_rotation, expected_rotation, atol=1e-12)
        ),
        "returned_translation_matches_known_poses": bool(
            np.allclose(returned_translation, expected_translation, atol=1e-12)
        ),
        "adapter_storage_matches_pinned_formula": bool(
            np.allclose(encoded.lidar2sensor, expected_storage.astype(np.float32), atol=1e-7)
        ),
        "old_translation_materially_distinct": bool(
            np.linalg.norm(expected_storage[:3, 3] - old) > 0.5
        ),
        "transformed_point_exact": bool(np.array_equal(observed_point, expected_point)),
    }
    return {
        "fixed_frame": "map",
        "source": {
            "frame": "lidar",
            "time_sec": 1,
            "map_translation_xyz": source_translation.tolist(),
            "map_yaw_degrees": source_yaw_degrees,
        },
        "target": {
            "frame": "lidar",
            "time_sec": 2,
            "map_translation_xyz": target_translation.tolist(),
            "map_yaw_degrees": target_yaw_degrees,
        },
        "returned_source_to_target_rotation": returned_rotation.tolist(),
        "returned_source_to_target_translation": returned_translation.tolist(),
        "returned_quaternion_xyzw": list(quaternion),
        "correct_translation_storage": expected_storage[:3, 3].tolist(),
        "old_incorrect_translation_storage": old.tolist(),
        "expected_storage_float32": expected_storage.astype(np.float32).tolist(),
        "observed_storage_float32": encoded.lidar2sensor.tolist(),
        "test_point_source_xyz": source_point.tolist(),
        "test_point_expected_xyz": expected_point.tolist(),
        "test_point_observed_xyz": observed_point.tolist(),
        "checks": checks,
        "exact": all(checks.values()),
    }


def _timestamp(timestamp_microseconds: int) -> TimeStamp:
    seconds, microseconds = divmod(timestamp_microseconds, 1_000_000)
    return TimeStamp(seconds, microseconds * 1_000)


def _raw_points(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size == 0 or values.size % 5 != 0:
        raise RuntimeError("raw sweep is empty or malformed")
    return np.ascontiguousarray(values.reshape(-1, 5))


def _membership_mask(points: np.ndarray, transform: SweepTransform) -> np.ndarray:
    transformed = points.copy()
    matrix = np.array(transform.lidar2sensor.tolist())
    transformed[:, :3] = transformed[:, :3] @ matrix[:3, :3]
    transformed[:, :3] -= matrix[:3, 3]
    return (
        (transformed[:, 0] > -50.0)
        & (transformed[:, 0] < 50.0)
        & (transformed[:, 1] > -50.0)
        & (transformed[:, 1] < 50.0)
        & (transformed[:, 2] > -5.0)
        & (transformed[:, 2] < 3.0)
    )


def _w1_membership(nusc: Any, sample_token: str) -> dict[str, Any]:
    acquisitions = _acquisitions(nusc, sample_token, 10)
    current = acquisitions[-1]
    historical = list(reversed(acquisitions[:-1]))
    current_stamp = _timestamp(current.timestamp_microseconds)
    current_identity = acquisition_identity(LIDAR_FRAME, current_stamp)
    current_pose = _pose(current)
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
            "m45b_repair_membership",
        )
        buffer.set_transform(
            _transform(
                parent=EGO_FRAME,
                child=LIDAR_FRAME,
                stamp=stamp,
                record=acquisition.calibration,
            ),
            "m45b_repair_membership",
        )
    per_history: list[dict[str, Any]] = []
    total_live_only = 0
    total_reference_only = 0
    for history_index, acquisition in enumerate(historical):
        stamp = _timestamp(acquisition.timestamp_microseconds)
        identity = acquisition_identity(LIDAR_FRAME, stamp)
        reference = SweepTransform.from_poses(
            source_id=identity,
            target_id=current_identity,
            sweep_pose=_pose(acquisition),
            current_pose=current_pose,
        )
        returned = buffer.lookup_transform_full(
            LIDAR_FRAME,
            Time(nanoseconds=current.timestamp_microseconds * 1_000),
            LIDAR_FRAME,
            Time(nanoseconds=acquisition.timestamp_microseconds * 1_000),
            FIXED_FRAME,
        )
        live = sweep_transform_from_ros(
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
            target_id=current_identity,
        )
        raw = _raw_points(acquisition.path)
        reference_mask = _membership_mask(raw, reference)
        live_mask = _membership_mask(raw, live)
        live_only = int(np.count_nonzero(live_mask & ~reference_mask))
        reference_only = int(np.count_nonzero(reference_mask & ~live_mask))
        total_live_only += live_only
        total_reference_only += reference_only
        per_history.append(
            {
                "history_index": history_index,
                "acquisition_identity": identity,
                "live_only_count": live_only,
                "reference_only_count": reference_only,
                "exact": bool(np.array_equal(reference_mask, live_mask)),
            }
        )
    return {
        "history_depth": len(historical),
        "per_history": per_history,
        "live_only_count": total_live_only,
        "reference_only_count": total_reference_only,
        "previous_failure_live_only_count": 7,
        "previous_failure_reference_only_count": 5,
        "previous_failure_net_count": 2,
        "exact": total_live_only == 0
        and total_reference_only == 0
        and all(bool(record["exact"]) for record in per_history),
    }


def main() -> None:
    args = _arguments()
    if not str(args.data_root) or not args.data_root.is_dir():
        raise SystemExit("set --data-root or LASERPERCEPTION_NUSCENES_ROOT")
    for path in (args.reference, args.failure_diagnostic, args.transform_ledger):
        if not path.is_file():
            raise SystemExit(f"required evidence is missing: {path}")
    if len(args.repair_commit) != 40 or any(
        value not in "0123456789abcdef" for value in args.repair_commit
    ):
        raise SystemExit("--repair-commit must be a lowercase 40-character Git SHA")

    from nuscenes.nuscenes import NuScenes

    references = _reference_samples(args.reference)
    nusc = NuScenes(version="v1.0-mini", dataroot=str(args.data_root), verbose=False)
    selected, selection = _select_sentinels(nusc, references)
    unit_regression = _unit_regression()
    pure_translation = _same_frame_time_travel()
    pure_translation["formula_discrimination"] = False
    pure_translation["limitation"] = (
        "Identity rotation makes -t equal -R.T@t; this regression proves cross-time lookup only."
    )
    rotation_time_travel = _rotation_time_travel()
    scene_start = _run_case(
        0,
        args.data_root,
        timeout_sec=args.timeout_sec,
        expected=_expected(references[0]),
    )
    w1 = _run_case(
        W1_INDEX,
        args.data_root,
        timeout_sec=args.timeout_sec,
        expected=_expected(references[W1_INDEX]),
    )
    membership = _w1_membership(nusc, str(references[W1_INDEX]["sample_token"]))
    w1["retained_row_membership"] = membership

    sentinels: list[dict[str, Any]] = []
    for item in selected:
        record = item["record"]
        result = _run_case(
            int(record["sample_index"]),
            args.data_root,
            timeout_sec=args.timeout_sec,
            expected=_expected(record),
        )
        result["selection_reason"] = item["selection_reason"]
        result["max_historical_to_current_rotation_degrees"] = item["metric"]
        sentinels.append(result)

    required_exact = [
        bool(unit_regression["exact"]),
        bool(unit_regression["old_behavior_materially_distinct"]),
        bool(pure_translation["exact"]),
        bool(rotation_time_travel["exact"]),
        bool(scene_start["exact"]),
        bool(w1["exact"]),
        bool(membership["exact"]),
        int(w1["acquisition_count"]) == 11,
        int(w1["final_history_depth"]) == 10,
        all(
            bool(result["exact"])
            and int(result["acquisition_count"]) == 11
            and int(result["final_history_depth"]) == 10
            for result in sentinels
        ),
    ]
    passed = all(required_exact)
    evidence = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "base_main": BASE_MAIN,
        "failure_wip_commit": FAILURE_WIP_COMMIT,
        "diagnostic_commit": DIAGNOSTIC_COMMIT,
        "repair_commit": args.repair_commit,
        "adapter_formulas": {
            "old": "translation_storage = -t",
            "corrected": "translation_storage = -R.T @ t",
            "rotation_storage": "R.T",
            "calculation_and_storage": "float64 calculation then one float32 SweepTransform cast",
        },
        "rotation_bearing_unit_regression": unit_regression,
        "same_frame_different_time_pure_translation": pure_translation,
        "same_frame_different_time_rotation_translation": rotation_time_travel,
        "scene_start": scene_start,
        "w1": w1,
        "additional_full_history_selection": selection,
        "additional_full_history_sentinels": sentinels,
        "chronology": {
            "previous_failure_diagnostic": {
                "logical_path": "benchmarks/m45b/diagnostics/w1_raw_ros_hash_failure.json",
                "sha256": _sha256(args.failure_diagnostic),
            },
            "transform_ledger": {
                "logical_path": "benchmarks/m45b/diagnostics/w1_tf_transform_ledger.json",
                "sha256": _sha256(args.transform_ledger),
            },
            "repaired_w1_exact": bool(w1["exact"] and membership["exact"]),
        },
        "environment": {
            "ros_distro": os.environ.get("ROS_DISTRO", "unknown"),
            "rmw_implementation": rclpy.utilities.get_rmw_implementation_identifier(),
            "tf2_ros_version": _tf2_version(),
            "dataset": "nuScenes v1.0-mini mini_val",
        },
        "scope_guards": {
            "exact_gate_changed": False,
            "tolerance_or_tier_b_added": False,
            "tf2_lookup_changed": False,
            "replay_tf_publication_changed": False,
            "timestamp_handling_changed": False,
            "history_ordering_changed": False,
            "range_filtering_changed": False,
            "multisweep_builder_changed": False,
            "detector_chain_run": False,
            "detector_changed": False,
            "model_changed": False,
            "onnx_changed": False,
            "tensorrt_engine_changed": False,
            "exact_fast_changed": False,
            "performance_campaign_run": False,
        },
    }
    payload = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not passed:
        raise SystemExit("M4.5b TF adapter repair exactness gate failed")


if __name__ == "__main__":
    main()
