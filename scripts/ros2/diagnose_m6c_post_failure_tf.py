"""Diagnose the preserved M6c frame-1 platform, quaternion, and tf2 boundaries."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from laserperception_ros.conversion import pointcloud2_to_model_ready
from laserperception_ros.kitti_raw_replay_node import KittiRawReplayNode
from laserperception_ros.multisweep_node import LaserPerceptionMultiSweepNode, _time_message
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2

from laserperception.datasets.kitti_raw import KittiRawSequence
from laserperception.datasets.kitti_ros_replay import model_lidar_pose_to_world_transform
from laserperception.detection.live_multisweep import (
    LiveRawSweep,
    sweep_transform_from_ros,
)
from laserperception.detection.mmdet3d_backend import sha256_file
from laserperception.detection.multisweep import (
    POINTPILLARS_POINT_CLOUD_RANGE,
    POINTPILLARS_USE_DIM,
    HistoricalSweep,
    SweepTransform,
)
from laserperception.evaluation.m6c_representation import (
    array_sha256,
    builder_matrix_from_ros_transform,
    compare_float32_arrays,
    compare_voxel_structures,
    quaternion_to_rotation_matrix_xyzw,
    rotation_summary,
    voxel_structure,
)

BASE_MAIN_SHA = "ebbbc0bbc4423e3be476abcd1165f75a136fa54c"
PLAN_COMMIT = "6a00cdc8fc2fa950ca7f8a4bf4261fdeeefbc6d9"
R2_PROTOCOL_COMMIT = "0a8419978d265571b51f943ffc797b5fcc78c4ca"
FAILURE_SHA256 = "fd319dfcb13da570c62d83b4475531bbbd735845808225ebffee4afe03f1cea4"
M6A_SHA256 = "a62da9808079994d971c4d47bbdf04f2c50d44ddcfd0c7958f7534512552155b"
DRIVE = "2011_09_26_drive_0001"
FRAME_INDEX = 1
EXPECTED_MODEL_READY_SHA256 = "4088c7ca546aa4b9a00f485153d4a00fd7ed92cde1e7c70f3a24bb6ab883bf7e"
OBSERVED_MODEL_READY_SHA256 = "5bd1d66a1cfe553ae91493b7eb48f36233afe0947f8ab096576f40d2557f16f7"
EXPECTED_TRANSFORM_SHA256 = "c0c66df4237968a1c0ced2f3bc260d01158e97ef5a5e4bb359efaace9369e733"
OBSERVED_TRANSFORM_SHA256 = "a57ade3532cca9ff0e6a3eb8998a5ba57c882482f10f97d0fc6112abd5336f9e"
FIXED_FRAME = "kitti_world"
LIDAR_FRAME = "kitti_model_aligned_lidar"


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=_root(), check=True, capture_output=True, text=True
    ).stdout.strip()


def _require_identity(diagnostic_commit: str) -> None:
    if _git("rev-parse", "HEAD") != diagnostic_commit:
        raise RuntimeError("D1 must run at the exact committed diagnostic implementation")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("D1 requires a clean tracked worktree")
    for ancestor in (BASE_MAIN_SHA, R2_PROTOCOL_COMMIT, PLAN_COMMIT):
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, diagnostic_commit],
            cwd=_root(),
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"D1 commit does not descend from required commit {ancestor}")


class _CaptureNode(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("m6c_d1_model_ready_capture")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._condition = threading.Condition()
        self._messages: deque[PointCloud2] = deque()
        self._subscription = self.create_subscription(PointCloud2, topic, self._capture, qos)

    def _capture(self, message: PointCloud2) -> None:
        with self._condition:
            self._messages.append(message)
            self._condition.notify_all()

    def wait(self, timeout_sec: float) -> PointCloud2:
        deadline = time.monotonic() + timeout_sec
        with self._condition:
            while not self._messages:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("timed out waiting for D1 model-ready PointCloud2")
                self._condition.wait(remaining)
            return self._messages.popleft()


class _TransformCaptureBuilder(LaserPerceptionMultiSweepNode):
    """Observe the real tf2 lookup without changing the production conversion."""

    def __init__(self, *, parameter_overrides: list[Any]) -> None:
        super().__init__(parameter_overrides=parameter_overrides)
        self.lookup_translation_xyz: tuple[float, float, float] | None = None
        self.lookup_quaternion_xyzw: tuple[float, float, float, float] | None = None
        self.encoded_transform: SweepTransform | None = None

    def _historical_sweep(
        self,
        historical: LiveRawSweep,
        current: LiveRawSweep,
        target_frame: str,
    ) -> HistoricalSweep:
        transform = self._tf_buffer.lookup_transform_full(
            target_frame,
            Time.from_msg(_time_message(current)),
            historical.frame_id,
            Time.from_msg(_time_message(historical)),
            self._fixed_frame,
            timeout=Duration(seconds=self._transform_timeout_sec),
        )
        translation = (
            float(transform.transform.translation.x),
            float(transform.transform.translation.y),
            float(transform.transform.translation.z),
        )
        quaternion = (
            float(transform.transform.rotation.x),
            float(transform.transform.rotation.y),
            float(transform.transform.rotation.z),
            float(transform.transform.rotation.w),
        )
        encoded = sweep_transform_from_ros(
            translation_xyz=translation,
            quaternion_xyzw=quaternion,
            source_id=historical.sweep.source_id,
            target_id=current.sweep.source_id,
        )
        self.lookup_translation_xyz = translation
        self.lookup_quaternion_xyzw = quaternion
        self.encoded_transform = encoded
        return HistoricalSweep(historical.sweep, encoded)


def _capture_real_ros(
    data_root: Path,
    *,
    timeout_sec: float,
) -> tuple[np.ndarray, tuple[float, ...], tuple[float, ...], np.ndarray, dict[str, int]]:
    raw_topic = "/laserperception/m6c/d1/raw"
    model_topic = "/laserperception/m6c/d1/model_ready"
    replay = KittiRawReplayNode(
        parameter_overrides=[
            Parameter("raw_points_topic", value=raw_topic),
            Parameter("data_root", value=str(data_root)),
            Parameter("drive_id", value=DRIVE),
            Parameter("start_frame", value=0),
            Parameter("end_frame", value=FRAME_INDEX),
            Parameter("auto_start", value=False),
            Parameter("fixed_frame", value=FIXED_FRAME),
            Parameter("lidar_frame", value=LIDAR_FRAME),
        ]
    )
    builder = _TransformCaptureBuilder(
        parameter_overrides=[
            Parameter("raw_points_topic", value=raw_topic),
            Parameter("model_ready_topic", value=model_topic),
            Parameter("fixed_frame", value=FIXED_FRAME),
            Parameter("target_frame", value=LIDAR_FRAME),
            Parameter("max_historical_sweeps", value=10),
            Parameter("transform_timeout_sec", value=0.5),
            Parameter("tf_cache_time_sec", value=60.0),
        ]
    )
    capture = _CaptureNode(model_topic)
    executor = MultiThreadedExecutor(num_threads=3)
    for node in (replay, builder, capture):
        executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5.0
        while replay._publisher.get_subscription_count() == 0:
            if time.monotonic() >= deadline:
                raise TimeoutError("D1 replay did not discover the live builder")
            time.sleep(0.02)
        target: PointCloud2 | None = None
        for frame_index in range(FRAME_INDEX + 1):
            if replay.publish_next() != frame_index:
                raise RuntimeError("D1 replay lost chronological identity")
            target = capture.wait(timeout_sec)
        if target is None or builder.encoded_transform is None:
            raise RuntimeError("D1 did not capture the historical ROS transform")
        if builder.lookup_translation_xyz is None or builder.lookup_quaternion_xyzw is None:
            raise RuntimeError("D1 did not capture the raw tf2 transform")
        cloud = pointcloud2_to_model_ready(target)
        counters = {
            "raw_frames_received": builder.raw_frames_received,
            "valid_raw_frames": builder.valid_raw_frames,
            "model_ready_outputs": builder.model_ready_frames_published,
            "rejected_frames": builder.rejected_frames,
            "tf_failures": builder.tf_failures,
            "history_depth": builder.current_history_depth,
        }
        return (
            cloud.points_xyzt,
            builder.lookup_translation_xyz,
            builder.lookup_quaternion_xyzw,
            builder.encoded_transform.lidar2sensor,
            counters,
        )
    finally:
        executor.shutdown(timeout_sec=2.0)
        thread.join(timeout=2.0)
        capture.destroy_node()
        builder.destroy_node()
        replay.destroy_node()


def _world_pose(
    sequence: KittiRawSequence, index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose = sequence.lidar_pose(index)
    original_rotation = pose.ego_to_global_rotation @ pose.lidar_to_ego_rotation
    translation, quaternion_values = model_lidar_pose_to_world_transform(pose)
    quaternion = np.asarray(quaternion_values, dtype=np.float64)
    reconstructed = quaternion_to_rotation_matrix_xyzw(quaternion)
    return reconstructed, np.asarray(translation, dtype=np.float64), original_rotation


def _direct_builder_matrix_float64(sequence: KittiRawSequence) -> np.ndarray:
    """Expose the pre-cast arithmetic transcribed by ``SweepTransform.from_poses``."""

    sweep_pose = sequence.lidar_pose(0)
    current_pose = sequence.lidar_pose(FRAME_INDEX)
    l2e_r_s_mat = sweep_pose.lidar_to_ego_rotation
    e2g_r_s_mat = sweep_pose.ego_to_global_rotation
    l2e_t_s = sweep_pose.lidar_to_ego_translation
    e2g_t_s = sweep_pose.ego_to_global_translation
    l2e_r_mat = current_pose.lidar_to_ego_rotation
    e2g_r_mat = current_pose.ego_to_global_rotation
    l2e_t = current_pose.lidar_to_ego_translation
    e2g_t = current_pose.ego_to_global_translation

    rotation = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    translation = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    translation -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    )
    sensor2lidar_rotation = rotation.T
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = sensor2lidar_rotation.T
    result[:3, 3:4] = -sensor2lidar_rotation.T @ translation.reshape(3, 1)
    return result


def _pre_range(
    sequence: KittiRawSequence,
    transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    current = sequence.frame(FRAME_INDEX).to_raw_sweep()
    historical = sequence.frame(0).to_raw_sweep()
    current_points = current.points.copy()
    current_points[:, 4] = np.float32(0.0)
    historical_points = historical.points.copy()
    matrix = np.array(np.asarray(transform, dtype=np.float32).tolist())
    historical_points[:, :3] = historical_points[:, :3] @ matrix[:3, :3]
    historical_points[:, :3] -= matrix[:3, 3]
    historical_points[:, 4] = current.timestamp_seconds - historical.timestamp_seconds
    concatenated = np.concatenate([current_points, historical_points], axis=0)
    xyzt = np.ascontiguousarray(concatenated[:, POINTPILLARS_USE_DIM])
    minimum = POINTPILLARS_POINT_CLOUD_RANGE[:3]
    maximum = POINTPILLARS_POINT_CLOUD_RANGE[3:]
    mask = (
        (xyzt[:, 0] > minimum[0])
        & (xyzt[:, 0] < maximum[0])
        & (xyzt[:, 1] > minimum[1])
        & (xyzt[:, 1] < maximum[1])
        & (xyzt[:, 2] > minimum[2])
        & (xyzt[:, 2] < maximum[2])
    )
    current_retained = int(np.count_nonzero(mask[: len(current_points)]))
    return xyzt, mask, np.ascontiguousarray(xyzt[mask]), current_retained


def _stage(name: str, float64_matrix: np.ndarray, float32_matrix: np.ndarray) -> dict[str, object]:
    return {
        "name": name,
        "float64": np.asarray(float64_matrix, dtype=np.float64).tolist(),
        "float32": np.asarray(float32_matrix, dtype=np.float32).tolist(),
        "float32_sha256": array_sha256(np.asarray(float32_matrix, dtype=np.float32)),
        "rotation": rotation_summary(np.asarray(float64_matrix, dtype=np.float64)[:3, :3]),
    }


def _float64_difference(first: np.ndarray, second: np.ndarray) -> dict[str, object]:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    delta = np.abs(left - right)
    return {
        "exact": np.array_equal(left, right),
        "differing_elements": int(np.count_nonzero(left != right)),
        "maximum_absolute_difference": float(np.max(delta)) if delta.size else 0.0,
    }


def _adjacent_record(
    name: str,
    first_matrix_float64: np.ndarray,
    second_matrix_float64: np.ndarray,
    first_matrix: np.ndarray,
    second_matrix: np.ndarray,
    first_points: np.ndarray,
    second_points: np.ndarray,
) -> dict[str, object]:
    transform = compare_float32_arrays(first_matrix, second_matrix)
    return {
        "mechanism": name,
        "transform_float64": _float64_difference(first_matrix_float64, second_matrix_float64),
        "transform_float32": transform,
        "rotation_only": compare_float32_arrays(first_matrix[:3, :3], second_matrix[:3, :3]),
        "translation_only": compare_float32_arrays(first_matrix[:3, 3], second_matrix[:3, 3]),
        "resulting_model_ready": compare_float32_arrays(first_points, second_points),
    }


def _history_point_summary(
    expected: np.ndarray,
    observed: np.ndarray,
    current_retained: int,
) -> dict[str, object]:
    historical_expected = expected[current_retained:]
    historical_observed = observed[current_retained:]
    comparison = compare_float32_arrays(historical_expected, historical_observed)
    differing = historical_expected != historical_observed
    return {
        "rows_compared": int(len(historical_expected)),
        "differing_rows": int(np.count_nonzero(np.any(differing, axis=1))),
        "differing_values_by_xyzt": [int(value) for value in np.count_nonzero(differing, axis=0)],
        "comparison": comparison,
    }


def _atomic_write(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--diagnostic-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--message-timeout-sec", type=float, default=15.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _require_identity(args.diagnostic_commit)
    root = _root()
    failure_path = root / "benchmarks/m6c/diagnostics/gate_a_failure_frame_0000000001.json"
    m6a_path = root / "benchmarks/m6a/results/kitti_raw_offline_reconstruction.json"
    if sha256_file(failure_path) != FAILURE_SHA256 or sha256_file(m6a_path) != M6A_SHA256:
        raise RuntimeError("frozen M6a/R2 evidence identity mismatch")
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    boundary = failure["first_differing_boundary"]
    t0 = np.asarray(boundary["frozen_expected_transform"], dtype=np.float32)
    if array_sha256(t0) != EXPECTED_TRANSFORM_SHA256:
        raise RuntimeError("frozen T0 transform identity mismatch")

    date_root = args.data_root.expanduser().resolve() / "2011_09_26"
    sequence = KittiRawSequence(date_root, date_root / f"{DRIVE}_sync")
    source = sequence.frame(0).to_raw_sweep()
    current = sequence.frame(FRAME_INDEX).to_raw_sweep()
    t1_transform = SweepTransform.from_poses(
        source_id=source.source_id,
        target_id=current.source_id,
        sweep_pose=sequence.lidar_pose(0),
        current_pose=sequence.lidar_pose(FRAME_INDEX),
    )
    t1_float64 = _direct_builder_matrix_float64(sequence)
    t1 = t1_float64.astype(np.float32)
    if not np.array_equal(t1, t1_transform.lidar2sensor):
        raise RuntimeError("instrumented T1 pre-cast arithmetic differs from production storage")

    source_rotation, source_translation, source_original_rotation = _world_pose(sequence, 0)
    current_rotation, current_translation, current_original_rotation = _world_pose(
        sequence, FRAME_INDEX
    )
    relative_rotation = current_rotation.T @ source_rotation
    relative_translation = current_rotation.T @ (source_translation - current_translation)
    t2_float64 = builder_matrix_from_ros_transform(relative_rotation, relative_translation)
    t2 = t2_float64.astype(np.float32)

    rclpy.init()
    try:
        ros_points, tf_translation, tf_quaternion, t4, counters = _capture_real_ros(
            args.data_root.expanduser().resolve(),
            timeout_sec=args.message_timeout_sec,
        )
    finally:
        rclpy.shutdown()
    tf_quaternion_array = np.asarray(tf_quaternion, dtype=np.float64)
    tf_rotation = quaternion_to_rotation_matrix_xyzw(tf_quaternion_array)
    t3_float64 = builder_matrix_from_ros_transform(
        tf_rotation, np.asarray(tf_translation, dtype=np.float64)
    )
    t3 = t3_float64.astype(np.float32)

    stage_matrices = {"T0": t0, "T1": t1, "T2": t2, "T3": t3, "T4": t4}
    stage_float64 = {
        "T0": t0.astype(np.float64),
        "T1": t1_float64,
        "T2": t2_float64,
        "T3": t3_float64,
        "T4": t4.astype(np.float64),
    }
    stage_points: dict[str, np.ndarray] = {}
    stage_pre_range: dict[str, np.ndarray] = {}
    stage_masks: dict[str, np.ndarray] = {}
    current_retained = -1
    for name, matrix in stage_matrices.items():
        pre_range, mask, points, retained = _pre_range(sequence, matrix)
        stage_pre_range[name] = pre_range
        stage_masks[name] = mask
        stage_points[name] = points
        if current_retained < 0:
            current_retained = retained
        elif current_retained != retained:
            raise RuntimeError("current-sweep range membership changed between transform stages")

    if array_sha256(stage_points["T0"]) != EXPECTED_MODEL_READY_SHA256:
        raise RuntimeError("T0 did not reproduce the frozen frame-1 model-ready input")
    if array_sha256(ros_points) != OBSERVED_MODEL_READY_SHA256:
        raise RuntimeError("real ROS frame-1 payload did not reproduce the preserved R2 failure")
    if not np.array_equal(stage_points["T4"], ros_points):
        raise RuntimeError("captured ROS output is not explained by the captured T4 transform")

    pairs = [
        ("platform_arithmetic", "T0", "T1"),
        ("unit_quaternion_projection", "T1", "T2"),
        ("tf2", "T2", "T3"),
        ("float32_storage", "T3", "T4"),
    ]
    contributions = {
        label: _adjacent_record(
            label,
            stage_float64[first],
            stage_float64[second],
            stage_matrices[first],
            stage_matrices[second],
            stage_points[first],
            stage_points[second],
        )
        for label, first, second in pairs
    }
    classifications: list[str] = []
    if not bool(contributions["platform_arithmetic"]["transform_float32"]["exact"]):  # type: ignore[index]
        classifications.append("PLATFORM_ARITHMETIC_PRESENT")
    if not bool(
        contributions["unit_quaternion_projection"]["transform_float32"]["exact"]  # type: ignore[index]
    ):
        classifications.append("UNIT_QUATERNION_PROJECTION_PRESENT")
    if not bool(contributions["tf2"]["transform_float64"]["exact"]):  # type: ignore[index]
        classifications.append("TF2_ADDITIONAL_DIVERGENCE_PRESENT_BELOW_FLOAT32")
    if not bool(contributions["float32_storage"]["transform_float64"]["exact"]):  # type: ignore[index]
        classifications.append("FLOAT32_STORAGE_ROUNDING_PRESENT")
    explained = array_sha256(t4) == OBSERVED_TRANSFORM_SHA256
    if not explained:
        classifications.append("UNRESOLVED_TRANSFORM_BOUNDARY")

    expected_structure = voxel_structure(stage_points["T0"])
    observed_structure = voxel_structure(stage_points["T4"])
    pre_mask_changes = int(np.count_nonzero(stage_masks["T0"] != stage_masks["T4"]))
    voxel_comparison = compare_voxel_structures(expected_structure, observed_structure)
    voxel_comparison["pre_builder_range_mask"] = {
        "exact": pre_mask_changes == 0,
        "expected_retained": int(np.count_nonzero(stage_masks["T0"])),
        "observed_retained": int(np.count_nonzero(stage_masks["T4"])),
        "points_changing_membership": pre_mask_changes,
    }

    dominant_point_values = max(
        contributions,
        key=lambda key: int(
            contributions[key]["resulting_model_ready"].get("differing_elements", -1)  # type: ignore[union-attr]
        ),
    )
    record: dict[str, object] = {
        "schema_version": 1,
        "status": "TRANSFORM_LADDER_EXPLAINED" if explained else "UNRESOLVED_TRANSFORM_BOUNDARY",
        "diagnostic_only": True,
        "r2_status": "FAILED",
        "r2_protocol_commit": R2_PROTOCOL_COMMIT,
        "diagnostic_plan_commit": PLAN_COMMIT,
        "diagnostic_implementation_commit": args.diagnostic_commit,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "ros_distro": "Humble",
            "rmw_implementation": "rmw_fastrtps_cpp",
        },
        "condition": {"drive": DRIVE, "frame": "0000000001", "condition": "H10"},
        "stages": {
            "T0": _stage("frozen Windows canonical", stage_float64["T0"], t0),
            "T1": _stage("WSL direct matrix", t1_float64, t1),
            "T2": _stage("WSL unit-quaternion composition", t2_float64, t2),
            "T3": _stage("real tf2 relative transform", t3_float64, t3),
            "T4": _stage("final builder storage", t4.astype(np.float64), t4),
        },
        "quaternion_diagnostic": {
            "source": {
                "input_rotation": rotation_summary(source_original_rotation),
                "quaternion": list(model_lidar_pose_to_world_transform(sequence.lidar_pose(0))[1]),
                "quaternion_norm": float(
                    np.linalg.norm(model_lidar_pose_to_world_transform(sequence.lidar_pose(0))[1])
                ),
                "reconstructed_rotation": rotation_summary(source_rotation),
                "input_vs_reconstructed_float64": _float64_difference(
                    source_original_rotation, source_rotation
                ),
            },
            "current": {
                "input_rotation": rotation_summary(current_original_rotation),
                "quaternion": list(
                    model_lidar_pose_to_world_transform(sequence.lidar_pose(FRAME_INDEX))[1]
                ),
                "quaternion_norm": float(
                    np.linalg.norm(
                        model_lidar_pose_to_world_transform(sequence.lidar_pose(FRAME_INDEX))[1]
                    )
                ),
                "reconstructed_rotation": rotation_summary(current_rotation),
                "input_vs_reconstructed_float64": _float64_difference(
                    current_original_rotation, current_rotation
                ),
            },
            "q_and_negative_q_reconstruct_exactly": np.array_equal(
                tf_rotation, quaternion_to_rotation_matrix_xyzw(-tf_quaternion_array)
            ),
            "tf2_relative_quaternion": list(tf_quaternion),
            "tf2_relative_quaternion_norm": float(np.linalg.norm(tf_quaternion_array)),
            "proper_rotation_checks_pass": (
                abs(rotation_summary(tf_rotation)["determinant"] - 1.0) < 1e-12
                and rotation_summary(tf_rotation)["maximum_orthonormality_residual"] < 1e-12
            ),
        },
        "contributions": contributions,
        "classifications": classifications,
        "dominant_by_resulting_differing_point_values": dominant_point_values,
        "frame_1_point_propagation": {
            "point_count_exact": len(stage_points["T0"]) == len(stage_points["T4"]),
            "shape_exact": stage_points["T0"].shape == stage_points["T4"].shape,
            "dtype_exact": stage_points["T0"].dtype == stage_points["T4"].dtype,
            "current_rows_exact": np.array_equal(
                stage_points["T0"][:current_retained], stage_points["T4"][:current_retained]
            ),
            "time_lag_exact": np.array_equal(stage_points["T0"][:, 3], stage_points["T4"][:, 3]),
            "historical": _history_point_summary(
                stage_points["T0"], stage_points["T4"], current_retained
            ),
            "expected_sha256": array_sha256(stage_points["T0"]),
            "observed_sha256": array_sha256(stage_points["T4"]),
        },
        "frame_1_voxel_consequence": voxel_comparison,
        "ros_builder_counters": counters,
        "downstream_authorized": explained,
        "scope": {
            "gate_a_rerun_as_success_attempt": False,
            "gate_b_started": False,
            "detector_initialized": False,
            "performance_campaign": False,
        },
    }
    _atomic_write(args.output.expanduser().resolve(), record)
    print(json.dumps({"status": record["status"], "classifications": classifications}))
    return 0 if explained else 2


if __name__ == "__main__":
    raise SystemExit(main())
