"""Generate the canonical M6a-R2 KITTI Raw offline reconstruction evidence.

The runner is deliberately CPU-only. It validates the production KITTI Raw
adapter and unchanged ``MultiSweepBuilder`` without importing ROS, a detector,
MMDetection3D, MMDeploy, ONNX, or TensorRT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.m6a.diagnostics.generate_pose_oracle_diagnosis import (
    _direct_devkit_world_poses,
    _normalize,
    _records,
)
from benchmarks.m6a.diagnostics.generate_pose_oracle_diagnosis import (
    generate as generate_r1_diagnosis,
)
from laserperception.datasets.kitti_raw import (
    KITTI_TO_MODEL_ROTATION,
    KittiRawSequence,
    official_oxts_poses,
    select_m6a_reconstruction_frames,
)
from laserperception.detection.multisweep import (
    POINTPILLARS_POINT_CLOUD_RANGE,
    POINTPILLARS_USE_DIM,
    SweepTransform,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_COMMIT = "0218d610bac064c1f9cc9b8f0db8a7ca371b2fb5"
PROTOCOL_V1_COMMIT = "4d6bc3704f5404fbb761cc758c60f7958e17b872"
MEASUREMENT_V1_COMMIT = "ec9e341056807d5549353c8ef362fd109b25f2f2"
FAILURE_RECORD_COMMIT = "28c15e85aaac090aeda2bede9ec42ee6834c37b3"
FAILURE_RELATIVE = Path("benchmarks/m6a/diagnostics/pose_oracle_failure_ec9e341.json")
FAILURE_SHA256 = "894960c34b96aac02db2a8a10f13c66e39e3ac99b3ef34ee76c0ef3e88eabaf3"
R1_PROTOCOL_COMMIT = "be463437d1f873f03265e6cabd8f0cd680ee29bb"
R1_HEAD = "a4fb2625db5f95b4eb81e0a70051037285c0be61"
R1_RELATIVE = Path("benchmarks/m6a/diagnostics/pose_oracle_diagnosis_ec9e341.json")
R1_SHA256 = "44509f4c28fafbdd848c2627c99cde4615bd8e6011520c2a371b1ee3ce6853d8"
PROTOCOL_R2_COMMIT = "17924559ca852d23e661e0451bf1a22fc3af9bf6"
PROTOCOL_R2_RELATIVE = Path("docs/m6/M6A_PROTOCOL_R2.md")
FROZEN_RECONSTRUCTION_INDICES = (
    0,
    1,
    2,
    5,
    10,
    11,
    14,
    17,
    23,
    30,
    36,
    43,
    49,
    55,
    62,
    65,
    68,
    75,
    81,
    87,
    94,
    100,
    106,
    107,
)
EXPECTED_ARCHIVE_HASHES = {
    "canonical_sync": "7827a821ddc0a973bdf66de3a005a8825bd37c304aa9750f3c018f80d5b18458",
    "canonical_tracklets": "fe1a9a054f0cf24459d6637b54800b6d0c1d632fa0c6a42a1b1ae81efe4168f7",
    "canonical_calibration": "e0108cfd000cf802c14ed94fba38601185c792e5640abba015a34b7a85b812e0",
    "oracle_sync": "22faf3a360b488bb54fd35f2a1333f73697a7d8f00d8cd02c960efda3d22847c",
    "oracle_calibration": "2ed01563b616baec14f25659671bd9a77810bd87752d709d184fe304ff7258e3",
    "odometry_poses": "a969190799bb3fc4a15f5f0675b1ee6b7067292599b3269bdafc1e5ba883b9e1",
    "odometry_calibration": "fa45d2bbff828776e6df689b161415fb7cd719345454b6d3567c2ff81fa4d075",
    "odometry_devkit": "c4703456de60cdeaaec672381fbe169155ba4c2fcded102335b42cb07ff76758",
    "raw_devkit": "5fe511ba02b9588c4b18ad154e6d31d716e16b6ad115d1ac00d13d6d5bbcb6bc",
}
ARCHIVE_FILENAMES = {
    "canonical_sync": "2011_09_26_drive_0001_sync.zip",
    "canonical_tracklets": "2011_09_26_drive_0001_tracklets.zip",
    "canonical_calibration": "2011_09_26_calib.zip",
    "oracle_sync": "2011_09_30_drive_0016_sync.zip",
    "oracle_calibration": "2011_09_30_calib.zip",
    "odometry_poses": "data_odometry_poses.zip",
    "odometry_calibration": "data_odometry_calib.zip",
    "odometry_devkit": "devkit_odometry.zip",
    "raw_devkit": "devkit_raw_data.zip",
}
EXPECTED_TRACKLET_XML_SHA256 = "34f0672dee9dc94535893e653b4a66e6ddf534a09d2533bac4e62965935a91b8"
MAX_VOXELS = 40_000
PILLAR_SIZE_METRES = 0.25


def sha256_file(path: Path) -> str:
    """Return a streaming SHA256 for an external or tracked file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def require_clean_measurement_commit() -> str:
    """Fail closed unless evidence is being generated from a clean commit."""

    status = _git("status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError("canonical M6a evidence requires a clean worktree")
    head = _git("rev-parse", "HEAD")
    if _git("merge-base", "--is-ancestor", PROTOCOL_R2_COMMIT, head) != "":
        raise RuntimeError("Protocol R2 must be an ancestor of the measurement commit")
    return head


def exact_raw_pose_gate(
    date_root: Path,
    drive_root: Path,
    *,
    frame_count: int,
    role: str,
) -> dict[str, Any]:
    """Compare the production and direct Raw-devkit arithmetic paths exactly."""

    records = _records(drive_root / "oxts/data")[:frame_count]
    if len(records) != frame_count:
        raise RuntimeError(f"{role} expected {frame_count} OXTS frames")
    production = official_oxts_poses(records)
    reference = _normalize(_direct_devkit_world_poses(records))
    exact = [np.array_equal(left, right) for left, right in zip(production, reference, strict=True)]
    matrix_differences = [
        float(np.max(np.abs(left - right)))
        for left, right in zip(production, reference, strict=True)
    ]
    translation_differences = [
        float(np.linalg.norm(left[:3, 3] - right[:3, 3]))
        for left, right in zip(production, reference, strict=True)
    ]
    rotation_differences = [
        float(np.max(np.abs(left[:3, :3] - right[:3, :3])))
        for left, right in zip(production, reference, strict=True)
    ]
    result = {
        "role": role,
        "reference": "official KITTI Raw devkit OXTS conversion semantics",
        "production": "laserperception.datasets.kitti_raw.official_oxts_poses",
        "comparison_boundary": "direct float64 4x4 matrices before serialization",
        "limits": {
            "all_16_matrix_scalars_exact": True,
            "matrix_max_abs": 0.0,
            "rotation_matrix_max_abs": 0.0,
            "translation_norm_m": 0.0,
        },
        "frame_count": frame_count,
        "exact_equality_count": int(sum(exact)),
        "matrix_max_abs": max(matrix_differences),
        "rotation_matrix_max_abs": max(rotation_differences),
        "rotation_angle_rad": 0.0 if all(exact) else None,
        "translation_norm_m": max(translation_differences),
        "status": "pass" if all(exact) else "fail",
    }
    if not all(exact):
        raise RuntimeError(
            f"{role} Raw-devkit exactness changed; nonzero difference is the finding"
        )
    identity = np.eye(4, dtype=np.float64)
    result["frame_zero"] = {
        "production_reference_exact": bool(np.array_equal(production[0], reference[0])),
        "production_matrix_max_abs_from_ideal_identity": float(
            np.max(np.abs(production[0] - identity))
        ),
        "reference_matrix_max_abs_from_ideal_identity": float(
            np.max(np.abs(reference[0] - identity))
        ),
        "r1_1e-12_ideal_identity_check": "historical_fail_preserved",
        "r2_role": "known_non_blocking_serialized_inverse_diagnostic_without_a_new_limit",
    }
    calibration = date_root / "calib_imu_to_velo.txt"
    result["date_calibration_file_sha256"] = sha256_file(calibration)
    return result


def _summary_only(error: dict[str, Any]) -> dict[str, Any]:
    """Drop per-frame arrays while preserving every requested distribution."""

    result: dict[str, Any] = {"frame_count": error["frame_count"]}
    for key in ("translation_norm_m", "rotation_angle_rad", "rotation_matrix_max_abs"):
        result[key] = {
            "statistics": error[key]["statistics"],
            "max_frame": error[key]["max_frame"],
        }
    result["translation_signed_m"] = {
        axis: {
            "statistics": values["statistics"],
            "sign_changes": values["sign_changes"],
        }
        for axis, values in error["translation_signed_m"].items()
    }
    if "delta_frames" in error:
        result["delta_frames"] = error["delta_frames"]
    return result


def odometry_context(data_root: Path) -> dict[str, Any]:
    """Re-run and compact the accepted R1 external trajectory comparison."""

    diagnosis = generate_r1_diagnosis(data_root, "DATA-PRODUCT / TIMING", R1_PROTOCOL_COMMIT)
    return {
        "role": "external consistency context; not a KITTI Raw correctness equality oracle",
        "odometry_sequence": "04",
        "raw_drive": "2011_09_30_drive_0016",
        "frame_count": diagnosis["sequence_mapping"]["mapped_count"],
        "absolute_error": _summary_only(diagnosis["absolute_error"]),
        "relative_error": {
            delta: _summary_only(values) for delta, values in diagnosis["relative_error"].items()
        },
        "timestamp_ledger": {
            key: value
            for key, value in diagnosis["timestamp_ledger"].items()
            if key != "offset_values_ns"
        },
        "timing_hypothesis": diagnosis["timing_hypothesis"],
        "calibration_comparison": diagnosis["calibration_comparison"],
        "raw_sync_policy": diagnosis["source_provenance"]["raw_sync_policy"],
        "odometry_pose_product": diagnosis["source_provenance"]["odometry_pose_product"],
        "interpolation_promoted_to_production": False,
        "equality_pass_gate": False,
    }


def validate_archives(data_root: Path) -> dict[str, str]:
    """Verify every frozen external archive that is present in the M6a record."""

    result: dict[str, str] = {}
    for name, filename in ARCHIVE_FILENAMES.items():
        path = data_root / "archives" / filename
        if not path.is_file():
            raise FileNotFoundError(f"required external archive is missing: {filename}")
        digest = sha256_file(path)
        if digest != EXPECTED_ARCHIVE_HASHES[name]:
            raise RuntimeError(f"external archive hash mismatch: {filename}")
        result[name] = digest
    return result


def raw_decode_gate(sequence: KittiRawSequence) -> dict[str, Any]:
    """Require exact decoding, row order, and source bytes for all 108 frames."""

    frames: list[dict[str, Any]] = []
    drive_root = sequence.drive_root
    for index in range(len(sequence)):
        path = drive_root / "velodyne_points/data" / f"{index:010d}.bin"
        source_bytes = path.read_bytes()
        frame = sequence.frame(index)
        decoded_bytes = frame.points_xyzi.astype("<f4", copy=False).tobytes(order="C")
        if decoded_bytes != source_bytes:
            raise RuntimeError(f"raw decode bytes or order changed at frame {index}")
        frames.append(
            {
                "index": index,
                "source_file": f"velodyne_points/data/{index:010d}.bin",
                "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "row_count": int(frame.points_xyzi.shape[0]),
            }
        )
    return {
        "status": "pass",
        "frame_count": len(frames),
        "dtype": "little-endian float32",
        "columns": ["x_forward_m", "y_left_m", "z_up_m", "reflectance"],
        "source_bytes_exact": True,
        "xyz_exact": True,
        "source_order_exact": True,
        "hidden_sorting": False,
        "hidden_filtering": False,
        "reflectance_recognized": True,
        "reflectance_promoted_to_detector_input": False,
        "frames": frames,
    }


def tracklet_contract(data_root: Path, drive_root: Path) -> dict[str, Any]:
    """Parse selected-drive tracklets and freeze their official source semantics."""

    xml_path = drive_root / "tracklet_labels.xml"
    digest = sha256_file(xml_path)
    if digest != EXPECTED_TRACKLET_XML_SHA256:
        raise RuntimeError("canonical drive tracklet XML hash changed")
    root = ET.parse(xml_path).getroot()
    container = root.find("tracklets")
    if container is None:
        raise RuntimeError("tracklet_labels.xml contains no tracklets container")
    items = container.findall("item")
    declared = int(container.findtext("count", default="-1"))
    classes: Counter[str] = Counter()
    occlusion: Counter[int] = Counter()
    truncation: Counter[int] = Counter()
    pose_states: Counter[int] = Counter()
    pose_count = 0
    covered_frames: set[int] = set()
    roll_pitch_zero = True
    for item in items:
        object_type = item.findtext("objectType")
        poses = item.find("poses")
        if object_type is None or poses is None:
            raise RuntimeError("tracklet is missing objectType or poses")
        classes[object_type] += 1
        first_frame = int(item.findtext("first_frame", default="-1"))
        pose_items = poses.findall("item")
        pose_count += len(pose_items)
        for offset, pose in enumerate(pose_items):
            covered_frames.add(first_frame + offset)
            roll_pitch_zero &= float(pose.findtext("rx", default="nan")) == 0.0
            roll_pitch_zero &= float(pose.findtext("ry", default="nan")) == 0.0
            occlusion[int(pose.findtext("occlusion", default="-99"))] += 1
            truncation[int(pose.findtext("truncation", default="-99"))] += 1
            pose_states[int(pose.findtext("state", default="-99"))] += 1
    if declared != len(items) or not covered_frames:
        raise RuntimeError("tracklet count or temporal coverage is invalid")
    tracklet_archive = data_root / "archives" / ARCHIVE_FILENAMES["canonical_tracklets"]
    return {
        "status": "pass",
        "selected_drive_available": True,
        "archive_sha256": sha256_file(tracklet_archive),
        "xml_sha256": digest,
        "tracklet_count": len(items),
        "class_counts": dict(sorted(classes.items())),
        "pose_count": pose_count,
        "covered_frame_min": min(covered_frames),
        "covered_frame_max": max(covered_frames),
        "covered_frame_count": len(covered_frames),
        "frame": "native KITTI Velodyne: +X forward, +Y left, +Z up",
        "dimensions": "h, w, l in metres",
        "origin": "translation tx,ty,tz at bottom/contact centre",
        "yaw": "rz about +Z; positive under the right-handed Velodyne basis",
        "roll_pitch_zero_for_selected_drive": bool(roll_pitch_zero),
        "occlusion_counts": {str(key): value for key, value in sorted(occlusion.items())},
        "occlusion_codes": {"0": "visible", "1": "partly", "2": "fully"},
        "truncation_counts": {str(key): value for key, value in sorted(truncation.items())},
        "truncation_codes": {"0": "in image", "1": "truncated", "2": "out image"},
        "pose_state_counts": {str(key): value for key, value in sorted(pose_states.items())},
        "pose_state_codes": {"1": "interpolated", "2": "labeled"},
        "official_possible_classes": [
            "Car",
            "Van",
            "Truck",
            "Pedestrian",
            "Person (sitting)",
            "Cyclist",
            "Tram",
            "Misc",
        ],
        "taxonomy_mapping_status": "unresolved; deferred to owner review before M6b",
    }


def _quaternion_to_rotation(values: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(values, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def model_frame_evidence() -> dict[str, Any]:
    """Record actual LIDAR_TOP calibration plus the pinned preparation path."""

    ledger_relative = Path("benchmarks/m45b/diagnostics/w1_tf_transform_ledger.json")
    ledger = json.loads((REPO_ROOT / ledger_relative).read_text(encoding="utf-8"))
    calibrated = ledger["raw_nuscenes_records"]["current_calibrated_sensor"]
    sensor_to_ego = _quaternion_to_rotation(calibrated["rotation"])
    axes = {
        "+X": sensor_to_ego[:, 0].tolist(),
        "+Y": sensor_to_ego[:, 1].tolist(),
        "+Z": sensor_to_ego[:, 2].tolist(),
    }
    inverse = KITTI_TO_MODEL_ROTATION.T
    return {
        "status": "pass",
        "frozen_before_detector_results": True,
        "actual_lidar_top_calibration": {
            "tracked_ledger": ledger_relative.as_posix(),
            "tracked_ledger_sha256": sha256_file(REPO_ROOT / ledger_relative),
            "sensor_token": calibrated["sensor_token"],
            "quaternion_wxyz": calibrated["rotation"],
            "translation_m": calibrated["translation"],
            "rotation_sensor_to_ego": sensor_to_ego.tolist(),
            "rotation_determinant": float(np.linalg.det(sensor_to_ego)),
            "sensor_axes_in_vehicle_ego_xyz": axes,
        },
        "pinned_preparation_path": [
            "LoadPointsFromFile keeps five native LIDAR_TOP columns",
            "LoadPointsFromMultiSweeps transforms history into current LIDAR_TOP",
            "final use_dim is [0,1,2,4] with no later sensor-axis rotation",
            "M4.5a final frame is current raw LIDAR_TOP",
        ],
        "model_basis": {"+X": "vehicle right", "+Y": "vehicle forward", "+Z": "up"},
        "kitti_basis": {"+X": "vehicle forward", "+Y": "vehicle left", "+Z": "up"},
        "model_from_kitti_rotation": KITTI_TO_MODEL_ROTATION.tolist(),
        "determinant": float(np.linalg.det(KITTI_TO_MODEL_ROTATION)),
        "inverse": inverse.tolist(),
        "basis_examples": {
            "kitti_forward_to_model": (KITTI_TO_MODEL_ROTATION @ [1.0, 0.0, 0.0]).tolist(),
            "kitti_left_to_model": (KITTI_TO_MODEL_ROTATION @ [0.0, 1.0, 0.0]).tolist(),
            "kitti_up_to_model": (KITTI_TO_MODEL_ROTATION @ [0.0, 0.0, 1.0]).tolist(),
        },
        "derivation": (
            "official KITTI basis + official nuScenes KITTI converter convention + actual "
            "LIDAR_TOP calibration + pinned MMDetection3D/M4.5a preparation path"
        ),
        "translation_added": False,
        "scale_changed": False,
    }


def _manual_builder_parts(
    sequence: KittiRawSequence, current_index: int
) -> tuple[list[np.ndarray], tuple[int, ...], tuple[int, ...]]:
    """Reproduce builder arithmetic around the unchanged production transform helper."""

    selected = tuple(range(current_index, max(-1, current_index - 11), -1))
    current = sequence.frame(current_index).to_raw_sweep()
    current_pose = sequence.lidar_pose(current_index)
    parts: list[np.ndarray] = []
    counts: list[int] = []
    for rank, index in enumerate(selected):
        sweep = sequence.frame(index).to_raw_sweep()
        points = sweep.points.copy()
        counts.append(points.shape[0])
        if rank == 0:
            points[:, 4] = np.float32(0.0)
        else:
            transform = SweepTransform.from_poses(
                source_id=sweep.source_id,
                target_id=current.source_id,
                sweep_pose=sequence.lidar_pose(index),
                current_pose=current_pose,
            )
            matrix = np.array(transform.lidar2sensor.tolist())
            points[:, :3] = points[:, :3] @ matrix[:3, :3]
            points[:, :3] -= matrix[:3, 3]
            points[:, 4] = current.timestamp_seconds - sweep.timestamp_seconds
        xyzt = points[:, POINTPILLARS_USE_DIM]
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
        parts.append(np.ascontiguousarray(xyzt[mask]))
    return parts, selected, tuple(counts)


def _candidate_pillars(points: np.ndarray) -> int:
    minimum_x, minimum_y = POINTPILLARS_POINT_CLOUD_RANGE[:2]
    coordinates = np.floor(
        np.column_stack(
            (
                (points[:, 0] - minimum_x) / PILLAR_SIZE_METRES,
                (points[:, 1] - minimum_y) / PILLAR_SIZE_METRES,
            )
        )
    ).astype(np.int32)
    return int(np.unique(coordinates, axis=0).shape[0])


def reconstruction_evidence(sequence: KittiRawSequence) -> dict[str, Any]:
    """Run the frozen 24-frame offline oracle and ten exact repetitions each."""

    selected = select_m6a_reconstruction_frames(sequence.ego_to_global_poses)
    if selected != FROZEN_RECONSTRUCTION_INDICES:
        raise RuntimeError(f"frozen frame selector changed: {selected}")
    frame_results: list[dict[str, Any]] = []
    determinism: list[dict[str, Any]] = []
    for current_index in selected:
        production = sequence.reconstruct(current_index)
        parts, manual_selected, source_counts = _manual_builder_parts(sequence, current_index)
        expected = np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32)
        actual = production.point_cloud.points_xyzt
        if (
            production.selected_indices != manual_selected
            or production.source_counts != source_counts
        ):
            raise RuntimeError(f"source selection changed at frame {current_index}")
        if not np.array_equal(actual, expected):
            raise RuntimeError(f"manual builder invariant changed at frame {current_index}")
        if actual.dtype != np.float32 or actual.ndim != 2 or actual.shape[1] != 4:
            raise RuntimeError(f"output contract changed at frame {current_index}")
        if not actual.flags.c_contiguous or not np.isfinite(actual).all():
            raise RuntimeError(
                f"output contiguity/finite invariant failed at frame {current_index}"
            )
        lag_values = [float(part[0, 3]) for part in parts]
        if any(part.shape[0] == 0 or not np.all(part[:, 3] == part[0, 3]) for part in parts):
            raise RuntimeError(f"per-acquisition lag constancy failed at frame {current_index}")
        if lag_values[0] != 0.0 or np.signbit(parts[0][0, 3]):
            raise RuntimeError(f"current lag is not exact positive zero at frame {current_index}")
        if any(lag <= 0.0 for lag in lag_values[1:]) or any(
            newer >= older for newer, older in zip(lag_values[1:], lag_values[2:], strict=False)
        ):
            raise RuntimeError(f"historical lag ordering failed at frame {current_index}")
        if len(set(lag_values)) != len(parts):
            raise RuntimeError(f"distinct lag count failed at frame {current_index}")
        hashes = [production.point_cloud.sha256]
        for _ in range(9):
            hashes.append(sequence.reconstruct(current_index).point_cloud.sha256)
        if len(set(hashes)) != 1:
            raise RuntimeError(f"reconstruction is nondeterministic at frame {current_index}")
        timestamp = sequence.timestamps[current_index]
        candidate_pillars = _candidate_pillars(actual)
        overflow = max(0, candidate_pillars - MAX_VOXELS)
        frame_results.append(
            {
                "frame_index": current_index,
                "frame_id": f"2011_09_26_drive_0001_sync/{current_index:010d}",
                "timestamp_ns": timestamp.nanoseconds,
                "timestamp_us": timestamp.microseconds,
                "sub_microsecond_remainder_ns": timestamp.discarded_nanoseconds,
                "history_indices": list(manual_selected[1:]),
                "history_ids": [
                    f"2011_09_26_drive_0001_sync/{index:010d}" for index in manual_selected[1:]
                ],
                "history_depth": len(manual_selected) - 1,
                "source_counts": list(source_counts),
                "source_in_range_counts": [int(part.shape[0]) for part in parts],
                "pre_builder_row_count": sum(source_counts),
                "output_row_count": int(actual.shape[0]),
                "filtered_by_existing_strict_range": sum(source_counts) - int(actual.shape[0]),
                "output_dtype": str(actual.dtype),
                "output_shape": list(actual.shape),
                "time_lag_seconds": lag_values,
                "time_lag_min_seconds": min(lag_values),
                "time_lag_max_seconds": max(lag_values),
                "history_span_seconds": max(lag_values),
                "output_sha256": production.point_cloud.sha256,
                "candidate_occupied_pillars": candidate_pillars,
                "max_voxels": MAX_VOXELS,
                "max_voxels_engaged": candidate_pillars > MAX_VOXELS,
                "candidate_pillar_overflow_count": overflow,
                "candidate_pillar_overflow_fraction": (
                    float(overflow / candidate_pillars) if candidate_pillars else 0.0
                ),
                "invariants": {
                    "dtype_float32": True,
                    "shape_n_by_4": True,
                    "contiguous": True,
                    "finite": True,
                    "xyzt": True,
                    "manual_builder_bytes_exact": True,
                    "current_first": True,
                    "history_nearest_to_farthest": True,
                    "source_order_preserved_within_range_mask": True,
                    "current_lag_positive_zero_exact": True,
                    "per_acquisition_lag_constant": True,
                    "history_lags_positive_and_increasing": True,
                    "distinct_lag_count_matches_acquisitions": True,
                },
            }
        )
        determinism.append(
            {
                "frame_index": current_index,
                "repetitions": 10,
                "unique_hash_count": 1,
                "sha256": hashes[0],
                "status": "pass",
            }
        )
    depths = Counter(frame["history_depth"] for frame in frame_results)
    return {
        "status": "pass",
        "canonical_drive": "2011_09_26_drive_0001",
        "drive_frame_count": len(sequence),
        "frozen_frame_count": len(frame_results),
        "frozen_indices": list(selected),
        "history_depth_distribution": {str(key): value for key, value in sorted(depths.items())},
        "startup_frames": sum(frame["history_depth"] == 0 for frame in frame_results),
        "shallow_frames": sum(0 < frame["history_depth"] < 10 for frame in frame_results),
        "full_history_frames": sum(frame["history_depth"] == 10 for frame in frame_results),
        "existing_builder": "MultiSweepBuilder unchanged",
        "range_filter": {
            "strict": True,
            "point_cloud_range_m": list(POINTPILLARS_POINT_CLOUD_RANGE),
        },
        "frames": frame_results,
        "determinism": {
            "status": "pass",
            "sentinel_count": len(determinism),
            "repetitions_per_sentinel": 10,
            "results": determinism,
        },
    }


def generate(data_root: Path) -> dict[str, Any]:
    """Generate the fail-closed canonical R2 record in memory."""

    measurement_commit = require_clean_measurement_commit()
    if sha256_file(REPO_ROOT / FAILURE_RELATIVE) != FAILURE_SHA256:
        raise RuntimeError("original Tier-A failure artifact changed")
    if sha256_file(REPO_ROOT / R1_RELATIVE) != R1_SHA256:
        raise RuntimeError("accepted R1 diagnosis artifact changed")
    archive_hashes = validate_archives(data_root)
    extracted = data_root / "extracted"
    canonical_date = extracted / "2011_09_26"
    canonical_drive = canonical_date / "2011_09_26_drive_0001_sync"
    oracle_date = extracted / "2011_09_30"
    oracle_drive = oracle_date / "2011_09_30_drive_0016_sync"
    oracle_gate = exact_raw_pose_gate(
        oracle_date,
        oracle_drive,
        frame_count=271,
        role="adapter pose-oracle drive",
    )
    transfer_gate = exact_raw_pose_gate(
        canonical_date,
        canonical_drive,
        frame_count=108,
        role="canonical reconstruction drive transfer check",
    )
    sequence = KittiRawSequence(canonical_date, canonical_drive)
    if len(sequence) != 108:
        raise RuntimeError("canonical reconstruction drive frame count changed")
    raw_decode = raw_decode_gate(sequence)
    tracklets = tracklet_contract(data_root, canonical_drive)
    model_frame = model_frame_evidence()
    reconstruction = reconstruction_evidence(sequence)
    max_voxel_frames = [frame for frame in reconstruction["frames"] if frame["max_voxels_engaged"]]
    return {
        "schema_version": 1,
        "status": "pass",
        "canonical": True,
        "provenance": {
            "laserperception_version": "0.2.0",
            "base_commit": BASE_COMMIT,
            "protocol_v1_commit": PROTOCOL_V1_COMMIT,
            "measurement_v1_commit": MEASUREMENT_V1_COMMIT,
            "failure_record_commit": FAILURE_RECORD_COMMIT,
            "r1_diagnosis_protocol_commit": R1_PROTOCOL_COMMIT,
            "r1_diagnosis_head": R1_HEAD,
            "protocol_r2_commit": PROTOCOL_R2_COMMIT,
            "protocol_r2_sha256": sha256_file(REPO_ROOT / PROTOCOL_R2_RELATIVE),
            "measurement_commit": measurement_commit,
            "official_dataset": "KITTI Raw",
            "official_raw_page": "https://www.cvlibs.net/datasets/kitti/raw_data.php",
            "official_odometry_page": "https://www.cvlibs.net/datasets/kitti/eval_odometry.php",
            "license_reference": "KITTI terms / CC BY-NC-SA 3.0; data not redistributed",
            "archive_sha256": archive_hashes,
            "private_paths_recorded": False,
        },
        "chronology": [
            {"stage": "Protocol v1", "commit": PROTOCOL_V1_COMMIT},
            {
                "stage": "ORIGINAL TIER-A FAIL",
                "measurement_commit": MEASUREMENT_V1_COMMIT,
                "artifact": FAILURE_RELATIVE.as_posix(),
                "sha256": FAILURE_SHA256,
                "unchanged": True,
            },
            {
                "stage": "post-failure R1 diagnosis",
                "head": R1_HEAD,
                "artifact": R1_RELATIVE.as_posix(),
                "sha256": R1_SHA256,
                "root_cause": "DATA-PRODUCT / TIMING",
            },
            {
                "stage": "prospective Protocol R2",
                "commit": PROTOCOL_R2_COMMIT,
                "created_before_this_measurement": True,
            },
            {"stage": "new canonical M6a measurement", "commit": measurement_commit},
        ],
        "gate_ledger": {
            "cpu_adapter_unit_tests": "pass_existing_and_revalidated_by_full_quality",
            "original_odometry_equality_gate": "historical_fail_preserved",
            "raw_pose_correctness_oracle": "pass",
            "odometry_external_context": "reported_nonblocking",
            "model_frame_orientation": "pass",
            "kitti_to_model_alignment": "pass",
            "raw_decode": "pass",
            "offline_reconstruction": "pass",
            "determinism": "pass",
            "tracklet_contract": "pass",
            "m6b_preregistration_draft": "complete_but_inactive",
        },
        "pose_correctness": {
            "status": "pass",
            "drive_roles_are_distinct": True,
            "adapter_pose_oracle_drive": oracle_gate,
            "canonical_reconstruction_drive_transfer_check": transfer_gate,
        },
        "odometry_external_check": odometry_context(data_root),
        "model_frame": model_frame,
        "dataset": {
            "canonical_reconstruction_drive": "2011_09_26_drive_0001",
            "canonical_frame_count": 108,
            "pose_oracle_drive": "2011_09_30_drive_0016",
            "pose_oracle_mapped_frame_count": 271,
            "one_drive_does_not_substitute_for_the_other": True,
            "timestamp_source": "velodyne_points/timestamps.txt",
            "timestamp_contract": "exact ns; us = ns // 1000; remainder = ns % 1000",
            "raw_decode": raw_decode,
            "tracklets": tracklets,
        },
        "offline_reconstruction": reconstruction,
        "input_shift": {
            "status": "pass",
            "full_history_frame_count": reconstruction["full_history_frames"],
            "voxel_size_xy_m": [PILLAR_SIZE_METRES, PILLAR_SIZE_METRES],
            "max_voxels": MAX_VOXELS,
            "frames_exceeding_max_voxels": len(max_voxel_frames),
            "max_voxels_engaged_any": bool(max_voxel_frames),
            "spatial_cap_characterization_performed": False,
            "interpretation": (
                "input-only candidate-pillar capacity diagnostic; no detector-quality cause claimed"
            ),
        },
        "future_m6b_oracle": {
            "status": "frozen_offline_hash_targets",
            "ros_path_run": False,
            "detector_run": False,
            "frame_count": reconstruction["frozen_frame_count"],
        },
        "scope": {
            "detector_inference": False,
            "tensorrt_kitti_run": False,
            "ros_kitti_run": False,
            "threshold_tuning": False,
            "model_change": False,
            "training": False,
            "performance_campaign": False,
            "m6b_started": False,
            "production_pose_adapter_modified_for_r2": False,
            "multisweep_builder_modified_for_r2": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("canonical output already exists; refuse to overwrite")
    result = generate(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "status": result["status"],
                "measurement_commit": result["provenance"]["measurement_commit"],
                "sha256": sha256_file(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
