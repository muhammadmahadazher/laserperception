"""Run the preregistered frozen-detector KITTI Raw M6b characterization."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import statistics
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from laserperception.datasets.kitti_raw import KittiRawSequence, KittiReconstructionResult
from laserperception.detection.m1_assets import resolve_m1_asset_paths
from laserperception.detection.m2_assets import resolve_m2_asset_paths
from laserperception.detection.m2_backend import M2Backend
from laserperception.detection.mmdet3d_backend import sha256_file
from laserperception.detection.multisweep import MultiSweepBuilder, MultiSweepBuilderConfig
from laserperception.detection.runtime_metadata import repository_git_sha
from laserperception.detection.types import Detection3D, DetectionFrame
from laserperception.evaluation.kitti_m6b import (
    KittiReferenceCamera,
    KittiTrackletPose,
    M6bGroundTruthBox,
    bev_iou,
    convert_tracklet_pose,
    match_detections,
    model_box_corners,
    model_to_native_corners,
    native_box_corners,
    parse_kitti_tracklets,
    visible_in_reference_camera,
)
from laserperception.evaluation.m6b_input_oracle import reconstruct_from_frozen_transforms
from laserperception.evaluation.m6b_metrics import (
    RankedDisposition,
    all_points_average_precision,
    count_metrics,
    descriptive_statistics,
    drop_homogeneity_test,
    longest_consecutive_runs,
    wrapped_absolute_yaw_error,
)
from laserperception.evaluation.m6b_pillars import (
    PillarAudit,
    analyze_pillars,
    pillar_box_overlap_mask,
    pillar_centres,
    spatial_regions,
)

PROTOCOL_COMMIT = "16e2f7734061a5d0c2c2dec7b44f8b31e21591ae"
EXPECTED_CONFIG_PATH = "configs/m6/kitti_m6b.yaml"
EXPECTED_LEDGER_SHA256 = "2c41c9b21f9d30016ca22c46f75650e753cfe2a9b825077e715d65803610b480"
BASE_MAIN_COMMIT = "91fecf94dc5373c77d614b042e2db58cbe5f7063"
M6A_EVIDENCE_PATH = "benchmarks/m6a/results/kitti_raw_offline_reconstruction.json"
M6A_EVIDENCE_SHA256 = "a62da9808079994d971c4d47bbdf04f2c50d44ddcfd0c7958f7534512552155b"
M6A_MEASUREMENT_COMMIT = "1ab832df89109546abedc9f4e7f21c16c4cd0dca"
PROJECT_VERSION = "0.2.0"
RAW_NAMES = ("cls_score", "bbox_pred", "dir_cls_pred")
CONDITIONS = {"H10": 10, "H5": 5}
CLASSES = ("car", "pedestrian")
IOU_THRESHOLDS = (0.30, 0.50, 0.70)
OPERATING_SCORE = 0.25


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected mapping in {path.name}")
    return value


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _raw_arrays(raw: Mapping[str, list[Any]]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name in RAW_NAMES:
        values = raw[name]
        if len(values) != 1:
            raise RuntimeError(f"raw output {name} must contain one feature level")
        result[name] = values[0].detach().cpu().contiguous().numpy()
    return result


def _raw_hashes(raw: Mapping[str, list[Any]]) -> dict[str, str]:
    return {name: _array_sha256(array) for name, array in _raw_arrays(raw).items()}


def _raw_metadata(raw: Mapping[str, list[Any]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in RAW_NAMES:
        values = raw[name]
        if len(values) != 1:
            raise RuntimeError(f"raw output {name} must contain one feature level")
        tensor = values[0]
        if str(tensor.device) != "cuda:0":
            raise RuntimeError(f"raw output {name} is not on cuda:0")
        result[name] = {
            "device": str(tensor.device),
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
    return result


def _gt_geometry_fixture() -> dict[str, object]:
    pose = KittiTrackletPose(
        track_id=0,
        frame_index=0,
        object_type="Car",
        height=2.0,
        width=2.5,
        length=4.0,
        translation_xyz=(1.0, 2.0, 3.0),
        rotation_xyz=(0.0, 0.0, 0.0),
        state=2,
        occlusion=0,
        truncation=0,
    )
    converted = convert_tracklet_pose(pose)
    expected_center = np.asarray((-2.0, 1.0, 4.0))
    accepted = {
        "bottom_to_geometric_centre": bool(
            np.array_equal(np.asarray(converted.center_xyz), expected_center)
        ),
        "dimensions_hwl_to_lwh": converted.size_lwh == (4.0, 2.5, 2.0),
        "yaw_plus_pi_over_2": converted.yaw_rad == math.pi / 2.0,
        "wrong_yaw_identity_rejected": converted.yaw_rad != 0.0,
        "wrong_yaw_minus_pi_over_2_rejected": converted.yaw_rad != -math.pi / 2.0,
    }
    basis = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    recovered_native_centre = basis.T @ np.asarray(converted.center_xyz)
    expected_native_centre = np.asarray((1.0, 2.0, 4.0))
    inverse_error = float(np.max(np.abs(recovered_native_centre - expected_native_centre)))
    if not all(accepted.values()) or inverse_error != 0.0:
        raise RuntimeError("analytic GT geometry fixture failed")
    return {
        "passed": True,
        "checks": accepted,
        "inverse_max_absolute_error": inverse_error,
    }


def _clean_measurement_tree(root: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("M6b measurement requires a clean committed worktree")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, "HEAD"],
        cwd=root,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("measurement commit does not descend from the frozen protocol commit")


def _artifact(path: Path, expected: str) -> dict[str, object]:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"frozen artifact mismatch: {path.name}")
    return {"logical_name": path.name, "sha256": actual, "size_bytes": path.stat().st_size}


def _gpu_record() -> dict[str, object]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,pci.bus_id,temperature.gpu,pstate,clocks.sm,clocks.mem,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    fields = [value.strip() for value in query.split(",")]
    names = (
        "name",
        "driver_version",
        "pci_bus_id",
        "temperature_c",
        "performance_state",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "power_draw_w",
        "power_limit_w",
    )
    return dict(zip(names, fields, strict=True))


def _eligible_poses(
    poses: Sequence[KittiTrackletPose],
    camera: KittiReferenceCamera,
) -> tuple[KittiTrackletPose, ...]:
    return tuple(
        pose
        for pose in poses
        if pose.valid_labelled_pose
        and pose.evaluation_role in {"target", "neighbour_ignore"}
        and visible_in_reference_camera(native_box_corners(pose), camera)
    )


def _prediction_fov(
    predictions: Sequence[Detection3D], camera: KittiReferenceCamera
) -> tuple[tuple[Detection3D, ...], tuple[Detection3D, ...]]:
    inside: list[Detection3D] = []
    outside: list[Detection3D] = []
    for prediction in predictions:
        native = model_to_native_corners(model_box_corners(prediction))
        (inside if visible_in_reference_camera(native, camera) else outside).append(prediction)
    return tuple(inside), tuple(outside)


def _source_ignore_reason(
    prediction: Detection3D,
    ignores: Sequence[M6bGroundTruthBox],
    threshold: float,
) -> str:
    candidates = sorted(
        ((bev_iou(prediction, box), box.source_type) for box in ignores),
        key=lambda item: (-item[0], item[1]),
    )
    if not candidates or candidates[0][0] < threshold:
        return "unresolved"
    return {
        "Van": "ignored_van",
        "Person (sitting)": "ignored_person_sitting",
    }.get(candidates[0][1], f"ignored_{candidates[0][1].lower().replace(' ', '_')}")


def _class_evaluation(
    frame_id: str,
    class_name: str,
    inside_predictions: Sequence[Detection3D],
    outside_predictions: Sequence[Detection3D],
    poses: Sequence[KittiTrackletPose],
    audit: PillarAudit,
    track_labelled_counts: Mapping[tuple[str, int], int],
) -> dict[str, object]:
    drive_id = frame_id.split("/", 1)[0]
    frame_index = int(frame_id.rsplit("/", 1)[1])
    target_poses = [
        pose
        for pose in poses
        if pose.evaluation_role == "target" and pose.evaluation_class == class_name
    ]
    ignore_poses = [
        pose
        for pose in poses
        if pose.evaluation_role == "neighbour_ignore" and pose.evaluation_class == class_name
    ]
    targets = tuple(convert_tracklet_pose(pose) for pose in target_poses)
    ignores = tuple(convert_tracklet_pose(pose) for pose in ignore_poses)
    thresholds: dict[str, object] = {}
    primary_summary = None
    for threshold in IOU_THRESHOLDS:
        summary = match_detections(
            inside_predictions,
            targets,
            ignores,
            class_name=class_name,
            iou_threshold=threshold,
            score_threshold=OPERATING_SCORE,
        )
        metrics = count_metrics(
            summary.true_positives, summary.false_positives, summary.false_negatives
        )
        thresholds[f"{threshold:.2f}"] = {
            **metrics,
            "ignored_predictions": summary.ignored_predictions,
        }
        if threshold == 0.50:
            primary_summary = summary
    assert primary_summary is not None

    pose_by_track = {pose.track_id: pose for pose in target_poses}
    matched_by_track: dict[int, tuple[Detection3D, float]] = {}
    ignored_reasons: dict[str, int] = defaultdict(int)
    for record in primary_summary.records:
        prediction = inside_predictions[record.prediction_index]
        if record.disposition == "true_positive" and record.gt_track_id is not None:
            matched_by_track[record.gt_track_id] = (prediction, record.bev_iou)
        elif record.disposition == "ignored_neighbour":
            ignored_reasons[_source_ignore_reason(prediction, ignores, 0.50)] += 1

    target_observations: list[dict[str, object]] = []
    for box in targets:
        pose = pose_by_track[box.track_id]
        matched = matched_by_track.get(box.track_id)
        candidate_mask = pillar_box_overlap_mask(
            audit.candidate_xy_indices,
            center_xy=box.center_xyz[:2],
            size_lw=box.size_lwh[:2],
            yaw_rad=box.yaw_rad,
        )
        discarded_mask = pillar_box_overlap_mask(
            audit.discarded_xy_indices,
            center_xy=box.center_xyz[:2],
            size_lw=box.size_lwh[:2],
            yaw_rad=box.yaw_rad,
        )
        retained_mask = pillar_box_overlap_mask(
            audit.retained_xy_indices,
            center_xy=box.center_xyz[:2],
            size_lw=box.size_lwh[:2],
            yaw_rad=box.yaw_rad,
        )
        candidate_inside = int(candidate_mask.sum())
        retained_inside = int(retained_mask.sum())
        discarded_inside = int(discarded_mask.sum())
        observation: dict[str, object] = {
            "object_key": f"{drive_id}/track_{box.track_id}",
            "track_id": box.track_id,
            "frame_index": frame_index,
            "track_labelled_frame_count": track_labelled_counts[(drive_id, box.track_id)],
            "source_type": box.source_type,
            "center_xyz": list(box.center_xyz),
            "range_forward_m": box.center_xyz[1],
            "occlusion": pose.occlusion,
            "truncation": pose.truncation,
            "matched": matched is not None,
            "candidate_pillars_in_GT": candidate_inside,
            "retained_pillars_in_GT": retained_inside,
            "discarded_pillars_in_GT": discarded_inside,
            "discarded_fraction_in_GT": (
                discarded_inside / candidate_inside if candidate_inside else 0.0
            ),
        }
        if matched is not None:
            prediction, iou = matched
            delta = np.asarray(prediction.center_xyz) - np.asarray(box.center_xyz)
            observation.update(
                {
                    "prediction_score": prediction.score,
                    "bev_iou": iou,
                    "centre_error_3d_m": float(np.linalg.norm(delta)),
                    "centre_error_bev_m": float(np.linalg.norm(delta[:2])),
                    "absolute_yaw_error_rad": wrapped_absolute_yaw_error(
                        prediction.yaw_rad, box.yaw_rad
                    ),
                }
            )
        target_observations.append(observation)

    ranked_summary = match_detections(
        inside_predictions,
        targets,
        ignores,
        class_name=class_name,
        iou_threshold=0.50,
        score_threshold=0.0,
    )
    ranked = [
        {
            "score": record.score,
            "frame_id": frame_id,
            "prediction_index": record.prediction_index,
            "true_positive": record.disposition == "true_positive",
        }
        for record in ranked_summary.records
        if record.disposition != "ignored_neighbour"
    ]
    primary = thresholds["0.50"]
    assert isinstance(primary, dict)
    inside_operating = sum(
        prediction.class_name == class_name and prediction.score >= OPERATING_SCORE
        for prediction in inside_predictions
    )
    outside_operating = sum(
        prediction.class_name == class_name and prediction.score >= OPERATING_SCORE
        for prediction in outside_predictions
    )
    return {
        "eligible_GT_count": len(targets),
        "inside_FOV_prediction_count_score_0_25": inside_operating,
        "outside_annotation_fov_predictions_score_0_25": outside_operating,
        "neighbour_ignore_GT_count": len(ignores),
        "thresholds": thresholds,
        "primary": {
            **primary,
            "ignored_by_reason": dict(sorted(ignored_reasons.items())),
            "median_matched_iou": (
                statistics.median(
                    observation["bev_iou"]
                    for observation in target_observations
                    if observation["matched"]
                )
                if any(observation["matched"] for observation in target_observations)
                else None
            ),
        },
        "target_observations": target_observations,
        "ranked_dispositions": ranked,
    }


def _pillar_spatial(audit: PillarAudit) -> dict[str, object]:
    candidate_regions = spatial_regions(audit.candidate_xy_indices)
    discarded_regions = spatial_regions(audit.discarded_xy_indices)

    def counts(values: np.ndarray, size: int) -> list[int]:
        return [int(value) for value in np.bincount(values, minlength=size)]

    discarded_centres = pillar_centres(audit.discarded_xy_indices)
    bounding_box = (
        {
            "minimum_xy": [float(value) for value in discarded_centres.min(axis=0)],
            "maximum_xy": [float(value) for value in discarded_centres.max(axis=0)],
        }
        if len(discarded_centres)
        else None
    )
    return {
        "candidate": {
            "azimuth_sector": counts(candidate_regions["azimuth_sector"], 12),
            "cartesian_quadrant": counts(candidate_regions["cartesian_quadrant"], 4),
            "radial_bin": counts(candidate_regions["radial_bin"], 3),
        },
        "discarded": {
            "azimuth_sector": counts(discarded_regions["azimuth_sector"], 12),
            "cartesian_quadrant": counts(discarded_regions["cartesian_quadrant"], 4),
            "radial_bin": counts(discarded_regions["radial_bin"], 3),
        },
        "discarded_spatial_bounding_box": bounding_box,
    }


def _reconstruct_input(
    sequence: KittiRawSequence,
    frame_index: int,
    history: int,
    expected: Mapping[str, object],
) -> KittiReconstructionResult:
    records = expected.get("frozen_sweep_transforms")
    if not isinstance(records, list):
        raise RuntimeError("input ledger frozen transform records are malformed")
    validated_records: list[Mapping[str, object]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise RuntimeError("input ledger frozen transform records are malformed")
        validated_records.append(record)
    return reconstruct_from_frozen_transforms(
        sequence,
        frame_index,
        validated_records,
        builder=MultiSweepBuilder(MultiSweepBuilderConfig(max_historical_sweeps=history)),
    )


def _verify_input_oracles(
    sequences: Mapping[str, KittiRawSequence],
    ledger_frames: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    verified_conditions = 0
    for position, expected in enumerate(ledger_frames, start=1):
        frame_id = str(expected["frame_id"])
        drive, raw_index = frame_id.split("/", 1)
        for condition, history in CONDITIONS.items():
            reconstruction = _reconstruct_input(
                sequences[drive],
                int(raw_index),
                history,
                expected,
            )
            points = reconstruction.point_cloud.points_xyzt
            expected_condition = expected[condition.lower()]
            if not isinstance(expected_condition, Mapping):
                raise RuntimeError("input ledger condition is malformed")
            if _array_sha256(points) != expected_condition["model_ready_sha256"]:
                raise RuntimeError(f"model-ready input hash mismatch at {frame_id} {condition}")
            if int(len(points)) != expected_condition["point_count"]:
                raise RuntimeError(f"model-ready point-count mismatch at {frame_id} {condition}")
            if list(reconstruction.selected_indices) != expected_condition["selected_indices"]:
                raise RuntimeError(f"history ID mismatch at {frame_id} {condition}")
            actual_lags = [float(value) for value in np.unique(points[:, 3])]
            if actual_lags != expected_condition["time_lag_values"]:
                raise RuntimeError(f"time-lag mismatch at {frame_id} {condition}")
            verified_conditions += 1
        if position == 1 or position % 50 == 0 or position == len(ledger_frames):
            print(f"verified input oracles for {position}/{len(ledger_frames)} frames", flush=True)
    return {
        "passed": True,
        "frame_count": len(ledger_frames),
        "condition_count": verified_conditions,
        "conditions": list(CONDITIONS),
        "detector_inference_performed": False,
    }


def _run_inference(
    backend: M2Backend,
    engine: Path,
    sequence: KittiRawSequence,
    frame_index: int,
    history: int,
    expected: Mapping[str, object],
) -> tuple[DetectionFrame, dict[str, object], PillarAudit, dict[str, list[Any]]]:
    reconstruction = _reconstruct_input(sequence, frame_index, history, expected)
    points = reconstruction.point_cloud.points_xyzt
    condition = f"h{history}"
    expected_condition = expected[condition]
    if not isinstance(expected_condition, Mapping):
        raise RuntimeError("input ledger condition is malformed")
    actual_input_hash = _array_sha256(points)
    if actual_input_hash != expected_condition["model_ready_sha256"]:
        raise RuntimeError(f"model-ready input hash mismatch at {expected['frame_id']} {condition}")
    if list(reconstruction.selected_indices) != expected_condition["selected_indices"]:
        raise RuntimeError(f"history ID mismatch at {expected['frame_id']} {condition}")
    audit = analyze_pillars(points)
    expected_pillars = expected_condition["pillars"]
    if not isinstance(expected_pillars, Mapping):
        raise RuntimeError("input ledger pillar record is malformed")
    if audit.candidate_count != expected_pillars["candidate_occupied_pillars"]:
        raise RuntimeError(f"candidate pillar mismatch at {expected['frame_id']} {condition}")
    prepared = backend.prepare_model_ready_points(
        reconstruction.point_cloud,
        sample_id=str(expected["frame_id"]),
        coordinate_frame="kitti_model_aligned_lidar",
    )
    voxelized = backend.voxelize(prepared)
    if voxelized.voxel_count != audit.retained_count:
        raise RuntimeError(f"exact_fast retained pillar mismatch at {expected['frame_id']}")
    shared_cuda_inputs = backend.assert_shared_cuda_inputs(voxelized)
    raw = backend.run_tensorrt_raw(voxelized, engine)
    frame = backend.postprocess_raw(
        raw,
        voxelized,
        backend_name="tensorrt",
        precision="fp16",
        provenance_mode="full",
    )
    execution = {
        "model_ready_sha256": actual_input_hash,
        "point_count": int(len(points)),
        "history_indices": list(reconstruction.selected_indices),
        "time_lag_values": [float(value) for value in np.unique(points[:, 3])],
        "voxel_count": voxelized.voxel_count,
        "voxel_hashes": voxelized.hashes(),
        "shared_cuda_inputs": shared_cuda_inputs,
        "raw_output_hashes": _raw_hashes(raw),
        "raw_output_tensors": _raw_metadata(raw),
        "detection_frame_sha256": _json_sha256(frame.to_dict()),
        "detection_count_all_postprocessed_scores": len(frame.detections),
        "pillars": {
            **audit.summary(),
            "spatial": _pillar_spatial(audit),
            "coordinate_hashes": {
                "candidate_xy": _array_sha256(audit.candidate_xy_indices),
                "retained_xy": _array_sha256(audit.retained_xy_indices),
                "discarded_xy": _array_sha256(audit.discarded_xy_indices),
            },
        },
    }
    return frame, execution, audit, raw


def _repeatability(
    backend: M2Backend,
    engine: Path,
    sequences: Mapping[str, KittiRawSequence],
    ledger_by_id: Mapping[str, Mapping[str, object]],
    sentinels: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    results = []
    for sentinel in sentinels:
        frame_id = str(sentinel["frame_id"])
        drive, raw_index = frame_id.split("/", 1)
        raw_hashes: list[dict[str, str]] = []
        frame_hashes: list[str] = []
        for repetition in range(10):
            frame, execution, _, _ = _run_inference(
                backend,
                engine,
                sequences[drive],
                int(raw_index),
                10,
                ledger_by_id[frame_id],
            )
            raw_hashes.append(execution["raw_output_hashes"])
            frame_hashes.append(_json_sha256(frame.to_dict()))
            print(f"repeat {frame_id} {repetition + 1}/10", flush=True)
        raw_exact = all(value == raw_hashes[0] for value in raw_hashes)
        frame_exact = len(set(frame_hashes)) == 1
        results.append(
            {
                "role": sentinel["role"],
                "frame_id": frame_id,
                "repetitions": 10,
                "raw_output_hashes": raw_hashes,
                "detection_frame_hashes": frame_hashes,
                "raw_outputs_exact": raw_exact,
                "detection_frames_exact": frame_exact,
            }
        )
        if not raw_exact or not frame_exact:
            return {
                "passed": False,
                "sentinel_count": len(results),
                "samples": results,
            }
    return {"passed": True, "sentinel_count": len(results), "samples": results}


def _condition_run(
    condition: str,
    backend: M2Backend,
    engine: Path,
    sequences: Mapping[str, KittiRawSequence],
    poses_by_frame: Mapping[str, Sequence[KittiTrackletPose]],
    camera: KittiReferenceCamera,
    ledger_frames: Sequence[Mapping[str, object]],
    track_labelled_counts: Mapping[tuple[str, int], int],
) -> list[dict[str, object]]:
    history = CONDITIONS[condition]
    results: list[dict[str, object]] = []
    for rank, expected in enumerate(ledger_frames, start=1):
        frame_id = str(expected["frame_id"])
        drive, raw_index = frame_id.split("/", 1)
        frame, execution, audit, _ = _run_inference(
            backend,
            engine,
            sequences[drive],
            int(raw_index),
            history,
            expected,
        )
        eligible = _eligible_poses(poses_by_frame.get(frame_id, ()), camera)
        inside, outside = _prediction_fov(frame.detections, camera)
        class_results = {
            class_name: _class_evaluation(
                frame_id,
                class_name,
                inside,
                outside,
                eligible,
                audit,
                track_labelled_counts,
            )
            for class_name in CLASSES
        }
        results.append(
            {
                "frame_id": frame_id,
                "frame_index": int(raw_index),
                "condition": condition,
                "execution": execution,
                "outside_annotation_fov_predictions_all_classes": len(outside),
                "classes": class_results,
                "_detections": frame.detections,
            }
        )
        if rank == 1 or rank % 25 == 0 or rank == len(ledger_frames):
            print(f"{condition} {rank}/{len(ledger_frames)} {frame_id}", flush=True)
    return results


def _aggregate_class(frames: Sequence[Mapping[str, object]], class_name: str) -> dict[str, object]:
    class_frames = [frame["classes"][class_name] for frame in frames]
    thresholds: dict[str, object] = {}
    for threshold in ("0.30", "0.50", "0.70"):
        tp = sum(int(frame["thresholds"][threshold]["true_positives"]) for frame in class_frames)
        fp = sum(int(frame["thresholds"][threshold]["false_positives"]) for frame in class_frames)
        fn = sum(int(frame["thresholds"][threshold]["false_negatives"]) for frame in class_frames)
        ignored = sum(
            int(frame["thresholds"][threshold]["ignored_predictions"]) for frame in class_frames
        )
        thresholds[threshold] = {**count_metrics(tp, fp, fn), "ignored_predictions": ignored}
    observations = [
        observation for frame in class_frames for observation in frame["target_observations"]
    ]
    matched = [observation for observation in observations if observation["matched"]]
    ignored_by_reason: dict[str, int] = defaultdict(int)
    for frame in class_frames:
        for reason, count in frame["primary"]["ignored_by_reason"].items():
            ignored_by_reason[str(reason)] += int(count)
    ranked = [
        RankedDisposition(**record)
        for frame in class_frames
        for record in frame["ranked_dispositions"]
    ]
    range_slices = []
    for lower, upper in ((0.0, 20.0), (20.0, 35.0), (35.0, 50.0)):
        selected = [
            item
            for item in observations
            if lower <= float(item["range_forward_m"]) < upper
            or (upper == 50.0 and float(item["range_forward_m"]) == upper)
        ]
        hits = sum(bool(item["matched"]) for item in selected)
        range_slices.append(
            {
                "range_m": [lower, upper],
                "eligible_GT": len(selected),
                "true_positives": hits,
                "false_negatives": len(selected) - hits,
                "recall": hits / len(selected) if selected else None,
            }
        )
    strata: dict[str, object] = {}
    for field in ("occlusion", "truncation"):
        field_result = []
        for value in (0, 1):
            selected = [item for item in observations if int(item[field]) == value]
            hits = sum(bool(item["matched"]) for item in selected)
            field_result.append(
                {
                    "value": value,
                    "denominator": len(selected),
                    "reported": len(selected) >= 10,
                    "recall": hits / len(selected) if len(selected) >= 10 else None,
                }
            )
        strata[field] = field_result

    tracks: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for item in observations:
        tracks[str(item["object_key"])].append(item)
    track_results = []
    for key, items in sorted(tracks.items()):
        ordered = sorted(items, key=lambda item: int(item["frame_index"]))
        frame_indices = [int(item["frame_index"]) for item in ordered]
        detected = [bool(item["matched"]) for item in ordered]
        longest_hit, longest_miss = longest_consecutive_runs(frame_indices, detected)
        scores = [float(item["prediction_score"]) for item in ordered if item["matched"]]
        ranges = [float(item["range_forward_m"]) for item in ordered]
        track_results.append(
            {
                "object_key": key,
                "labelled_frame_count": int(ordered[0]["track_labelled_frame_count"]),
                "eligible_eval_frame_count": len(ordered),
                "detected_frames": sum(detected),
                "detection_continuity_fraction": sum(detected) / len(ordered),
                "longest_consecutive_detected_run": longest_hit,
                "longest_consecutive_miss_run": longest_miss,
                "matched_score_median": statistics.median(scores) if scores else None,
                "forward_range_span_m": [min(ranges), max(ranges)],
            }
        )

    def group_metrics(selected: Sequence[Mapping[str, object]]) -> dict[str, object]:
        summaries = [frame["classes"][class_name]["primary"] for frame in selected]
        tp = sum(int(item["true_positives"]) for item in summaries)
        fp = sum(int(item["false_positives"]) for item in summaries)
        fn = sum(int(item["false_negatives"]) for item in summaries)
        ignored = sum(int(item["ignored_predictions"]) for item in summaries)
        return {
            "frame_count": len(selected),
            **count_metrics(tp, fp, fn),
            "ignored_predictions": ignored,
        }

    overflow_frames = [frame for frame in frames if frame["execution"]["pillars"]["overflow"]]
    non_overflow_frames = [
        frame for frame in frames if not frame["execution"]["pillars"]["overflow"]
    ]
    return {
        "eligible_GT_count": len(observations),
        "inside_FOV_prediction_count_score_0_25": sum(
            int(frame["inside_FOV_prediction_count_score_0_25"]) for frame in class_frames
        ),
        "outside_annotation_fov_predictions_score_0_25": sum(
            int(frame["outside_annotation_fov_predictions_score_0_25"]) for frame in class_frames
        ),
        "neighbour_ignore_GT_count": sum(
            int(frame["neighbour_ignore_GT_count"]) for frame in class_frames
        ),
        "ignored_predictions_by_reason": dict(sorted(ignored_by_reason.items())),
        "thresholds": thresholds,
        "primary_matched_distributions": {
            "BEV_IoU": descriptive_statistics([float(item["bev_iou"]) for item in matched]),
            "centre_error_3d_m": descriptive_statistics(
                [float(item["centre_error_3d_m"]) for item in matched]
            ),
            "centre_error_bev_m": descriptive_statistics(
                [float(item["centre_error_bev_m"]) for item in matched]
            ),
            "absolute_yaw_error_rad": descriptive_statistics(
                [float(item["absolute_yaw_error_rad"]) for item in matched]
            ),
            "score": descriptive_statistics([float(item["prediction_score"]) for item in matched]),
        },
        "score_ranked_PR": all_points_average_precision(
            ranked, ground_truth_count=len(observations)
        ),
        "range_slices": range_slices,
        "occlusion_truncation_strata": strata,
        "track_level": track_results,
        "overflow_association": {
            "overflow_frames": group_metrics(overflow_frames),
            "non_overflow_frames": group_metrics(non_overflow_frames),
        },
        "targets_overlapping_retained_pillars": sum(
            int(item["retained_pillars_in_GT"]) > 0 for item in observations
        ),
        "targets_overlapping_discarded_pillars": sum(
            int(item["discarded_pillars_in_GT"]) > 0 for item in observations
        ),
    }


def _sum_spatial_counts(
    frames: Sequence[Mapping[str, object]],
    population: str,
    region: str,
    size: int,
) -> list[int]:
    if not frames:
        return [0] * size
    values = [frame["execution"]["pillars"]["spatial"][population][region] for frame in frames]
    return np.sum(values, axis=0, dtype=np.int64).tolist()


def _sum_first_touch_histograms(
    frames: Sequence[Mapping[str, object]], population: str
) -> dict[str, int]:
    result: dict[int, int] = defaultdict(int)
    for frame in frames:
        histogram = frame["execution"]["pillars"]["first_touch_sweep_histogram"][population]
        for key, value in histogram.items():
            result[int(key)] += int(value)
    return {str(key): result[key] for key in sorted(result)}


def _aggregate_pillars(frames: Sequence[Mapping[str, object]]) -> dict[str, object]:
    overflow = [frame for frame in frames if frame["execution"]["pillars"]["overflow"]]
    candidate = (
        np.sum(
            [
                frame["execution"]["pillars"]["spatial"]["candidate"]["azimuth_sector"]
                for frame in overflow
            ],
            axis=0,
            dtype=np.int64,
        )
        if overflow
        else np.zeros(12, dtype=np.int64)
    )
    discarded = (
        np.sum(
            [
                frame["execution"]["pillars"]["spatial"]["discarded"]["azimuth_sector"]
                for frame in overflow
            ],
            axis=0,
            dtype=np.int64,
        )
        if overflow
        else np.zeros(12, dtype=np.int64)
    )
    test = drop_homogeneity_test(candidate, discarded)
    rates = np.divide(discarded, candidate, out=np.zeros(12), where=candidate > 0)
    nonzero = rates[rates > 0]
    ratio = float(rates.max() / nonzero.min()) if len(nonzero) else None
    aggregate_sector = int(np.argmax(rates)) if len(overflow) else None
    matching = 0
    for frame in overflow:
        frame_candidate = np.asarray(
            frame["execution"]["pillars"]["spatial"]["candidate"]["azimuth_sector"]
        )
        frame_discarded = np.asarray(
            frame["execution"]["pillars"]["spatial"]["discarded"]["azimuth_sector"]
        )
        frame_rates = np.divide(
            frame_discarded,
            frame_candidate,
            out=np.zeros(12, dtype=np.float64),
            where=frame_candidate > 0,
        )
        matching += int(int(np.argmax(frame_rates)) == aggregate_sector)
    consistency = matching / len(overflow) if overflow else None
    practical = bool(
        ratio is not None and ratio >= 2.0 and consistency is not None and consistency > 0.5
    )

    def region_summary(region: str, size: int) -> dict[str, object]:
        region_candidate = np.asarray(
            _sum_spatial_counts(overflow, "candidate", region, size), dtype=np.int64
        )
        region_discarded = np.asarray(
            _sum_spatial_counts(overflow, "discarded", region, size), dtype=np.int64
        )
        region_rates = np.divide(
            region_discarded,
            region_candidate,
            out=np.zeros(size, dtype=np.float64),
            where=region_candidate > 0,
        )
        return {
            "candidate_counts": region_candidate.tolist(),
            "discarded_counts": region_discarded.tolist(),
            "drop_fractions": region_rates.tolist(),
        }

    return {
        "frame_count": len(frames),
        "overflow_frame_count": len(overflow),
        "overflow_frame_fraction": len(overflow) / len(frames),
        "candidate_pillar_count": descriptive_statistics(
            [float(frame["execution"]["pillars"]["candidate_occupied_pillars"]) for frame in frames]
        ),
        "discarded_pillar_count": descriptive_statistics(
            [float(frame["execution"]["pillars"]["discarded_pillars"]) for frame in frames]
        ),
        "aggregate_overflow_sector_candidate_counts": candidate.tolist(),
        "aggregate_overflow_sector_discarded_counts": discarded.tolist(),
        "aggregate_sector_drop_rates": rates.tolist(),
        "homogeneity_test": test,
        "largest_to_smallest_nonzero_drop_rate_ratio": ratio,
        "aggregate_highest_drop_sector": aggregate_sector,
        "directional_consistency_fraction": consistency,
        "practical_concentration_gate": practical,
        "spatially_non_uniform_truncation": bool(test["reject_homogeneity"] and practical),
        "maximum_discarded_pillars": max(
            (int(frame["execution"]["pillars"]["discarded_pillars"]) for frame in frames),
            default=0,
        ),
        "overflow_spatial": {
            "azimuth_sector": region_summary("azimuth_sector", 12),
            "cartesian_quadrant": region_summary("cartesian_quadrant", 4),
            "radial_bin": region_summary("radial_bin", 3),
        },
        "first_touch_sweep_histogram": {
            population: _sum_first_touch_histograms(frames, population)
            for population in ("candidate", "retained", "discarded")
        },
    }


def _aggregate_inputs(frames: Sequence[Mapping[str, object]]) -> dict[str, object]:
    executions = [frame["execution"] for frame in frames]
    lag_values = [
        [float(value) for value in execution["time_lag_values"]] for execution in executions
    ]
    return {
        "point_count": descriptive_statistics(
            [float(execution["point_count"]) for execution in executions]
        ),
        "in_range_point_count": descriptive_statistics(
            [float(execution["pillars"]["in_range_points"]) for execution in executions]
        ),
        "history_sweep_count": descriptive_statistics(
            [float(len(execution["history_indices"]) - 1) for execution in executions]
        ),
        "distinct_time_lag_count": descriptive_statistics(
            [float(len(values)) for values in lag_values]
        ),
        "time_lag_range_seconds": [
            min(min(values) for values in lag_values),
            max(max(values) for values in lag_values),
        ],
        "per_frame_time_span_seconds": descriptive_statistics(
            [max(values) - min(values) for values in lag_values]
        ),
        "voxel_count": descriptive_statistics(
            [float(execution["voxel_count"]) for execution in executions]
        ),
    }


def _aggregate_condition(frames: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "frame_count": len(frames),
        "input_summary": _aggregate_inputs(frames),
        "classes": {class_name: _aggregate_class(frames, class_name) for class_name in CLASSES},
        "pillar_cap": _aggregate_pillars(frames),
    }


def _paired_analysis(
    h10: Sequence[Mapping[str, object]], h5: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    h5_by_id = {str(frame["frame_id"]): frame for frame in h5}
    class_changes: dict[str, object] = {}
    car_frame_deltas = []
    for class_name in CLASSES:
        transitions: dict[str, int] = defaultdict(int)
        for first in h10:
            second = h5_by_id[str(first["frame_id"])]
            first_obs = {
                str(item["object_key"]): bool(item["matched"])
                for item in first["classes"][class_name]["target_observations"]
            }
            second_obs = {
                str(item["object_key"]): bool(item["matched"])
                for item in second["classes"][class_name]["target_observations"]
            }
            if first_obs.keys() != second_obs.keys():
                raise RuntimeError("paired H10/H5 GT sets differ")
            for key in first_obs:
                transitions[
                    {
                        (False, False): "missed_both",
                        (True, True): "detected_both",
                        (True, False): "lost_in_H5",
                        (False, True): "gained_in_H5",
                    }[(first_obs[key], second_obs[key])]
                ] += 1
            if class_name == "car" and int(first["classes"]["car"]["eligible_GT_count"]) > 0:
                first_primary = first["classes"]["car"]["primary"]
                second_primary = second["classes"]["car"]["primary"]
                car_frame_deltas.append(
                    {
                        "frame_id": first["frame_id"],
                        "H10_recall": first_primary["recall"],
                        "H5_recall": second_primary["recall"],
                        "absolute_recall_change": abs(
                            float(first_primary["recall"]) - float(second_primary["recall"])
                        ),
                    }
                )
        class_changes[class_name] = dict(sorted(transitions.items()))
    input_deltas = []
    for first in h10:
        second = h5_by_id[str(first["frame_id"])]
        first_exec, second_exec = first["execution"], second["execution"]
        input_deltas.append(
            {
                "frame_id": first["frame_id"],
                "point_count_delta_H10_minus_H5": first_exec["point_count"]
                - second_exec["point_count"],
                "candidate_pillar_delta_H10_minus_H5": (
                    first_exec["pillars"]["candidate_occupied_pillars"]
                    - second_exec["pillars"]["candidate_occupied_pillars"]
                ),
                "overflow_H10": first_exec["pillars"]["overflow"],
                "overflow_H5": second_exec["pillars"]["overflow"],
            }
        )
    return {
        "paired_frame_count": len(h10),
        "eligible_car_frame_count": len(car_frame_deltas),
        "input_delta_summary": {
            "point_count_H10_minus_H5": descriptive_statistics(
                [float(item["point_count_delta_H10_minus_H5"]) for item in input_deltas]
            ),
            "candidate_pillars_H10_minus_H5": descriptive_statistics(
                [float(item["candidate_pillar_delta_H10_minus_H5"]) for item in input_deltas]
            ),
        },
        "class_detection_transitions": class_changes,
        "input_deltas": input_deltas,
        "car_frame_recall_deltas": car_frame_deltas,
    }


def _visualization_selection(
    h10: Sequence[Mapping[str, object]],
    h5: Sequence[Mapping[str, object]],
    sentinels: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    eligible = [frame for frame in h10 if int(frame["classes"]["car"]["eligible_GT_count"]) > 0]
    if not eligible:
        raise RuntimeError("no Car GT frame exists for frozen visualization selection")

    def recall(frame: Mapping[str, object]) -> float:
        return float(frame["classes"]["car"]["primary"]["recall"])

    def median_iou(frame: Mapping[str, object]) -> float:
        value = frame["classes"]["car"]["primary"]["median_matched_iou"]
        return float(value) if value is not None else -1.0

    best = sorted(
        eligible, key=lambda frame: (-recall(frame), -median_iou(frame), str(frame["frame_id"]))
    )[0]
    worst = sorted(
        eligible, key=lambda frame: (recall(frame), median_iou(frame), str(frame["frame_id"]))
    )[0]
    ordered = sorted(
        eligible, key=lambda frame: (recall(frame), median_iou(frame), str(frame["frame_id"]))
    )
    median = ordered[(len(ordered) - 1) // 2]
    highest_overflow = sorted(
        h10,
        key=lambda frame: (
            -int(frame["execution"]["pillars"]["overflow_count"]),
            str(frame["frame_id"]),
        ),
    )[0]
    high_count = int(highest_overflow["execution"]["pillars"]["candidate_occupied_pillars"])
    non_overflow = sorted(
        (frame for frame in h10 if not bool(frame["execution"]["pillars"]["overflow"])),
        key=lambda frame: (
            abs(int(frame["execution"]["pillars"]["candidate_occupied_pillars"]) - high_count),
            str(frame["frame_id"]),
        ),
    )[0]
    h5_by_id = {str(frame["frame_id"]): frame for frame in h5}
    paired = sorted(
        eligible,
        key=lambda frame: (
            -abs(
                recall(frame)
                - float(h5_by_id[str(frame["frame_id"])]["classes"]["car"]["primary"]["recall"])
            ),
            str(frame["frame_id"]),
        ),
    )[0]
    return {
        "preselected_sentinels": [dict(item) for item in sentinels],
        "metric_selected": {
            "car_best": best["frame_id"],
            "car_median": median["frame_id"],
            "car_worst": worst["frame_id"],
            "highest_overflow": highest_overflow["frame_id"],
            "non_overflow_comparison": non_overflow["frame_id"],
            "paired_H10_H5": paired["frame_id"],
        },
    }


def _box_polygon(box: Detection3D | M6bGroundTruthBox) -> np.ndarray:
    length, width, _ = box.size_lwh
    local = np.array(
        [
            [length / 2, width / 2],
            [-length / 2, width / 2],
            [-length / 2, -width / 2],
            [length / 2, -width / 2],
        ]
    )
    rotation = np.array(
        [
            [math.cos(box.yaw_rad), -math.sin(box.yaw_rad)],
            [math.sin(box.yaw_rad), math.cos(box.yaw_rad)],
        ]
    )
    return local @ rotation.T + np.asarray(box.center_xyz[:2])


def _render_visualizations(
    output_directory: Path,
    selection: Mapping[str, object],
    frame_lookup: Mapping[tuple[str, str], Mapping[str, object]],
    sequences: Mapping[str, KittiRawSequence],
    poses_by_frame: Mapping[str, Sequence[KittiTrackletPose]],
    camera: KittiReferenceCamera,
    ledger_by_id: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    roles: dict[str, str] = {
        f"sentinel_{item['role']}": str(item["frame_id"])
        for item in selection["preselected_sentinels"]
    }
    roles.update({str(key): str(value) for key, value in selection["metric_selected"].items()})
    output_directory.mkdir(parents=True, exist_ok=True)
    records = []
    for role, frame_id in roles.items():
        conditions = ("H10", "H5") if role == "paired_H10_H5" else ("H10",)
        figure, axes = plt.subplots(
            1, len(conditions), figsize=(7 * len(conditions), 7), squeeze=False
        )
        drive, raw_index = frame_id.split("/", 1)
        eligible = _eligible_poses(poses_by_frame.get(frame_id, ()), camera)
        for axis, condition in zip(axes[0], conditions, strict=True):
            reconstruction = _reconstruct_input(
                sequences[drive],
                int(raw_index),
                CONDITIONS[condition],
                ledger_by_id[frame_id],
            )
            points = reconstruction.point_cloud.points_xyzt
            axis.scatter(points[::5, 0], points[::5, 1], s=0.08, c="#777777", alpha=0.45)
            audit = analyze_pillars(points)
            if audit.overflow:
                dropped = pillar_centres(audit.discarded_xy_indices)
                axis.scatter(dropped[:, 0], dropped[:, 1], s=0.5, c="#ff7f0e", alpha=0.5)
            for pose in eligible:
                if pose.evaluation_role != "target":
                    continue
                box = convert_tracklet_pose(pose)
                axis.add_patch(
                    Polygon(
                        _box_polygon(box),
                        closed=True,
                        fill=False,
                        edgecolor="#2ca02c",
                        linewidth=1.2,
                    )
                )
            detections = frame_lookup[(condition, frame_id)]["_detections"]
            for detection in detections:
                if detection.class_name not in CLASSES or detection.score < OPERATING_SCORE:
                    continue
                axis.add_patch(
                    Polygon(
                        _box_polygon(detection),
                        closed=True,
                        fill=False,
                        edgecolor="#d62728" if detection.class_name == "car" else "#9467bd",
                        linewidth=0.8,
                    )
                )
            axis.set(xlim=(-50, 50), ylim=(-50, 50), aspect="equal", title=f"{role} — {condition}")
            axis.set_xlabel("model x right (m)")
            axis.set_ylabel("model y forward (m)")
        figure.suptitle(f"Offline KITTI Raw cross-domain replay\n{frame_id}")
        figure.tight_layout()
        path = output_directory / f"{role}.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        records.append(
            {
                "role": role,
                "frame_id": frame_id,
                "conditions": list(conditions),
                "path": path.relative_to(_root()).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "real_detector_output": True,
                "offline": True,
            }
        )
    return records


def _write_json(path: Path, record: Mapping[str, object]) -> None:
    encoded = json.dumps(record, indent=2, sort_keys=True) + "\n"
    forbidden = (str(Path.home()),)
    if any(value and value in encoded for value in forbidden):
        raise RuntimeError("refusing to write evidence containing a private absolute path")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=_root() / "benchmarks/m6b/results/kitti_raw_cross_domain_characterization.json",
    )
    parser.add_argument(
        "--visual-directory",
        type=Path,
        default=_root() / "docs/assets/m6b",
    )
    parser.add_argument("--verify-input-oracles-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _root()
    _clean_measurement_tree(root)
    measurement_commit = repository_git_sha(root)
    config_path = root / EXPECTED_CONFIG_PATH
    config = _load_yaml(config_path)
    if config["status"] != "preregistered_before_detector_inference":
        raise RuntimeError("M6b protocol is not frozen")
    gt_geometry_fixture = _gt_geometry_fixture()
    installed_version = importlib.import_module("importlib.metadata").version("laserperception")
    if installed_version != PROJECT_VERSION:
        raise RuntimeError(f"expected LaserPerception {PROJECT_VERSION}, found {installed_version}")
    ledger_path = root / str(config["input_oracles"]["committed_ledger"])
    if sha256_file(ledger_path) != EXPECTED_LEDGER_SHA256:
        raise RuntimeError("committed pre-inference ledger hash mismatch")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger["detector_inference_performed"] is not False:
        raise RuntimeError("pre-inference ledger is contaminated")
    ledger_frames = ledger["frames"]
    ledger_by_id = {str(frame["frame_id"]): frame for frame in ledger_frames}
    m6a_evidence = root / M6A_EVIDENCE_PATH
    if sha256_file(m6a_evidence) != M6A_EVIDENCE_SHA256:
        raise RuntimeError("canonical M6a evidence hash mismatch")

    date_root = args.data_root.expanduser().resolve() / "2011_09_26"
    camera = KittiReferenceCamera.from_date_root(date_root)
    drive_ids = tuple(
        config["dataset"][key] for key in ("canonical_drive", "selected_secondary_drive")
    )
    sequences = {
        drive: KittiRawSequence(date_root, date_root / f"{drive}_sync") for drive in drive_ids
    }
    input_verification = _verify_input_oracles(sequences, ledger_frames)
    if args.verify_input_oracles_only:
        print(json.dumps(input_verification, indent=2, sort_keys=True), flush=True)
        return 0
    all_poses: dict[str, tuple[KittiTrackletPose, ...]] = {}
    poses_by_frame: dict[str, list[KittiTrackletPose]] = defaultdict(list)
    track_labelled_counts: dict[tuple[str, int], int] = defaultdict(int)
    for drive in drive_ids:
        path = date_root / f"{drive}_sync/tracklet_labels.xml"
        poses = parse_kitti_tracklets(path)
        all_poses[drive] = poses
        if (
            sha256_file(path)
            != config["dataset"]["tracklets"][
                "canonical_sha256" if drive == drive_ids[0] else "secondary_sha256"
            ]
        ):
            raise RuntimeError(f"tracklet hash mismatch for {drive}")
        for pose in poses:
            poses_by_frame[f"{drive}/{pose.frame_index:010d}"].append(pose)
            if pose.object_type in {"Car", "Pedestrian"} and pose.valid_labelled_pose:
                track_labelled_counts[(drive, pose.track_id)] += 1

    m1 = _load_yaml(root / "configs/detection/m1_pointpillars_nuscenes.yaml")
    m2 = _load_yaml(root / "configs/detection/m2_pointpillars_tensorrt.yaml")
    m1_assets = resolve_m1_asset_paths(m1)
    m2_assets = resolve_m2_asset_paths(m2)
    onnx = m2_assets.artifact_directory / "pointpillars.onnx"
    engine = m2_assets.engine_directory / "pointpillars_fp16.engine"
    artifacts = {
        "checkpoint": _artifact(
            m1_assets.checkpoint_path, config["frozen_detector"]["checkpoint_sha256"]
        ),
        "onnx": _artifact(onnx, config["frozen_detector"]["onnx_sha256"]),
        "tensorrt_engine": _artifact(engine, config["frozen_detector"]["tensorrt_engine_sha256"]),
    }
    backend = M2Backend(
        m1_assets.mmdet3d_root / str(m1["model"]["upstream_config"]),
        m1_assets.checkpoint_path,
        m2_assets.mmdeploy_root / str(m2["deployment"]["official_deployment_config"]),
        checkpoint_sha256=str(m1["model"]["checkpoint"]["sha256"]),
        voxelization_mode="exact_fast",
    )
    backend.initialize()
    repeatability = _repeatability(
        backend,
        engine,
        sequences,
        ledger_by_id,
        config["repeatability"]["sentinels"],
    )
    if not repeatability["passed"]:
        failed_record = {
            "schema_version": 1,
            "milestone": "M6b",
            "status": "FAILED_TENSORRT_REPEATABILITY",
            "measurement_commit": measurement_commit,
            "protocol_commit": PROTOCOL_COMMIT,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_config": {"path": EXPECTED_CONFIG_PATH, "sha256": sha256_file(config_path)},
            "pre_inference_ledger": {
                "path": str(config["input_oracles"]["committed_ledger"]),
                "sha256": sha256_file(ledger_path),
            },
            "pre_inference_input_verification": input_verification,
            "environment": {"gpu": _gpu_record(), "cuda_device": "cuda:0"},
            "artifacts": artifacts,
            "repeatability": repeatability,
            "full_characterization_started": False,
        }
        _write_json(args.output, failed_record)
        print("repeatability failed; full characterization was not started", flush=True)
        return 2

    h10 = _condition_run(
        "H10",
        backend,
        engine,
        sequences,
        poses_by_frame,
        camera,
        ledger_frames,
        track_labelled_counts,
    )
    h5 = _condition_run(
        "H5",
        backend,
        engine,
        sequences,
        poses_by_frame,
        camera,
        ledger_frames,
        track_labelled_counts,
    )
    aggregated = {"H10": _aggregate_condition(h10), "H5": _aggregate_condition(h5)}
    paired = _paired_analysis(h10, h5)
    selection = _visualization_selection(h10, h5, config["repeatability"]["sentinels"])
    lookup = {(str(frame["condition"]), str(frame["frame_id"])): frame for frame in (*h10, *h5)}
    visuals = _render_visualizations(
        args.visual_directory, selection, lookup, sequences, poses_by_frame, camera, ledger_by_id
    )
    for frame in (*h10, *h5):
        frame.pop("_detections")
    result: dict[str, object] = {
        "schema_version": 1,
        "milestone": "M6b",
        "status": "M6b FROZEN CROSS-DOMAIN CHARACTERIZATION COMPLETE",
        "measurement_commit": measurement_commit,
        "protocol_commit": PROTOCOL_COMMIT,
        "base_main_commit": BASE_MAIN_COMMIT,
        "project_version": PROJECT_VERSION,
        "m6a_evidence": {
            "path": M6A_EVIDENCE_PATH,
            "sha256": M6A_EVIDENCE_SHA256,
            "measurement_commit": M6A_MEASUREMENT_COMMIT,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_config": {
            "path": EXPECTED_CONFIG_PATH,
            "sha256": sha256_file(config_path),
        },
        "pre_inference_ledger": {
            "path": str(config["input_oracles"]["committed_ledger"]),
            "sha256": sha256_file(ledger_path),
            "frame_count": len(ledger_frames),
        },
        "pre_inference_input_verification": input_verification,
        "environment": {
            "gpu": _gpu_record(),
            "laserperception": installed_version,
            "backend_versions": dict(backend.versions),
            "mmdeploy": str(importlib.import_module("mmdeploy").__version__),
            "tensorrt": str(importlib.import_module("tensorrt").__version__),
            "cuda_device": "cuda:0",
        },
        "artifacts": artifacts,
        "dataset": {
            "name": "KITTI Raw",
            "drives": list(drive_ids),
            "evaluation_frame_count": len(ledger_frames),
            "annotation_region": "rectified_reference_camera_0_FOV",
            "whole_lidar_precision_claimed": False,
            "second_drive_selection_policy": config["dataset"]["selection_policy"],
            "candidate_census": config["candidate_census"],
            "evaluation_frame_ranges": config["evaluation_frames"]["exact_ranges"],
            "GT_counts": {
                class_name: aggregated["H10"]["classes"][class_name]["eligible_GT_count"]
                for class_name in CLASSES
            },
            "pedestrian_low_n": bool(config["evaluation_frames"]["pedestrian_low_n"]),
        },
        "annotation_contract": {
            "reference_camera_region": config["reference_camera_region"],
            "taxonomy": config["taxonomy"],
            "matching": config["matching"],
            "dontcare_available": False,
            "terminology": "LaserPerception M6b cross-domain Raw-tracklet metrics",
            "whole_lidar_precision_claimed": False,
            "outside_annotation_fov_predictions_all_classes": {
                "H10": sum(
                    int(frame["outside_annotation_fov_predictions_all_classes"]) for frame in h10
                ),
                "H5": sum(
                    int(frame["outside_annotation_fov_predictions_all_classes"]) for frame in h5
                ),
            },
        },
        "GT_geometry": {
            "contract": config["ground_truth"],
            "analytic_fixture": gt_geometry_fixture,
        },
        "frozen_detector_contract": config["frozen_detector"],
        "input_oracles": {
            "contract": config["input_oracles"],
            "H10_summary": aggregated["H10"]["input_summary"],
            "H5_summary": aggregated["H5"]["input_summary"],
        },
        "repeatability": repeatability,
        "condition_results": aggregated,
        "paired_H10_H5": paired,
        "visualization_selection": selection,
        "visualizations": visuals,
        "frame_results": {"H10": h10, "H5": h5},
        "interpretation_guards": {
            "official_KITTI_benchmark": False,
            "threshold_tuned": False,
            "causal_claim": False,
            "H10_H5_compound_ablation": True,
            "ROS_used": False,
            "model_changed": False,
            "engine_changed": False,
        },
    }
    _write_json(args.output, result)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
