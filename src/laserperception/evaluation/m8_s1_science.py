"""Future-authorized M8 P1-S1 scientific process implementation.

This module imports KITTI ground truth and evaluator code by design. The
narrow CLI imports it only *after* the separate owner authorization and every
static runtime binding have passed.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from laserperception.detection.m8_backend import DsvtBackend
from laserperception.detection.m8_s1_preflight import FrozenInputSource
from laserperception.detection.m8_s1_runtime import (
    CANDIDATE_MANIFEST_PATH,
    EVALUATOR_IDENTITY,
    AtomicAttempt,
    AttemptIdentity,
    atomic_write_json,
    canonical_condition_ids,
    canonical_json_sha256,
    stage_r_condition_ids,
    validate_scientific_condition_payload,
    zero_intensity_copy,
)
from laserperception.detection.measurement_telemetry import (
    NvidiaSmiSampler,
    summarize_gpu_telemetry,
)
from laserperception.detection.types import Detection3D, DetectionFrame
from laserperception.evaluation.kitti_m6b import (
    KittiReferenceCamera,
    KittiTrackletPose,
    convert_tracklet_pose,
    match_detections,
    model_box_corners,
    model_to_native_corners,
    native_box_corners,
    parse_kitti_tracklets,
    visible_in_reference_camera,
)
from laserperception.evaluation.m6b_metrics import count_metrics

CLASSES = ("car", "pedestrian")
IOU_THRESHOLDS = (0.30, 0.50, 0.70)
SCORE_THRESHOLD = 0.25


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _inside_camera(box: Detection3D, camera: KittiReferenceCamera) -> bool:
    native = model_to_native_corners(model_box_corners(box))
    return visible_in_reference_camera(native, camera)


def _eligible_poses(
    poses: Sequence[KittiTrackletPose], camera: KittiReferenceCamera
) -> tuple[KittiTrackletPose, ...]:
    return tuple(
        pose
        for pose in poses
        if pose.valid_labelled_pose
        and pose.evaluation_role in {"target", "neighbour_ignore"}
        and visible_in_reference_camera(native_box_corners(pose), camera)
    )


def _class_evidence(
    *,
    frame_id: str,
    class_name: str,
    predictions: Sequence[Detection3D],
    poses: Sequence[KittiTrackletPose],
) -> dict[str, object]:
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
    primary = None
    for threshold in IOU_THRESHOLDS:
        summary = match_detections(
            predictions,
            targets,
            ignores,
            class_name=class_name,
            iou_threshold=threshold,
            score_threshold=SCORE_THRESHOLD,
        )
        matched = sorted(
            f"{frame_id.split('/', 1)[0]}/track_{record.gt_track_id}"
            for record in summary.records
            if record.disposition == "true_positive" and record.gt_track_id is not None
        )
        thresholds[f"{threshold:.2f}"] = {
            **count_metrics(
                summary.true_positives,
                summary.false_positives,
                summary.false_negatives,
            ),
            "ignored_predictions": summary.ignored_predictions,
            "matched_gt_identity_set": matched,
        }
        if threshold == 0.50:
            primary = summary
    assert primary is not None
    primary_by_prediction = {record.prediction_index: record for record in primary.records}
    pose_by_track = {pose.track_id: pose for pose in target_poses}
    matched_tracks = {
        record.gt_track_id: record
        for record in primary.records
        if record.disposition == "true_positive" and record.gt_track_id is not None
    }
    target_observations = []
    drive_id = frame_id.split("/", 1)[0]
    for target in targets:
        pose = pose_by_track[target.track_id]
        match = matched_tracks.get(target.track_id)
        target_observations.append(
            {
                "gt_identity": f"{drive_id}/track_{target.track_id}",
                "track_id": target.track_id,
                "frame_index": target.frame_index,
                "source_type": target.source_type,
                "range_forward_m": target.center_xyz[1],
                "range_band_metres": (
                    "0_20"
                    if target.center_xyz[1] < 20.0
                    else "20_35"
                    if target.center_xyz[1] < 35.0
                    else "35_50"
                ),
                "occlusion": pose.occlusion,
                "truncation": pose.truncation,
                "matched": match is not None,
                "matched_iou": None if match is None else match.bev_iou,
            }
        )
    ranked = match_detections(
        predictions,
        targets,
        ignores,
        class_name=class_name,
        iou_threshold=0.50,
        score_threshold=0.0,
    )
    return {
        "eligible_GT_count": len(targets),
        "neighbour_ignore_GT_count": len(ignores),
        "thresholded_prediction_count": sum(
            prediction.class_name == class_name and prediction.score >= SCORE_THRESHOLD
            for prediction in predictions
        ),
        "thresholds": thresholds,
        "target_observations": target_observations,
        "ranked_dispositions": [
            {
                "score": record.score,
                "frame_id": frame_id,
                "prediction_index": record.prediction_index,
                "true_positive": record.disposition == "true_positive",
            }
            for record in ranked.records
            if record.disposition != "ignored_neighbour"
        ],
        "primary_prediction_dispositions": {
            str(index): {
                "disposition": record.disposition,
                "matched_gt_identity": (
                    None if record.gt_track_id is None else f"{drive_id}/track_{record.gt_track_id}"
                ),
                "matched_iou": record.bev_iou,
            }
            for index, record in primary_by_prediction.items()
        },
    }


def _condition_evidence(
    frame: DetectionFrame,
    *,
    frame_id: str,
    history: str,
    input_sha256: str,
    poses: Sequence[KittiTrackletPose],
    camera: KittiReferenceCamera,
) -> dict[str, object]:
    inside_indices = [
        index
        for index, detection in enumerate(frame.detections)
        if _inside_camera(detection, camera)
    ]
    inside = tuple(frame.detections[index] for index in inside_indices)
    classes = {
        class_name: _class_evidence(
            frame_id=frame_id,
            class_name=class_name,
            predictions=inside,
            poses=poses,
        )
        for class_name in CLASSES
    }
    prediction_records = []
    inside_position = {original: position for position, original in enumerate(inside_indices)}
    for stable_index, detection in enumerate(frame.detections):
        class_record = classes.get(detection.class_name)
        disposition: Mapping[str, object] | None = None
        if isinstance(class_record, Mapping) and stable_index in inside_position:
            dispositions = class_record["primary_prediction_dispositions"]
            assert isinstance(dispositions, Mapping)
            value = dispositions.get(str(inside_position[stable_index]))
            if isinstance(value, Mapping):
                disposition = value
        box = {
            "center_xyz": list(detection.center_xyz),
            "size_lwh": list(detection.size_lwh),
            "yaw_rad": detection.yaw_rad,
            "velocity_xy": None if detection.velocity_xy is None else list(detection.velocity_xy),
        }
        prediction_records.append(
            {
                "stable_prediction_index": stable_index,
                "class_name": detection.class_name,
                "score": detection.score,
                "box_lidar": box,
                "box_sha256": canonical_json_sha256(box),
                "inside_annotation_fov": stable_index in inside_position,
                "primary_disposition": None if disposition is None else disposition["disposition"],
                "matched_gt_identity": (
                    None if disposition is None else disposition["matched_gt_identity"]
                ),
                "matched_iou": None if disposition is None else disposition["matched_iou"],
            }
        )
    for value in classes.values():
        assert isinstance(value, dict)
        value.pop("primary_prediction_dispositions")
    frame_payload = frame.to_dict()
    payload = {
        "frame_id": frame_id,
        "history": history,
        "input_sha256": input_sha256,
        "detection_frame_sha256": canonical_json_sha256(frame_payload),
        "predictions": prediction_records,
        "classes": classes,
        "outside_annotation_fov_prediction_count": len(frame.detections) - len(inside),
        "evaluator_provenance": {
            "identity": EVALUATOR_IDENTITY,
            "score_threshold": SCORE_THRESHOLD,
            "iou_thresholds": list(IOU_THRESHOLDS),
            "annotation_conditioned": True,
        },
    }
    validate_scientific_condition_payload(payload)
    return payload


def _load_gt(
    date_root: Path,
) -> tuple[KittiReferenceCamera, Mapping[str, Sequence[KittiTrackletPose]]]:
    camera = KittiReferenceCamera.from_date_root(date_root)
    by_frame: dict[str, list[KittiTrackletPose]] = defaultdict(list)
    for drive_id in ("2011_09_26_drive_0001", "2011_09_26_drive_0091"):
        for pose in parse_kitti_tracklets(date_root / f"{drive_id}_sync/tracklet_labels.xml"):
            by_frame[f"{drive_id}/{pose.frame_index:010d}"].append(pose)
    return camera, by_frame


def _revalidate_all(source: FrozenInputSource) -> dict[str, object]:
    h10 = h5 = 0
    for frame_id in source.frames:
        pair = source.pair(frame_id)
        h10 += int(pair[0][1]["history"] == "H10")
        h5 += int(pair[1][1]["history"] == "H5")
    if (h10, h5) != (428, 428):
        raise RuntimeError("future S1 process input revalidation is incomplete")
    return {"H10_exact": h10, "H5_exact": h5, "conditions_exact": h10 + h5}


def run_scientific_attempt(
    *,
    mode: str,
    repository_root: Path,
    full_ledger: Path,
    date_root: Path,
    runtime_commit: str,
    attempt_root: Path,
    logical_pass_id: str,
    attempt_id: str,
) -> dict[str, object]:
    """Execute one future-authorized, uninterrupted fresh-process attempt."""

    process_uuid = str(uuid.uuid4())
    identity = AttemptIdentity(
        mode=mode,
        logical_pass_id=logical_pass_id,
        attempt_id=attempt_id,
        process_uuid=process_uuid,
        process_id=os.getpid(),
        runtime_commit=runtime_commit,
    )
    attempt = AtomicAttempt(attempt_root, identity)
    try:
        source = FrozenInputSource.load(
            date_root=date_root,
            full_ledger=full_ledger,
            accepted_ledger=repository_root
            / "benchmarks/m8/diagnostics/m8_input_projection_ledger.json",
        )
        revalidation = _revalidate_all(source)
        atomic_write_json(attempt_root / "input_revalidation.json", revalidation)
        camera, poses_by_frame = _load_gt(date_root)
        backend = DsvtBackend.from_environment(
            manifest_path=repository_root / CANDIDATE_MANIFEST_PATH
        )
        atomic_write_json(attempt_root / "runtime_state.json", backend.runtime_state())
        sampler = NvidiaSmiSampler(interval_seconds=1.0)
        sampler.start()
        payloads = []
        try:
            condition_ids = (
                stage_r_condition_ids() if mode == "stage-r" else canonical_condition_ids()
            )
            sampler.begin_block("scientific_attempt")
            for condition_id in condition_ids:
                frame_id, history = condition_id.rsplit("/", 1)
                pair = source.pair(frame_id)
                points, input_identity = pair[0 if history == "H10" else 1]
                if mode == "zero-intensity-pass":
                    points = zero_intensity_copy(points)
                    input_identity = {
                        **input_identity,
                        "primary_input_sha256": input_identity["input_sha256"],
                        "input_sha256": _sha256_array(points),
                        "intervention": "candidate intensity float32 +0",
                    }
                frame = backend.infer(points, sample_id=condition_id)
                poses = _eligible_poses(poses_by_frame.get(frame_id, ()), camera)
                payload = _condition_evidence(
                    frame,
                    frame_id=frame_id,
                    history=history,
                    input_sha256=str(input_identity["input_sha256"]),
                    poses=poses,
                    camera=camera,
                )
                attempt.record(condition_id, payload)
                payloads.append(payload)
            sampler.end_block("scientific_attempt")
        finally:
            sampler.stop()
        raw = {
            "schema_version": "laserperception.m8.s1.raw-pass.v1",
            "status": "COMPLETE",
            **identity.to_dict(),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "conditions": payloads,
            "telemetry": summarize_gpu_telemetry(sampler.samples),
        }
        raw["result_sha256"] = canonical_json_sha256(raw)
        atomic_write_json(attempt_root / "raw_pass.json", raw)
        final = attempt.finalize()
        return {"raw_pass": raw, "final_manifest": final}
    except Exception as error:
        attempt.fail(f"{type(error).__name__}: {error}")
        raise
