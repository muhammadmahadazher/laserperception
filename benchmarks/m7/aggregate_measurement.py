"""Aggregate completed M7 checkpoints with the frozen M6b evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from benchmarks.m7.execution import factorial_contrasts
from benchmarks.m7.protocol import (
    CHECKPOINT_SHA256,
    ENGINE_SHA256,
    EVALUATOR_IDENTITY,
    M6B_RESULT_FULL_BYTES,
    M6B_RESULT_FULL_SHA256,
    ONNX_SHA256,
    PROTOCOL_FREEZE_COMMIT,
    Arm,
    ProtocolViolation,
    canonical_condition_ids,
)
from benchmarks.m7.provenance import atomic_write_json, canonical_json_sha256
from laserperception.detection.types import Detection3D
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
from laserperception.evaluation.m6b_metrics import (
    RankedDisposition,
    all_points_average_precision,
    count_metrics,
)

IMPLEMENTATION_COMMIT = "c989f7df5ca8c5ac8148c0ed3a2e91de48b754b2"
MEASUREMENT_RUNTIME_COMMIT = "5a8c02e8ba279ee44a8bb87eb2ec2984ca95e729"
INPUT_LEDGER_SHA256 = "577a7ee3da5495611592ca3226a2adefd577fa54821bb859d25892d0cbcbb8ea"
PAIRED_GT_SHA256 = "0f4ecf564bff30913a0cb35b2043a9a5cd0c8fdb26b220c4cb12072e186f8ba5"
TRACKLET_SHA256 = {
    "2011_09_26_drive_0001": "34f0672dee9dc94535893e653b4a66e6ddf534a09d2533bac4e62965935a91b8",
    "2011_09_26_drive_0091": "3d363ee40129e51aaf44764b9637bc7e946b6e3ec628784adcdedd395505feab",
}
CLASSES = ("car", "pedestrian")
NEW_ARMS = (Arm.B, Arm.C, Arm.D, Arm.F)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_checkpoint_identity() -> dict[str, str]:
    return {
        "protocol_commit": PROTOCOL_FREEZE_COMMIT,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "measurement_runtime_commit": MEASUREMENT_RUNTIME_COMMIT,
        "input_ledger_sha256": INPUT_LEDGER_SHA256,
        "engine_sha256": ENGINE_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "onnx_sha256": ONNX_SHA256,
        "evaluator_identity": EVALUATOR_IDENTITY,
    }


def _checkpoint_path(root: Path, condition: str) -> Path:
    frame_id, arm = condition.split("|", 1)
    drive_id, frame_index = frame_id.split("/", 1)
    return root / "conditions" / drive_id / f"{frame_index}_{arm}.json"


def _detection(record: Mapping[str, object]) -> Detection3D:
    velocity = record.get("velocity_xy")
    return Detection3D(
        center_xyz=tuple(float(value) for value in record["center_xyz"]),  # type: ignore[arg-type]
        size_lwh=tuple(float(value) for value in record["size_lwh"]),  # type: ignore[arg-type]
        yaw_rad=float(record["yaw_rad"]),
        score=float(record["score"]),
        class_id=int(record["class_id"]),
        class_name=str(record["class_name"]),
        velocity_xy=(
            None if velocity is None else tuple(float(value) for value in velocity)  # type: ignore[arg-type]
        ),
    )


def _load_completed_frames(
    checkpoint_root: Path,
) -> dict[Arm, dict[str, tuple[Detection3D, ...]]]:
    progress_path = checkpoint_root / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    expected_conditions = list(canonical_condition_ids())
    if progress.get("schema_version") != "laserperception.m7.progress.v2":
        raise ProtocolViolation("M7 progress schema differs")
    if progress.get("identity") != _expected_checkpoint_identity():
        raise ProtocolViolation("M7 progress identity differs")
    if progress.get("condition_ids") != expected_conditions:
        raise ProtocolViolation("M7 progress corpus differs")
    statuses = progress.get("conditions")
    if not isinstance(statuses, Mapping) or set(statuses) != set(expected_conditions):
        raise ProtocolViolation("M7 progress condition records are malformed")

    frames: dict[Arm, dict[str, tuple[Detection3D, ...]]] = {arm: {} for arm in NEW_ARMS}
    for condition in expected_conditions:
        status = statuses[condition]
        if not isinstance(status, Mapping) or status.get("status") != "COMPLETE":
            raise ProtocolViolation(f"M7 condition is not complete: {condition}")
        path = _checkpoint_path(checkpoint_root, condition)
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "COMPLETE":
            raise ProtocolViolation(f"M7 checkpoint is not complete: {condition}")
        if record.get("identity") != _expected_checkpoint_identity():
            raise ProtocolViolation(f"M7 checkpoint identity differs: {condition}")
        if record.get("condition_id") != condition:
            raise ProtocolViolation(f"M7 checkpoint condition differs: {condition}")
        payload_sha = canonical_json_sha256(
            {key: value for key, value in record.items() if key != "checkpoint_payload_sha256"}
        )
        if record.get("checkpoint_payload_sha256") != payload_sha:
            raise ProtocolViolation(f"M7 checkpoint payload hash differs: {condition}")
        if status.get("checkpoint_payload_sha256") != payload_sha:
            raise ProtocolViolation(f"M7 progress payload hash differs: {condition}")
        payload = record.get("payload")
        if not isinstance(payload, Mapping):
            raise ProtocolViolation(f"M7 checkpoint payload is malformed: {condition}")
        frame = payload.get("detection_frame")
        if not isinstance(frame, Mapping) or not isinstance(frame.get("detections"), list):
            raise ProtocolViolation(f"M7 DetectionFrame payload is malformed: {condition}")
        frame_id, arm_text = condition.split("|", 1)
        arm = Arm(arm_text)
        frames[arm][frame_id] = tuple(
            _detection(value) for value in frame["detections"] if isinstance(value, Mapping)
        )
    if any(len(values) != 428 for values in frames.values()):
        raise ProtocolViolation("M7 complete-corpus arm counts differ from 428 each")
    return frames


def _load_poses(
    dataset_date_root: Path,
) -> tuple[KittiReferenceCamera, dict[str, tuple[KittiTrackletPose, ...]]]:
    camera = KittiReferenceCamera.from_date_root(dataset_date_root)
    poses_by_frame: dict[str, list[KittiTrackletPose]] = defaultdict(list)
    for drive_id, expected_sha in TRACKLET_SHA256.items():
        path = dataset_date_root / f"{drive_id}_sync" / "tracklet_labels.xml"
        if _sha256_file(path) != expected_sha:
            raise ProtocolViolation(f"M7 tracklet identity differs: {drive_id}")
        for pose in parse_kitti_tracklets(path):
            poses_by_frame[f"{drive_id}/{pose.frame_index:010d}"].append(pose)
    return camera, {key: tuple(value) for key, value in poses_by_frame.items()}


def _visible_poses(
    poses: Sequence[KittiTrackletPose], camera: KittiReferenceCamera
) -> tuple[KittiTrackletPose, ...]:
    return tuple(
        pose
        for pose in poses
        if pose.valid_labelled_pose
        and pose.evaluation_role in {"target", "neighbour_ignore"}
        and visible_in_reference_camera(native_box_corners(pose), camera)
    )


def _visible_predictions(
    predictions: Sequence[Detection3D], camera: KittiReferenceCamera
) -> tuple[Detection3D, ...]:
    return tuple(
        prediction
        for prediction in predictions
        if visible_in_reference_camera(
            model_to_native_corners(model_box_corners(prediction)), camera
        )
    )


def _evaluate_arm_class(
    frames: Mapping[str, tuple[Detection3D, ...]],
    poses_by_frame: Mapping[str, Sequence[KittiTrackletPose]],
    camera: KittiReferenceCamera,
    class_name: str,
) -> tuple[dict[str, object], set[tuple[str, int, int]]]:
    tp = fp = fn = ignored = ground_truth_count = 0
    ranked: list[RankedDisposition] = []
    detected: set[tuple[str, int, int]] = set()
    for frame_id in sorted(frames):
        drive_id, frame_text = frame_id.split("/", 1)
        eligible = _visible_poses(poses_by_frame.get(frame_id, ()), camera)
        targets = tuple(
            convert_tracklet_pose(pose)
            for pose in eligible
            if pose.evaluation_role == "target" and pose.evaluation_class == class_name
        )
        neighbour_ignores = tuple(
            convert_tracklet_pose(pose)
            for pose in eligible
            if pose.evaluation_role == "neighbour_ignore" and pose.evaluation_class == class_name
        )
        predictions = _visible_predictions(frames[frame_id], camera)
        primary = match_detections(
            predictions,
            targets,
            neighbour_ignores,
            class_name=class_name,
            iou_threshold=0.50,
            score_threshold=0.25,
        )
        tp += primary.true_positives
        fp += primary.false_positives
        fn += primary.false_negatives
        ignored += primary.ignored_predictions
        ground_truth_count += len(targets)
        detected.update(
            (drive_id, int(frame_text), int(record.gt_track_id))
            for record in primary.records
            if record.disposition == "true_positive" and record.gt_track_id is not None
        )
        ranked_summary = match_detections(
            predictions,
            targets,
            neighbour_ignores,
            class_name=class_name,
            iou_threshold=0.50,
            score_threshold=0.0,
        )
        ranked.extend(
            RankedDisposition(
                score=record.score,
                frame_id=frame_id,
                prediction_index=record.prediction_index,
                true_positive=record.disposition == "true_positive",
            )
            for record in ranked_summary.records
            if record.disposition != "ignored_neighbour"
        )
    metrics = count_metrics(tp, fp, fn)
    return (
        {
            **metrics,
            "annotation_conditioned_precision": metrics["precision"],
            "average_precision": all_points_average_precision(
                ranked, ground_truth_count=ground_truth_count
            )["average_precision"],
            "eligible_ground_truth_count": ground_truth_count,
            "ignored_predictions": ignored,
        },
        detected,
    )


def _pose_key(value: Mapping[str, object]) -> tuple[str, int, int]:
    return (str(value["drive_id"]), int(value["frame_index"]), int(value["gt_track_id"]))


def _paired_summary(
    detected: set[tuple[str, int, int]], class_record: Mapping[str, object]
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in ("e_only", "a_only", "shared", "neither"):
        raw = class_record.get(name)
        if not isinstance(raw, list):
            raise ProtocolViolation(f"M7 paired-GT set is malformed: {name}")
        values = tuple(_pose_key(value) for value in raw if isinstance(value, Mapping))
        hits = tuple(sorted(detected.intersection(values)))
        misses = tuple(sorted(set(values).difference(detected)))
        result[name] = {
            "detected_count": len(hits),
            "denominator": len(values),
            "rate": len(hits) / len(values) if values else 0.0,
            "detected_pose_keys": [list(value) for value in hits],
            "missed_pose_keys": [list(value) for value in misses],
        }
    return result


def _m6_raw_table(
    result: Mapping[str, object], condition: str, class_name: str
) -> dict[str, object]:
    conditions = result.get("condition_results")
    if not isinstance(conditions, Mapping) or not isinstance(conditions.get(condition), Mapping):
        raise ProtocolViolation("frozen M6b condition result is malformed")
    classes = conditions[condition].get("classes")
    if not isinstance(classes, Mapping) or not isinstance(classes.get(class_name), Mapping):
        raise ProtocolViolation("frozen M6b class result is malformed")
    record = classes[class_name]
    thresholds = record.get("thresholds")
    score_ranked = record.get("score_ranked_PR")
    if not isinstance(thresholds, Mapping) or not isinstance(thresholds.get("0.50"), Mapping):
        raise ProtocolViolation("frozen M6b primary metrics are malformed")
    if not isinstance(score_ranked, Mapping):
        raise ProtocolViolation("frozen M6b AP result is malformed")
    primary = thresholds["0.50"]
    return {
        "true_positives": int(primary["true_positives"]),
        "false_positives": int(primary["false_positives"]),
        "false_negatives": int(primary["false_negatives"]),
        "precision": float(primary["precision"]),
        "annotation_conditioned_precision": float(primary["precision"]),
        "recall": float(primary["recall"]),
        "f1": float(primary["f1"]),
        "average_precision": float(score_ranked["average_precision"]),
        "eligible_ground_truth_count": int(record["eligible_GT_count"]),
        "ignored_predictions": int(primary["ignored_predictions"]),
    }


def aggregate(
    *,
    checkpoint_root: Path,
    dataset_date_root: Path,
    m6b_result_path: Path,
    paired_gt_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Return the two compact raw-only M7 result records."""

    if (
        m6b_result_path.stat().st_size != M6B_RESULT_FULL_BYTES
        or _sha256_file(m6b_result_path) != M6B_RESULT_FULL_SHA256
    ):
        raise ProtocolViolation("frozen full M6b result identity differs")
    if _sha256_file(paired_gt_path) != PAIRED_GT_SHA256:
        raise ProtocolViolation("frozen paired-GT identity differs")
    m6b_result = json.loads(m6b_result_path.read_text(encoding="utf-8"))
    paired = json.loads(paired_gt_path.read_text(encoding="utf-8"))
    if not isinstance(paired.get("classes"), Mapping):
        raise ProtocolViolation("frozen paired-GT classes are malformed")

    frames = _load_completed_frames(checkpoint_root)
    camera, poses_by_frame = _load_poses(dataset_date_root)
    arm_table: dict[str, dict[str, object]] = {class_name: {} for class_name in CLASSES}
    for class_name in CLASSES:
        arm_table[class_name]["A"] = _m6_raw_table(m6b_result, "H10", class_name)
        arm_table[class_name]["E"] = _m6_raw_table(m6b_result, "H5", class_name)
        expected_baselines = {
            "car": {"A": (16, 66), "E": (48, 66)},
            "pedestrian": {"A": (219, 396), "E": (268, 396)},
        }
        for baseline_arm, (expected_tp, expected_gt) in expected_baselines[class_name].items():
            baseline_record = arm_table[class_name][baseline_arm]
            if (
                baseline_record["true_positives"] != expected_tp
                or baseline_record["eligible_ground_truth_count"] != expected_gt
            ):
                raise ProtocolViolation(
                    f"frozen M6b {class_name} Arm-{baseline_arm} baseline differs"
                )
        class_paired = paired["classes"].get(class_name)
        if not isinstance(class_paired, Mapping):
            raise ProtocolViolation(f"paired-GT class is malformed: {class_name}")
        for arm in NEW_ARMS:
            raw, detected = _evaluate_arm_class(frames[arm], poses_by_frame, camera, class_name)
            paired_result = _paired_summary(detected, class_paired)
            baseline, gap = (16, 32) if class_name == "car" else (219, 49)
            raw["gap_recovery"] = (int(raw["true_positives"]) - baseline) / gap
            raw["paired_sets"] = paired_result
            if class_name == "car":
                raw["g_car"] = raw["gap_recovery"]
                raw["r_gain"] = paired_result["e_only"]["rate"]
                raw["r_shared"] = paired_result["shared"]["rate"]
                raw["r_novel"] = paired_result["neither"]["rate"]
                raw["preregistered_booleans"] = {
                    "g_car_at_least_0_50": raw["g_car"] >= 0.50,
                    "r_gain_at_least_0_50": raw["r_gain"] >= 0.50,
                    "r_shared_at_least_15_over_16": raw["r_shared"] >= 15 / 16,
                }
                gates = raw["preregistered_booleans"]
                raw["preregistered_booleans"]["all_three_pass"] = all(gates.values())
            else:
                raw["g_ped"] = raw["gap_recovery"]
                raw["e_only_recovery"] = paired_result["e_only"]["rate"]
                raw["a_only_retention"] = paired_result["a_only"]["rate"]
                raw["shared_retention"] = paired_result["shared"]["rate"]
                raw["neither_recovery"] = paired_result["neither"]["rate"]
            arm_table[class_name][arm.name] = raw

    car_recall = {key: float(arm_table["car"][key]["recall"]) for key in ("A", "B", "C", "D")}
    contrasts = {
        "schema_version": "laserperception.m7.raw-factorial-contrasts.v1",
        "status": "M7 MEASUREMENT COMPLETE — RAW PREREGISTERED RESULTS ONLY.",
        "interpretation_status": "SCIENTIFIC INTERPRETATION NOT YET FROZEN.",
        "outcome": "car_recall_score_gte_0.25_oriented_bev_iou_gte_0.50",
        "values": car_recall,
        "contrasts": factorial_contrasts(
            a=car_recall["A"],
            b=car_recall["B"],
            c=car_recall["C"],
            d=car_recall["D"],
        ),
        "F_excluded": True,
    }
    table = {
        "schema_version": "laserperception.m7.raw-arm-table.v1",
        "status": "M7 MEASUREMENT COMPLETE — RAW PREREGISTERED RESULTS ONLY.",
        "interpretation_status": "SCIENTIFIC INTERPRETATION NOT YET FROZEN.",
        "operating_point": {
            "score_threshold": 0.25,
            "oriented_bev_iou_threshold": 0.50,
            "evaluator_identity": EVALUATOR_IDENTITY,
        },
        "arms": arm_table,
        "f_raw_comparison": {
            class_name: {
                baseline: {
                    metric: float(arm_table[class_name]["F"][metric])
                    - float(arm_table[class_name][baseline][metric])
                    for metric in (
                        "true_positives",
                        "false_positives",
                        "false_negatives",
                        "recall",
                        "precision",
                        "f1",
                        "average_precision",
                    )
                }
                for baseline in ("A", "E")
            }
            for class_name in CLASSES
        },
        "f_context": (
            "F is a natural, unthinned, long-span comparator at matched history-sweep count; "
            "it does not isolate temporal span as a unique cause."
        ),
        "input_structural_context": {
            "A_B_overflow_frames": 68,
            "C_D_E_F_overflow_frames": 0,
            "C_E_exact_matched_total_point_count": True,
            "C_has_more_candidate_pillars_than_E": True,
            "F_point_population": "near_E",
            "F_pillar_count": "intermediate",
        },
    }
    return table, contrasts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dataset-date-root", type=Path, required=True)
    parser.add_argument("--m6b-result", type=Path, required=True)
    parser.add_argument("--paired-gt", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    table, contrasts = aggregate(
        checkpoint_root=args.checkpoint_root,
        dataset_date_root=args.dataset_date_root,
        m6b_result_path=args.m6b_result,
        paired_gt_path=args.paired_gt,
    )
    atomic_write_json(args.output_directory / "m7_raw_arm_table.json", table)
    atomic_write_json(args.output_directory / "m7_raw_factorial_contrasts.json", contrasts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
