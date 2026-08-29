"""Complete M7 secondary characterization from frozen DetectionFrame checkpoints."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from benchmarks.m7.aggregate_measurement import (
    CLASSES,
    NEW_ARMS,
    _load_completed_frames,
    _load_poses,
    _sha256_file,
    _visible_poses,
)
from benchmarks.m7.execution import factorial_contrasts
from benchmarks.m7.protocol import (
    EVALUATOR_IDENTITY,
    M6B_RESULT_FULL_BYTES,
    M6B_RESULT_FULL_SHA256,
    ProtocolViolation,
)
from benchmarks.m7.provenance import atomic_write_json
from laserperception.detection.types import Detection3D
from laserperception.evaluation.kitti_m6b import (
    KittiReferenceCamera,
    KittiTrackletPose,
    M6bGroundTruthBox,
    bev_iou,
    convert_tracklet_pose,
    match_detections,
    model_box_corners,
    model_to_native_corners,
    visible_in_reference_camera,
)
from laserperception.evaluation.m6b_metrics import (
    RankedDisposition,
    all_points_average_precision,
    count_metrics,
    longest_consecutive_runs,
)

IOU_THRESHOLDS = (0.30, 0.50, 0.70)
OPERATING_SCORE = 0.25
PRIMARY_ARM_TABLE_SHA256 = "2539286bc4ddf05e0526e0301aeb93e295afa1d549140d2ef341edc6cb725f44"
PRIMARY_CONSISTENCY_FIELDS = (
    "true_positives",
    "false_positives",
    "false_negatives",
    "precision",
    "recall",
    "f1",
    "ignored_predictions",
)


def _prediction_fov(
    predictions: Sequence[Detection3D], camera: KittiReferenceCamera
) -> tuple[tuple[Detection3D, ...], tuple[Detection3D, ...]]:
    """Partition predictions with the frozen M6b reference-camera FOV rule."""

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
    """Return the deterministic M6b source label for an ignored prediction."""

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


def _prediction_population(
    inside: Sequence[Detection3D],
    outside: Sequence[Detection3D],
    *,
    class_name: str,
) -> tuple[int, int]:
    """Count class predictions at the frozen score threshold by FOV disposition."""

    def count(values: Sequence[Detection3D]) -> int:
        return sum(
            prediction.class_name == class_name and prediction.score >= OPERATING_SCORE
            for prediction in values
        )

    return count(inside), count(outside)


def _track_labelled_counts(
    poses_by_frame: Mapping[str, Sequence[KittiTrackletPose]],
) -> dict[tuple[str, int], int]:
    counts: dict[tuple[str, int], int] = defaultdict(int)
    for frame_id, poses in poses_by_frame.items():
        drive_id = frame_id.split("/", 1)[0]
        for pose in poses:
            if pose.object_type in {"Car", "Pedestrian"} and pose.valid_labelled_pose:
                counts[(drive_id, pose.track_id)] += 1
    return dict(counts)


def _evaluate_frame_class(
    *,
    frame_id: str,
    predictions: Sequence[Detection3D],
    poses: Sequence[KittiTrackletPose],
    camera: KittiReferenceCamera,
    class_name: str,
    labelled_counts: Mapping[tuple[str, int], int],
) -> dict[str, object]:
    drive_id, frame_text = frame_id.split("/", 1)
    frame_index = int(frame_text)
    eligible = _visible_poses(poses, camera)
    target_poses = [
        pose
        for pose in eligible
        if pose.evaluation_role == "target" and pose.evaluation_class == class_name
    ]
    ignore_poses = [
        pose
        for pose in eligible
        if pose.evaluation_role == "neighbour_ignore" and pose.evaluation_class == class_name
    ]
    targets = tuple(convert_tracklet_pose(pose) for pose in target_poses)
    ignores = tuple(convert_tracklet_pose(pose) for pose in ignore_poses)
    inside, outside = _prediction_fov(predictions, camera)

    thresholds: dict[str, dict[str, float | int]] = {}
    primary_summary = None
    for threshold in IOU_THRESHOLDS:
        summary = match_detections(
            inside,
            targets,
            ignores,
            class_name=class_name,
            iou_threshold=threshold,
            score_threshold=OPERATING_SCORE,
        )
        thresholds[f"{threshold:.2f}"] = {
            **count_metrics(
                summary.true_positives,
                summary.false_positives,
                summary.false_negatives,
            ),
            "ignored_predictions": summary.ignored_predictions,
        }
        if threshold == 0.50:
            primary_summary = summary
    assert primary_summary is not None

    matched_by_track: dict[int, Detection3D] = {}
    ignored_reasons: dict[str, int] = defaultdict(int)
    for record in primary_summary.records:
        prediction = inside[record.prediction_index]
        if record.disposition == "true_positive" and record.gt_track_id is not None:
            matched_by_track[record.gt_track_id] = prediction
        elif record.disposition == "ignored_neighbour":
            ignored_reasons[_source_ignore_reason(prediction, ignores, 0.50)] += 1

    observations: list[dict[str, object]] = []
    for box in targets:
        prediction = matched_by_track.get(box.track_id)
        observation: dict[str, object] = {
            "object_key": f"{drive_id}/track_{box.track_id}",
            "frame_index": frame_index,
            "track_labelled_frame_count": labelled_counts[(drive_id, box.track_id)],
            "range_forward_m": box.center_xyz[1],
            "matched": prediction is not None,
        }
        if prediction is not None:
            observation["prediction_score"] = prediction.score
        observations.append(observation)

    ranked_summary = match_detections(
        inside,
        targets,
        ignores,
        class_name=class_name,
        iou_threshold=0.50,
        score_threshold=0.0,
    )
    ranked = [
        RankedDisposition(
            score=record.score,
            frame_id=frame_id,
            prediction_index=record.prediction_index,
            true_positive=record.disposition == "true_positive",
        )
        for record in ranked_summary.records
        if record.disposition != "ignored_neighbour"
    ]
    inside_count, outside_count = _prediction_population(inside, outside, class_name=class_name)
    return {
        "eligible_GT_count": len(targets),
        "inside_FOV_prediction_count_score_0_25": inside_count,
        "outside_annotation_fov_predictions_score_0_25": outside_count,
        "neighbour_ignore_GT_count": len(ignores),
        "thresholds": thresholds,
        "ignored_predictions_by_reason": dict(sorted(ignored_reasons.items())),
        "target_observations": observations,
        "ranked_dispositions": ranked,
    }


def _range_slices(observations: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for lower, upper in ((0.0, 20.0), (20.0, 35.0), (35.0, 50.0)):
        selected = [
            item
            for item in observations
            if lower <= float(item["range_forward_m"]) < upper
            or (upper == 50.0 and float(item["range_forward_m"]) == upper)
        ]
        hits = sum(bool(item["matched"]) for item in selected)
        result.append(
            {
                "range_m": [lower, upper],
                "eligible_GT": len(selected),
                "true_positives": hits,
                "false_negatives": len(selected) - hits,
                "recall": hits / len(selected) if selected else None,
            }
        )
    return result


def _track_continuity(
    observations: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    tracks: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for item in observations:
        tracks[str(item["object_key"])].append(item)

    result: list[dict[str, object]] = []
    for key, items in sorted(tracks.items()):
        ordered = sorted(items, key=lambda item: int(item["frame_index"]))
        frame_indices = [int(item["frame_index"]) for item in ordered]
        detected = [bool(item["matched"]) for item in ordered]
        longest_hit, longest_miss = longest_consecutive_runs(frame_indices, detected)
        scores = [float(item["prediction_score"]) for item in ordered if item["matched"]]
        ranges = [float(item["range_forward_m"]) for item in ordered]
        result.append(
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
    return result


def _compact_ranked_summary(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "method": summary["method"],
        "ground_truth_count": summary["ground_truth_count"],
        "prediction_count": summary["prediction_count"],
        "average_precision": summary["average_precision"],
    }


def _aggregate_class_frames(
    frames: Sequence[Mapping[str, object]],
    *,
    total_postprocessed_predictions: int,
) -> dict[str, object]:
    thresholds: dict[str, object] = {}
    for threshold in ("0.30", "0.50", "0.70"):
        tp = sum(int(frame["thresholds"][threshold]["true_positives"]) for frame in frames)  # type: ignore[index]
        fp = sum(int(frame["thresholds"][threshold]["false_positives"]) for frame in frames)  # type: ignore[index]
        fn = sum(int(frame["thresholds"][threshold]["false_negatives"]) for frame in frames)  # type: ignore[index]
        ignored = sum(
            int(frame["thresholds"][threshold]["ignored_predictions"]) for frame in frames
        )  # type: ignore[index]
        thresholds[threshold] = {
            **count_metrics(tp, fp, fn),
            "ignored_predictions": ignored,
        }

    observations = [
        observation
        for frame in frames
        for observation in frame["target_observations"]  # type: ignore[union-attr]
    ]
    ignored_by_reason: dict[str, int] = defaultdict(int)
    for frame in frames:
        for reason, count in frame["ignored_predictions_by_reason"].items():  # type: ignore[union-attr]
            ignored_by_reason[str(reason)] += int(count)
    ranked = [
        record
        for frame in frames
        for record in frame["ranked_dispositions"]  # type: ignore[union-attr]
    ]
    ranked_summary = all_points_average_precision(ranked, ground_truth_count=len(observations))
    primary = thresholds["0.50"]
    assert isinstance(primary, Mapping)
    prediction_population = {
        "total_postprocessed_prediction_population_all_scores_all_classes": (
            total_postprocessed_predictions
        ),
        "inside_FOV_prediction_count_score_0_25": sum(
            int(frame["inside_FOV_prediction_count_score_0_25"]) for frame in frames
        ),
        "outside_annotation_fov_predictions_score_0_25": sum(
            int(frame["outside_annotation_fov_predictions_score_0_25"]) for frame in frames
        ),
        "neighbour_ignore_GT_count": sum(
            int(frame["neighbour_ignore_GT_count"]) for frame in frames
        ),
    }
    return {
        "eligible_GT_count": len(observations),
        "thresholds": thresholds,
        "range_slices": _range_slices(observations),
        "prediction_population": prediction_population,
        "neighbour_ignore": {
            "ignored_predictions": int(primary["ignored_predictions"]),
            "ignored_by_reason": dict(sorted(ignored_by_reason.items())),
        },
        "track_level": _track_continuity(observations),
        "score_ranked_PR_summary": _compact_ranked_summary(ranked_summary),
    }


def _m6_secondary_class(
    m6b_result: Mapping[str, object], condition: str, class_name: str
) -> dict[str, object]:
    conditions = m6b_result.get("condition_results")
    frames = m6b_result.get("frame_results")
    if not isinstance(conditions, Mapping) or not isinstance(conditions.get(condition), Mapping):
        raise ProtocolViolation("frozen M6b condition result is malformed")
    if not isinstance(frames, Mapping) or not isinstance(frames.get(condition), list):
        raise ProtocolViolation("frozen M6b frame results are malformed")
    classes = conditions[condition].get("classes")
    if not isinstance(classes, Mapping) or not isinstance(classes.get(class_name), Mapping):
        raise ProtocolViolation("frozen M6b class result is malformed")
    record = classes[class_name]
    ranked = record.get("score_ranked_PR")
    if not isinstance(ranked, Mapping):
        raise ProtocolViolation("frozen M6b ranked result is malformed")
    total_predictions = sum(
        int(frame["execution"]["detection_count_all_postprocessed_scores"])
        for frame in frames[condition]
    )
    primary = record["thresholds"]["0.50"]  # type: ignore[index]
    return {
        "eligible_GT_count": int(record["eligible_GT_count"]),
        "thresholds": record["thresholds"],
        "range_slices": record["range_slices"],
        "prediction_population": {
            "total_postprocessed_prediction_population_all_scores_all_classes": total_predictions,
            "inside_FOV_prediction_count_score_0_25": int(
                record["inside_FOV_prediction_count_score_0_25"]
            ),
            "outside_annotation_fov_predictions_score_0_25": int(
                record["outside_annotation_fov_predictions_score_0_25"]
            ),
            "neighbour_ignore_GT_count": int(record["neighbour_ignore_GT_count"]),
        },
        "neighbour_ignore": {
            "ignored_predictions": int(primary["ignored_predictions"]),
            "ignored_by_reason": record["ignored_predictions_by_reason"],
        },
        "track_level": record["track_level"],
        "score_ranked_PR_summary": _compact_ranked_summary(ranked),
    }


def _primary_record(
    primary: Mapping[str, object], class_name: str, arm_name: str
) -> Mapping[str, object]:
    arms = primary.get("arms")
    if not isinstance(arms, Mapping) or not isinstance(arms.get(class_name), Mapping):
        raise ProtocolViolation("accepted primary arm table is malformed")
    record = arms[class_name].get(arm_name)
    if not isinstance(record, Mapping):
        raise ProtocolViolation("accepted primary arm record is malformed")
    return record


def _validate_primary_consistency(
    secondary: Mapping[str, object], primary: Mapping[str, object]
) -> None:
    thresholds = secondary.get("thresholds")
    ranked = secondary.get("score_ranked_PR_summary")
    if not isinstance(thresholds, Mapping) or not isinstance(thresholds.get("0.50"), Mapping):
        raise ProtocolViolation("secondary primary-threshold result is malformed")
    if not isinstance(ranked, Mapping):
        raise ProtocolViolation("secondary ranked result is malformed")
    threshold = thresholds["0.50"]
    for field in PRIMARY_CONSISTENCY_FIELDS:
        if threshold.get(field) != primary.get(field):
            raise ProtocolViolation(f"secondary primary result differs: {field}")
    if ranked.get("average_precision") != primary.get("average_precision"):
        raise ProtocolViolation("secondary average precision differs from accepted primary result")


def _pedestrian_factorial(arms: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    values = {name: float(arms[name]["thresholds"]["0.50"]["recall"]) for name in "ABCD"}  # type: ignore[index]
    return {
        "outcome": "pedestrian_recall_score_gte_0.25_oriented_bev_iou_gte_0.50",
        "values": values,
        "contrasts": factorial_contrasts(
            a=values["A"],
            b=values["B"],
            c=values["C"],
            d=values["D"],
        ),
        "descriptive_only": True,
        "F_excluded": True,
    }


def aggregate_secondary(
    *,
    checkpoint_root: Path,
    dataset_date_root: Path,
    m6b_result_path: Path,
    primary_arm_table_path: Path,
    secondary_implementation_commit: str,
) -> dict[str, object]:
    """Return the preregistered secondary record without detector execution."""

    if len(secondary_implementation_commit) != 40 or any(
        value not in "0123456789abcdef" for value in secondary_implementation_commit
    ):
        raise ProtocolViolation("secondary aggregation implementation commit is invalid")
    if (
        m6b_result_path.stat().st_size != M6B_RESULT_FULL_BYTES
        or _sha256_file(m6b_result_path) != M6B_RESULT_FULL_SHA256
    ):
        raise ProtocolViolation("frozen full M6b result identity differs")
    if _sha256_file(primary_arm_table_path) != PRIMARY_ARM_TABLE_SHA256:
        raise ProtocolViolation("accepted M7 primary arm-table identity differs")

    m6b_result = json.loads(m6b_result_path.read_text(encoding="utf-8"))
    primary = json.loads(primary_arm_table_path.read_text(encoding="utf-8"))
    frames = _load_completed_frames(checkpoint_root)
    camera, poses_by_frame = _load_poses(dataset_date_root)
    labelled_counts = _track_labelled_counts(poses_by_frame)

    results: dict[str, dict[str, object]] = {class_name: {} for class_name in CLASSES}
    baseline_conditions = {"A": "H10", "E": "H5"}
    for class_name in CLASSES:
        for arm_name, condition in baseline_conditions.items():
            result = _m6_secondary_class(m6b_result, condition, class_name)
            _validate_primary_consistency(result, _primary_record(primary, class_name, arm_name))
            results[class_name][arm_name] = result

        for arm in NEW_ARMS:
            class_frames = [
                _evaluate_frame_class(
                    frame_id=frame_id,
                    predictions=frames[arm][frame_id],
                    poses=poses_by_frame.get(frame_id, ()),
                    camera=camera,
                    class_name=class_name,
                    labelled_counts=labelled_counts,
                )
                for frame_id in sorted(frames[arm])
            ]
            result = _aggregate_class_frames(
                class_frames,
                total_postprocessed_predictions=sum(
                    len(predictions) for predictions in frames[arm].values()
                ),
            )
            _validate_primary_consistency(result, _primary_record(primary, class_name, arm.name))
            results[class_name][arm.name] = result

        results[class_name] = {
            name: results[class_name][name] for name in ("A", "B", "C", "D", "E", "F")
        }

    pedestrian_descriptive = {
        arm.name: {
            field: _primary_record(primary, "pedestrian", arm.name)[field]
            for field in (
                "g_ped",
                "e_only_recovery",
                "a_only_retention",
                "shared_retention",
                "neither_recovery",
            )
        }
        for arm in NEW_ARMS
    }
    return {
        "schema_version": "laserperception.m7.raw-secondary-characterization.v1",
        "status": "M7 MEASUREMENT COMPLETE — RAW PREREGISTERED RESULTS ONLY.",
        "interpretation_status": "SCIENTIFIC INTERPRETATION NOT YET FROZEN.",
        "detector_calls_added": 0,
        "secondary_aggregation_implementation_commit": secondary_implementation_commit,
        "source_identities": {
            "m6b_full_result_sha256": M6B_RESULT_FULL_SHA256,
            "primary_arm_table_sha256": PRIMARY_ARM_TABLE_SHA256,
        },
        "operating_points": {
            "score_threshold": OPERATING_SCORE,
            "oriented_bev_iou_thresholds": list(IOU_THRESHOLDS),
            "primary_oriented_bev_iou_threshold": 0.50,
            "evaluator_identity": EVALUATOR_IDENTITY,
            "precision_and_f1_scope": "annotation_conditioned_inside_reference_camera_FOV",
        },
        "arms": results,
        "pedestrian_recall_factorial": _pedestrian_factorial(results["pedestrian"]),
        "pedestrian_paired_descriptive": pedestrian_descriptive,
        "f_context": (
            "F is a natural, unthinned, long-span comparator at matched history-sweep count; "
            "it does not isolate temporal span as a unique cause."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dataset-date-root", type=Path, required=True)
    parser.add_argument("--m6b-result", type=Path, required=True)
    parser.add_argument("--primary-arm-table", type=Path, required=True)
    parser.add_argument("--secondary-implementation-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = aggregate_secondary(
        checkpoint_root=args.checkpoint_root,
        dataset_date_root=args.dataset_date_root,
        m6b_result_path=args.m6b_result,
        primary_arm_table_path=args.primary_arm_table,
        secondary_implementation_commit=args.secondary_implementation_commit,
    )
    atomic_write_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
