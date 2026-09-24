"""Build compact M8 S1 primary and secondary raw-result records offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CLASSES = ("car", "pedestrian")
HISTORIES = ("H10", "H5")
ARM_BY_HISTORY = {"H10": "A2", "H5": "E2"}
THRESHOLDS = ("0.30", "0.50", "0.70")
RANGE_BANDS = (
    ("0_20", [0.0, 20.0]),
    ("20_35", [20.0, 35.0]),
    ("35_50", [35.0, 50.0]),
)
SCORE_THRESHOLD = 0.25
TOTAL_POPULATION_KEY = "total_postprocessed_prediction_population_all_scores_all_classes"
POINTPILLARS_RECALL = {
    "H10": {"car": 16 / 66, "pedestrian": 219 / 396},
    "H5": {"car": 48 / 66, "pedestrian": 268 / 396},
}
METRIC_FIELDS = (
    "true_positives",
    "false_positives",
    "false_negatives",
    "recall",
    "precision",
    "f1",
    "ignored_predictions",
)


def summarize_three(values: Sequence[Any]) -> dict[str, Any]:
    if len(values) != 3:
        raise ValueError("M8 S1 summaries require exactly three values")
    return {
        "pass_values": list(values),
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
    }


def _longest_runs(frame_indices: Sequence[int], detected: Sequence[bool]) -> tuple[int, int]:
    longest_hit = longest_miss = current_hit = current_miss = 0
    previous: int | None = None
    for frame, hit in zip(frame_indices, detected, strict=True):
        if previous is None or frame != previous + 1:
            current_hit = current_miss = 0
        if hit:
            current_hit += 1
            current_miss = 0
            longest_hit = max(longest_hit, current_hit)
        else:
            current_miss += 1
            current_hit = 0
            longest_miss = max(longest_miss, current_miss)
        previous = frame
    return longest_hit, longest_miss


def _class_conditions(
    conditions: Sequence[Mapping[str, object]], history: str, class_name: str
) -> list[tuple[Mapping[str, object], Mapping[str, object]]]:
    selected: list[tuple[Mapping[str, object], Mapping[str, object]]] = []
    for condition in conditions:
        if condition.get("history") != history:
            continue
        classes = condition.get("classes")
        if not isinstance(classes, Mapping) or not isinstance(classes.get(class_name), Mapping):
            raise ValueError("malformed M8 S1 class evidence")
        selected.append((condition, classes[class_name]))
    if not selected:
        raise ValueError(f"no {history} conditions")
    return selected


def _range_rows(
    records: Sequence[tuple[Mapping[str, object], Mapping[str, object]]], threshold: str
) -> list[dict[str, object]]:
    counts = {key: [0, 0] for key, _bounds in RANGE_BANDS}
    for _condition, class_record in records:
        thresholds = class_record["thresholds"]
        matched = set(thresholds[threshold]["matched_gt_identity_set"])  # type: ignore[index]
        observations = class_record["target_observations"]
        seen: set[str] = set()
        for observation in observations:  # type: ignore[union-attr]
            identity = str(observation["gt_identity"])
            if identity in seen:
                raise ValueError("duplicate GT identity within one condition")
            seen.add(identity)
            band = str(observation["range_band_metres"])
            if band not in counts:
                raise ValueError(f"unexpected M8 S1 range band: {band}")
            counts[band][0] += 1
            counts[band][1] += int(identity in matched)
        if len(matched) != int(thresholds[threshold]["true_positives"]):  # type: ignore[index]
            raise ValueError("matched identity count differs from primary threshold count")
    result = []
    for key, bounds in RANGE_BANDS:
        eligible, tp = counts[key]
        result.append(
            {
                "range_m": bounds,
                "eligible_GT": eligible,
                "true_positives": tp,
                "false_negatives": eligible - tp,
                "recall": tp / eligible if eligible else None,
            }
        )
    return result


def _track_rows(
    records: Sequence[tuple[Mapping[str, object], Mapping[str, object]]], threshold: str
) -> list[dict[str, object]]:
    tracks: dict[str, list[dict[str, object]]] = defaultdict(list)
    for _condition, class_record in records:
        matched = set(class_record["thresholds"][threshold]["matched_gt_identity_set"])  # type: ignore[index]
        for observation in class_record["target_observations"]:  # type: ignore[union-attr]
            identity = str(observation["gt_identity"])
            tracks[identity].append(
                {
                    "frame_index": int(observation["frame_index"]),
                    "range_forward_m": float(observation["range_forward_m"]),
                    "matched": identity in matched,
                }
            )
    result = []
    for identity, observations in sorted(tracks.items()):
        ordered = sorted(observations, key=lambda row: int(row["frame_index"]))
        frames = [int(row["frame_index"]) for row in ordered]
        detected = [bool(row["matched"]) for row in ordered]
        ranges = [float(row["range_forward_m"]) for row in ordered]
        longest_hit, longest_miss = _longest_runs(frames, detected)
        result.append(
            {
                "object_key": identity,
                "eligible_eval_frame_count": len(ordered),
                "detected_frames": sum(detected),
                "detection_continuity_fraction": sum(detected) / len(ordered),
                "longest_consecutive_detected_run": longest_hit,
                "longest_consecutive_miss_run": longest_miss,
                "forward_range_span_m": [min(ranges), max(ranges)],
            }
        )
    return result


def reduce_history(conditions: Sequence[Mapping[str, object]], history: str) -> dict[str, object]:
    selected_conditions = [row for row in conditions if row.get("history") == history]
    total_predictions = sum(len(row["predictions"]) for row in selected_conditions)  # type: ignore[arg-type]
    outside_all = sum(
        int(row["outside_annotation_fov_prediction_count"]) for row in selected_conditions
    )
    classes: dict[str, object] = {}
    for class_name in CLASSES:
        records = _class_conditions(conditions, history, class_name)
        inside = outside = 0
        for condition, _class_record in records:
            for prediction in condition["predictions"]:  # type: ignore[union-attr]
                if (
                    prediction["class_name"] == class_name
                    and float(prediction["score"]) >= SCORE_THRESHOLD
                ):
                    if bool(prediction["inside_annotation_fov"]):
                        inside += 1
                    else:
                        outside += 1
        threshold_totals: dict[str, object] = {}
        for threshold in THRESHOLDS:
            totals = {
                field: sum(int(record["thresholds"][threshold][field]) for _c, record in records)  # type: ignore[index]
                for field in (
                    "true_positives",
                    "false_positives",
                    "false_negatives",
                    "ignored_predictions",
                )
            }
            threshold_totals[threshold] = totals
        classes[class_name] = {
            "range_by_iou": {
                threshold: _range_rows(records, threshold) for threshold in THRESHOLDS
            },
            "track_continuity_by_iou": {
                threshold: _track_rows(records, threshold) for threshold in THRESHOLDS
            },
            "prediction_population": {
                TOTAL_POPULATION_KEY: total_predictions,
                "inside_FOV_prediction_count_score_0_25": inside,
                "outside_annotation_fov_predictions_score_0_25": outside,
                "outside_annotation_fov_predictions_all_scores_all_classes": outside_all,
            },
            "neighbour_ignore": {
                "neighbour_ignore_GT_count": sum(
                    int(record["neighbour_ignore_GT_count"]) for _c, record in records
                ),
                "ignored_predictions_by_iou": {
                    threshold: threshold_totals[threshold]["ignored_predictions"]  # type: ignore[index]
                    for threshold in THRESHOLDS
                },
            },
        }
    return {"condition_count": len(selected_conditions), "classes": classes}


def _secondary_spread(pass_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for history in HISTORIES:
        arm: dict[str, object] = {}
        for class_name in CLASSES:
            records = [
                row["arms"][ARM_BY_HISTORY[history]]["classes"][class_name] for row in pass_rows
            ]  # type: ignore[index]
            range_spread: dict[str, object] = {}
            track_spread: dict[str, object] = {}
            for threshold in THRESHOLDS:
                range_rows = []
                for index in range(len(RANGE_BANDS)):
                    row = {
                        "range_m": records[0]["range_by_iou"][threshold][index]["range_m"],  # type: ignore[index]
                    }
                    row.update(
                        {
                            field: summarize_three(
                                [
                                    record["range_by_iou"][threshold][index][field]
                                    for record in records  # type: ignore[index]
                                ]
                            )
                            for field in (
                                "eligible_GT",
                                "true_positives",
                                "false_negatives",
                                "recall",
                            )
                        }
                    )
                    range_rows.append(row)
                range_spread[threshold] = range_rows
                per_track: dict[str, list[Mapping[str, object]]] = defaultdict(list)
                for record in records:
                    for track in record["track_continuity_by_iou"][threshold]:  # type: ignore[index]
                        per_track[str(track["object_key"])].append(track)
                track_spread[threshold] = []
                for key, rows in sorted(per_track.items()):
                    if len(rows) != 3:
                        raise ValueError("track identity differs across accepted passes")
                    track_spread[threshold].append(
                        {
                            "object_key": key,
                            "eligible_eval_frame_count": rows[0]["eligible_eval_frame_count"],
                            "forward_range_span_m": rows[0]["forward_range_span_m"],
                            **{
                                field: summarize_three([row[field] for row in rows])
                                for field in (
                                    "detected_frames",
                                    "detection_continuity_fraction",
                                    "longest_consecutive_detected_run",
                                    "longest_consecutive_miss_run",
                                )
                            },
                        }
                    )
            population = {
                field: summarize_three(
                    [record["prediction_population"][field] for record in records]
                )  # type: ignore[index]
                for field in records[0]["prediction_population"]  # type: ignore[union-attr]
            }
            neighbour = {
                "neighbour_ignore_GT_count": summarize_three(
                    [
                        record["neighbour_ignore"]["neighbour_ignore_GT_count"]
                        for record in records  # type: ignore[index]
                    ]
                ),
                "ignored_predictions_by_iou": {
                    threshold: summarize_three(
                        [
                            record["neighbour_ignore"]["ignored_predictions_by_iou"][threshold]  # type: ignore[index]
                            for record in records
                        ]
                    )
                    for threshold in THRESHOLDS
                },
            }
            arm[class_name] = {
                "range_by_iou": range_spread,
                "track_continuity_by_iou": track_spread,
                "prediction_population": population,
                "neighbour_ignore": neighbour,
            }
        output[ARM_BY_HISTORY[history]] = {"classes": arm}
    return output


def _validate_commit(value: str) -> None:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("publication implementation commit must be a lowercase Git SHA")


def build_secondary(
    raw_passes: Sequence[Mapping[str, object]], *, publication_implementation_commit: str
) -> dict[str, object]:
    _validate_commit(publication_implementation_commit)
    if len(raw_passes) != 3:
        raise ValueError("M8 S1 secondary reduction requires exactly three passes")
    process_ids = [str(row.get("process_uuid")) for row in raw_passes]
    if len(set(process_ids)) != 3 or any(row.get("status") != "COMPLETE" for row in raw_passes):
        raise ValueError("secondary reduction requires three distinct complete processes")
    passes = []
    for number, raw in enumerate(raw_passes, 1):
        conditions = raw.get("conditions")
        if not isinstance(conditions, list) or len(conditions) != 856:
            raise ValueError("each M8 S1 pass must contain 856 conditions")
        arms = {
            ARM_BY_HISTORY[history]: reduce_history(conditions, history) for history in HISTORIES
        }
        passes.append(
            {
                "pass": number,
                "logical_pass_id": raw["logical_pass_id"],
                "attempt_id": raw["attempt_id"],
                "process_uuid": raw["process_uuid"],
                "raw_pass_result_sha256": raw["result_sha256"],
                "arms": arms,
            }
        )
    return {
        "schema_version": "laserperception.m8.s1.secondary-raw.v1",
        "status": "RAW PREREGISTERED MEASUREMENT; SCIENTIFIC INTERPRETATION NOT FROZEN",
        "detector_calls_added": 0,
        "publication_implementation_commit": publication_implementation_commit,
        "operating_points": {
            "score_threshold": SCORE_THRESHOLD,
            "oriented_bev_iou_thresholds": [0.30, 0.50, 0.70],
            "range_bands_metres": [bounds for _key, bounds in RANGE_BANDS],
            "precision_scope": "annotation-conditioned inside reference-camera FOV",
        },
        "passes": passes,
        "spread": _secondary_spread(passes),
        "three_pass_interpretation": "observed runtime/numerical spread only",
    }


def _compact_ap(record: Mapping[str, object]) -> dict[str, object]:
    return {
        field: record[field]
        for field in (
            "method",
            "ground_truth_count",
            "prediction_count",
            "average_precision",
            "interpretation",
        )
    }


def build_primary(
    frozen_aggregate: Mapping[str, object],
    raw_passes: Sequence[Mapping[str, object]],
    *,
    frozen_aggregate_sha256: str,
    publication_implementation_commit: str,
) -> dict[str, object]:
    _validate_commit(publication_implementation_commit)
    aggregate_passes = frozen_aggregate.get("passes")
    if not isinstance(aggregate_passes, list) or len(aggregate_passes) != 3 or len(raw_passes) != 3:
        raise ValueError("compact primary publication requires three frozen aggregate passes")
    passes = []
    for number, (aggregate_pass, raw) in enumerate(
        zip(aggregate_passes, raw_passes, strict=True), 1
    ):
        arms: dict[str, object] = {}
        for history in HISTORIES:
            classes: dict[str, object] = {}
            for class_name in CLASSES:
                source = aggregate_pass[history]["classes"][class_name]  # type: ignore[index]
                classes[class_name] = {
                    "thresholds": {
                        threshold: {
                            field: source["thresholds"][threshold][field] for field in METRIC_FIELDS
                        }
                        for threshold in THRESHOLDS
                    },
                    "annotation_conditioned_AP_at_iou_0_50": _compact_ap(
                        source["annotation_conditioned_AP"]
                    ),
                }
            arms[ARM_BY_HISTORY[history]] = {
                "history": history,
                "condition_count": aggregate_pass[history]["condition_count"],  # type: ignore[index]
                "classes": classes,
            }
        passes.append(
            {
                "pass": number,
                "logical_pass_id": raw["logical_pass_id"],
                "attempt_id": raw["attempt_id"],
                "process_uuid": raw["process_uuid"],
                "raw_pass_result_sha256": raw["result_sha256"],
                "arms": arms,
            }
        )

    spread: dict[str, object] = {}
    baseline: dict[str, object] = {}
    for history in HISTORIES:
        arm_name = ARM_BY_HISTORY[history]
        arm_spread: dict[str, object] = {}
        arm_baseline: dict[str, object] = {}
        for class_name in CLASSES:
            records = [row["arms"][arm_name]["classes"][class_name] for row in passes]  # type: ignore[index]
            arm_spread[class_name] = {
                "thresholds": {
                    threshold: {
                        field: summarize_three(
                            [
                                record["thresholds"][threshold][field]
                                for record in records  # type: ignore[index]
                            ]
                        )
                        for field in METRIC_FIELDS
                    }
                    for threshold in THRESHOLDS
                },
                "annotation_conditioned_AP_at_iou_0_50": summarize_three(
                    [
                        record["annotation_conditioned_AP_at_iou_0_50"]["average_precision"]  # type: ignore[index]
                        for record in records
                    ]
                ),
            }
            deltas = [
                float(record["thresholds"]["0.50"]["recall"])  # type: ignore[index]
                - POINTPILLARS_RECALL[history][class_name]
                for record in records
            ]
            arm_baseline[class_name] = {
                "formula": (f"{arm_name}_pass_recall - frozen_PointPillars_{history}_recall"),
                "frozen_PointPillars_recall": POINTPILLARS_RECALL[history][class_name],
                **summarize_three(deltas),
            }
        spread[arm_name] = {"classes": arm_spread}
        baseline[arm_name] = {"classes": arm_baseline}
    return {
        "schema_version": "laserperception.m8.s1.primary-raw.v1",
        "status": "RAW PREREGISTERED MEASUREMENT; SCIENTIFIC INTERPRETATION NOT FROZEN",
        "detector_calls_added": 0,
        "publication_implementation_commit": publication_implementation_commit,
        "frozen_aggregate_sha256": frozen_aggregate_sha256,
        "frozen_aggregation_schema": frozen_aggregate["schema_version"],
        "operating_points": {
            "score_threshold": SCORE_THRESHOLD,
            "oriented_bev_iou_thresholds": [0.30, 0.50, 0.70],
            "annotation_conditioned_AP_iou_threshold": 0.50,
            "precision_scope": "annotation-conditioned inside reference-camera FOV",
        },
        "passes": passes,
        "spread": spread,
        "paired_history_contrast": frozen_aggregate["paired_history_contrast"],
        "pointpillars_historical_recall_contrast": baseline,
        "three_pass_interpretation": frozen_aggregate["three_pass_interpretation"],
        "boxes_averaged": frozen_aggregate["boxes_averaged"],
    }


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass-input", action="append", type=Path, required=True)
    parser.add_argument("--frozen-aggregate", type=Path)
    parser.add_argument("--primary-output", type=Path)
    parser.add_argument("--secondary-output", type=Path)
    parser.add_argument("--publication-implementation-commit", required=True)
    args = parser.parse_args()
    passes = [json.loads(path.read_text(encoding="utf-8")) for path in args.pass_input]
    if args.secondary_output is not None:
        write_json(
            args.secondary_output,
            build_secondary(
                passes,
                publication_implementation_commit=args.publication_implementation_commit,
            ),
        )
    if args.primary_output is not None:
        if args.frozen_aggregate is None:
            parser.error("--primary-output requires --frozen-aggregate")
        aggregate_bytes = args.frozen_aggregate.read_bytes()
        aggregate = json.loads(aggregate_bytes)
        write_json(
            args.primary_output,
            build_primary(
                aggregate,
                passes,
                frozen_aggregate_sha256=hashlib.sha256(aggregate_bytes).hexdigest(),
                publication_implementation_commit=args.publication_implementation_commit,
            ),
        )
    if args.secondary_output is None and args.primary_output is None:
        parser.error("at least one output is required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
