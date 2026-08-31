"""Deterministic offline aggregation for frozen M8 P1-S1 raw pass evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from laserperception.detection.m8_s1_runtime import (
    M8S1ProtocolViolation,
    paired_history_delta,
    summarize_three,
)
from laserperception.evaluation.m6b_metrics import (
    RankedDisposition,
    all_points_average_precision,
    count_metrics,
)

CLASSES = ("car", "pedestrian")
HISTORIES = ("H10", "H5")
THRESHOLDS = ("0.30", "0.50", "0.70")
PROHIBITED_INFERENTIAL_FIELDS = frozenset(
    {"p_value", "p_values", "confidence_interval", "confidence_intervals", "standard_error"}
)


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise M8S1ProtocolViolation("M8 S1 metric must be numeric")
    return float(value)


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise M8S1ProtocolViolation("M8 S1 count must be an integer")
    return value


def _require_three_passes(passes: Sequence[Mapping[str, object]]) -> None:
    if len(passes) != 3:
        raise M8S1ProtocolViolation("M8 S1 aggregation requires exactly three passes")
    process_ids = {record.get("process_uuid") for record in passes}
    if None in process_ids or len(process_ids) != 3:
        raise M8S1ProtocolViolation("M8 S1 passes require three distinct process identities")
    if any(record.get("status") != "COMPLETE" for record in passes):
        raise M8S1ProtocolViolation("incomplete M8 S1 attempts are not aggregatable")


def aggregate_ranked_ap(
    records: Sequence[Mapping[str, object]], *, ground_truth_count: int
) -> dict[str, object]:
    """Delegate annotation-conditioned AP exactly to the frozen M6b implementation."""

    ranked = [
        RankedDisposition(
            score=_number(record["score"]),
            frame_id=str(record["frame_id"]),
            prediction_index=_integer(record["prediction_index"]),
            true_positive=bool(record["true_positive"]),
        )
        for record in records
    ]
    result = all_points_average_precision(ranked, ground_truth_count=ground_truth_count)
    result["interpretation"] = "annotation-conditioned descriptive AP"
    return result


def aggregate_one_pass(raw: Mapping[str, object]) -> dict[str, object]:
    """Aggregate per-condition frozen evidence for one process realization."""

    conditions = raw.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != 856:
        raise M8S1ProtocolViolation("one canonical S1 pass must contain 856 conditions")
    output: dict[str, object] = {}
    for history in HISTORIES:
        selected = [record for record in conditions if record.get("history") == history]
        if len(selected) != 428:
            raise M8S1ProtocolViolation(f"one S1 pass must contain 428 {history} conditions")
        class_output: dict[str, object] = {}
        for class_name in CLASSES:
            class_records = []
            for record in selected:
                classes = record.get("classes")
                if not isinstance(classes, Mapping) or not isinstance(
                    classes.get(class_name), Mapping
                ):
                    raise M8S1ProtocolViolation("S1 condition class evidence is malformed")
                class_records.append(classes[class_name])
            threshold_output: dict[str, object] = {}
            for threshold in THRESHOLDS:
                summaries = []
                for record in class_records:
                    thresholds = record.get("thresholds")
                    if not isinstance(thresholds, Mapping) or not isinstance(
                        thresholds.get(threshold), Mapping
                    ):
                        raise M8S1ProtocolViolation("S1 threshold evidence is malformed")
                    summaries.append(thresholds[threshold])
                tp = sum(_integer(summary["true_positives"]) for summary in summaries)
                fp = sum(_integer(summary["false_positives"]) for summary in summaries)
                fn = sum(_integer(summary["false_negatives"]) for summary in summaries)
                ignored = sum(_integer(summary["ignored_predictions"]) for summary in summaries)
                matched = sorted(
                    {
                        str(identity)
                        for summary in summaries
                        for identity in summary["matched_gt_identity_set"]
                    }
                )
                threshold_output[threshold] = {
                    **count_metrics(tp, fp, fn),
                    "ignored_predictions": ignored,
                    "matched_gt_identity_set": matched,
                }
            ranked = [
                disposition
                for record in class_records
                for disposition in record["ranked_dispositions"]
            ]
            primary = threshold_output["0.50"]
            assert isinstance(primary, Mapping)
            class_output[class_name] = {
                "thresholds": threshold_output,
                "annotation_conditioned_AP": aggregate_ranked_ap(
                    ranked,
                    ground_truth_count=_integer(primary["true_positives"])
                    + _integer(primary["false_negatives"]),
                ),
            }
        output[history] = {"condition_count": len(selected), "classes": class_output}
    return output


def aggregate_three_passes(passes: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Return pass 1/2/3 plus min/median/max and paired history contrasts."""

    _require_three_passes(passes)
    per_pass = [aggregate_one_pass(record) for record in passes]
    spread: dict[str, object] = {}
    history: dict[str, object] = {}
    for class_name in CLASSES:
        class_spread: dict[str, object] = {}
        h10_recalls: list[float] = []
        h5_recalls: list[float] = []
        for condition in HISTORIES:
            recalls = []
            for pass_result in per_pass:
                condition_result = cast(Mapping[str, object], pass_result[condition])
                classes = cast(Mapping[str, object], condition_result["classes"])
                class_result = cast(Mapping[str, object], classes[class_name])
                thresholds = cast(Mapping[str, object], class_result["thresholds"])
                primary = cast(Mapping[str, object], thresholds["0.50"])
                recalls.append(_number(primary["recall"]))
            class_spread[condition] = {"primary_recall": summarize_three(recalls)}
            (h10_recalls if condition == "H10" else h5_recalls).extend(recalls)
        spread[class_name] = class_spread
        history[class_name] = paired_history_delta(h10_recalls, h5_recalls)
    result = {
        "schema_version": "laserperception.m8.s1.aggregate.v1",
        "passes": per_pass,
        "spread": spread,
        "paired_history_contrast": history,
        "three_pass_interpretation": "observed runtime/numerical spread only",
        "boxes_averaged": False,
    }
    if any(key in PROHIBITED_INFERENTIAL_FIELDS for key in result):
        raise AssertionError("prohibited inferential field escaped M8 S1 aggregation")
    return result
