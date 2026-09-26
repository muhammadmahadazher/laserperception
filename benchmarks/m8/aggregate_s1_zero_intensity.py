"""Publish compact zero-intensity S1 measurements from verified raw passes, CPU-only."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.m8.aggregate_s1_measurement import (
    CLASSES,
    HISTORIES,
    METRIC_FIELDS,
    RANGE_BANDS,
    SCORE_THRESHOLD,
    THRESHOLDS,
    reduce_history,
    summarize_three,
    write_json,
)
from laserperception.evaluation.m8_s1_aggregation import aggregate_three_passes

ZEROI_ARMS = {"H10": "A2_zeroI", "H5": "E2_zeroI"}
INTERVENTION = "candidate intensity float32 +0"
ZEROI_COMMIT = "95fb66ac1f57c41f06f05bd9ef5dac27b1e3ea54"
PRIMARY_COMMIT = "6994d72c3e7691a86116d1417ac3ae08256d163f"
PROTOCOL_SHA256 = "c132f60257c6a39debb548461c79bd59c98325484d233db6095b441c638d8e88"
PRIMARY_RAW_SHA256 = "500f45b80a27f94c2e235840a7edbd76aa5bb620e6a0820baef7b4e7576e9a58"
PRIMARY_UUIDS = frozenset(
    {
        "3256b511-921e-4456-92c6-1fd5d5a8c380",
        "21f32e37-58c3-42e2-b9c1-2011b20e47b7",
        "b9c1ab4b-9ea2-428a-a41e-e70fb8528d87",
    }
)
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def canonical_sha(record: Mapping[str, object]) -> str:
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_passes(
    passes: Sequence[Mapping[str, Any]], ledger: Mapping[str, Any]
) -> list[tuple[str, str, str]]:
    """Require complete, independent passes and exact frozen input provenance."""

    if len(passes) != 3:
        raise ValueError("zero-intensity publication requires exactly three passes")
    ledger_rows = ledger.get("conditions")
    if ledger.get("zeroi_execution_commit") != ZEROI_COMMIT or not isinstance(ledger_rows, list):
        raise ValueError("wrong zero-intensity transformation ledger")
    mapping = [
        (row["condition_id"], row["primary_input_sha256"], row["zeroI_input_sha256"])
        for row in ledger_rows
    ]
    if len(mapping) != 856 or len({row[0] for row in mapping}) != 856:
        raise ValueError("transformation ledger is not one canonical 856-condition mapping")
    if any(not SHA256.fullmatch(value) for row in mapping for value in row[1:]):
        raise ValueError("malformed transformation ledger SHA256")
    uuids: set[str] = set()
    for number, raw in enumerate(passes, 1):
        uuid = raw.get("process_uuid")
        if not isinstance(uuid, str) or uuid in uuids or uuid in PRIMARY_UUIDS:
            raise ValueError(
                "zero-intensity processes must be distinct from each other and primary"
            )
        uuids.add(uuid)
        if (
            raw.get("status") != "COMPLETE"
            or raw.get("runtime_commit") != ZEROI_COMMIT
            or raw.get("logical_pass_id") != f"zero-intensity-pass-{number}"
        ):
            raise ValueError("wrong zero-intensity process status or execution identity")
        result_sha = raw.get("result_sha256")
        if (
            not isinstance(result_sha, str)
            or canonical_sha({key: value for key, value in raw.items() if key != "result_sha256"})
            != result_sha
        ):
            raise ValueError("raw pass result SHA256 mismatch")
        conditions = raw.get("conditions")
        if not isinstance(conditions, list) or len(conditions) != 856:
            raise ValueError("each pass requires 856 conditions")
        if Counter(row.get("history") for row in conditions) != {"H10": 428, "H5": 428}:
            raise ValueError("each pass requires 428 conditions per history")
        for index, condition in enumerate(conditions):
            observed = (
                f"{condition.get('frame_id')}/{condition.get('history')}",
                condition.get("primary_input_sha256"),
                condition.get("input_sha256"),
            )
            if observed != mapping[index] or condition.get("intervention") != INTERVENTION:
                raise ValueError(f"zero-intensity provenance mismatch at condition {index}")
    return mapping


def _compact_ap(record: Mapping[str, Any]) -> dict[str, object]:
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


def build_zeroi_raw(
    passes: Sequence[Mapping[str, Any]], ledger: Mapping[str, Any]
) -> dict[str, object]:
    validate_passes(passes, ledger)
    aggregate = aggregate_three_passes(passes)
    rows: list[dict[str, object]] = []
    for number, (raw, aggregate_pass) in enumerate(
        zip(passes, aggregate["passes"], strict=True), 1
    ):
        arms: dict[str, object] = {}
        for history, arm in ZEROI_ARMS.items():
            source_history = aggregate_pass[history]
            classes = {}
            for class_name in CLASSES:
                source = source_history["classes"][class_name]
                classes[class_name] = {
                    "thresholds": {
                        threshold: {
                            **{
                                field: source["thresholds"][threshold][field]
                                for field in METRIC_FIELDS
                            },
                            "matched_gt_identity_set": source["thresholds"][threshold][
                                "matched_gt_identity_set"
                            ],
                        }
                        for threshold in THRESHOLDS
                    },
                    "annotation_conditioned_AP_at_iou_0_50": _compact_ap(
                        source["annotation_conditioned_AP"]
                    ),
                }
            arms[arm] = {
                "history": history,
                "condition_count": source_history["condition_count"],
                "classes": classes,
            }
        rows.append(
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
    for arm in ZEROI_ARMS.values():
        classes = {}
        for class_name in CLASSES:
            records = [row["arms"][arm]["classes"][class_name] for row in rows]
            classes[class_name] = {
                "thresholds": {
                    threshold: {
                        field: summarize_three(
                            [record["thresholds"][threshold][field] for record in records]
                        )
                        for field in METRIC_FIELDS
                    }
                    for threshold in THRESHOLDS
                },
                "annotation_conditioned_AP_at_iou_0_50": summarize_three(
                    [
                        record["annotation_conditioned_AP_at_iou_0_50"]["average_precision"]
                        for record in records
                    ]
                ),
            }
        spread[arm] = {"classes": classes}
    contrast = {}
    for class_name, value in aggregate["paired_history_contrast"].items():
        contrast[class_name] = {
            **value,
            "formula": "recall(E2_zeroI_i) - recall(A2_zeroI_i)",
        }
    return {
        "schema_version": "laserperception.m8.s1.zero-intensity-raw.v1",
        "status": "RAW MEASUREMENT; SCIENTIFIC INTERPRETATION PENDING",
        "zeroi_execution_commit": ZEROI_COMMIT,
        "primary_historical_execution_commit": PRIMARY_COMMIT,
        "frozen_protocol_sha256": PROTOCOL_SHA256,
        "intervention": {"definition": INTERVENTION, "ieee754_float32_word": "0x00000000"},
        "accepted_processes": 3,
        "accepted_canonical_calls": 2568,
        "detector_calls_added": 0,
        "boxes_averaged": False,
        "operating_points": {
            "score_threshold": SCORE_THRESHOLD,
            "oriented_bev_iou_thresholds": [0.30, 0.50, 0.70],
            "annotation_conditioned_AP_iou_threshold": 0.50,
            "precision_scope": "annotation-conditioned inside reference-camera FOV",
        },
        "passes": rows,
        "spread": spread,
        "within_zeroi_history_contrast": contrast,
    }


def build_zeroi_secondary(
    passes: Sequence[Mapping[str, Any]], ledger: Mapping[str, Any]
) -> dict[str, object]:
    validate_passes(passes, ledger)
    rows = []
    for number, raw in enumerate(passes, 1):
        rows.append(
            {
                "pass": number,
                "logical_pass_id": raw["logical_pass_id"],
                "attempt_id": raw["attempt_id"],
                "process_uuid": raw["process_uuid"],
                "arms": {
                    ZEROI_ARMS[history]: reduce_history(raw["conditions"], history)
                    for history in HISTORIES
                },
            }
        )
    spread: dict[str, object] = {}
    for arm in ZEROI_ARMS.values():
        classes = {}
        for class_name in CLASSES:
            records = [row["arms"][arm]["classes"][class_name] for row in rows]
            ranges: dict[str, object] = {}
            tracks: dict[str, object] = {}
            for threshold in THRESHOLDS:
                ranges[threshold] = [
                    {
                        "range_m": records[0]["range_by_iou"][threshold][index]["range_m"],
                        **{
                            field: summarize_three(
                                [
                                    record["range_by_iou"][threshold][index][field]
                                    for record in records
                                ]
                            )
                            for field in (
                                "eligible_GT",
                                "true_positives",
                                "false_negatives",
                                "recall",
                            )
                        },
                    }
                    for index in range(len(RANGE_BANDS))
                ]
                track_sets = [
                    {
                        track["object_key"]: track
                        for track in record["track_continuity_by_iou"][threshold]
                    }
                    for record in records
                ]
                if not (track_sets[0].keys() == track_sets[1].keys() == track_sets[2].keys()):
                    raise ValueError("track identities differ across zero-intensity processes")
                tracks[threshold] = [
                    {
                        "object_key": key,
                        "eligible_eval_frame_count": track_sets[0][key][
                            "eligible_eval_frame_count"
                        ],
                        "forward_range_span_m": track_sets[0][key]["forward_range_span_m"],
                        **{
                            field: summarize_three(
                                [track_set[key][field] for track_set in track_sets]
                            )
                            for field in (
                                "detected_frames",
                                "detection_continuity_fraction",
                                "longest_consecutive_detected_run",
                                "longest_consecutive_miss_run",
                            )
                        },
                    }
                    for key in sorted(track_sets[0])
                ]
            classes[class_name] = {
                "range_by_iou": ranges,
                "track_continuity_by_iou": tracks,
                "prediction_population": {
                    field: summarize_three(
                        [record["prediction_population"][field] for record in records]
                    )
                    for field in records[0]["prediction_population"]
                },
                "neighbour_ignore": {
                    "neighbour_ignore_GT_count": summarize_three(
                        [
                            record["neighbour_ignore"]["neighbour_ignore_GT_count"]
                            for record in records
                        ]
                    ),
                    "ignored_predictions_by_iou": {
                        threshold: summarize_three(
                            [
                                record["neighbour_ignore"]["ignored_predictions_by_iou"][threshold]
                                for record in records
                            ]
                        )
                        for threshold in THRESHOLDS
                    },
                },
            }
        spread[arm] = {"classes": classes}
    return {
        "schema_version": "laserperception.m8.s1.zero-intensity-secondary-raw.v1",
        "status": "RAW MEASUREMENT; SCIENTIFIC INTERPRETATION PENDING",
        "detector_rerun": False,
        "detector_calls_added": 0,
        "operating_points": {
            "score_threshold": SCORE_THRESHOLD,
            "oriented_bev_iou_thresholds": [0.30, 0.50, 0.70],
            "range_bands_metres": [bounds for _name, bounds in RANGE_BANDS],
            "precision_scope": "annotation-conditioned inside reference-camera FOV",
        },
        "passes": rows,
        "spread": spread,
    }


def build_descriptive_comparison(
    primary: Mapping[str, Any], zeroi: Mapping[str, Any]
) -> dict[str, object]:
    """Compare separate realization distributions; never pair process indices."""

    if primary.get("schema_version") != "laserperception.m8.s1.primary-raw.v1":
        raise ValueError("comparison requires the frozen primary result schema")
    if [row.get("process_uuid") for row in primary.get("passes", [])] != [
        "3256b511-921e-4456-92c6-1fd5d5a8c380",
        "21f32e37-58c3-42e2-b9c1-2011b20e47b7",
        "b9c1ab4b-9ea2-428a-a41e-e70fb8528d87",
    ]:
        raise ValueError("comparison requires the frozen primary process identities")
    if len(zeroi.get("passes", [])) != 3:
        raise ValueError("comparison requires three primary and three zero-intensity passes")
    arms = {}
    for history, zero_arm in ZEROI_ARMS.items():
        primary_arm = {"H10": "A2", "H5": "E2"}[history]
        classes = {}
        for class_name in CLASSES:
            primary_values = [
                row["arms"][primary_arm]["classes"][class_name]["thresholds"]["0.50"]["recall"]
                for row in primary["passes"]
            ]
            zero_values = [
                row["arms"][zero_arm]["classes"][class_name]["thresholds"]["0.50"]["recall"]
                for row in zeroi["passes"]
            ]
            primary_summary = summarize_three(primary_values)
            zero_summary = summarize_three(zero_values)
            classes[class_name] = {
                "primary_recall": primary_summary,
                "zeroi_recall": zero_summary,
                "descriptive_difference_of_observed_medians": zero_summary["median"]
                - primary_summary["median"],
            }
        arms[zero_arm] = {"history": history, "primary_arm": primary_arm, "classes": classes}
    return {
        "schema_version": "laserperception.m8.s1.zero-intensity-comparison.v1",
        "primary_execution_commit": PRIMARY_COMMIT,
        "zeroi_execution_commit": ZEROI_COMMIT,
        "process_indices_paired": False,
        "inference": "descriptive distributions only; no causal or significance estimate",
        "arms": arms,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass-input", action="append", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--reconciliation", type=Path, required=True)
    parser.add_argument("--primary-raw", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--secondary-output", type=Path, required=True)
    parser.add_argument("--comparison-output", type=Path, required=True)
    parser.add_argument("--core-output", type=Path, required=True)
    args = parser.parse_args()
    passes = [json.loads(path.read_text(encoding="utf-8")) for path in args.pass_input]
    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    reconciliation = json.loads(args.reconciliation.read_text(encoding="utf-8"))
    if file_sha(args.primary_raw) != PRIMARY_RAW_SHA256:
        raise ValueError("primary raw artifact does not match the frozen published SHA256")
    primary = json.loads(args.primary_raw.read_text(encoding="utf-8"))
    validate_passes(passes, ledger)
    core = aggregate_three_passes(passes)
    raw = build_zeroi_raw(passes, ledger)
    descriptors = reconciliation.get("records")
    if (
        reconciliation.get("status") != "VALIDATED"
        or not isinstance(descriptors, list)
        or len(descriptors) != 3
    ):
        raise ValueError("three validated archive descriptors are required")
    for index, (descriptor, raw_pass, path) in enumerate(
        zip(descriptors, passes, args.pass_input, strict=True), 1
    ):
        if (
            descriptor["pass"] != index
            or descriptor["attempt_id"] != raw_pass["attempt_id"]
            or descriptor["process_uuid"] != raw_pass["process_uuid"]
            or descriptor["raw_pass_sha256"] != file_sha(path)
        ):
            raise ValueError("archive descriptor does not bind to raw pass")
    raw["process_descriptors"] = [
        {
            "pass": row["pass"],
            "logical_pass_id": f"zero-intensity-pass-{row['pass']}",
            "attempt_id": row["attempt_id"],
            "process_uuid": row["process_uuid"],
            "status": "COMPLETE",
            "accepted_canonical_calls": 856,
            "archive_sha256": row["archive_sha256"],
            "raw_pass_file_sha256": row["raw_pass_sha256"],
            "final_result_sha256": row["result_sha256"],
        }
        for row in descriptors
    ]
    secondary = build_zeroi_secondary(passes, ledger)
    comparison = build_descriptive_comparison(primary, raw)
    for path, record in (
        (args.core_output, core),
        (args.raw_output, raw),
        (args.secondary_output, secondary),
        (args.comparison_output, comparison),
    ):
        write_json(path, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
