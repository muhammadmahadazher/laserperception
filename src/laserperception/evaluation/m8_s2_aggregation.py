"""Deterministic, CPU-only review of future S2 condition evidence.

No detector, accelerator, or real ground-truth loader is imported here. The
three-pass formulas and paired Car sets are fixed by the frozen S2 protocol.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import cast

from laserperception.detection.m8_s1_runtime import canonical_json_sha256, sha256_file
from laserperception.detection.m8_s2_runtime import (
    ARMS,
    CONDITION_SCHEMA,
    FULL_LEDGER_SHA256,
    FULL_PASS_IDS,
    PARTITIONS_PATH,
    PARTITIONS_SHA256,
    PROTOCOL_SHA256,
    REPEATABILITY_IDS,
    M8S2ProtocolViolation,
    condition_ids,
)
from laserperception.evaluation.m6b_metrics import count_metrics
from laserperception.evaluation.m8_s1_aggregation import aggregate_ranked_ap

CLASSES = ("car", "pedestrian")
THRESHOLDS = ("0.30", "0.50", "0.70")
RANGE_BANDS = ("0_20", "20_35", "35_50")
SCORE_THRESHOLD = 0.25


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise M8S2ProtocolViolation(f"S2 {label} must be a nonnegative integer")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise M8S2ProtocolViolation(f"S2 {label} must be numeric")
    return float(value)


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise M8S2ProtocolViolation(f"S2 {label} must be an ordered list")
    return value


def _spread(values: Sequence[int | float]) -> dict[str, object]:
    if len(values) != 3:
        raise M8S2ProtocolViolation("S2 spread requires three complete passes")
    return {
        "pass_values": list(values),
        "minimum": min(values),
        "median": median(values),
        "maximum": max(values),
    }


def _optional_spread(values: Sequence[int | float | None]) -> dict[str, object]:
    if len(values) != 3:
        raise M8S2ProtocolViolation("S2 spread requires three complete passes")
    if all(value is None for value in values):
        return {"pass_values": list(values), "minimum": None, "median": None, "maximum": None}
    if any(value is None for value in values):
        raise M8S2ProtocolViolation("S2 denominator availability differs across passes")
    return _spread(cast(Sequence[int | float], values))


def validate_attempts(passes: Sequence[Mapping[str, object]], *, mode: str) -> None:
    """Reject missing, partial, reordered, spliced, or reused process evidence."""

    logical_ids = (
        REPEATABILITY_IDS
        if mode == "repeatability"
        else FULL_PASS_IDS
        if mode == "full-pass"
        else ()
    )
    if len(passes) != len(logical_ids):
        raise M8S2ProtocolViolation("S2 process count differs")
    if tuple(row.get("logical_pass_id") for row in passes) != logical_ids:
        raise M8S2ProtocolViolation("S2 logical pass order differs")
    process_ids = [row.get("process_uuid") for row in passes]
    attempt_ids = [row.get("attempt_id") for row in passes]
    if any(not isinstance(value, str) or not value for value in process_ids + attempt_ids):
        raise M8S2ProtocolViolation("S2 process or attempt identity is absent")
    if len(set(process_ids)) != len(passes) or len(set(attempt_ids)) != len(passes):
        raise M8S2ProtocolViolation("S2 process or attempt identity was reused")
    for key in (
        "execution_commit",
        "runtime_policy_sha256",
        "input_gate_receipt_sha256",
        "qualification_receipt_sha256",
        "full_ledger_sha256",
        "protocol_sha256",
    ):
        if len({row.get(key) for row in passes}) != 1:
            raise M8S2ProtocolViolation(f"S2 process binding differs: {key}")
    if any(
        row.get("full_ledger_sha256") != FULL_LEDGER_SHA256
        or row.get("protocol_sha256") != PROTOCOL_SHA256
        for row in passes
    ):
        raise M8S2ProtocolViolation("S2 process frozen input or protocol identity differs")
    expected = condition_ids(mode)
    for row in passes:
        authorization_sha = row.get("authorization_sha256")
        qualification_sha = row.get("qualification_receipt_sha256")
        if (
            not isinstance(row.get("authorization_id"), str)
            or not row["authorization_id"]
            or not isinstance(authorization_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", authorization_sha) is None
            or not isinstance(qualification_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", qualification_sha) is None
        ):
            raise M8S2ProtocolViolation("S2 attempt authorization chain is absent")
        result_sha = row.get("result_sha256")
        if not isinstance(result_sha, str) or re.fullmatch(r"[0-9a-f]{64}", result_sha) is None:
            raise M8S2ProtocolViolation("S2 atomic attempt result identity is absent")
        conditions = row.get("conditions")
        if (
            row.get("mode") != mode
            or row.get("status") != "COMPLETE"
            or row.get("expected_calls") != len(expected)
            or row.get("completed_calls") != len(expected)
            or row.get("accepted_canonical_calls") != len(expected)
            or row.get("failed_calls") != 0
            or not isinstance(conditions, list)
            or tuple(item.get("condition_id") for item in conditions) != expected
        ):
            raise M8S2ProtocolViolation("S2 incomplete or reordered process is not aggregatable")


def load_completed_attempt(root: Path, *, mode: str) -> dict[str, object]:
    """Verify every condition file against one atomic final manifest."""

    try:
        manifest = json.loads((root / "final_pass_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise M8S2ProtocolViolation("S2 complete attempt manifest is absent") from error
    if not isinstance(manifest, dict) or manifest.get("status") != "COMPLETE":
        raise M8S2ProtocolViolation("S2 attempt is incomplete")
    unsigned = dict(manifest)
    digest = unsigned.pop("result_sha256", None)
    if digest != canonical_json_sha256(unsigned):
        raise M8S2ProtocolViolation("S2 final attempt identity differs")
    expected = condition_ids(mode)
    hashes = manifest.get("condition_file_sha256")
    if not isinstance(hashes, list) or len(hashes) != len(expected):
        raise M8S2ProtocolViolation("S2 condition file hashes are incomplete")
    if manifest.get("completed_condition_ids") != list(expected):
        raise M8S2ProtocolViolation("S2 final condition order differs")
    records: list[dict[str, object]] = []
    for index, condition_id in enumerate(expected):
        path = root / "conditions" / f"{index:04d}.json"
        if sha256_file(path) != hashes[index]:
            raise M8S2ProtocolViolation("S2 condition file SHA256 differs")
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(record, dict)
            or record.get("schema_version") != CONDITION_SCHEMA
            or record.get("condition_id") != condition_id
        ):
            raise M8S2ProtocolViolation("S2 condition file identity differs")
        unsigned_record = dict(record)
        record_sha = unsigned_record.pop("record_sha256", None)
        if record_sha != canonical_json_sha256(unsigned_record):
            raise M8S2ProtocolViolation("S2 condition record self-identity differs")
        payload = record.get("payload")
        if not isinstance(payload, dict) or payload.get("condition_id") != condition_id:
            raise M8S2ProtocolViolation("S2 condition payload differs")
        identity_keys = (
            "mode",
            "logical_pass_id",
            "attempt_id",
            "process_uuid",
            "process_id",
            "execution_commit",
            "runtime_policy_sha256",
            "input_gate_receipt_sha256",
            "qualification_receipt_sha256",
            "authorization_id",
            "authorization_sha256",
            "full_ledger_sha256",
            "protocol_sha256",
        )
        if record.get("identity") != {key: manifest.get(key) for key in identity_keys}:
            raise M8S2ProtocolViolation("S2 condition process binding differs")
        records.append(payload)
    if len(list((root / "conditions").glob("*.json"))) != len(expected):
        raise M8S2ProtocolViolation("S2 condition file population differs")
    return {**manifest, "conditions": records}


def _class_record(condition: Mapping[str, object], class_name: str) -> Mapping[str, object]:
    classes = condition.get("classes")
    if not isinstance(classes, Mapping) or not isinstance(classes.get(class_name), Mapping):
        raise M8S2ProtocolViolation("S2 class evidence is absent")
    return cast(Mapping[str, object], classes[class_name])


def _threshold(record: Mapping[str, object], threshold: str) -> Mapping[str, object]:
    thresholds = record.get("thresholds")
    if not isinstance(thresholds, Mapping) or not isinstance(thresholds.get(threshold), Mapping):
        raise M8S2ProtocolViolation("S2 threshold evidence is absent")
    return cast(Mapping[str, object], thresholds[threshold])


def _repeatability_signature(
    condition: Mapping[str, object], class_name: str
) -> tuple[object, ...]:
    evidence = _class_record(condition, class_name)
    primary = _threshold(evidence, "0.50")
    signature: list[object] = [
        _integer(evidence.get("thresholded_prediction_count"), "prediction count")
    ]
    for name in ("true_positives", "false_positives", "false_negatives", "ignored_predictions"):
        signature.append(_integer(primary.get(name), name))
    signature.append(
        tuple(
            sorted(str(item) for item in _list(primary.get("matched_gt_identity_set"), "matches"))
        )
    )
    for threshold in ("0.30", "0.70"):
        record = _threshold(evidence, threshold)
        signature.append(_integer(record.get("true_positives"), "TP"))
        signature.append(
            tuple(
                sorted(
                    str(item) for item in _list(record.get("matched_gt_identity_set"), "matches")
                )
            )
        )
    return tuple(signature)


def review_repeatability(
    passes: Sequence[Mapping[str, object]], *, aggregation_commit: str | None = None
) -> dict[str, object]:
    """Require exact frozen discrete agreement across ten fresh processes."""

    validate_attempts(passes, mode="repeatability")
    if aggregation_commit is None:
        aggregation_commit = str(passes[0]["execution_commit"])
    if aggregation_commit != passes[0]["execution_commit"]:
        raise M8S2ProtocolViolation("S2 aggregation checkout differs from science commit")
    first = cast(list[Mapping[str, object]], passes[0]["conditions"])
    comparisons = 0
    for later in passes[1:]:
        conditions = cast(list[Mapping[str, object]], later["conditions"])
        for reference, current in zip(first, conditions, strict=True):
            for class_name in CLASSES:
                if _repeatability_signature(reference, class_name) != _repeatability_signature(
                    current, class_name
                ):
                    raise M8S2ProtocolViolation(
                        f"S2 repeatability differs: {reference['condition_id']}/{class_name}"
                    )
                comparisons += 1
    result: dict[str, object] = {
        "schema_version": "laserperception.m8.s2.repeatability-review.v1",
        "status": "ACCEPTED",
        "processes": 10,
        "calls_per_process": 28,
        "accepted_calls": 280,
        "discrete_comparisons": comparisons,
        "process_uuids": [row["process_uuid"] for row in passes],
        "source_attempts": [
            {
                "logical_pass_id": row["logical_pass_id"],
                "attempt_id": row["attempt_id"],
                "process_uuid": row["process_uuid"],
                "result_sha256": row["result_sha256"],
                "qualification_receipt_sha256": row["qualification_receipt_sha256"],
                "authorization_id": row["authorization_id"],
                "authorization_sha256": row["authorization_sha256"],
            }
            for row in passes
        ],
        "execution_commit": passes[0]["execution_commit"],
        "aggregation_commit": aggregation_commit,
        "runtime_policy_sha256": passes[0]["runtime_policy_sha256"],
        "input_gate_receipt_sha256": passes[0]["input_gate_receipt_sha256"],
        "qualification_receipt_sha256": passes[0]["qualification_receipt_sha256"],
        "owner_reviewed": False,
        "full_corpus_authorized": False,
    }
    result["result_sha256"] = canonical_json_sha256(result)
    return result


def _pose_key(condition: Mapping[str, object], observation: Mapping[str, object]) -> str:
    frame_id = str(condition["frame_id"])
    return f"{frame_id}/track_{_integer(observation['track_id'], 'track ID')}"


def _outside_fov_count(condition: Mapping[str, object], class_name: str) -> int:
    predictions = _list(condition.get("predictions"), "stable predictions")
    if any(not isinstance(item, Mapping) for item in predictions):
        raise M8S2ProtocolViolation("S2 stable prediction record is malformed")
    outside = [
        item
        for item in predictions
        if isinstance(item, Mapping) and item.get("inside_annotation_fov") is False
    ]
    if len(outside) != _integer(
        condition.get("outside_annotation_fov_prediction_count"), "outside FOV"
    ):
        raise M8S2ProtocolViolation("S2 outside-FOV count differs from stable predictions")
    return sum(
        item.get("class_name") == class_name
        and _number(item.get("score"), "prediction score") >= SCORE_THRESHOLD
        for item in outside
    )


def _class_aggregate(
    selected: Sequence[Mapping[str, object]], class_name: str
) -> dict[str, object]:
    evidence = [_class_record(row, class_name) for row in selected]
    thresholds: dict[str, object] = {}
    for threshold in THRESHOLDS:
        rows = [_threshold(record, threshold) for record in evidence]
        tp = sum(_integer(row.get("true_positives"), "TP") for row in rows)
        fp = sum(_integer(row.get("false_positives"), "FP") for row in rows)
        fn = sum(_integer(row.get("false_negatives"), "FN") for row in rows)
        ignored = sum(_integer(row.get("ignored_predictions"), "ignored") for row in rows)
        matched_ids = sorted(
            {
                str(value)
                for row in rows
                for value in _list(row.get("matched_gt_identity_set"), "matches")
            }
        )
        thresholds[threshold] = {
            **count_metrics(tp, fp, fn),
            "ignored_predictions": ignored,
            "matched_gt_identity_set": matched_ids,
        }
    primary = cast(Mapping[str, object], thresholds["0.50"])
    raw_ranked = [
        item
        for record in evidence
        for item in _list(record.get("ranked_dispositions"), "ranked dispositions")
    ]
    if any(not isinstance(item, Mapping) for item in raw_ranked):
        raise M8S2ProtocolViolation("S2 ranked disposition is malformed")
    ranked = cast(list[Mapping[str, object]], raw_ranked)
    observations_by_iou: dict[str, dict[str, bool]] = {value: {} for value in THRESHOLDS}
    range_counts: dict[str, dict[str, dict[str, int | float | None]]] = {
        value: {band: {"targets": 0, "matched": 0} for band in RANGE_BANDS} for value in THRESHOLDS
    }
    tracks: dict[str, dict[str, list[tuple[int, bool]]]] = {
        value: defaultdict(list) for value in THRESHOLDS
    }
    for condition, record in zip(selected, evidence, strict=True):
        matched_by_iou = {}
        for threshold in THRESHOLDS:
            identities = _list(
                _threshold(record, threshold).get("matched_gt_identity_set"), "matches"
            )
            if any(not isinstance(value, str) or not value for value in identities):
                raise M8S2ProtocolViolation("S2 matched GT identity is malformed")
            if len(set(identities)) != len(identities):
                raise M8S2ProtocolViolation("S2 matched GT identity is duplicated")
            matched_by_iou[threshold] = set(identities)
        observed_identities: set[str] = set()
        for raw_item in _list(record.get("target_observations"), "target observations"):
            if not isinstance(raw_item, Mapping):
                raise M8S2ProtocolViolation("S2 target observation is malformed")
            item = raw_item
            key = _pose_key(condition, item)
            if key in observations_by_iou["0.50"]:
                raise M8S2ProtocolViolation("S2 target pose was counted twice")
            identity = str(item["gt_identity"])
            if identity in observed_identities:
                raise M8S2ProtocolViolation("S2 target identity was counted twice")
            observed_identities.add(identity)
            forward_m = _number(item.get("range_forward_m"), "forward range")
            if not math.isfinite(forward_m):
                raise M8S2ProtocolViolation("S2 forward range must be finite")
            band = (
                "0_20"
                if 0.0 <= forward_m < 20.0
                else "20_35"
                if forward_m < 35.0 and forward_m >= 20.0
                else "35_50"
                if forward_m < 50.0 and forward_m >= 35.0
                else None
            )
            for threshold in THRESHOLDS:
                is_matched = identity in matched_by_iou[threshold]
                if threshold == "0.50" and item.get("matched") is not is_matched:
                    raise M8S2ProtocolViolation("S2 primary match differs from GT identities")
                observations_by_iou[threshold][key] = is_matched
                if band is not None:
                    counts = range_counts[threshold][band]
                    counts["targets"] = _integer(counts["targets"], "range targets") + 1
                    counts["matched"] = _integer(counts["matched"], "range matches") + int(
                        is_matched
                    )
                tracks[threshold][identity].append(
                    (_integer(item["frame_index"], "frame index"), is_matched)
                )
        if any(not identities <= observed_identities for identities in matched_by_iou.values()):
            raise M8S2ProtocolViolation("S2 matched GT identity lacks a target observation")
    for threshold in THRESHOLDS:
        for band in RANGE_BANDS:
            counts = range_counts[threshold][band]
            targets = _integer(counts["targets"], "range targets")
            matched = _integer(counts["matched"], "range matches")
            counts["false_negatives"] = targets - matched
            counts["recall"] = matched / targets if targets else None
    continuity: dict[str, dict[str, dict[str, int]]] = {}
    for threshold in THRESHOLDS:
        if sum(observations_by_iou[threshold].values()) != _integer(
            cast(Mapping[str, object], thresholds[threshold])["true_positives"], "TP"
        ):
            raise M8S2ProtocolViolation("S2 matched target count differs from TP")
        continuity[threshold] = {}
        for track, poses in sorted(tracks[threshold].items()):
            ordered = sorted(poses)
            longest = current = 0
            prior = None
            for frame_index, is_matched in ordered:
                current = (
                    current + 1 if is_matched and prior == frame_index - 1 else int(is_matched)
                )
                longest = max(longest, current)
                prior = frame_index
            continuity[threshold][track] = {
                "target_poses": len(ordered),
                "matched_poses": sum(item[1] for item in ordered),
                "longest_consecutive_matched": longest,
            }
    return {
        "thresholds": thresholds,
        "annotation_conditioned_AP": aggregate_ranked_ap(
            ranked,
            ground_truth_count=_integer(primary["true_positives"], "TP")
            + _integer(primary["false_negatives"], "FN"),
        ),
        "thresholded_prediction_count": sum(
            _integer(item.get("thresholded_prediction_count"), "prediction count")
            for item in evidence
        ),
        "outside_annotation_fov_prediction_count": sum(
            _outside_fov_count(row, class_name) for row in selected
        ),
        "neighbour_ignore_GT_count": sum(
            _integer(item.get("neighbour_ignore_GT_count"), "neighbour GT") for item in evidence
        ),
        "range": range_counts,
        "track_continuity": continuity,
        "target_pose_detection": observations_by_iou["0.50"],
        "target_pose_detection_by_iou": observations_by_iou,
    }


def _continuity_spread(class_results: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_pass = [cast(Mapping[str, object], row["track_continuity"]) for row in class_results]
    spread: dict[str, object] = {}
    for threshold in THRESHOLDS:
        tracks = [cast(Mapping[str, object], row[threshold]) for row in by_pass]
        if any(set(row) != set(tracks[0]) for row in tracks[1:]):
            raise M8S2ProtocolViolation("S2 continuity track set differs across passes")
        spread[threshold] = {
            track: {
                key: _spread(
                    [
                        _integer(
                            cast(Mapping[str, object], row[track]).get(key),
                            f"track continuity {key}",
                        )
                        for row in tracks
                    ]
                )
                for key in (
                    "target_poses",
                    "matched_poses",
                    "longest_consecutive_matched",
                )
            }
            for track in sorted(tracks[0])
        }
    return spread


def _range_value(
    class_result: Mapping[str, object], threshold: str, band: str, key: str
) -> float | None:
    ranges = cast(Mapping[str, object], class_result["range"])
    bands = cast(Mapping[str, object], ranges[threshold])
    counts = cast(Mapping[str, object], bands[band])
    value = counts.get(key)
    return None if value is None else _number(value, key)


def aggregate_one_full_pass(raw: Mapping[str, object]) -> dict[str, object]:
    """Aggregate one complete process without mixing condition outputs."""

    conditions = cast(list[Mapping[str, object]], raw["conditions"])
    result: dict[str, object] = {}
    for arm in ARMS:
        selected = [row for row in conditions if row.get("arm") == arm]
        if len(selected) != 428:
            raise M8S2ProtocolViolation("S2 arm count differs")
        result[arm] = {
            "condition_count": 428,
            "classes": {
                class_name: _class_aggregate(selected, class_name) for class_name in CLASSES
            },
        }
    return result


def _partitions(root: Path) -> dict[str, set[str]]:
    if sha256_file(root / PARTITIONS_PATH) != PARTITIONS_SHA256:
        raise M8S2ProtocolViolation("S2 frozen partitions SHA256 differs")
    payload = json.loads((root / PARTITIONS_PATH).read_text(encoding="utf-8"))
    car = payload["classes"]["car"]
    expected = {"shared": 19, "e2_only": 24, "a2_only": 0, "neither": 23, "unstable": 0}
    result = {}
    for name, count in expected.items():
        bucket = car.get(name)
        if not isinstance(bucket, Mapping) or bucket.get("count") != count:
            raise M8S2ProtocolViolation(f"S2 Car partition differs: {name}")
        result[name] = {
            f"{item['drive_id']}/{int(item['frame_index']):010d}/track_{item['gt_track_id']}"
            for item in bucket["identities"]
        }
        if len(result[name]) != count:
            raise M8S2ProtocolViolation("S2 Car partition contains duplicate poses")
    if len(set.union(*result.values())) != 66:
        raise M8S2ProtocolViolation("S2 Car partitions overlap")
    return result


def _car_recovery(
    class_result: Mapping[str, object], partitions: Mapping[str, set[str]]
) -> dict[str, object]:
    observed = cast(Mapping[str, bool], class_result["target_pose_detection"])
    if set(observed) != set.union(*partitions.values()):
        raise M8S2ProtocolViolation("S2 Car pose denominator differs from frozen partitions")
    detected = {key for key, matched in observed.items() if matched}
    primary = cast(
        Mapping[str, object], cast(Mapping[str, object], class_result["thresholds"])["0.50"]
    )
    tp = _integer(primary["true_positives"], "Car TP")
    if tp != len(detected):
        raise M8S2ProtocolViolation("S2 Car TP differs from matched pose identities")
    counts = {name: len(detected & identities) for name, identities in partitions.items()}
    lost_shared = sorted(partitions["shared"] - detected)
    gain = (tp - 19) / 24
    r_gain = counts["e2_only"] / 24
    return {
        "G_car": gain,
        "R_gain": r_gain,
        "R_shared": counts["shared"] / 19,
        "R_Aonly": None,
        "R_neither": counts["neither"] / 23,
        "detected_counts_by_partition": counts,
        "detected_pose_ids": sorted(detected),
        "gained_E2_only_pose_ids": sorted(detected & partitions["e2_only"]),
        "lost_shared_pose_ids": lost_shared,
        "shared_losses": len(lost_shared),
        "interpretation_gate": gain >= 0.5 and r_gain >= 0.5 and len(lost_shared) <= 1,
    }


def factorial_contrasts(b2: float, c2: float, d2: float, a_ref: float) -> dict[str, float]:
    """Frozen 2x2 lag/count contrasts; F2 is excluded."""

    return {
        "L": ((b2 - a_ref) + (d2 - c2)) / 2,
        "P": ((c2 - a_ref) + (d2 - b2)) / 2,
        "I": d2 - b2 - c2 + a_ref,
    }


def aggregate_three_full_passes(
    passes: Sequence[Mapping[str, object]],
    *,
    repository_root: Path,
    aggregation_commit: str | None = None,
) -> dict[str, object]:
    """Report pass 1/2/3 first, then descriptive spread and frozen gates."""

    validate_attempts(passes, mode="full-pass")
    if aggregation_commit is None:
        aggregation_commit = str(passes[0]["execution_commit"])
    if aggregation_commit != passes[0]["execution_commit"]:
        raise M8S2ProtocolViolation("S2 aggregation checkout differs from science commit")
    per_pass = [aggregate_one_full_pass(row) for row in passes]
    partitions = _partitions(repository_root)
    recovery: list[dict[str, object]] = []
    factorial: list[dict[str, object]] = []
    for result in per_pass:
        arm_recovery: dict[str, object] = {}
        car_tp = {}
        car_recall = {}
        for arm in ARMS:
            class_result = cast(
                Mapping[str, object], cast(Mapping[str, object], result[arm])["classes"]
            )["car"]
            assert isinstance(class_result, Mapping)
            arm_recovery[arm] = _car_recovery(class_result, partitions)
            primary = cast(
                Mapping[str, object], cast(Mapping[str, object], class_result["thresholds"])["0.50"]
            )
            car_tp[arm] = _number(primary.get("true_positives"), "Car TP")
            car_recall[arm] = _number(primary.get("recall"), "Car recall")
        recovery.append(arm_recovery)
        factorial.append(
            {
                "Car_TP": factorial_contrasts(car_tp["B2"], car_tp["C2"], car_tp["D2"], 19.0),
                "Car_recall": factorial_contrasts(
                    car_recall["B2"], car_recall["C2"], car_recall["D2"], 19 / 66
                ),
                "F2_excluded": True,
            }
        )
    spread: dict[str, object] = {}
    for arm in ARMS:
        classes: dict[str, object] = {}
        for class_name in CLASSES:
            class_results: list[Mapping[str, object]] = [
                cast(
                    Mapping[str, object],
                    cast(Mapping[str, object], cast(Mapping[str, object], row[arm])["classes"])[
                        class_name
                    ],
                )
                for row in per_pass
            ]
            threshold_spread = {}
            for threshold in THRESHOLDS:
                threshold_spread[threshold] = {
                    key: _spread(
                        [
                            _number(
                                cast(
                                    Mapping[str, object],
                                    cast(Mapping[str, object], item["thresholds"])[threshold],
                                ).get(key),
                                key,
                            )
                            for item in class_results
                        ]
                    )
                    for key in (
                        "true_positives",
                        "false_positives",
                        "false_negatives",
                        "precision",
                        "recall",
                        "f1",
                        "ignored_predictions",
                    )
                }
            population_spread = {
                key: _spread([_number(item.get(key), key) for item in class_results])
                for key in (
                    "thresholded_prediction_count",
                    "outside_annotation_fov_prediction_count",
                    "neighbour_ignore_GT_count",
                )
            }
            ap_spread = {
                key: _spread(
                    [
                        _number(
                            cast(Mapping[str, object], item["annotation_conditioned_AP"]).get(key),
                            key,
                        )
                        for item in class_results
                    ]
                )
                for key in ("average_precision",)
            }
            range_spread = {
                threshold: {
                    band: {
                        key: _optional_spread(
                            [_range_value(item, threshold, band, key) for item in class_results]
                        )
                        for key in ("targets", "matched", "false_negatives", "recall")
                    }
                    for band in RANGE_BANDS
                }
                for threshold in THRESHOLDS
            }
            classes[class_name] = {
                "thresholds": threshold_spread,
                "annotation_conditioned_AP": ap_spread,
                "prediction_population_and_fov": population_spread,
                "range": range_spread,
                "track_continuity_by_pass": [item["track_continuity"] for item in class_results],
                "track_continuity": _continuity_spread(class_results),
            }
        spread[arm] = {"classes": classes}
    recovery_spread = {
        arm: {
            key: _optional_spread(
                [
                    None
                    if (value := cast(Mapping[str, object], row[arm]).get(key)) is None
                    else _number(value, key)
                    for row in recovery
                ]
            )
            for key in ("G_car", "R_gain", "R_shared", "R_Aonly", "R_neither")
        }
        for arm in ARMS
    }
    gates = {
        arm: all(
            cast(Mapping[str, object], row[arm])["interpretation_gate"] is True for row in recovery
        )
        for arm in ARMS
    }
    output: dict[str, object] = {
        "schema_version": "laserperception.m8.s2.aggregate.v1",
        "status": "COMPLETE_INPUT_EVIDENCE_AGGREGATION",
        "aggregation_commit": aggregation_commit,
        "execution_binding": {
            key: passes[0][key]
            for key in (
                "execution_commit",
                "runtime_policy_sha256",
                "input_gate_receipt_sha256",
                "qualification_receipt_sha256",
                "full_ledger_sha256",
                "protocol_sha256",
            )
        },
        "source_attempts": [
            {
                "logical_pass_id": row["logical_pass_id"],
                "attempt_id": row["attempt_id"],
                "process_uuid": row["process_uuid"],
                "result_sha256": row["result_sha256"],
                "qualification_receipt_sha256": row["qualification_receipt_sha256"],
                "authorization_id": row["authorization_id"],
                "authorization_sha256": row["authorization_sha256"],
            }
            for row in passes
        ],
        "passes": per_pass,
        "spread": spread,
        "car_recovery_by_pass": recovery,
        "car_recovery_spread": recovery_spread,
        "car_factorial_by_pass": factorial,
        "interpretation_gate_all_three": gates,
        "accepted_processes": 3,
        "accepted_calls": 5136,
        "denominator": {"A_REF": 19, "E_REF": 43, "D_REF": 24, "fixed": True},
        "annotation_conditioned": True,
        "causal_claim": False,
    }
    output["result_sha256"] = canonical_json_sha256(output)
    return output
