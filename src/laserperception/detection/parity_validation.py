"""Frozen M2 threshold diagnostics and aggregate acceptance evaluation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from math import ceil
from statistics import median
from typing import Any, cast

from laserperception.detection.parity import DetectionMatch, MatchingResult, match_detections
from laserperception.detection.types import DetectionFrame


def is_threshold_edge_score(
    score: float, *, minimum_inclusive: float = 0.20, maximum_inclusive: float = 0.30
) -> bool:
    """Return whether a score lies in the frozen inclusive diagnostic band."""

    return minimum_inclusive <= float(score) <= maximum_inclusive


def analyze_sample(
    reference: DetectionFrame,
    candidate: DetectionFrame,
    *,
    sample_index: int,
    exported_threshold: float = 0.25,
    high_confidence_threshold: float = 0.30,
    minimum_bev_iou: float = 0.50,
) -> dict[str, object]:
    """Match one sample and serialize every frozen final-detection diagnostic."""

    if reference.sample_id != candidate.sample_id:
        raise ValueError("parity frames must refer to the same sample ID")
    matching = match_detections(
        reference.detections,
        candidate.detections,
        minimum_bev_iou=minimum_bev_iou,
    )
    reference_exported = tuple(
        detection for detection in reference.detections if detection.score >= exported_threshold
    )
    candidate_exported = tuple(
        detection for detection in candidate.detections if detection.score >= exported_threshold
    )
    matches = [_match_record(match, high_confidence_threshold) for match in matching.matches]
    threshold_edges = _threshold_edges(matching, exported_threshold)
    return {
        "sample_index": sample_index,
        "sample_id": reference.sample_id,
        "counts": {
            "pytorch_raw_postprocess": len(reference.detections),
            "tensorrt_raw_postprocess": len(candidate.detections),
            "pytorch_exported": len(reference_exported),
            "tensorrt_exported": len(candidate_exported),
            "absolute_exported_difference": abs(len(candidate_exported) - len(reference_exported)),
            "allowed_exported_difference": max(1, ceil(0.05 * len(reference_exported))),
            "pytorch_high_confidence": sum(
                detection.score >= high_confidence_threshold for detection in reference.detections
            ),
            "tensorrt_high_confidence": sum(
                detection.score >= high_confidence_threshold for detection in candidate.detections
            ),
        },
        "per_class_exported_counts": {
            "pytorch": dict(Counter(item.class_name for item in reference_exported)),
            "tensorrt": dict(Counter(item.class_name for item in candidate_exported)),
        },
        "matches": matches,
        "unmatched": {
            "pytorch": [item.to_dict() for item in matching.unmatched_reference],
            "tensorrt": [item.to_dict() for item in matching.unmatched_candidate],
        },
        "threshold_edge_disagreements": threshold_edges,
    }


def aggregate_acceptance(
    reports: Sequence[Mapping[str, Any]],
    *,
    minimum_coverage: float = 0.99,
    maximum_xy_m: float = 0.25,
    maximum_z_m: float = 0.25,
    maximum_dimension_relative_error: float = 0.05,
    maximum_yaw_degrees: float = 5.0,
    maximum_score_difference: float = 0.05,
    maximum_aggregate_count_relative_difference: float = 0.05,
) -> dict[str, object]:
    """Apply every locked M2 acceptance criterion across the frozen sample set."""

    reference_high_total = 0
    candidate_high_total = 0
    reference_high_matched = 0
    candidate_high_matched = 0
    reference_exported_total = 0
    candidate_exported_total = 0
    count_disagreements: list[dict[str, object]] = []
    threshold_edges: list[dict[str, object]] = []
    high_matches: list[Mapping[str, Any]] = []
    per_sample_count_pass = True

    for report in reports:
        counts = _mapping(report, "counts")
        reference_high_total += int(counts["pytorch_high_confidence"])
        candidate_high_total += int(counts["tensorrt_high_confidence"])
        reference_exported_total += int(counts["pytorch_exported"])
        candidate_exported_total += int(counts["tensorrt_exported"])
        difference = int(counts["absolute_exported_difference"])
        allowed = int(counts["allowed_exported_difference"])
        if difference > allowed:
            per_sample_count_pass = False
        if difference:
            count_disagreements.append(
                {
                    "sample_index": int(report["sample_index"]),
                    "pytorch_count": int(counts["pytorch_exported"]),
                    "tensorrt_count": int(counts["tensorrt_exported"]),
                    "absolute_difference": difference,
                    "allowed_difference": allowed,
                }
            )
        threshold_edges.extend(
            {
                "sample_index": int(report["sample_index"]),
                **dict(item),
            }
            for item in _sequence(report, "threshold_edge_disagreements")
        )
        for item in _sequence(report, "matches"):
            match = dict(item)
            if bool(match["reference_high_confidence"]):
                reference_high_matched += 1
            if bool(match["candidate_high_confidence"]):
                candidate_high_matched += 1
            if bool(match["high_confidence"]):
                high_matches.append(match)

    reference_coverage = _coverage(reference_high_matched, reference_high_total)
    candidate_coverage = _coverage(candidate_high_matched, candidate_high_total)
    aggregate_count_difference = abs(candidate_exported_total - reference_exported_total)
    aggregate_count_relative = (
        aggregate_count_difference / reference_exported_total
        if reference_exported_total
        else (0.0 if candidate_exported_total == 0 else float("inf"))
    )
    xy_values = [float(item["center_displacement_xy_m"]) for item in high_matches]
    z_values = [float(item["center_displacement_z_absolute_m"]) for item in high_matches]
    dimension_values = [
        float(value) for item in high_matches for value in item["dimension_relative_error_lwh"]
    ]
    yaw_values = [float(item["circular_yaw_difference_degrees"]) for item in high_matches]
    score_values = [float(item["confidence_score_absolute_difference"]) for item in high_matches]
    class_mismatches = sum(not bool(item["class_equal"]) for item in high_matches)
    checks = {
        "pytorch_to_tensorrt_high_confidence_coverage": reference_coverage >= minimum_coverage,
        "tensorrt_to_pytorch_high_confidence_coverage": candidate_coverage >= minimum_coverage,
        "high_confidence_xy_center": _maximum(xy_values) <= maximum_xy_m,
        "high_confidence_z_center": _maximum(z_values) <= maximum_z_m,
        "high_confidence_dimensions": _maximum(dimension_values)
        <= maximum_dimension_relative_error,
        "high_confidence_yaw": _maximum(yaw_values) <= maximum_yaw_degrees,
        "high_confidence_score": _maximum(score_values) <= maximum_score_difference,
        "high_confidence_classes": class_mismatches == 0,
        "per_sample_exported_counts": per_sample_count_pass,
        "aggregate_exported_counts": aggregate_count_relative
        <= maximum_aggregate_count_relative_difference,
    }
    return {
        "overall_pass": all(checks.values()),
        "checks": checks,
        "high_confidence_coverage": {
            "pytorch_to_tensorrt": reference_coverage,
            "tensorrt_to_pytorch": candidate_coverage,
            "pytorch_total": reference_high_total,
            "pytorch_matched": reference_high_matched,
            "tensorrt_total": candidate_high_total,
            "tensorrt_matched": candidate_high_matched,
        },
        "exported_counts": {
            "pytorch_total": reference_exported_total,
            "tensorrt_total": candidate_exported_total,
            "absolute_difference": aggregate_count_difference,
            "relative_difference_from_pytorch": aggregate_count_relative,
            "per_sample_disagreements": count_disagreements,
        },
        "matched_high_confidence_metrics": {
            "match_count": len(high_matches),
            "xy_center_displacement_m": _statistics(xy_values),
            "absolute_z_center_difference_m": _statistics(z_values),
            "dimension_relative_error": _statistics(dimension_values),
            "circular_yaw_difference_degrees": _statistics(yaw_values),
            "absolute_score_difference": _statistics(score_values),
            "class_name_mismatches": class_mismatches,
        },
        "threshold_edge_disagreements": threshold_edges,
    }


def _match_record(match: DetectionMatch, high_threshold: float) -> dict[str, object]:
    record = match.to_dict()
    reference_high = match.reference.score >= high_threshold
    candidate_high = match.candidate.score >= high_threshold
    return {
        **record,
        "reference_high_confidence": reference_high,
        "candidate_high_confidence": candidate_high,
        "high_confidence": reference_high or candidate_high,
    }


def _threshold_edges(
    matching: MatchingResult, exported_threshold: float
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for match in matching.matches:
        reference_exported = match.reference.score >= exported_threshold
        candidate_exported = match.candidate.score >= exported_threshold
        if reference_exported != candidate_exported and (
            is_threshold_edge_score(match.reference.score)
            or is_threshold_edge_score(match.candidate.score)
        ):
            records.append(
                {
                    "kind": "matched_threshold_crossing",
                    "pytorch": match.reference.to_dict(),
                    "tensorrt": match.candidate.to_dict(),
                    "bev_iou": match.bev_iou,
                }
            )
    for detection in matching.unmatched_reference:
        if detection.score >= exported_threshold and is_threshold_edge_score(detection.score):
            records.append(
                {
                    "kind": "pytorch_only_near_threshold",
                    "pytorch": detection.to_dict(),
                    "tensorrt": None,
                }
            )
    for detection in matching.unmatched_candidate:
        if detection.score >= exported_threshold and is_threshold_edge_score(detection.score):
            records.append(
                {
                    "kind": "tensorrt_only_near_threshold",
                    "pytorch": None,
                    "tensorrt": detection.to_dict(),
                }
            )
    return records


def _mapping(parent: Mapping[str, object], key: str) -> Mapping[str, Any]:
    value = parent[key]
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return cast(Mapping[str, Any], value)


def _sequence(parent: Mapping[str, object], key: str) -> tuple[Mapping[str, Any], ...]:
    value = parent[key]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{key} must be a sequence")
    if not all(isinstance(item, Mapping) for item in value):
        raise TypeError(f"{key} must contain mappings")
    return tuple(cast(Mapping[str, Any], item) for item in value)


def _coverage(matched: int, total: int) -> float:
    return matched / total if total else 1.0


def _maximum(values: Sequence[float]) -> float:
    return max(values, default=0.0)


def _statistics(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "maximum": None}
    return {"count": len(values), "median": median(values), "maximum": max(values)}
