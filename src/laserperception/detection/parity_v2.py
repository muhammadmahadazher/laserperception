"""Versioned M2 parity-v2 acceptance and diagnostic helpers.

The module is NumPy-only so the scientific acceptance rules remain CPU-testable without
MMDetection3D, MMDeploy, PyTorch, CUDA, ONNX, or TensorRT.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import pi
from typing import Any, Literal, cast

import numpy as np

Statistic = dict[str, float | int | None]
DiscreteDivergence = Literal[
    "confirmed_nms_survivor_swap",
    "other_discrete_output_divergence",
    "unexplained_outlier",
]


def full_heading_difference_degrees(reference_yaw_rad: float, candidate_yaw_rad: float) -> float:
    """Return the smallest full circular yaw difference in degrees."""

    difference = (float(candidate_yaw_rad) - float(reference_yaw_rad) + pi) % (2.0 * pi) - pi
    return abs(float(np.degrees(difference)))


def axis_yaw_difference_degrees(reference_yaw_rad: float, candidate_yaw_rad: float) -> float:
    """Return rectangular box-axis yaw difference modulo pi in degrees."""

    difference = (float(candidate_yaw_rad) - float(reference_yaw_rad) + pi / 2.0) % pi - pi / 2.0
    return abs(float(np.degrees(difference)))


def is_direction_flip(reference_yaw_rad: float, candidate_yaw_rad: float) -> bool:
    """Infer whether final headings occupy opposite directions on the same box axis.

    The pinned final detection contract does not retain anchor provenance or direction logits.
    A full circular difference greater than 90 degrees is therefore the preregistered final-frame
    direction-disagreement rule. It does not claim which absolute direction class was selected.
    """

    return full_heading_difference_degrees(reference_yaw_rad, candidate_yaw_rad) > 90.0


def direction_flip_classification(
    reference_yaw_rad: float,
    candidate_yaw_rad: float,
    *,
    maximum_axis_yaw_degrees: float = 5.0,
) -> str:
    """Classify final yaw agreement without hiding heading divergence."""

    if not is_direction_flip(reference_yaw_rad, candidate_yaw_rad):
        return "heading_agreement"
    if (
        axis_yaw_difference_degrees(reference_yaw_rad, candidate_yaw_rad)
        <= maximum_axis_yaw_degrees
    ):
        return "geometrically_axis_equivalent_but_heading_divergent"
    return "heading_divergent_with_axis_error"


def classify_discrete_divergence(
    *,
    same_class: bool = False,
    competing_candidates_overlap: bool = False,
    different_survivors_selected: bool = False,
    candidate_ordering_changed: bool = False,
    other_discrete_decision_evidence: bool = False,
) -> DiscreteDivergence:
    """Classify an outlier without calling it an NMS swap absent complete evidence."""

    if (
        same_class
        and competing_candidates_overlap
        and different_survivors_selected
        and candidate_ordering_changed
    ):
        return "confirmed_nms_survivor_swap"
    if other_discrete_decision_evidence:
        return "other_discrete_output_divergence"
    return "unexplained_outlier"


def distribution_statistics(
    values: Sequence[float] | np.ndarray,
    *,
    include_p90: bool = False,
) -> Statistic:
    """Return deterministic distribution statistics for finite scalar values."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError("statistics values must all be finite")
    keys = ["count", "median", "p95", "p99", "maximum", "mean"]
    if include_p90:
        keys.insert(2, "p90")
    if array.size == 0:
        return {key: 0 if key == "count" else None for key in keys}
    result: Statistic = {
        "count": int(array.size),
        "median": float(np.percentile(array, 50.0)),
        "p95": float(np.percentile(array, 95.0)),
        "p99": float(np.percentile(array, 99.0)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }
    if include_p90:
        result["p90"] = float(np.percentile(array, 90.0))
    return result


def tolerance_statistics(
    values: Sequence[float] | np.ndarray,
    *,
    tolerance: float,
    minimum_pass_fraction: float = 0.99,
) -> dict[str, float | int | bool | None]:
    """Evaluate a per-detection tolerance using a frozen pass fraction."""

    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    if not 0.0 <= minimum_pass_fraction <= 1.0:
        raise ValueError("minimum_pass_fraction must be between zero and one")
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    statistics = distribution_statistics(array)
    passes = array <= tolerance
    pass_count = int(np.count_nonzero(passes))
    count = int(array.size)
    pass_fraction = pass_count / count if count else 1.0
    return {
        **statistics,
        "tolerance": float(tolerance),
        "pass_count": pass_count,
        "failure_count": count - pass_count,
        "pass_fraction": pass_fraction,
        "minimum_pass_fraction": float(minimum_pass_fraction),
        "accepted": pass_fraction >= minimum_pass_fraction,
    }


def raw_tensor_difference_statistics(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> tuple[dict[str, object], np.ndarray]:
    """Compare one corresponding raw output tensor and retain absolute differences."""

    reference_array = np.asarray(reference)
    candidate_array = np.asarray(candidate)
    if reference_array.shape != candidate_array.shape:
        raise ValueError(
            f"raw tensor shape mismatch: {reference_array.shape} versus {candidate_array.shape}"
        )
    difference_dtype = np.result_type(
        reference_array.dtype, candidate_array.dtype, np.dtype(np.float32)
    )
    difference = np.abs(
        candidate_array.astype(difference_dtype, copy=False)
        - reference_array.astype(difference_dtype, copy=False)
    )
    record: dict[str, object] = {
        "shape": [int(value) for value in reference_array.shape],
        "shape_consistent": True,
        "pytorch_dtype": str(reference_array.dtype),
        "tensorrt_dtype": str(candidate_array.dtype),
        "dtype_consistent": reference_array.dtype == candidate_array.dtype,
        "absolute_difference": distribution_statistics(difference),
    }
    return record, difference.reshape(-1)


def reshape_anchor_logits(tensor: np.ndarray, *, values_per_anchor: int) -> np.ndarray:
    """Reshape one NCHW head output to official HWA-by-values anchor order."""

    array = np.asarray(tensor)
    if array.ndim != 4 or array.shape[0] != 1:
        raise ValueError("raw head output must have shape (1, channels, height, width)")
    if values_per_anchor <= 0 or array.shape[1] % values_per_anchor:
        raise ValueError("head channels must be divisible by values_per_anchor")
    return array.transpose(0, 2, 3, 1).reshape(-1, values_per_anchor).astype(np.float64, copy=False)


def official_nms_pre_union(
    reference_class_logits: np.ndarray,
    candidate_class_logits: np.ndarray,
    *,
    nms_pre: int,
) -> np.ndarray:
    """Return the union of runtime-specific official pre-NMS top-anchor pools."""

    reference = np.asarray(reference_class_logits)
    candidate = np.asarray(candidate_class_logits)
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("class logits must have matching (anchors, classes) shapes")
    if nms_pre == 0:
        raise ValueError("nms_pre must be positive or negative for all anchors")
    limit = len(reference) if nms_pre < 0 else min(nms_pre, len(reference))
    reference_indices = _top_anchor_indices(reference, limit)
    candidate_indices = _top_anchor_indices(candidate, limit)
    return np.union1d(reference_indices, candidate_indices).astype(np.int64, copy=False)


def direction_population_summary(
    reference_direction_logits: np.ndarray,
    candidate_direction_logits: np.ndarray,
) -> dict[str, object]:
    """Summarize argmax agreement and winning margins for one anchor population."""

    reference = np.asarray(reference_direction_logits, dtype=np.float64)
    candidate = np.asarray(candidate_direction_logits, dtype=np.float64)
    if reference.shape != candidate.shape or reference.ndim != 2 or reference.shape[1] != 2:
        raise ValueError("direction logits must have matching (anchors, 2) shapes")
    reference_classes = np.argmax(reference, axis=1)
    candidate_classes = np.argmax(candidate, axis=1)
    flipped = reference_classes != candidate_classes
    agreeing = ~flipped
    reference_margins = np.abs(reference[:, 0] - reference[:, 1])
    candidate_margins = np.abs(candidate[:, 0] - candidate[:, 1])
    count = int(len(reference))
    flip_count = int(np.count_nonzero(flipped))
    return {
        "count": count,
        "direction_argmax_disagreement_count": flip_count,
        "direction_argmax_disagreement_fraction": flip_count / count if count else 0.0,
        "winning_margins": {
            "agreeing_anchors": {
                "pytorch": distribution_statistics(reference_margins[agreeing], include_p90=True),
                "tensorrt": distribution_statistics(candidate_margins[agreeing], include_p90=True),
            },
            "disagreeing_anchors": {
                "pytorch": distribution_statistics(reference_margins[flipped], include_p90=True),
                "tensorrt": distribution_statistics(candidate_margins[flipped], include_p90=True),
            },
        },
    }


def direction_disagreement_records(
    reference_direction_logits: np.ndarray,
    candidate_direction_logits: np.ndarray,
    *,
    sample_index: int,
    sample_id: str,
    anchor_indices: np.ndarray | None = None,
) -> list[dict[str, object]]:
    """Record every direction disagreement in a selected anchor population."""

    reference = np.asarray(reference_direction_logits, dtype=np.float64)
    candidate = np.asarray(candidate_direction_logits, dtype=np.float64)
    if reference.shape != candidate.shape or reference.ndim != 2 or reference.shape[1] != 2:
        raise ValueError("direction logits must have matching (anchors, 2) shapes")
    indices = (
        np.arange(len(reference), dtype=np.int64)
        if anchor_indices is None
        else np.asarray(anchor_indices, dtype=np.int64)
    )
    selected_reference = reference[indices]
    selected_candidate = candidate[indices]
    reference_classes = np.argmax(selected_reference, axis=1)
    candidate_classes = np.argmax(selected_candidate, axis=1)
    records: list[dict[str, object]] = []
    for local_index in np.flatnonzero(reference_classes != candidate_classes):
        pytorch_logits = selected_reference[local_index]
        tensorrt_logits = selected_candidate[local_index]
        records.append(
            {
                "sample_index": int(sample_index),
                "sample_id": sample_id,
                "anchor_index": int(indices[local_index]),
                "pytorch_logits": [float(value) for value in pytorch_logits],
                "tensorrt_logits": [float(value) for value in tensorrt_logits],
                "pytorch_margin": float(abs(pytorch_logits[0] - pytorch_logits[1])),
                "tensorrt_margin": float(abs(tensorrt_logits[0] - tensorrt_logits[1])),
                "pytorch_direction_class": int(reference_classes[local_index]),
                "tensorrt_direction_class": int(candidate_classes[local_index]),
            }
        )
    return records


def aggregate_acceptance_v2(
    reports: Sequence[Mapping[str, Any]],
    *,
    minimum_coverage: float = 0.99,
    minimum_metric_pass_fraction: float = 0.99,
    maximum_xy_m: float = 0.25,
    maximum_z_m: float = 0.25,
    maximum_dimension_relative_error: float = 0.05,
    maximum_axis_yaw_degrees: float = 5.0,
    maximum_score_difference: float = 0.05,
    minimum_direction_agreement: float = 0.99,
    maximum_aggregate_count_relative_difference: float = 0.05,
) -> dict[str, object]:
    """Apply only the preregistered Stage-1 parity-v2 acceptance gates."""

    reference_high_total = 0
    candidate_high_total = 0
    reference_high_matched = 0
    candidate_high_matched = 0
    reference_exported_total = 0
    candidate_exported_total = 0
    per_sample_count_pass = True
    count_disagreements: list[dict[str, object]] = []
    threshold_edges: list[dict[str, object]] = []
    high_matches: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []

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
                    "sample_id": str(report["sample_id"]),
                    "pytorch_count": int(counts["pytorch_exported"]),
                    "tensorrt_count": int(counts["tensorrt_exported"]),
                    "absolute_difference": difference,
                    "allowed_difference": allowed,
                }
            )
        threshold_edges.extend(
            {"sample_index": int(report["sample_index"]), **dict(item)}
            for item in _sequence(report, "threshold_edge_disagreements")
        )
        for item in _sequence(report, "matches"):
            if bool(item["reference_high_confidence"]):
                reference_high_matched += 1
            if bool(item["candidate_high_confidence"]):
                candidate_high_matched += 1
            if bool(item["high_confidence"]):
                high_matches.append((report, item))

    reference_coverage = _coverage(reference_high_matched, reference_high_total)
    candidate_coverage = _coverage(candidate_high_matched, candidate_high_total)
    aggregate_count_difference = abs(candidate_exported_total - reference_exported_total)
    aggregate_count_relative = (
        aggregate_count_difference / reference_exported_total
        if reference_exported_total
        else (0.0 if candidate_exported_total == 0 else float("inf"))
    )

    xy_values: list[float] = []
    z_values: list[float] = []
    dimension_values: list[float] = []
    score_values: list[float] = []
    axis_yaw_values: list[float] = []
    full_heading_values: list[float] = []
    direction_agreements: list[bool] = []
    direction_flips: list[dict[str, object]] = []
    continuous_outliers: list[dict[str, object]] = []
    class_mismatches = 0

    for report, item in high_matches:
        xy = float(item["center_displacement_xy_m"])
        z = float(item["center_displacement_z_absolute_m"])
        dimensions = tuple(float(value) for value in item["dimension_relative_error_lwh"])
        dimension = max(dimensions, default=0.0)
        score = float(item["confidence_score_absolute_difference"])
        reference_detection = _mapping(item, "reference")
        candidate_detection = _mapping(item, "candidate")
        reference_yaw = float(reference_detection["yaw_rad"])
        candidate_yaw = float(candidate_detection["yaw_rad"])
        axis_yaw = axis_yaw_difference_degrees(reference_yaw, candidate_yaw)
        full_heading = full_heading_difference_degrees(reference_yaw, candidate_yaw)
        direction_agreement = not is_direction_flip(reference_yaw, candidate_yaw)

        xy_values.append(xy)
        z_values.append(z)
        dimension_values.append(dimension)
        score_values.append(score)
        axis_yaw_values.append(axis_yaw)
        full_heading_values.append(full_heading)
        direction_agreements.append(direction_agreement)
        if not bool(item["class_equal"]):
            class_mismatches += 1

        failed_metrics = [
            name
            for name, failed in (
                ("xy", xy > maximum_xy_m),
                ("z", z > maximum_z_m),
                ("dimensions", dimension > maximum_dimension_relative_error),
                ("score", score > maximum_score_difference),
                ("axis_yaw", axis_yaw > maximum_axis_yaw_degrees),
            )
            if failed
        ]
        if failed_metrics:
            continuous_outliers.append(
                {
                    "sample_index": int(report["sample_index"]),
                    "sample_id": str(report["sample_id"]),
                    "class_name": str(reference_detection["class_name"]),
                    "failed_metrics": failed_metrics,
                    "match": dict(item),
                    "axis_yaw_difference_degrees": axis_yaw,
                    "full_heading_difference_degrees": full_heading,
                }
            )
        if not direction_agreement:
            direction_flips.append(
                {
                    "sample_index": int(report["sample_index"]),
                    "sample_id": str(report["sample_id"]),
                    "class_name": str(reference_detection["class_name"]),
                    "pytorch_final_yaw_rad": reference_yaw,
                    "tensorrt_final_yaw_rad": candidate_yaw,
                    "full_circular_yaw_difference_degrees": full_heading,
                    "axis_yaw_difference_modulo_pi_degrees": axis_yaw,
                    "classification": direction_flip_classification(
                        reference_yaw,
                        candidate_yaw,
                        maximum_axis_yaw_degrees=maximum_axis_yaw_degrees,
                    ),
                    "anchor_provenance_recovered": False,
                    "pytorch_direction_class": None,
                    "tensorrt_direction_class": None,
                    "pytorch_direction_logits": None,
                    "tensorrt_direction_logits": None,
                    "pytorch_winning_logit_margin": None,
                    "tensorrt_winning_logit_margin": None,
                }
            )

    xy_report = tolerance_statistics(
        xy_values, tolerance=maximum_xy_m, minimum_pass_fraction=minimum_metric_pass_fraction
    )
    z_report = tolerance_statistics(
        z_values, tolerance=maximum_z_m, minimum_pass_fraction=minimum_metric_pass_fraction
    )
    dimension_report = tolerance_statistics(
        dimension_values,
        tolerance=maximum_dimension_relative_error,
        minimum_pass_fraction=minimum_metric_pass_fraction,
    )
    score_report = tolerance_statistics(
        score_values,
        tolerance=maximum_score_difference,
        minimum_pass_fraction=minimum_metric_pass_fraction,
    )
    axis_yaw_report = tolerance_statistics(
        axis_yaw_values,
        tolerance=maximum_axis_yaw_degrees,
        minimum_pass_fraction=minimum_metric_pass_fraction,
    )
    direction_count = len(direction_agreements)
    direction_agreement_count = sum(direction_agreements)
    direction_agreement_fraction = (
        direction_agreement_count / direction_count if direction_count else 1.0
    )
    checks = {
        "per_sample_exported_counts": per_sample_count_pass,
        "aggregate_exported_counts": aggregate_count_relative
        <= maximum_aggregate_count_relative_difference,
        "pytorch_to_tensorrt_high_confidence_coverage": reference_coverage >= minimum_coverage,
        "tensorrt_to_pytorch_high_confidence_coverage": candidate_coverage >= minimum_coverage,
        "high_confidence_xy_center": bool(xy_report["accepted"]),
        "high_confidence_z_center": bool(z_report["accepted"]),
        "high_confidence_dimensions_per_detection": bool(dimension_report["accepted"]),
        "high_confidence_score": bool(score_report["accepted"]),
        "high_confidence_axis_yaw": bool(axis_yaw_report["accepted"]),
        "high_confidence_heading_direction_agreement": (
            direction_agreement_fraction >= minimum_direction_agreement
        ),
        "high_confidence_classes": class_mismatches == 0,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    overall_pass = not failed_checks
    return {
        "stage": 1,
        "overall_pass": overall_pass,
        "checks": checks,
        "failed_checks": failed_checks,
        "stage_2_required": not overall_pass,
        "recommended_next_experiment": (
            "TARGETED DIRECTION-HEAD FP32 DIAGNOSTIC"
            if failed_checks == ["high_confidence_heading_direction_agreement"]
            else None
        ),
        "high_confidence_match_denominator": len(high_matches),
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
        "continuous_metrics": {
            "xy_center_displacement_m": xy_report,
            "absolute_z_center_difference_m": z_report,
            "maximum_dimension_relative_error_per_detection": dimension_report,
            "absolute_score_difference": score_report,
            "axis_yaw_difference_modulo_pi_degrees": axis_yaw_report,
        },
        "full_heading_diagnostics": {
            **distribution_statistics(full_heading_values),
            "agreement_count": direction_agreement_count,
            "disagreement_count": direction_count - direction_agreement_count,
            "agreement_fraction": direction_agreement_fraction,
            "minimum_agreement_fraction": minimum_direction_agreement,
            "accepted": direction_agreement_fraction >= minimum_direction_agreement,
            "direction_flips": direction_flips,
        },
        "distinct_high_confidence_continuous_outliers": {
            "count": len(continuous_outliers),
            "denominator": len(high_matches),
            "fraction": len(continuous_outliers) / len(high_matches) if high_matches else 0.0,
            "detections": continuous_outliers,
        },
        "class_name_mismatches": class_mismatches,
        "threshold_edge_disagreements": threshold_edges,
    }


def _top_anchor_indices(class_logits: np.ndarray, limit: int) -> np.ndarray:
    if limit >= len(class_logits):
        return np.arange(len(class_logits), dtype=np.int64)
    scores = np.max(class_logits, axis=1)
    indices = np.argpartition(scores, -limit)[-limit:]
    order = np.argsort(-scores[indices], kind="stable")
    return indices[order].astype(np.int64, copy=False)


def _coverage(matched: int, total: int) -> float:
    return matched / total if total else 1.0


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
