"""Frozen statistical summaries for the M6b cross-domain characterization."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RankedDisposition:
    """One non-ignored score-ranked prediction for global PR aggregation."""

    score: float
    frame_id: str
    prediction_index: int
    true_positive: bool


def count_metrics(
    true_positives: int, false_positives: int, false_negatives: int
) -> dict[str, float | int]:
    """Return frozen operating-point precision, recall, and F1 with counts."""

    if min(true_positives, false_positives, false_negatives) < 0:
        raise ValueError("metric counts must be non-negative")
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def all_points_average_precision(
    records: Sequence[RankedDisposition],
    *,
    ground_truth_count: int,
) -> dict[str, object]:
    """Compute the preregistered monotonic precision-envelope area."""

    if ground_truth_count < 0:
        raise ValueError("ground_truth_count must be non-negative")
    ordered = sorted(
        records,
        key=lambda item: (-item.score, item.frame_id, item.prediction_index),
    )
    if not ordered or ground_truth_count == 0:
        return {
            "method": "all_points_monotonic_precision_envelope_area",
            "ground_truth_count": ground_truth_count,
            "prediction_count": len(ordered),
            "average_precision": 0.0,
            "curve": [],
        }
    true_positive = np.fromiter((item.true_positive for item in ordered), dtype=np.int64)
    false_positive = 1 - true_positive
    cumulative_tp = np.cumsum(true_positive)
    cumulative_fp = np.cumsum(false_positive)
    recall = cumulative_tp / ground_truth_count
    precision = cumulative_tp / (cumulative_tp + cumulative_fp)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    changes = np.flatnonzero(mrec[1:] != mrec[:-1])
    average_precision = float(np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1]))
    curve = [
        {
            "rank": index + 1,
            "score": item.score,
            "frame_id": item.frame_id,
            "prediction_index": item.prediction_index,
            "true_positive": item.true_positive,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
        }
        for index, item in enumerate(ordered)
    ]
    return {
        "method": "all_points_monotonic_precision_envelope_area",
        "ground_truth_count": ground_truth_count,
        "prediction_count": len(ordered),
        "average_precision": average_precision,
        "curve": curve,
    }


def descriptive_statistics(values: Sequence[float]) -> dict[str, float | int | None]:
    """Return deterministic count/mean/median/p90/p95/min/max/population std."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("descriptive values must be a finite vector")
    if len(array) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "minimum": None,
            "maximum": None,
            "population_std": None,
        }
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "population_std": float(np.std(array, ddof=0)),
    }


def longest_consecutive_runs(
    frame_indices: Sequence[int],
    detected: Sequence[bool],
) -> tuple[int, int]:
    """Return longest hit/miss runs, resetting across nonconsecutive GT frames."""

    if len(frame_indices) != len(detected):
        raise ValueError("frame and detection vectors must have equal length")
    if any(
        current <= previous
        for previous, current in zip(frame_indices, frame_indices[1:], strict=False)
    ):
        raise ValueError("frame indices must be strictly increasing")
    longest_hit = longest_miss = current_length = 0
    current_value: bool | None = None
    previous_frame: int | None = None
    for frame, value in zip(frame_indices, detected, strict=True):
        if previous_frame is None or frame != previous_frame + 1 or value != current_value:
            current_length = 1
        else:
            current_length += 1
        current_value = value
        if value:
            longest_hit = max(longest_hit, current_length)
        else:
            longest_miss = max(longest_miss, current_length)
        previous_frame = frame
    return longest_hit, longest_miss


def drop_homogeneity_test(
    candidate_counts: Sequence[int],
    discarded_counts: Sequence[int],
    *,
    alpha: float = 0.01,
    critical_statistic: float = 24.724970311318277,
    minimum_expected_count: float = 5.0,
) -> dict[str, object]:
    """Compare discarded sector counts with candidate-proportional expectation."""

    candidate = np.asarray(candidate_counts, dtype=np.float64)
    discarded = np.asarray(discarded_counts, dtype=np.float64)
    if candidate.ndim != 1 or discarded.shape != candidate.shape or len(candidate) < 2:
        raise ValueError("candidate and discarded counts must be equal-length vectors")
    if (candidate < 0).any() or (discarded < 0).any() or (discarded > candidate).any():
        raise ValueError("sector counts must satisfy 0 <= discarded <= candidate")
    if not 0.0 < alpha < 1.0 or critical_statistic <= 0 or minimum_expected_count <= 0:
        raise ValueError("test parameters are invalid")
    total_candidate = float(candidate.sum())
    total_discarded = float(discarded.sum())
    expected = (
        total_discarded * candidate / total_candidate
        if total_candidate > 0.0
        else np.zeros_like(candidate)
    )
    adequate = bool(total_discarded > 0.0 and (expected >= minimum_expected_count).all())
    statistic = float(np.sum((discarded - expected) ** 2 / expected)) if adequate else None
    return {
        "method": "Pearson_discarded_vs_candidate_proportions",
        "alpha": alpha,
        "degrees_of_freedom": len(candidate) - 1,
        "critical_statistic": critical_statistic,
        "minimum_expected_count": minimum_expected_count,
        "expected_counts": [float(value) for value in expected],
        "adequate_expected_counts": adequate,
        "statistic": statistic,
        "reject_homogeneity": bool(statistic is not None and statistic > critical_statistic),
    }


def wrapped_absolute_yaw_error(first: float, second: float) -> float:
    """Return absolute full-heading error in ``[0, pi]``."""

    if not math.isfinite(first) or not math.isfinite(second):
        raise ValueError("yaw values must be finite")
    return abs((first - second + math.pi) % (2.0 * math.pi) - math.pi)
