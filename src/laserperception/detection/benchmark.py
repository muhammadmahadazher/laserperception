"""CPU-testable helpers for honest M1 latency reporting."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

import numpy as np


def latency_statistics_ms(values: Sequence[float]) -> dict[str, float | int]:
    """Summarize positive millisecond observations with population statistics."""

    samples = np.asarray(tuple(values), dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("latency values must be a non-empty one-dimensional sequence")
    if not np.isfinite(samples).all() or not bool(np.all(samples > 0.0)):
        raise ValueError("latency values must be finite and positive")
    median = float(np.median(samples))
    return {
        "count": int(samples.size),
        "mean_ms": float(np.mean(samples)),
        "median_ms": median,
        "p90_ms": float(np.percentile(samples, 90)),
        "p95_ms": float(np.percentile(samples, 95)),
        "min_ms": float(np.min(samples)),
        "max_ms": float(np.max(samples)),
        "population_std_ms": float(np.std(samples, ddof=0)),
        "fps_from_median_latency": 1000.0 / median,
    }


def half_run_backlog_summary(
    entry_intervals_ms: Sequence[float],
    *,
    first_half_drops: int,
    second_half_drops: int,
) -> dict[str, object]:
    """Expose first/second-half interval and drop growth without hiding overload."""

    if len(entry_intervals_ms) < 2:
        raise ValueError("at least two callback entry intervals are required")
    if first_half_drops < 0 or second_half_drops < 0:
        raise ValueError("half-run drop counts must be non-negative")
    midpoint = len(entry_intervals_ms) // 2
    first = latency_statistics_ms(entry_intervals_ms[:midpoint])
    second = latency_statistics_ms(entry_intervals_ms[midpoint:])
    interval_growth = second["median_ms"] > first["median_ms"]
    drop_growth = second_half_drops > first_half_drops
    return {
        "first_half": {"callback_entry_intervals": first, "input_drops": first_half_drops},
        "second_half": {
            "callback_entry_intervals": second,
            "input_drops": second_half_drops,
        },
        "callback_entry_interval_median_grew": interval_growth,
        "input_drops_grew": drop_growth,
        "falling_behind_between_halves": interval_growth or drop_growth,
    }


def bytes_to_gib(value: int) -> float:
    """Convert a non-negative byte count to binary GiB."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("value must be an integer")
    if value < 0:
        raise ValueError("value must be non-negative")
    result = value / (1024**3)
    if not isfinite(result):
        raise ValueError("converted value must be finite")
    return result
