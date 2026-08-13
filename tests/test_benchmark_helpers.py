import math

import pytest

from laserperception.detection.benchmark import (
    bytes_to_gib,
    half_run_backlog_summary,
    latency_statistics_ms,
)


def test_latency_statistics_are_deterministic() -> None:
    result = latency_statistics_ms([1.0, 2.0, 3.0, 4.0])

    assert result["count"] == 4
    assert result["mean_ms"] == 2.5
    assert result["median_ms"] == 2.5
    assert result["p90_ms"] == pytest.approx(3.7)
    assert result["p95_ms"] == pytest.approx(3.85)
    assert result["min_ms"] == 1.0
    assert result["max_ms"] == 4.0
    assert result["population_std_ms"] == pytest.approx(math.sqrt(1.25))
    assert result["fps_from_median_latency"] == 400.0


@pytest.mark.parametrize("values", [[], [0.0], [-1.0], [float("nan")], [float("inf")]])
def test_latency_statistics_reject_invalid_values(values: list[float]) -> None:
    with pytest.raises(ValueError):
        latency_statistics_ms(values)


def test_half_run_backlog_summary_exposes_interval_and_drop_growth() -> None:
    result = half_run_backlog_summary(
        [40.0, 45.0, 55.0, 60.0], first_half_drops=1, second_half_drops=3
    )

    assert result["first_half"]["input_drops"] == 1
    assert result["second_half"]["input_drops"] == 3
    assert result["callback_entry_interval_median_grew"] is True
    assert result["input_drops_grew"] is True
    assert result["falling_behind_between_halves"] is True


def test_half_run_backlog_summary_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        half_run_backlog_summary([1.0], first_half_drops=0, second_half_drops=0)
    with pytest.raises(ValueError):
        half_run_backlog_summary([1.0, 2.0], first_half_drops=-1, second_half_drops=0)


def test_bytes_to_gib_validates_and_converts() -> None:
    assert bytes_to_gib(2 * 1024**3) == 2.0
    with pytest.raises(TypeError):
        bytes_to_gib(True)
    with pytest.raises(ValueError):
        bytes_to_gib(-1)
