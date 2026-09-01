"""CPU tests for diagnostic NVIDIA telemetry parsing and eligibility."""

from __future__ import annotations

import pytest

from laserperception.detection.measurement_telemetry import (
    NvidiaSmiSampler,
    paired_gpu_state_eligibility,
    parse_nvidia_smi_row,
    summarize_gpu_telemetry,
)


def _sample(*, block: str, pstate: str, sm_clock: float, mem_clock: float) -> dict[str, object]:
    return {
        "available": True,
        "block": block,
        "name": "Synthetic GPU",
        "driver_version": "1.2.3",
        "pstate": pstate,
        "temperature.gpu": 60.0,
        "power.draw": 50.0,
        "power.limit": 100.0,
        "clocks.sm": sm_clock,
        "clocks.mem": mem_clock,
        "utilization.gpu": 80.0,
        "utilization.memory": 20.0,
        "memory.used": 2_000.0,
        "memory.total": 8_000.0,
    }


def test_parse_nvidia_smi_row_handles_numeric_and_unsupported_values() -> None:
    result = parse_nvidia_smi_row(
        "NVIDIA GeForce RTX 4060 Laptop GPU, 610.88, P0, 61, 49.2, N/A, 2250, 8001, "
        "98, 14, 2048, 8188"
    )

    assert result["available"] is True
    assert result["name"] == "NVIDIA GeForce RTX 4060 Laptop GPU"
    assert result["pstate"] == "P0"
    assert result["temperature.gpu"] == 61.0
    assert result["power.limit"] is None
    assert result["clocks.sm"] == 2250.0
    assert result["memory.used"] == 2048.0
    assert result["memory.total"] == 8188.0


def test_parse_nvidia_smi_row_rejects_incomplete_rows() -> None:
    with pytest.raises(ValueError, match="one complete GPU row"):
        parse_nvidia_smi_row("GPU, driver, P0")


def test_gpu_summary_retains_state_counts_and_numeric_ranges() -> None:
    summary = summarize_gpu_telemetry(
        [
            _sample(block="reference", pstate="P0", sm_clock=2100.0, mem_clock=8000.0),
            _sample(block="reference", pstate="P2", sm_clock=2200.0, mem_clock=8000.0),
        ]
    )

    assert summary["available"] is True
    assert summary["pstate_counts"] == {"P0": 1, "P2": 1}
    numeric = summary["numeric"]
    assert isinstance(numeric, dict)
    assert numeric["clocks.sm"]["minimum"] == 2100.0
    assert numeric["clocks.sm"]["maximum"] == 2200.0


def test_paired_state_eligibility_accepts_overlapping_observations() -> None:
    samples = [
        _sample(block="reference", pstate="P0", sm_clock=2100.0, mem_clock=8000.0),
        _sample(block="reference", pstate="P2", sm_clock=2300.0, mem_clock=8000.0),
        _sample(block="candidate", pstate="P0", sm_clock=2200.0, mem_clock=8000.0),
    ]

    result = paired_gpu_state_eligibility(samples, [("layer", "reference", "candidate")])

    assert result["eligible"] is True
    assert result["rejection_reasons"] == []


def test_paired_state_does_not_invent_clock_only_rejection_threshold() -> None:
    samples = [
        _sample(block="reference", pstate="P0", sm_clock=2100.0, mem_clock=8000.0),
        _sample(block="candidate", pstate="P0", sm_clock=2200.0, mem_clock=8100.0),
    ]

    result = paired_gpu_state_eligibility(samples, [("layer", "reference", "candidate")])

    assert result["eligible"] is True
    assert result["pairs"][0]["clock_range_overlap"] == {
        "clocks.sm": False,
        "clocks.mem": False,
    }


def test_paired_state_eligibility_rejects_obvious_disjoint_states() -> None:
    samples = [
        _sample(block="reference", pstate="P0", sm_clock=2100.0, mem_clock=8000.0),
        _sample(block="candidate", pstate="P8", sm_clock=300.0, mem_clock=400.0),
    ]

    result = paired_gpu_state_eligibility(samples, [("layer", "reference", "candidate")])

    assert result["eligible"] is False
    assert result["rejection_reasons"] == [
        "layer:disjoint_performance_states_with_disjoint_clock_ranges"
    ]


def test_sampler_labels_synchronous_block_boundary_samples() -> None:
    query_calls = 0

    def query() -> dict[str, object]:
        nonlocal query_calls
        query_calls += 1
        return _sample(block="ignored", pstate="P0", sm_clock=2000.0, mem_clock=8000.0)

    sampler = NvidiaSmiSampler(interval_seconds=10.0, query=query)
    sampler.start()
    sampler.begin_block("measured")
    sampler.end_block("measured")
    sampler.stop()

    samples = sampler.samples
    assert query_calls == 4
    assert [sample["block"] for sample in samples] == [None, "measured", "measured", None]
