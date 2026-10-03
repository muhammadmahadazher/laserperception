"""CPU-only, input-stratified GT-blind sizing and prospective cost arithmetic."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from statistics import median

from laserperception.detection.m8_s2_runtime import (
    ARMS,
    COMPACT_MANIFEST_PATH,
    M8S2ProtocolViolation,
)


def qualification_plan(repository_root: Path) -> dict[str, object]:
    """Fix sixteen input-only conditions before any detector output is known."""

    manifest = json.loads((repository_root / COMPACT_MANIFEST_PATH).read_text(encoding="utf-8"))
    rows = manifest["conditions"]
    chosen: list[dict[str, object]] = []
    for arm in ARMS:
        candidates = [row for row in rows if row["arm"] == arm]
        if len(candidates) != 428:
            raise M8S2ProtocolViolation("S2 sizing arm input population differs")
        by_pillars = sorted(
            candidates,
            key=lambda row: (row["cpu_analytic_candidate_pillar_count"], row["condition_id"]),
        )
        by_points = sorted(candidates, key=lambda row: (-row["point_count"], row["condition_id"]))
        targets = (
            ("low_pillars", by_pillars),
            ("median_pillars", by_pillars[214:] + by_pillars[:214]),
            ("high_pillars", list(reversed(by_pillars))),
            ("high_point_count", by_points),
        )
        seen: set[str] = set()
        for stratum, pool in targets:
            row = next(item for item in pool if item["condition_id"] not in seen)
            seen.add(str(row["condition_id"]))
            chosen.append(
                {
                    "condition_id": row["condition_id"],
                    "arm": arm,
                    "stratum": stratum,
                    "cpu_analytic_candidate_pillar_count": row[
                        "cpu_analytic_candidate_pillar_count"
                    ],
                    "point_count": row["point_count"],
                    "full_XYZIT_sha256": row["full_XYZIT_sha256"],
                }
            )
    if len(chosen) != 16 or len({row["condition_id"] for row in chosen}) != 16:
        raise AssertionError("S2 bounded sizing selection is not unique")
    return {
        "schema_version": "laserperception.m8.s2.qualification-plan.v1",
        "status": "STATIC PLAN ONLY; NOT EXECUTED OR AUTHORIZED",
        "selection_basis": (
            "frozen CPU-only point and candidate-pillar identities; no GT or detector output"
        ),
        "fresh_processes": 2,
        "excluded_warmup_calls_per_process": 2,
        "measured_calls_per_process": 16,
        "future_warmup_calls": 4,
        "future_measured_calls": 32,
        "future_engineering_calls": 36,
        "measured_conditions_per_process": chosen,
        "warmup_condition_ids_per_process": [chosen[0]["condition_id"], chosen[-1]["condition_id"]],
        "outputs_discarded_immediately": True,
        "ground_truth_loaded": False,
        "semantic_predictions_retained": False,
        "engineering_calls_executed": 0,
        "scientific_calls_executed": 0,
    }


def estimate_future_cost(
    *,
    measured_call_seconds: Sequence[float],
    initialization_seconds: Sequence[float],
    usd_per_hour: float,
) -> dict[str, object]:
    """Use future observed min/median/max and fresh-model overhead, never CI."""

    if len(measured_call_seconds) != 32 or len(initialization_seconds) != 2:
        raise M8S2ProtocolViolation(
            "S2 cost sizing requires 32 calls and two fresh initializations"
        )
    times = [float(value) for value in measured_call_seconds]
    starts = [float(value) for value in initialization_seconds]
    if min(times + starts) < 0 or any(
        value != value or value == float("inf") for value in times + starts
    ):
        raise M8S2ProtocolViolation("S2 sizing durations must be finite and nonnegative")
    price = float(usd_per_hour)
    if price < 0 or price != price or price == float("inf"):
        raise M8S2ProtocolViolation("S2 sizing price must be finite and nonnegative")
    scenarios = {
        "observed_minimum": (min(times), min(starts)),
        "median": (median(times), median(starts)),
        "observed_maximum": (max(times), max(starts)),
    }
    workloads = {
        "repeatability": (280, 10),
        "one_full_pass": (1712, 1),
        "three_full_passes": (5136, 3),
        "total_accepted_science": (5416, 13),
    }
    estimates = {
        workload: {
            name: {
                "seconds": calls * call_seconds + processes * initialization,
                "usd_at_input_rate": (calls * call_seconds + processes * initialization)
                / 3600
                * price,
            }
            for name, (call_seconds, initialization) in scenarios.items()
        }
        for workload, (calls, processes) in workloads.items()
    }
    return {
        "schema_version": "laserperception.m8.s2.cost-estimate.v1",
        "status": "DESCRIPTIVE ENGINEERING ESTIMATE; NOT A CONFIDENCE INTERVAL",
        "observed_measured_calls": 32,
        "observed_fresh_initializations": 2,
        "usd_per_hour_input": price,
        "workloads": {
            name: {"calls": calls, "fresh_initializations": processes}
            for name, (calls, processes) in workloads.items()
        },
        "estimates": estimates,
    }
