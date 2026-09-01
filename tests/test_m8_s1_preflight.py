from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

from laserperception.detection.m8_s1_preflight import (
    combine_sizing_workers,
    select_preflight_frames,
    select_warmup_frame,
)
from laserperception.detection.m8_s1_runtime import STAGE_R_FRAMES

ROOT = Path(__file__).resolve().parents[1]
CENSUS = ROOT / "benchmarks/m8/diagnostics/m8_h10_capacity_census.json"


def _worker(index: int) -> dict[str, object]:
    calls = []
    for frame in range(10):
        for history, seconds in (("H10", 0.10 + frame / 1000), ("H5", 0.05 + frame / 1000)):
            calls.append(
                {
                    "frame_id": f"frame-{frame}",
                    "history": history,
                    "wall_seconds": seconds + index / 10000,
                }
            )
    return {
        "status": "COMPLETE",
        "process_uuid": f"uuid-{index}",
        "process_id": 100 + index,
        "initialization_seconds": 3.0 + index,
        "measured_calls": calls,
        "ground_truth_loaded": False,
        "evaluator_loaded": False,
        "semantic_output_retained": False,
        "runtime_state": {
            "cuda_memory": {
                "max_reserved_bytes": 4_000,
                "device_total_bytes": 10_000,
            }
        },
        "resource_after": {
            "max_reserved_bytes": 6_000,
            "device_total_bytes": 10_000,
        },
        "host_memory": {"process_max_rss_bytes": 2_000, "total_bytes": 10_000},
        "telemetry": {"summary": {"available": False}, "by_block": {}},
    }


def test_preflight_selection_is_input_only_deterministic_and_excludes_sentinels() -> None:
    census = json.loads(CENSUS.read_text(encoding="utf-8"))
    first = select_preflight_frames(census)
    second = select_preflight_frames(census)
    assert first == second
    assert len(first) == 10
    assert [record["target_quantile"] for record in first] == [
        0.05,
        0.15,
        0.25,
        0.35,
        0.45,
        0.55,
        0.65,
        0.75,
        0.85,
        0.95,
    ]
    assert not {str(record["frame_id"]) for record in first}.intersection(STAGE_R_FRAMES)
    assert select_warmup_frame(first) not in {str(record["frame_id"]) for record in first}


def test_preflight_dependency_graph_contains_no_gt_or_evaluator_import() -> None:
    import laserperception.detection.m8_s1_preflight as preflight

    tree = ast.parse(inspect.getsource(preflight))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("evaluation" in name or "tracklet" in name for name in imported)


def test_preflight_combines_exactly_two_processes_without_semantics() -> None:
    result = combine_sizing_workers(
        [_worker(1), _worker(2)],
        runtime_commit="a" * 40,
        census_path=CENSUS,
    )
    assert result["process_count"] == 2
    assert result["warmup_call_count"] == 4
    assert result["measured_call_count"] == 40
    assert result["scientific_campaign_call_count"] == 0
    assert result["ground_truth_loaded"] is False
    assert result["evaluator_loaded"] is False
    assert result["semantic_output_retained"] is False
    assert result["operational_feasibility"]["classification"] == "OPERATIONALLY PLAUSIBLE"
    assert result["resources"]["maximum_cuda_reserved_fraction"] == 0.6
    assert (
        result["resources"]["workers"][0]["initialization_cuda_memory"]["max_reserved_bytes"]
        == 4_000
    )
    assert (
        result["resources"]["workers"][0]["post_sizing_cuda_memory"]["max_reserved_bytes"] == 6_000
    )


def test_preflight_does_not_create_or_require_scientific_authorization() -> None:
    assert not (ROOT / "benchmarks/m8/preregistration/m8_s1_inference_authorization.json").exists()
