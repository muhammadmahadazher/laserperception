from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path

from laserperception.detection.m8_s1_preflight import (
    MAX_PILLAR_CONDITION_ID,
    classify_max_pillar_capacity,
    combine_sizing_workers,
    select_capacity_replay_frames,
    select_preflight_frames,
    select_warmup_frame,
)
from laserperception.detection.m8_s1_runtime import STAGE_R_FRAMES

ROOT = Path(__file__).resolve().parents[1]
CENSUS = ROOT / "benchmarks/m8/diagnostics/m8_h10_capacity_census.json"
SIZING = ROOT / "benchmarks/m8/diagnostics/m8_s1_runtime_sizing.json"
RUNTIME_MANIFEST = ROOT / "benchmarks/m8/preregistration/m8_s1_measurement_runtime.json"


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


def test_capacity_replay_uses_frozen_corpus_order_not_pillar_rank_order() -> None:
    census = json.loads(CENSUS.read_text(encoding="utf-8"))
    replay = select_capacity_replay_frames(census)
    ordinals = [record["frozen_corpus_ordinal_1_based"] for record in replay]
    pillars = [record["H10_candidate_dynamic_pillars"] for record in replay]
    assert ordinals == sorted(ordinals)
    assert pillars != sorted(pillars)
    assert [record["frame_id"] for record in replay] == [
        "2011_09_26_drive_0001/0000000039",
        "2011_09_26_drive_0091/0000000015",
        "2011_09_26_drive_0091/0000000029",
        "2011_09_26_drive_0091/0000000058",
        "2011_09_26_drive_0091/0000000096",
        "2011_09_26_drive_0091/0000000157",
        "2011_09_26_drive_0091/0000000221",
        "2011_09_26_drive_0091/0000000234",
        "2011_09_26_drive_0091/0000000326",
        "2011_09_26_drive_0091/0000000332",
    ]
    assert MAX_PILLAR_CONDITION_ID not in {f"{record['frame_id']}/H10" for record in replay}


def test_capacity_classification_separates_allocated_and_reserved_memory() -> None:
    passed = classify_max_pillar_capacity(
        peak_allocated_bytes=40,
        peak_reserved_bytes=89,
        device_total_bytes=100,
    )
    reserved_review = classify_max_pillar_capacity(
        peak_allocated_bytes=40,
        peak_reserved_bytes=90,
        device_total_bytes=100,
    )
    allocated_review = classify_max_pillar_capacity(
        peak_allocated_bytes=90,
        peak_reserved_bytes=90,
        device_total_bytes=100,
    )
    assert passed["classification"] == "MAX-PILLAR CAPACITY REVIEW — PASS"
    assert reserved_review["classification"] == "OWNER MEMORY-MARGIN REVIEW REQUIRED"
    assert allocated_review["classification"] == "OWNER MEMORY-MARGIN REVIEW REQUIRED"
    assert passed["peak_allocated_fraction_of_device_total"] == 0.4
    assert passed["peak_reserved_fraction_of_device_total"] == 0.89


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


def test_tracked_preflight_is_gt_blind_and_manifest_bound() -> None:
    sizing_bytes = SIZING.read_bytes()
    sizing = json.loads(sizing_bytes)
    manifest = json.loads(RUNTIME_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["preflight"]["artifact_bytes"] == len(sizing_bytes)
    assert manifest["preflight"]["artifact_sha256"] == hashlib.sha256(sizing_bytes).hexdigest()
    assert sizing["process_count"] == 2
    assert sizing["warmup_call_count"] == 4
    assert sizing["measured_call_count"] == 40
    assert sizing["scientific_campaign_call_count"] == 0
    assert sizing["ground_truth_loaded"] is False
    assert sizing["evaluator_loaded"] is False
    assert sizing["semantic_output_retained"] is False
    assert sizing["prediction_count_observed_or_stored"] is False
    assert len({worker["process_uuid"] for worker in sizing["workers"]}) == 2
    assert len({worker["process_id"] for worker in sizing["workers"]}) == 2
    prohibited = {"box", "boxes", "score", "scores", "label", "labels", "prediction_count"}
    for worker in sizing["workers"]:
        for call in worker["measured_calls"]:
            assert prohibited.isdisjoint(call)
