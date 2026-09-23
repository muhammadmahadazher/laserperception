from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _json(relative: str) -> dict[str, object]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_external_omnilink_summary_preserves_claim_boundary() -> None:
    value = _json("benchmarks/external/omnilink_omnisim_2026_09/summary.json")
    assert value["external"] is True
    assert value["laserperception_source_commit"] == ("0f93c480acb6c98bc07781db8ed64b8433ec9238")
    assert value["score_threshold"] == 0.25
    detector = value["detector"]
    runtime = value["runtime"]
    assert isinstance(detector, dict) and isinstance(runtime, dict)
    assert detector["config_sha256"] is None
    assert runtime["warmup_policy"] is None
    assert runtime["measurement_timestamp"] is None
    assert runtime["memory_measurement_method"] is None
    assert runtime["operating_system"] is None
    assert runtime["nvidia_driver"] is None
    assert runtime["cuda_runtime"] is None
    sparse = value["sparse"]
    native = value["native"]
    assert isinstance(sparse, dict) and isinstance(native, dict)
    for result in (
        sparse["one_sweep"],
        sparse["eleven_sweeps"],
        native["one_sweep"],
        native["eleven_sweeps"],
    ):
        assert isinstance(result, dict)
        assert result["intended_match"] is None
    assert sparse["one_sweep"]["input_points"] == 370
    assert sparse["eleven_sweeps"]["input_points"] == 4070
    assert native["one_sweep"]["input_points"] == 21483
    assert native["eleven_sweeps"]["input_points"] == 233950
    assert sparse["one_sweep"]["input_sha256"] is None
    assert sparse["eleven_sweeps"]["input_sha256"] is None
    assert native["one_sweep"]["input_sha256"] is None
    assert native["eleven_sweeps"]["input_sha256"] == (
        "3e8fdddf277e5a173a92f38a4b3d71557a84940153216a4bc5e46054e6b4d107"
    )
    transform = value["transform_verification"]
    assert isinstance(transform, dict)
    assert transform["rebuilt_value_sha256"] == (
        "e00d05c6562449d52a2ca7d84692e50b6ccc9c512d5cc5f2f4a3551383094472"
    )
    assert transform["preserved_npy_sha256"] == (
        "3e8fdddf277e5a173a92f38a4b3d71557a84940153216a4bc5e46054e6b4d107"
    )
    capture = value["capture"]
    assert isinstance(capture, dict)
    assert capture["binary_sha256"] is None
    assert value["claim_boundary"]


def test_m8_current_status_keeps_incomplete_accounting_explicit() -> None:
    incomplete = _json("benchmarks/m8/diagnostics/external_runtime_primary_incomplete_summary.json")
    assert incomplete["status"] == "INCOMPLETE"
    assert incomplete["attempted_conditions"] == 779
    assert incomplete["execution_commit"] == "c676db28bcb038663752793c02f205ac948e0bae"
    assert incomplete["logical_pass_id"] == "primary-pass-1"
    assert incomplete["process_uuid"] == "c12b4bf5-9475-47bc-89b5-aafb33c43216"
    assert incomplete["attempt_id"] == "e7cb3bf8-a173-4830-84fe-21b606c891ba-attempt-1-1"
    assert incomplete["completed_conditions"] == 779
    assert incomplete["expected_conditions"] == 856
    assert incomplete["failed_conditions"] == 0
    assert incomplete["failure_reason"] is None
    assert incomplete["execution_end_utc"] is None
    assert incomplete["accepted_complete_processes"] == 0
    assert incomplete["accepted_canonical_calls"] == 0
    assert incomplete["pass2_started"] is False
    assert incomplete["pass3_started"] is False
    assert incomplete["result_usable_as_primary_measurement"] is False

    capacity = _json("benchmarks/m8/diagnostics/external_runtime_capacity_status_20260922.json")
    assert capacity["pod_created"] is False
    assert capacity["detector_calls"] == 0
    assert capacity["spend_usd"] == 0.0

    status = (ROOT / "docs/PROJECT_STATUS.md").read_text(encoding="utf-8")
    assert "RunPod has been used operationally" in status
    assert "accepted canonical primary calls: 0" in status
    assert "M8 primary A2/E2" in status
    assert "pending" in status.lower()


def test_m8_external_stage_r_and_cpu_timing_provenance_are_compact_and_explicit() -> None:
    stage_r = _json("benchmarks/m8/diagnostics/external_runtime_stage_r_summary.json")
    assert stage_r["status"] == "COMPLETE_ACCEPTED"
    assert stage_r["accepted_processes"] == 10
    assert stage_r["accepted_calls"] == 140
    assert stage_r["execution_commit"] == "c676db28bcb038663752793c02f205ac948e0bae"
    assert stage_r["execution_start_utc"]
    assert stage_r["execution_end_utc"]
    assert stage_r["primary_a2_e2_calls"] == 0
    assert stage_r["zero_intensity_calls"] == 0

    cpu = _json("benchmarks/m8/diagnostics/primary_input_revalidation_cpu_benchmark.json")
    assert cpu["detector_execution"] is False
    assert cpu["detector_config_sha256"] is None
    assert cpu["repetitions_per_worker_count"] == 1
    assert cpu["selected_worker_count"] == 4
    assert cpu["working_tree_had_tracked_changes"] is True
    assert cpu["timing_boundary"]
    memory = cpu["memory_measurement"]
    assert isinstance(memory, dict)
    assert memory["combined_peak_method"] is None
    assert memory["parent_process_method"]
    measurements = cpu["measurements"]
    assert isinstance(measurements, list)
    assert [value["worker_count"] for value in measurements] == [1, 2, 4]
    assert measurements[2]["observed_combined_peak_working_set_bytes_approximate"] == 1368961024
