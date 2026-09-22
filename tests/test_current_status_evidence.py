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
