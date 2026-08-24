from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILURE_PATH = ROOT / "benchmarks/m6c/diagnostics/gate_a_failure_frame_0000000001.json"


def test_m6c_gate_a_failure_is_fail_closed_and_compact() -> None:
    record = json.loads(FAILURE_PATH.read_text(encoding="utf-8"))
    assert FAILURE_PATH.stat().st_size < 50_000
    assert record["status"] == "FAILED_GATE_A_M6A_ROS_INPUT_EXACTNESS"
    assert record["gate_a"]["pass"] == 1
    assert record["gate_a"]["fail"] == 1
    assert record["gate_a"]["pending_after_fail_closed_stop"] == 22
    assert record["gate_a"]["first_failure"]["byte_exact"] is False
    assert record["first_differing_boundary"]["current_rows_exact"] is True
    assert record["first_differing_boundary"]["time_lag_column_exact"] is True
    assert record["downstream_stop"]["gate_b_started"] is False
    assert record["downstream_stop"]["gate_b_pending"] == 856
    assert record["downstream_stop"]["detector_inference_performed"] is False
    assert record["downstream_stop"]["performance_campaign_performed"] is False


def test_m6c_failure_evidence_contains_no_private_paths() -> None:
    text = FAILURE_PATH.read_text(encoding="utf-8").lower()
    assert "j:\\" not in text
    assert "c:\\users" not in text
    assert "/root/" not in text
    assert ".local/" not in text
