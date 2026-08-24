from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "benchmarks/m6c/results/kitti_raw_ros_projected_validation_r3.json"
RESULT_DOC = ROOT / "docs/m6/M6C_RESULTS_R3.md"
R2_DOC = ROOT / "docs/m6/M6C_RESULTS.md"


def _load() -> dict[str, object]:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_final_r3_result_closes_m6_with_all_frozen_gates_passed() -> None:
    result = _load()
    assert result["status"] == "PASS"
    assert result["scientific_classification"] == (
        "M6C_R3_POSITIVE_PROJECTED_REFERENCE_ROS_VALIDATION"
    )
    assert result["gate_1"]["gate_1a"] == {"exact": 24, "required": 24, "failed": 0}
    assert result["gate_1"]["gate_1b"] == {"exact": 856, "required": 856, "failed": 0}
    assert result["gate_1"]["unique_live_conditions"] == {
        "exact": 860,
        "required": 860,
        "failed": 0,
    }
    assert result["gate_2"]["parity_v2_stage_1"]["overall_pass"] is True
    assert result["gate_2"]["parity_v2_stage_1"]["stage_2_required"] is False
    assert result["detection3darray_contract"]["passed"] == 10
    assert result["detection3darray_contract"]["required"] == 10
    assert result["governance_outcome"]["m6"] == "complete"
    assert result["governance_outcome"]["active_technical_submilestone"] is None
    assert result["governance_outcome"]["r4_authorized"] is False


def test_final_r3_result_preserves_frozen_inputs_and_prior_chronology() -> None:
    result = _load()
    assert result["identities"]["protocol_commit"] == ("07c4ba293c3d0efbf01c7efb18d389a67828c3fc")
    assert result["identities"]["measurement_implementation_commit"] == (
        "28d81f3f9d4a5ce92d2dde7b3a6635c5079d1f4b"
    )
    assert result["projected_reference"]["unique_conditions"] == 860
    assert (
        result["original_vs_projected_characterization"]["total"]["model_ready_sha_different"]
        == 856
    )
    assert result["original_vs_projected_characterization"]["total"]["point_count_different"] == 0
    assert result["preserved_chronology"]["r2"].startswith("FAIL:")
    assert result["preserved_chronology"]["d1"].startswith("DIAGNOSTIC:")
    assert result["performance_campaign"] is False


def test_final_r3_result_is_compact_sanitized_and_byte_frozen() -> None:
    assert RESULT.stat().st_size == 8_702
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == (
        "f563b8fa5fc090b1890f2dab6c94cb79e74d64d1c71b0c7ca8a7fd5c9885435d"
    )
    text = RESULT.read_text(encoding="utf-8").lower()
    for forbidden in ("j:\\", "c:\\users", "/root/", ".local/", '"points_xyzt"'):
        assert forbidden not in text


def test_results_docs_preserve_r2_and_state_the_r3_claim_boundary() -> None:
    result_text = RESULT_DOC.read_text(encoding="utf-8")
    r2_text = R2_DOC.read_text(encoding="utf-8")
    for required in (
        "860/860 unique live conditions",
        "Stage 2 was not triggered",
        "does not erase the original R2 failure",
        "does **not** establish that",
        "No performance campaign or tuning occurred",
    ):
        assert required in result_text
    assert "Preserved historical result" in r2_text
    assert "M6c NOT READY — M6A ROS INPUT EXACTNESS FAILED" in r2_text


def test_governance_closes_m6_without_activating_m5_or_a_next_milestone() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "docs/ROADMAP.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "M6c is complete with a positive final R3" in agents
    assert "No technical submilestone is" in agents
    assert "currently active" in agents
    assert "M5 remains conditional and inactive" in agents
    assert "## M6 — complete" in roadmap
    assert "860/860 unique live conditions" in roadmap
    assert "Final ROS integration reproduced 860/860" in readme
