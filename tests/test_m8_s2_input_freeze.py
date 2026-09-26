"""Tracked, CPU-only checks of the S2 input freeze and M7 identity proof."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.m7.protocol import canonical_frame_ids

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks/m8/inputs/m8_s2_input_manifest.json"
FREEZE = ROOT / "benchmarks/m8/preregistration/m8_s2_input_freeze.json"
CHARACTERIZATION = ROOT / "benchmarks/m8/diagnostics/m8_s2_input_characterization.json"
M7_MANIFEST = ROOT / "benchmarks/m7/preregistration/m7_input_manifest.json"
ARM_ORDER = ("B2", "C2", "D2", "F2")
M7_ARMS = (
    "H10_LAG_COMPRESSED",
    "H10_POINT_COUNT_MATCHED",
    "H10_LAG_COMPRESSED_POINT_COUNT_MATCHED",
    "H10_ALTERNATE_FULL_SPAN",
)


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_input_freeze_binds_final_implementation_and_private_ledger_identity() -> None:
    freeze = _json(FREEZE)
    manifest = _json(MANIFEST)
    characterization = _json(CHARACTERIZATION)
    implementation = freeze["implementation_commit"]
    assert isinstance(implementation, str) and len(implementation) == 40
    assert manifest["implementation_commit"] == implementation
    assert freeze["protocol_sha256"] == manifest["protocol_sha256"]
    assert freeze["m7_manifest_sha256"] == _sha(M7_MANIFEST)
    assert freeze["compact_manifest"]["bytes"] == MANIFEST.stat().st_size
    assert freeze["compact_manifest"]["sha256"] == _sha(MANIFEST)
    assert freeze["characterization"]["bytes"] == CHARACTERIZATION.stat().st_size
    assert freeze["characterization"]["sha256"] == _sha(CHARACTERIZATION)
    assert freeze["full_ledger"] == manifest["private_full_ledger"]
    assert freeze["full_ledger"]["logical_filename"] == "m8_s2_input_ledger.jsonl"
    assert freeze["full_ledger"]["record_count"] == 1712
    assert freeze["full_ledger"]["bytes"] > 0
    assert len(freeze["full_ledger"]["sha256"]) == 64
    assert freeze["replay_28_exact"] is True
    assert freeze["implementation_frozen"] is True
    assert freeze["input_ledger_frozen"] is True
    assert freeze["runtime_bound"] is False
    assert freeze["inference_authorized"] is False
    assert freeze["S2_calls"] == 0
    assert characterization["arithmetic_namespace"] == "CPU_ANALYTIC_STRUCTURAL_IDENTITY"
    assert characterization["detector_calls"] == 0
    assert "J:\\" not in MANIFEST.read_text(encoding="utf-8")
    assert "J:\\" not in FREEZE.read_text(encoding="utf-8")


def test_all_1712_conditions_match_the_frozen_m7_xyzt_and_row_hashes() -> None:
    manifest = _json(MANIFEST)
    freeze = _json(FREEZE)
    rows = manifest["conditions"]
    m7_rows = _json(M7_MANIFEST)["conditions"]
    assert len(rows) == len(m7_rows) == 1712
    assert manifest["condition_count"] == 1712
    assert manifest["detector_calls"] == 0
    assert [row["condition_id"] for row in rows] == [
        f"{frame}/{arm}" for frame in canonical_frame_ids() for arm in ARM_ORDER
    ]
    assert len({row["condition_id"] for row in rows}) == 1712
    for row, m7, frame in zip(
        rows,
        m7_rows,
        (frame for frame in canonical_frame_ids() for _ in ARM_ORDER),
        strict=True,
    ):
        assert row["frame_id"] == frame
        assert m7["condition_id"] == f"{frame}|{M7_ARMS[ARM_ORDER.index(row['arm'])]}"
        assert row["XYZT_projection_sha256"] == m7["model_ready_sha256"]
        assert row["M7_expected_XYZT_sha256"] == m7["model_ready_sha256"]
        assert row["selected_global_row_sha256"] == m7["selected_row_sha256"]
        assert row["M7_XYZT_exact"] is True
        assert row["point_count"] == m7["point_count"]
        assert row["cpu_analytic_candidate_pillar_count"] > 0
        assert len(row["cpu_analytic_candidate_coordinate_sha256"]) == 64
        assert len(row["intensity_sha256"]) == 64
        assert len(row["full_XYZIT_sha256"]) == 64
    assert freeze["M7_XYZT_proof"] == {"B2": 428, "C2": 428, "D2": 428, "F2": 428, "total": 1712}
    assert freeze["cpu_analytic_pairwise_coordinates"] == {"B2_A2": 428, "D2_C2": 428}
    assert set(freeze["arm_relation_gates"].values()) == {428}


def test_cpu_cuda_census_difference_is_characterized_without_equating_runtimes() -> None:
    record = _json(CHARACTERIZATION)["known_cross_runtime_example"]
    assert record["condition_id"] == "2011_09_26_drive_0001/0000000010/H10"
    assert record["cpu_analytic_candidate_pillar_count"] == 30623
    assert record["historical_cuda_candidate_pillar_count"] == 30624
    assert record["source_input_identity_matched"] is True
    assert record["count_equality_required"] is False
    assert record["exact_one_pillar_cause_localized"] is False
