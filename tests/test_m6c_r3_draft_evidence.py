from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "benchmarks/m6c/diagnostics/r3_projected_reference_feasibility.json"
DRAFT = ROOT / "docs/m6/M6C_PROTOCOL_R3_DRAFT.md"

FROZEN_ARTIFACTS = {
    ROOT / "benchmarks/m6a/results/kitti_raw_offline_reconstruction.json": (
        "a62da9808079994d971c4d47bbdf04f2c50d44ddcfd0c7958f7534512552155b"
    ),
    ROOT / "benchmarks/m6b/results/kitti_raw_cross_domain_characterization.json": (
        "b9d47120aabb38733d79f987f16277ebab626d2b7c13159c40c30a68b76c1d26"
    ),
    ROOT / "benchmarks/m6b/diagnostics/pre_inference_input_ledger.json": (
        "2413224808b0140856a0e00f884c53bc9b49ec6b545172f5e7fa57d00802dc15"
    ),
    ROOT / "benchmarks/m6c/diagnostics/gate_a_failure_frame_0000000001.json": (
        "fd319dfcb13da570c62d83b4475531bbbd735845808225ebffee4afe03f1cea4"
    ),
    ROOT / "benchmarks/m6c/diagnostics/post_failure_tf_representation.json": (
        "07ea0434fb5833c96d8e6c619a8459cb43c30bbde97d5cfdba96ac8288f3db5d"
    ),
    ROOT / "benchmarks/m6c/diagnostics/post_failure_downstream_frame10_h10.json": (
        "6346a9d0f9916ea4c6e2abb4e7f9c58587a49a5f3b4cbe7ac9d2a6b4b2c3cd3c"
    ),
    ROOT / "benchmarks/m6c/preregistration/detector_sentinels.json": (
        "e80e803fe8923a52ced1bdecd17841a4cf915352a56ea5427607746409cf3be3"
    ),
}


def _load_evidence() -> dict[str, object]:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def test_r3_feasibility_is_exact_diagnostic_evidence() -> None:
    record = _load_evidence()
    assert EVIDENCE.stat().st_size < 1_000_000
    assert record["status"] == "PROJECTED_REFERENCE_BYTE_GATE_FEASIBLE"
    assert record["diagnostic_only"] is True
    assert record["draft_protocol_frozen"] is False
    assert record["r2_status"] == "FAILED"

    h10 = record["h10"]
    assert h10["transform_comparisons_required"] == 30
    assert h10["transform_comparisons_exact"] == 30
    assert h10["transform_comparisons_non_exact"] == 0
    assert h10["maximum_float32_transform_delta"] == 0.0
    assert h10["model_ready_frames_required"] == 3
    assert h10["model_ready_frames_exact"] == 3
    assert h10["model_ready_frames_non_exact"] == 0

    conditions = h10["conditions"]
    assert len(conditions) == 3
    assert sum(len(condition["transforms"]) for condition in conditions) == 30
    for condition in conditions:
        model_ready = condition["model_ready"]
        assert model_ready["exact"] is True
        assert model_ready["projected_sha256"] == model_ready["live_ros_sha256"]
        assert model_ready["row_order_and_xyzt_bytes_exact"] is True
        assert model_ready["first_different_boundary"] is None
        for transform in condition["transforms"]:
            assert transform["rotation_exact"] is True
            assert transform["translation_exact"] is True
            assert transform["complete_transform_exact"] is True
            assert transform["differing_float32_elements"] == 0
            assert transform["maximum_absolute_delta"] == 0.0
            assert transform["ulp_distances"] == []


def test_optional_h5_was_eligible_and_exact() -> None:
    h5 = _load_evidence()["optional_h5"]
    assert h5["eligible"] is True
    assert h5["executed"] is True
    assert h5["exact"] is True
    assert len(h5["transforms"]) == 5
    assert h5["model_ready"]["projected_sha256"] == h5["model_ready"]["live_ros_sha256"]
    assert all(transform["complete_transform_exact"] for transform in h5["transforms"])


def test_r3_draft_is_not_frozen_and_scope_remains_cpu_ros_only() -> None:
    record = _load_evidence()
    assert all(value is False for value in record["scope"].values())
    text = DRAFT.read_text(encoding="utf-8")
    assert "DRAFT — NOT FROZEN — OWNER REVIEW REQUIRED" in text
    assert not (ROOT / "docs/m6/M6C_PROTOCOL_R3.md").exists()


def test_r3_feasibility_contains_no_private_paths_or_payloads() -> None:
    text = EVIDENCE.read_text(encoding="utf-8").lower()
    for forbidden in (
        "j:\\",
        "c:\\users",
        "/root/",
        ".local/",
        '"points_xyz"',
        '"raw_tensor_values"',
    ):
        assert forbidden not in text


def test_r3_draft_preserves_frozen_source_artifacts() -> None:
    for path, expected_sha256 in FROZEN_ARTIFACTS.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256
