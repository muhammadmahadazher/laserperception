from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TF_EVIDENCE = ROOT / "benchmarks/m6c/diagnostics/post_failure_tf_representation.json"
DOWNSTREAM_EVIDENCE = ROOT / "benchmarks/m6c/diagnostics/post_failure_downstream_frame10_h10.json"

FROZEN_ARTIFACTS = {
    ROOT / "benchmarks/m6c/diagnostics/gate_a_failure_frame_0000000001.json": (
        "fd319dfcb13da570c62d83b4475531bbbd735845808225ebffee4afe03f1cea4"
    ),
    ROOT / "benchmarks/m6a/results/kitti_raw_offline_reconstruction.json": (
        "a62da9808079994d971c4d47bbdf04f2c50d44ddcfd0c7958f7534512552155b"
    ),
    ROOT / "benchmarks/m6b/results/kitti_raw_cross_domain_characterization.json": (
        "b9d47120aabb38733d79f987f16277ebab626d2b7c13159c40c30a68b76c1d26"
    ),
    ROOT / "benchmarks/m6b/diagnostics/pre_inference_input_ledger.json": (
        "2413224808b0140856a0e00f884c53bc9b49ec6b545172f5e7fa57d00802dc15"
    ),
    ROOT / "benchmarks/m6c/preregistration/detector_sentinels.json": (
        "e80e803fe8923a52ced1bdecd17841a4cf915352a56ea5427607746409cf3be3"
    ),
}


def _artifact_bytes_for_frozen_identity(path: Path) -> bytes:
    payload = path.read_bytes()
    if path.name == "detector_sentinels.json":
        return payload.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    return payload


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_post_failure_transform_evidence_is_diagnostic_and_fail_closed() -> None:
    record = _load(TF_EVIDENCE)
    assert TF_EVIDENCE.stat().st_size < 1_000_000
    assert record["status"] == "TRANSFORM_LADDER_EXPLAINED"
    assert record["r2_status"] == "FAILED"
    assert record["diagnostic_only"] is True
    assert record["scope"]["gate_a_rerun_as_success_attempt"] is False
    assert record["scope"]["gate_b_started"] is False
    assert record["scope"]["detector_initialized_for_frame_1"] is False
    assert record["scope"]["performance_campaign"] is False
    assert record["classifications"] == [
        "PLATFORM_ARITHMETIC_PRESENT",
        "UNIT_QUATERNION_PROJECTION_PRESENT",
        "TF2_ADDITIONAL_DIVERGENCE_PRESENT_BELOW_FLOAT32",
        "FLOAT32_STORAGE_ROUNDING_PRESENT",
    ]


def test_transform_ladder_separates_representation_boundaries() -> None:
    record = _load(TF_EVIDENCE)
    stages = record["stages"]
    contributions = record["adjacent_contributions"]
    assert stages["T0"]["float32_sha256"] != stages["T1"]["float32_sha256"]
    assert stages["T1"]["float32_sha256"] != stages["T2"]["float32_sha256"]
    assert stages["T2"]["float32_sha256"] == stages["T3"]["float32_sha256"]
    assert stages["T3"]["float32_sha256"] == stages["T4"]["float32_sha256"]
    assert contributions["T0_to_T1_platform_arithmetic"]["float32"]["differing_elements"] == 1
    assert (
        contributions["T1_to_T2_unit_quaternion_projection"]["float32"]["differing_elements"] == 6
    )
    assert contributions["T2_to_T3_tf2"]["float64"]["differing_elements"] == 10
    assert contributions["T2_to_T3_tf2"]["float32"]["exact"] is True
    assert contributions["T3_to_T4_storage"]["float32"]["exact"] is True
    assert record["quaternion_check"]["q_and_negative_q_rotation_exact"] is True
    assert record["quaternion_check"]["proper_rotation_checks_passed"] is True


def test_frame_one_separates_voxel_structure_from_values() -> None:
    consequence = _load(TF_EVIDENCE)["frame_1_voxel_consequence"]
    assert consequence["range_mask"]["points_changing_membership"] == 0
    assert consequence["discrete_point_coordinates"]["points_changed"] == 0
    assert consequence["candidate_pillars"]["key_set_exact"] is True
    assert consequence["retained_pillars"]["ordering_exact"] is True
    assert consequence["coors"]["exact"] is True
    assert consequence["num_points"]["exact"] is True
    assert consequence["retained_point_membership"]["exact"] is True
    assert consequence["retained_point_membership"]["ordering_exact"] is True
    assert consequence["voxel_feature_values"]["exact"] is False


def test_one_frame_downstream_scope_and_result_are_preserved() -> None:
    record = _load(DOWNSTREAM_EVIDENCE)
    assert DOWNSTREAM_EVIDENCE.stat().st_size < 1_000_000
    assert record["status"] == "DOWNSTREAM_DIAGNOSTIC_COMPLETE"
    assert record["r2_status"] == "FAILED"
    assert record["canonical_control"]["status"] == "PASS"
    assert record["scope"]["authorized_detector_conditions"] == 1
    assert record["scope"]["detector_conditions_executed"] == 1
    assert record["scope"]["network_executions"] == 2
    assert record["scope"]["gate_b_started"] is False
    assert record["scope"]["remaining_sentinels_run"] is False
    assert record["scope"]["performance_campaign"] is False
    assert record["scope"]["r3_created"] is False
    variant = record["ros_variant"]
    assert variant["model_ready"]["exact"] is False
    assert variant["voxel_structure"]["discrete_point_coordinates"]["points_changed"] == 6
    assert variant["voxel_structure"]["coors"]["exact"] is True
    assert variant["voxel_structure"]["num_points"]["exact"] is False
    assert variant["voxel_structure"]["retained_point_membership"]["exact"] is False
    assert variant["exact_fast_outputs"]["voxel_feature_values"]["exact"] is False
    assert all(not item["exact"] for item in variant["raw_tensorrt_outputs"].values())
    assert variant["detection_frame"]["exact"] is False
    assert variant["detection_array"]["semantic_exact"] is False
    assert variant["detection_array"]["velocity_exposed"] is False


def test_post_failure_evidence_contains_no_private_paths_or_payloads() -> None:
    for path in (TF_EVIDENCE, DOWNSTREAM_EVIDENCE):
        text = path.read_text(encoding="utf-8").lower()
        assert "j:\\" not in text
        assert "c:\\users" not in text
        assert "/root/" not in text
        assert ".local/" not in text
        assert '"points_xyz"' not in text
        assert '"raw_tensor_values"' not in text


def test_m6a_m6b_and_r2_source_artifacts_remain_byte_identical() -> None:
    for path, expected_sha256 in FROZEN_ARTIFACTS.items():
        observed = hashlib.sha256(_artifact_bytes_for_frozen_identity(path)).hexdigest()
        assert observed == expected_sha256
