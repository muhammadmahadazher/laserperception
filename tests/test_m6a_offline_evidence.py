from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "benchmarks/m6a/results/kitti_raw_offline_reconstruction.json"
EXPECTED_SHA256 = "a62da9808079994d971c4d47bbdf04f2c50d44ddcfd0c7958f7534512552155b"
MEASUREMENT_COMMIT = "1ab832df89109546abedc9f4e7f21c16c4cd0dca"


def _load() -> dict[str, Any]:
    payload = EVIDENCE.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == EXPECTED_SHA256
    return json.loads(payload)


def test_m6a_r2_evidence_preserves_chronology_and_exact_pose_roles() -> None:
    result = _load()
    assert result["status"] == "pass"
    assert result["canonical"] is True
    assert result["provenance"]["measurement_commit"] == MEASUREMENT_COMMIT
    assert result["chronology"][1]["stage"] == "ORIGINAL TIER-A FAIL"
    assert result["chronology"][1]["unchanged"] is True
    assert (
        result["chronology"][1]["sha256"]
        == "894960c34b96aac02db2a8a10f13c66e39e3ac99b3ef34ee76c0ef3e88eabaf3"
    )
    assert result["chronology"][2]["root_cause"] == "DATA-PRODUCT / TIMING"

    pose = result["pose_correctness"]
    assert pose["drive_roles_are_distinct"] is True
    for name, count in (
        ("adapter_pose_oracle_drive", 271),
        ("canonical_reconstruction_drive_transfer_check", 108),
    ):
        gate = pose[name]
        assert gate["status"] == "pass"
        assert gate["frame_count"] == count
        assert gate["exact_equality_count"] == count
        assert gate["matrix_max_abs"] == 0.0
        assert gate["rotation_angle_rad"] == 0.0
        assert gate["translation_norm_m"] == 0.0
        assert gate["frame_zero"]["production_reference_exact"] is True
        assert gate["frame_zero"]["r1_1e-12_ideal_identity_check"] == ("historical_fail_preserved")
        assert gate["frame_zero"]["production_matrix_max_abs_from_ideal_identity"] > 1e-12

    external = result["odometry_external_check"]
    assert external["equality_pass_gate"] is False
    assert external["interpolation_promoted_to_production"] is False
    assert set(external["relative_error"]) == {"1", "2", "5", "10"}


def test_m6a_r2_evidence_freezes_reconstruction_hashes_and_input_shift() -> None:
    result = _load()
    reconstruction = result["offline_reconstruction"]
    assert reconstruction["status"] == "pass"
    assert reconstruction["drive_frame_count"] == 108
    assert reconstruction["frozen_frame_count"] == 24
    assert reconstruction["startup_frames"] == 1
    assert reconstruction["shallow_frames"] == 3
    assert reconstruction["full_history_frames"] == 20
    assert len(reconstruction["frames"]) == 24
    assert len({frame["output_sha256"] for frame in reconstruction["frames"]}) == 24
    for frame in reconstruction["frames"]:
        assert frame["output_dtype"] == "float32"
        assert frame["output_shape"][1] == 4
        assert frame["time_lag_seconds"][0] == 0.0
        assert all(frame["invariants"].values())

    determinism = reconstruction["determinism"]
    assert determinism["status"] == "pass"
    assert determinism["sentinel_count"] == 24
    assert determinism["repetitions_per_sentinel"] == 10
    assert all(item["unique_hash_count"] == 1 for item in determinism["results"])

    shift = result["input_shift"]
    assert shift["status"] == "pass"
    assert shift["max_voxels"] == 40_000
    assert shift["max_voxels_engaged_any"] is True
    assert shift["frames_exceeding_max_voxels"] == 5
    assert shift["spatial_cap_characterization_performed"] is False


def test_m6a_r2_evidence_raw_tracklet_and_scope_gates() -> None:
    result = _load()
    raw = result["dataset"]["raw_decode"]
    assert raw["status"] == "pass"
    assert raw["frame_count"] == 108
    assert raw["source_bytes_exact"] is True
    assert raw["source_order_exact"] is True
    assert raw["reflectance_promoted_to_detector_input"] is False

    tracklets = result["dataset"]["tracklets"]
    assert tracklets["status"] == "pass"
    assert tracklets["selected_drive_available"] is True
    assert tracklets["tracklet_count"] == 15
    assert tracklets["pose_count"] == 572
    assert tracklets["covered_frame_min"] == 0
    assert tracklets["covered_frame_max"] == 107
    assert tracklets["class_counts"] == {"Car": 12, "Cyclist": 2, "Tram": 1}

    assert all(value is False for value in result["scope"].values())
    evidence_text = EVIDENCE.read_text(encoding="utf-8")
    assert "C:\\Users\\" not in evidence_text
    assert "J:\\" not in evidence_text
