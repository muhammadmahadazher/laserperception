from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from laserperception.detection.m6b_engine_remediation import (
    CANDIDATE_ENGINE_LOGICAL_NAME,
    EXPECTED_ONNX_SHA256,
    M6B_EVALUATION_DRIVES,
    load_engine_manifest,
    profile_shapes,
    reject_evaluation_drive,
    resolve_build_manifest_path,
    select_repeatability_frames,
    select_third_drive_frames,
    validate_candidate_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "configs/detection/m2_pointpillars_tensorrt.yaml"
CANDIDATE = ROOT / "configs/detection/m6_pointpillars_tensorrt_40k.yaml"


def test_candidate_manifest_preserves_historical_profile_and_artifacts() -> None:
    historical = load_engine_manifest(HISTORICAL)
    candidate = load_engine_manifest(CANDIDATE)

    shapes = validate_candidate_manifest(candidate, historical)

    assert profile_shapes(historical)["voxels"]["max_shape"] == [30000, 64, 4]
    assert shapes["voxels"] == {
        "min_shape": [4352, 64, 4],
        "opt_shape": [18207, 64, 4],
        "max_shape": [40000, 64, 4],
    }
    assert shapes["num_points"]["max_shape"] == [40000]
    assert shapes["coors"]["max_shape"] == [40000, 4]
    assert candidate["voxel_contract"]["upstream_max_voxels"]["validation"] == 40000
    assert candidate["artifacts"]["onnx"]["sha256"] == EXPECTED_ONNX_SHA256
    assert candidate["artifacts"]["engine"]["logical_name"] == CANDIDATE_ENGINE_LOGICAL_NAME
    assert (
        candidate["artifacts"]["engine"]["logical_name"]
        != historical["artifacts"]["engine"]["logical_name"]
    )


def test_builder_default_resolves_historical_m2_manifest() -> None:
    assert resolve_build_manifest_path(ROOT, None) == HISTORICAL
    assert profile_shapes(load_engine_manifest(resolve_build_manifest_path(ROOT, None)))["voxels"][
        "max_shape"
    ] == [30000, 64, 4]


def test_candidate_reuses_frozen_m2_parity_protocol_without_mutation() -> None:
    candidate = load_engine_manifest(CANDIDATE)
    validation = candidate["validation"]
    parity_path = ROOT / validation["parity_protocol"]

    assert parity_path == ROOT / "configs/detection/m2_parity_v2.yaml"
    assert (
        hashlib.sha256(parity_path.read_bytes()).hexdigest() == validation["parity_protocol_sha256"]
    )
    assert validation["parity_thresholds_mutable"] is False


@pytest.mark.parametrize("drive_id", sorted(M6B_EVALUATION_DRIVES))
def test_m6b_evaluation_drives_are_rejected(drive_id: str) -> None:
    with pytest.raises(ValueError, match="forbids network output"):
        reject_evaluation_drive(drive_id)


def test_only_preregistered_non_evaluation_drive_is_allowed() -> None:
    reject_evaluation_drive("2011_09_30_drive_0016")
    with pytest.raises(ValueError, match="authorizes only"):
        reject_evaluation_drive("2011_09_28_drive_0001")


def _selection_fixture() -> list[dict[str, object]]:
    counts = [
        17000,
        18000,
        18207,
        21000,
        26000,
        29999,
        30001,
        32000,
        34000,
        36000,
        38000,
        39000,
        39500,
        40000,
        40000,
    ]
    return [
        {
            "frame_index": index + 10,
            "frame_id": f"2011_09_30_drive_0016/{index + 10:010d}",
            "voxel_count": count,
            "point_count": 100000 + index,
            "model_ready_sha256": f"{index:064x}",
        }
        for index, count in enumerate(counts)
    ]


def test_third_drive_selection_is_deterministic_and_covers_profile() -> None:
    frames = _selection_fixture()

    first = select_third_drive_frames(frames)
    second = select_third_drive_frames(list(reversed(frames)))

    assert first == second
    assert len(first) == 12
    assert len({record["frame_index"] for record in first}) == 12
    assert any(int(record["voxel_count"]) <= 30000 for record in first)
    assert sum(int(record["voxel_count"]) > 30000 for record in first) >= 4
    assert any(int(record["voxel_count"]) >= 39000 for record in first)


def test_repeatability_selection_covers_highest_and_mid_range() -> None:
    selected = select_third_drive_frames(_selection_fixture())

    result = select_repeatability_frames(selected)

    assert result["highest_shape"]["voxel_count"] == 40000
    assert result["mid_range_near_opt"]["voxel_count"] == 18207


def test_selection_fails_closed_without_new_envelope_coverage() -> None:
    frames = [{"frame_index": index, "voxel_count": 20000 + index} for index in range(12)]
    with pytest.raises(ValueError, match="PROFILE-COVERAGE INSUFFICIENT"):
        select_third_drive_frames(frames)
