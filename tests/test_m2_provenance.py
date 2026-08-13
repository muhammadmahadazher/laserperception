"""CPU tests for the explicit full and live voxel provenance policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from laserperception.detection.m2_backend import (
    M2Backend,
    validate_provenance_mode,
    validate_voxelization_mode,
)
from laserperception.detection.types import DetectionFrame


@dataclass
class _SyntheticVoxelSample:
    hash_calls: int = 0
    voxel_count: int = 7

    @property
    def shapes(self) -> dict[str, list[int]]:
        return {"voxels": [7, 64, 4], "num_points": [7], "coors": [7, 4]}

    def hashes(self) -> dict[str, str]:
        self.hash_calls += 1
        return {"voxels": "v", "num_points": "n", "coors": "c"}


def _frame() -> DetectionFrame:
    return DetectionFrame(
        detections=(),
        sample_id="sample",
        coordinate_frame="nuscenes_lidar_top",
        metadata={"source": "synthetic"},
    )


def _attach(sample: _SyntheticVoxelSample, mode: str) -> DetectionFrame:
    return M2Backend.attach_runtime_metadata(
        _frame(),
        sample,  # type: ignore[arg-type]
        backend_name="tensorrt",
        precision="fp16",
        provenance_mode=validate_provenance_mode(mode),
    )


def test_full_provenance_preserves_historical_metadata_and_hashes() -> None:
    sample = _SyntheticVoxelSample()

    frame = _attach(sample, "full")

    assert sample.hash_calls == 1
    assert dict(frame.metadata) == {
        "source": "synthetic",
        "backend": "tensorrt",
        "precision": "fp16",
        "voxel_count": 7,
        "shared_voxel_hashes": {"voxels": "v", "num_points": "n", "coors": "c"},
    }


def test_live_provenance_omits_tensor_hashing_and_keeps_semantic_output() -> None:
    sample = _SyntheticVoxelSample()

    frame = _attach(sample, "live")

    assert sample.hash_calls == 0
    assert frame.detections == _frame().detections
    assert frame.sample_id == _frame().sample_id
    assert frame.coordinate_frame == _frame().coordinate_frame
    assert dict(frame.metadata) == {
        "source": "synthetic",
        "backend": "tensorrt",
        "precision": "fp16",
        "voxel_count": 7,
        "voxel_provenance_mode": "live",
        "voxel_provenance_scope": "lightweight_semantic_metadata_only",
        "shared_voxel_hashes_omitted": True,
        "shared_voxel_shapes": {
            "voxels": [7, 64, 4],
            "num_points": [7],
            "coors": [7, 4],
        },
    }


@pytest.mark.parametrize("value", ["", "FULL", "diagnostic"])
def test_invalid_provenance_mode_fails_closed(value: str) -> None:
    with pytest.raises(ValueError, match="must be full or live"):
        validate_provenance_mode(value)


@pytest.mark.parametrize("value", ["", "fast", "deterministic_false", "EXACT_FAST"])
def test_invalid_voxelization_mode_fails_closed(value: str) -> None:
    with pytest.raises(ValueError):
        validate_voxelization_mode(value)


def test_m2_backend_preserves_official_voxelization_default(tmp_path: Path) -> None:
    backend = M2Backend(
        tmp_path / "model.py",
        tmp_path / "checkpoint.pth",
        tmp_path / "deploy.py",
        checkpoint_sha256="0" * 64,
    )

    assert backend.voxelization_mode == "official"
    assert validate_voxelization_mode("exact_fast") == "exact_fast"
