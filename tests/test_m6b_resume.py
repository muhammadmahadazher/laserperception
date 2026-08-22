from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from laserperception.detection.types import Detection3D, DetectionFrame

ROOT = Path(__file__).resolve().parents[1]


def _runner() -> ModuleType:
    path = ROOT / "benchmarks/m6b/run_characterization.py"
    spec = importlib.util.spec_from_file_location("m6b_run_characterization", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ledger() -> list[dict[str, object]]:
    return [
        {
            "frame_id": "2011_09_26_drive_0001/0000000010",
            "h10": {"model_ready_sha256": "a" * 64},
            "h5": {"model_ready_sha256": "b" * 64},
        }
    ]


def test_progress_ledger_is_atomic_resumable_and_identity_bound(tmp_path: Path) -> None:
    runner = _runner()
    identity = {"measurement_commit": "1" * 40, "candidate_engine_sha256": "2" * 64}
    (tmp_path / "predictions").mkdir()

    progress = runner._initialize_progress(tmp_path, identity, _ledger())
    assert progress["totals"] == {
        "completed_H10": 0,
        "remaining_H10": 1,
        "completed_H5": 0,
        "remaining_H5": 1,
    }
    runner._set_progress(
        tmp_path,
        progress,
        "2011_09_26_drive_0001/0000000010|H10",
        "COMPLETE",
        checkpoint_sha256="3" * 64,
    )
    resumed = runner._initialize_progress(tmp_path, identity, _ledger())
    assert resumed["totals"]["completed_H10"] == 1

    with pytest.raises(RuntimeError, match="identity differs"):
        runner._initialize_progress(
            tmp_path, {**identity, "measurement_commit": "4" * 40}, _ledger()
        )


def test_condition_checkpoint_roundtrip_validates_detection_hash(tmp_path: Path) -> None:
    runner = _runner()
    identity = {"measurement_commit": "1" * 40}
    detection = Detection3D(
        center_xyz=(1.0, 2.0, 3.0),
        size_lwh=(4.0, 2.0, 1.5),
        yaw_rad=0.25,
        score=0.8,
        class_id=0,
        class_name="car",
    )
    frame = DetectionFrame(
        detections=(detection,),
        sample_id="2011_09_26_drive_0001/0000000010",
        coordinate_frame="kitti_model_aligned_lidar",
    )
    result = {
        "frame_id": frame.sample_id,
        "frame_index": 10,
        "condition": "H10",
        "execution": {
            "model_ready_sha256": "a" * 64,
            "voxel_count": 40000,
            "raw_output_hashes": {"cls_score": "b" * 64},
        },
        "outside_annotation_fov_predictions_all_classes": 0,
        "classes": {},
        "_detections": frame.detections,
        "_detection_frame": frame.to_dict(),
    }

    runner._save_checkpoint(tmp_path, result, identity)
    loaded = runner._load_checkpoint(
        runner._checkpoint_path(tmp_path, frame.sample_id, "H10"),
        identity,
        _ledger()[0],
        "H10",
    )

    assert loaded["_detections"] == frame.detections
