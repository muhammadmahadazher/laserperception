"""Validate canonical M4.5b evidence without ROS or GPU dependencies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "benchmarks/m45b/results/raw_ros_multisweep_correctness.json"
MEASUREMENT_COMMIT = "9e0f4dfacbfc997945825d86a85a3609594a059e"
EVIDENCE_SHA256 = "09ec61bee8b005b7f006a3cb56186cdb08e4da7f8d822174a34e3185267f7224"


def _record() -> dict[str, object]:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def test_m45b_canonical_evidence_records_all_exact_gates() -> None:
    record = _record()
    assert record["schema_version"] == "1.0"
    assert record["milestone"] == "M4.5b"
    assert record["status"] == "pass"
    assert record["identity"]["measurement_commit"] == MEASUREMENT_COMMIT
    assert record["identity"]["branch"] == "feat/m45b-ros-multisweep"

    frozen = record["correctness"]["frozen_detector_chain"]
    assert frozen["required_sample_count"] == 20
    assert frozen["completed_sample_count"] == 20
    assert frozen["passed"] is True
    assert frozen["model_ready_inputs_exact"] is True
    assert frozen["voxel_tensors_exact"] is True
    assert frozen["raw_tensorrt_outputs_exact"] is True
    assert frozen["detection_frames_exact"] is True
    assert frozen["detection3darray_semantics_exact"] is True
    assert len(frozen["samples"]) == 20
    assert all(sample["status"] == "pass" for sample in frozen["samples"])
    assert all(sample["model_ready_input"]["exact"] for sample in frozen["samples"])
    assert all(sample["voxelization"]["exact"] for sample in frozen["samples"])
    assert all(sample["raw_tensorrt_outputs"]["exact"] for sample in frozen["samples"])
    assert all(sample["detection_frame"]["exact"] for sample in frozen["samples"])
    assert all(sample["detection3darray"]["exact"] for sample in frozen["samples"])


def test_m45b_repair_chronology_and_old_m3_path_are_retained() -> None:
    record = _record()
    correctness = record["correctness"]
    repair = correctness["repair_exactness"]
    assert repair["scene_start"]["exact"] is True
    assert repair["scene_start"]["sample_index"] == 0
    assert repair["w1"]["exact"] is True
    assert repair["w1"]["sample_index"] == 42
    assert repair["w1"]["final_point_count"] == 354_182
    assert repair["w1"]["observed_sha256"] == (
        "5c1f5590ce7925d9312fe9ef755d0d849f579ff3d7ae842af75d41a0252cb29a"
    )
    sentinels = repair["rotation_stratified_sentinels"]
    assert [sample["sample_index"] for sample in sentinels] == [21, 39, 58]
    assert all(sample["exact"] is True for sample in sentinels)

    smoke = correctness["legacy_model_ready_m3_smoke"]
    assert smoke["status"] == "pass"
    assert smoke["measurement_commit"] == MEASUREMENT_COMMIT
    assert smoke["message_counts"] == {
        "accepted_callbacks": 1,
        "published_detections": 1,
        "published_input": 1,
        "rejected": 0,
        "sink_received": 1,
    }

    history = record["regression_history"]
    assert history["failed_formula"] == "translation_storage = -t"
    assert history["corrected_formula"] == "translation_storage = -R.T @ t"
    assert history["fail_first_regression_passed"] is True
    assert history["evidence_sha256"] == {
        "adapter_repair_exactness": (
            "078ceb041bf0123cc82b0e2ca1c97e6f47cf081eaa702254564f9c13150e2a66"
        ),
        "original_w1_failure": ("d912eaa94cdb38ee1c8b6c6f4fc59831c31f37d33152b23d1d2a9f334a2fc8d6"),
        "tf_transform_ledger": ("0363fd23ff426aca7a9d88518203062a8e7440b0155a49879f639b3c96c18f2d"),
    }


def test_m45b_evidence_is_frozen_sanitized_and_scope_closed() -> None:
    content = EVIDENCE.read_bytes()
    text = content.decode("utf-8")
    assert hashlib.sha256(content).hexdigest() == EVIDENCE_SHA256
    for forbidden in ("C:\\Users", "J:\\", "/root/", "/mnt/", "My Drive"):
        assert forbidden not in text

    record = json.loads(text)
    assert record["artifacts"] == {
        "checkpoint": "f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0",
        "engine": "a005f75852097cd9b193750560b214cc3d5237ae9b6c106c7fca3d4fc348714b",
        "onnx": "61ce22a8ca31498675c32576bfb94f0093d31dc95d2762f7254bf915a59ecc16",
    }
    assert record["scope_guards"] == {
        "engine_changed": False,
        "exact_fast_changed": False,
        "model_changed": False,
        "onnx_changed": False,
        "performance_campaign_run": False,
        "threshold_changed": False,
        "voxel_geometry_changed": False,
    }
