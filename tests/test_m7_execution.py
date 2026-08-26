from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.m7.evidence import PairedSets
from benchmarks.m7.execution import (
    CheckpointIdentity,
    ExecutionIdentity,
    M7CheckpointStore,
    ObservationHashes,
    RuntimeArtifacts,
    car_interpretation,
    factorial_contrasts,
    frozen_primary_match,
    gap_recovery,
    observation_hashes,
    paired_recovery,
    repeatability_condition,
    run_authorized,
)
from benchmarks.m7.protocol import ProtocolViolation
from benchmarks.m7.run_measurement import run_measurement
from laserperception.detection.types import Detection3D
from laserperception.evaluation.kitti_m6b import M6bGroundTruthBox


def _authorization(identity: ExecutionIdentity, *, allowed: bool = True) -> dict[str, object]:
    return {
        "schema_version": "laserperception.m7.inference-authorization.v1",
        **identity.to_dict(),
        "authorized_for_inference": allowed,
    }


def _hashes(suffix: str = "a") -> ObservationHashes:
    return ObservationHashes(suffix * 64, "b" * 64, "c" * 64, "d" * 64)


def test_authorization_hard_stops_before_detector_factory() -> None:
    expected = ExecutionIdentity("1" * 40, "2" * 64)
    calls = 0

    def factory() -> object:
        nonlocal calls
        calls += 1
        return object()

    with pytest.raises(ProtocolViolation, match="not explicitly authorized"):
        run_authorized(
            _authorization(expected, allowed=False), expected, factory, lambda value: value
        )
    assert calls == 0

    mismatched = _authorization(expected)
    mismatched["engine_sha256"] = "0" * 64
    with pytest.raises(ProtocolViolation, match="engine_sha256"):
        run_authorized(mismatched, expected, factory, lambda value: value)
    assert calls == 0

    assert run_authorized(_authorization(expected), expected, factory, lambda _: "ran") == "ran"
    assert calls == 1


def test_measurement_verifies_files_after_authorization_but_before_factory(tmp_path: Path) -> None:
    files = {}
    for name, payload in (
        ("ledger", b"ledger"),
        ("engine", b"engine"),
        ("checkpoint", b"checkpoint"),
        ("onnx", b"onnx"),
    ):
        path = tmp_path / name
        path.write_bytes(payload)
        files[name] = path
    expected = ExecutionIdentity(
        "1" * 40,
        hashlib.sha256(b"ledger").hexdigest(),
        engine_sha256=hashlib.sha256(b"engine").hexdigest(),
        checkpoint_sha256=hashlib.sha256(b"checkpoint").hexdigest(),
        onnx_sha256=hashlib.sha256(b"onnx").hexdigest(),
    )
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(json.dumps(_authorization(expected)), encoding="utf-8")
    artifacts = RuntimeArtifacts(
        input_ledger=files["ledger"],
        engine=files["engine"],
        checkpoint=files["checkpoint"],
        onnx=files["onnx"],
        evaluator_identity=expected.evaluator_identity,
    )
    calls = 0

    def factory() -> object:
        nonlocal calls
        calls += 1
        return object()

    files["engine"].write_bytes(b"changed")
    with pytest.raises(ProtocolViolation, match="engine SHA256 mismatch"):
        run_measurement(authorization_path, expected, artifacts, factory, lambda value: value)
    assert calls == 0


def test_primary_matching_delegates_to_frozen_m6b_thresholds() -> None:
    target = M6bGroundTruthBox(
        track_id=1,
        frame_index=10,
        source_type="Car",
        evaluation_role="target",
        class_name="car",
        center_xyz=(0.0, 0.0, 0.0),
        size_lwh=(4.0, 2.0, 1.5),
        yaw_rad=0.0,
    )
    below_threshold = Detection3D(
        center_xyz=(0.0, 0.0, 0.0),
        size_lwh=(4.0, 2.0, 1.5),
        yaw_rad=0.0,
        score=0.249,
        class_id=0,
        class_name="car",
    )
    accepted = Detection3D(
        center_xyz=(0.0, 0.0, 0.0),
        size_lwh=(4.0, 2.0, 1.5),
        yaw_rad=0.0,
        score=0.25,
        class_id=0,
        class_name="car",
    )

    assert (
        frozen_primary_match((below_threshold,), (target,), (), class_name="car").true_positives
        == 0
    )
    assert frozen_primary_match((accepted,), (target,), (), class_name="car").true_positives == 1


def test_observation_hashes_freeze_names_dtype_shape_and_detection_frame() -> None:
    raw = {
        "cls_score": np.zeros((1, 2), dtype=np.float32),
        "bbox_pred": np.ones((1, 3), dtype=np.float32),
        "dir_cls_pred": np.zeros((1, 4), dtype=np.float16),
    }
    first = observation_hashes(raw, {"detections": [], "sample_id": "synthetic"})
    changed = dict(raw)
    changed["cls_score"] = np.zeros((2, 1), dtype=np.float32)

    assert (
        first.cls_score
        != observation_hashes(changed, {"detections": [], "sample_id": "synthetic"}).cls_score
    )
    with pytest.raises(ProtocolViolation, match="tensor names"):
        observation_hashes({"cls_score": raw["cls_score"]}, {})


def test_repeatability_passes_exactly_ten_calls_and_returns_repeat_one() -> None:
    calls = 0

    def execute() -> ObservationHashes:
        nonlocal calls
        calls += 1
        return _hashes()

    assert repeatability_condition(execute) == _hashes()
    assert calls == 10


@pytest.mark.parametrize("field", ["cls_score", "detection_frame"])
def test_repeatability_stops_on_one_raw_or_detection_difference(field: str) -> None:
    calls = 0

    def execute() -> ObservationHashes:
        nonlocal calls
        calls += 1
        values = _hashes().to_dict()
        if calls == 7:
            values[field] = "0" * 64
        return ObservationHashes(**values)

    with pytest.raises(ProtocolViolation, match=field):
        repeatability_condition(execute)
    assert calls == 10


def test_atomic_checkpoint_resume_preserves_order_and_refuses_duplicate(tmp_path: Path) -> None:
    conditions = (
        "2011_09_26_drive_0001/0000000010|H10_LAG_COMPRESSED",
        "2011_09_26_drive_0001/0000000010|H10_POINT_COUNT_MATCHED",
    )
    identity = CheckpointIdentity("1" * 40, "2" * 64)
    store = M7CheckpointStore(
        tmp_path,
        identity,
        conditions,
        _allow_synthetic_fixture=True,
    )
    store.save_complete(
        conditions[0], input_sha256="3" * 64, hashes=_hashes(), payload={"synthetic": True}
    )

    resumed = M7CheckpointStore(
        tmp_path,
        identity,
        conditions,
        _allow_synthetic_fixture=True,
    )

    assert resumed.load_complete(conditions[0], expected_input_sha256="3" * 64) is not None
    assert resumed.pending_condition_ids() == (conditions[1],)
    with pytest.raises(ProtocolViolation, match="cannot be rerun"):
        resumed.save_complete(conditions[0], input_sha256="3" * 64, hashes=_hashes(), payload={})
    with pytest.raises(ProtocolViolation, match="identity differs"):
        M7CheckpointStore(
            tmp_path,
            CheckpointIdentity("9" * 40, "2" * 64),
            conditions,
            _allow_synthetic_fixture=True,
        )
    with pytest.raises(ProtocolViolation, match="shorten or reorder"):
        M7CheckpointStore(tmp_path / "canonical", identity, conditions)


def test_checkpoint_rejects_malformed_payload_and_never_deletes(tmp_path: Path) -> None:
    condition = "2011_09_26_drive_0001/0000000010|H10_LAG_COMPRESSED"
    identity = CheckpointIdentity("1" * 40, "2" * 64)
    store = M7CheckpointStore(tmp_path, identity, (condition,), _allow_synthetic_fixture=True)
    store.save_complete(condition, input_sha256="3" * 64, hashes=_hashes(), payload={})
    path = next((tmp_path / "conditions").rglob("*.json"))
    record = json.loads(path.read_text(encoding="utf-8"))
    record["payload"] = {"tampered": True}
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ProtocolViolation, match="payload hash"):
        store.load_complete(condition, expected_input_sha256="3" * 64)
    assert path.exists()


def test_frozen_metric_arithmetic_has_no_near_pass_or_alternate_cutoff() -> None:
    shared = tuple(("drive", index, index) for index in range(16))
    e_only = tuple(("drive", 100 + index, index) for index in range(32))
    neither = tuple(("drive", 200 + index, index) for index in range(18))
    sets = PairedSets(shared=shared, e_only=e_only, a_only=(), neither=neither)
    detected = set(shared[:15] + e_only[:16] + neither[:2])
    rates = paired_recovery(detected, sets)

    assert gap_recovery(32, baseline_tp=16, gap=32) == 0.5
    assert rates == {"r_gain": 0.5, "r_shared": 15 / 16, "r_novel": 2 / 18}
    assert car_interpretation(0.5, rates["r_gain"], rates["r_shared"])
    assert not car_interpretation(0.5, 0.5, 14 / 16)
    assert gap_recovery(8, baseline_tp=16, gap=32) == -0.25
    assert factorial_contrasts(a=1.0, b=3.0, c=2.0, d=6.0) == {
        "L": 3.0,
        "P": 2.0,
        "I": 2.0,
    }
