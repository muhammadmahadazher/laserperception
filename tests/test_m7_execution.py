from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.m7.detector import (
    M1_MANIFEST_RELATIVE,
    M6B_BACKEND_CLASS,
    M6B_CONFIG_RELATIVE,
    M6B_COORDINATE_FRAME,
    M6B_DEPLOY_CONFIG_RELATIVE,
    M6B_DEPLOY_CONFIG_SHA256,
    M6B_DEPLOY_REPOSITORY_COMMIT,
    M6B_DEPLOYMENT_MANIFEST_RELATIVE,
    M6B_DEVICE,
    M6B_EVALUATION_SCORE_THRESHOLD,
    M6B_MODEL_CONFIG_RELATIVE,
    M6B_MODEL_CONFIG_SHA256,
    M6B_MODEL_REPOSITORY_COMMIT,
    M6B_PRECISION,
    M6B_PROVENANCE_MODE,
    M6B_VOXELIZATION_MODE,
    CanonicalM7Detector,
    CanonicalM7DetectorIdentity,
)
from benchmarks.m7.evidence import PairedSets, StrictInputLedger
from benchmarks.m7.execution import (
    FROZEN_M7_INPUT_LEDGER_SHA256,
    CheckpointIdentity,
    DetectorObservation,
    ExecutionIdentity,
    M7CheckpointStore,
    ObservationHashes,
    RuntimeArtifacts,
    _run_authorized_for_test,
    car_interpretation,
    factorial_contrasts,
    frozen_primary_match,
    gap_recovery,
    observation_hashes,
    paired_recovery,
    repeatability_condition,
    verify_inference_authorization,
)
from benchmarks.m7.prepare_inputs import GeneratedCondition, GeneratedFrame
from benchmarks.m7.protocol import Arm, ProtocolViolation, canonical_condition_ids
from benchmarks.m7.provenance import model_ready_sha256
from benchmarks.m7.run_measurement import (
    CorpusRunSummary,
    M7CorpusRunner,
    _run_measurement_internal,
    run_measurement,
)
from laserperception.detection.m2_backend import EXPECTED_MMDEPLOY_VERSION
from laserperception.detection.types import Detection3D
from laserperception.evaluation.kitti_m6b import M6bGroundTruthBox


def _authorization(identity: ExecutionIdentity, *, allowed: bool = True) -> dict[str, object]:
    return {
        "schema_version": "laserperception.m7.inference-authorization.v2",
        **identity.to_dict(),
        "authorized_for_inference": allowed,
    }


def _hashes(suffix: str = "a") -> ObservationHashes:
    return ObservationHashes(suffix * 64, "b" * 64, "c" * 64, "d" * 64)


def _condition_record(condition: str, input_sha256: str) -> dict[str, object]:
    frame_id, arm_text = condition.split("|")
    drive_id, frame_text = frame_id.split("/")
    ranks = [0, 2, 4, 6, 8, 10] if arm_text == Arm.F.value else [0]
    return {
        "condition_id": condition,
        "drive_id": drive_id,
        "frame_index": int(frame_text),
        "arm": arm_text,
        "generation_commit": "1" * 40,
        "source_a_sha256": "a" * 64,
        "source_e_sha256": "e" * 64,
        "point_count": 1,
        "xyz_sha256": "c" * 64,
        "model_ready_sha256": input_sha256,
        "selected_row_sha256": "f" * 64,
        "lag_bit_patterns": ["0x00000000"],
        "lag_support_count": 1,
        "lag_span_seconds": 0.0,
        "sweep_ids": [f"sweep-{rank}" for rank in ranks],
        "per_sweep_point_counts": {str(rank): 1 for rank in ranks},
        "provenance_schema": "laserperception.m7.sweep-provenance.v2",
        "rank_source_identities": [
            {
                "history_rank": rank,
                "source_sweep_id": f"sweep-{rank}",
                "source_index": 10 - rank,
                "timestamp_text": f"synthetic-{rank}",
                "timestamp_nanoseconds": (10 - rank) * 100_000_000,
                "timestamp_microseconds": (10 - rank) * 100_000,
                "lag_float32_bits": f"0x{rank:08x}",
            }
            for rank in ranks
        ],
        "rank_to_lag_bit_pattern": {str(rank): f"0x{rank:08x}" for rank in ranks},
        "pillar_structure": {"candidate_count": 1},
        "lag_scale_provenance": None,
        "quota_provenance": None,
        "seed_provenance": None,
        "f_history_ranks": [2, 4, 6, 8, 10] if arm_text == Arm.F.value else None,
        "runtime_versions": {"python": "test", "numpy": "test"},
    }


class _Detector:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.point_ids: list[int] = []

    def infer(self, points: np.ndarray, *, condition_id: str) -> DetectorObservation:
        assert not points.flags.writeable
        self.calls.append(condition_id)
        self.point_ids.append(id(points))
        return DetectorObservation(
            raw_outputs={
                "cls_score": np.zeros((1,), dtype=np.float32),
                "bbox_pred": np.zeros((1,), dtype=np.float32),
                "dir_cls_pred": np.zeros((1,), dtype=np.float32),
            },
            detection_frame={"detections": [], "condition_id": condition_id},
            payload={"synthetic": True},
        )


class _IdentityDetector(_Detector):
    def __init__(self, identity: CanonicalM7DetectorIdentity) -> None:
        super().__init__()
        self.identity = identity


def _detector_identity(
    artifacts: RuntimeArtifacts,
    expected: ExecutionIdentity,
    *,
    engine_path: Path | None = None,
) -> CanonicalM7DetectorIdentity:
    return CanonicalM7DetectorIdentity(
        engine_path=(artifacts.engine if engine_path is None else engine_path).resolve(),
        checkpoint_path=artifacts.checkpoint.resolve(),
        onnx_path=artifacts.onnx.resolve(),
        engine_sha256=expected.engine_sha256,
        checkpoint_sha256=expected.checkpoint_sha256,
        onnx_sha256=expected.onnx_sha256,
        onnx_role="verified_provenance_only_not_runtime_consumed",
        voxelization_mode=M6B_VOXELIZATION_MODE,
        m6b_config_path=M6B_CONFIG_RELATIVE,
        m6b_config_sha256="1" * 64,
        model_manifest_path=M1_MANIFEST_RELATIVE,
        model_manifest_sha256="2" * 64,
        model_config_path=M6B_MODEL_CONFIG_RELATIVE,
        model_config_sha256=M6B_MODEL_CONFIG_SHA256,
        model_repository_commit=M6B_MODEL_REPOSITORY_COMMIT,
        deployment_manifest_path=M6B_DEPLOYMENT_MANIFEST_RELATIVE,
        deployment_manifest_sha256="5" * 64,
        deploy_config_path=M6B_DEPLOY_CONFIG_RELATIVE,
        deploy_config_sha256=M6B_DEPLOY_CONFIG_SHA256,
        deploy_repository_commit=M6B_DEPLOY_REPOSITORY_COMMIT,
        backend_class=M6B_BACKEND_CLASS,
        device=M6B_DEVICE,
        expected_mmdeploy_version=EXPECTED_MMDEPLOY_VERSION,
        provenance_mode=M6B_PROVENANCE_MODE,
        precision=M6B_PRECISION,
        coordinate_frame=M6B_COORDINATE_FRAME,
        evaluation_score_threshold=M6B_EVALUATION_SCORE_THRESHOLD,
    )


def test_authorization_hard_stops_before_private_test_detector_builder() -> None:
    expected = ExecutionIdentity("1" * 40, "2" * 64, "3" * 40)
    calls = 0

    def factory() -> object:
        nonlocal calls
        calls += 1
        return object()

    with pytest.raises(ProtocolViolation, match="not explicitly authorized"):
        _run_authorized_for_test(
            _authorization(expected, allowed=False), expected, factory, lambda value: value
        )
    assert calls == 0

    mismatched = _authorization(expected)
    mismatched["engine_sha256"] = "0" * 64
    with pytest.raises(ProtocolViolation, match="engine_sha256"):
        _run_authorized_for_test(mismatched, expected, factory, lambda value: value)
    assert calls == 0

    assert (
        _run_authorized_for_test(_authorization(expected), expected, factory, lambda _: "ran")
        == "ran"
    )
    assert calls == 1


def test_authorization_v2_requires_measurement_runtime_identity() -> None:
    expected = ExecutionIdentity("1" * 40, "2" * 64, "3" * 40)
    authorization = _authorization(expected)
    authorization["schema_version"] = "laserperception.m7.inference-authorization.v1"
    with pytest.raises(ProtocolViolation, match="schema"):
        verify_inference_authorization(authorization, expected)

    authorization = _authorization(expected)
    authorization["m7_measurement_runtime_commit"] = "4" * 40
    with pytest.raises(ProtocolViolation, match="m7_measurement_runtime_commit"):
        verify_inference_authorization(authorization, expected)


def test_frozen_ledger_byte_count_is_checked_before_hash(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_bytes(b"not-the-frozen-ledger")
    expected = ExecutionIdentity(
        "1" * 40,
        FROZEN_M7_INPUT_LEDGER_SHA256,
        "3" * 40,
    )
    artifacts = RuntimeArtifacts(
        input_ledger=ledger,
        engine=tmp_path / "engine",
        checkpoint=tmp_path / "checkpoint",
        onnx=tmp_path / "onnx",
        evaluator_identity=expected.evaluator_identity,
    )

    with pytest.raises(ProtocolViolation, match="byte count mismatch"):
        artifacts.verify_input_ledger(expected)


def test_measurement_gates_all_run_before_private_detector_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = {}
    for name, payload in (
        ("ledger", b"{}"),
        ("engine", b"engine"),
        ("checkpoint", b"checkpoint"),
        ("onnx", b"onnx"),
    ):
        path = tmp_path / name
        path.write_bytes(payload)
        files[name] = path
    expected = ExecutionIdentity(
        "1" * 40,
        hashlib.sha256(b"{}").hexdigest(),
        "3" * 40,
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
    builder_calls = 0

    def builder(
        actual_artifacts: RuntimeArtifacts, actual_expected: ExecutionIdentity
    ) -> _IdentityDetector:
        nonlocal builder_calls
        builder_calls += 1
        return _IdentityDetector(_detector_identity(actual_artifacts, actual_expected))

    class StubRunner:
        def __init__(self, **_: object) -> None:
            pass

        def run(self) -> CorpusRunSummary:
            return CorpusRunSummary(1_712, 0, 0, 0)

    monkeypatch.setattr(
        "benchmarks.m7.run_measurement.load_strict_input_ledger", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        "benchmarks.m7.run_measurement.verify_external_asset", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "benchmarks.m7.run_measurement.CanonicalM7SourceAdapter",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "benchmarks.m7.run_measurement.M7CheckpointStore", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr("benchmarks.m7.run_measurement.M7CorpusRunner", StubRunner)

    def run() -> CorpusRunSummary:
        return _run_measurement_internal(
            authorization_path,
            expected,
            artifacts,
            dataset_root=tmp_path / "dataset",
            m6b_input_asset=tmp_path / "m6-input.json",
            m6b_result_asset=tmp_path / "m6-result.json",
            checkpoint_root=tmp_path / "checkpoints",
            output_root=tmp_path / "output",
            _detector_builder=builder,
        )

    assert run() == CorpusRunSummary(1_712, 0, 0, 0)
    assert builder_calls == 1

    files["engine"].write_bytes(b"changed")
    with pytest.raises(ProtocolViolation, match="engine SHA256 mismatch"):
        run()
    assert builder_calls == 1
    files["engine"].write_bytes(b"engine")

    files["ledger"].write_bytes(b"changed")
    with pytest.raises(ProtocolViolation, match="input ledger SHA256 mismatch"):
        run()
    assert builder_calls == 1
    files["ledger"].write_bytes(b"{}")

    authorization_path.write_text(
        json.dumps(_authorization(expected, allowed=False)), encoding="utf-8"
    )
    with pytest.raises(ProtocolViolation, match="not explicitly authorized"):
        run()
    assert builder_calls == 1
    authorization_path.write_text(json.dumps(_authorization(expected)), encoding="utf-8")

    def malformed_ledger(*_args: object, **_kwargs: object) -> object:
        raise ProtocolViolation("M7 input ledger top-level schema is malformed")

    monkeypatch.setattr("benchmarks.m7.run_measurement.load_strict_input_ledger", malformed_ledger)
    with pytest.raises(ProtocolViolation, match="top-level schema"):
        run()
    assert builder_calls == 1


def test_canonical_measurement_api_has_no_arbitrary_execute_or_condition_list() -> None:
    parameters = inspect.signature(run_measurement).parameters
    assert "execute" not in parameters
    assert "conditions" not in parameters
    assert "detector_factory" not in parameters


def test_private_detector_identity_mismatch_stops_before_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files: dict[str, Path] = {}
    payloads = {
        "ledger": b"ledger",
        "engine": b"engine",
        "checkpoint": b"checkpoint",
        "onnx": b"onnx",
    }
    for name, payload in payloads.items():
        path = tmp_path / name
        path.write_bytes(payload)
        files[name] = path
    expected = ExecutionIdentity(
        "1" * 40,
        hashlib.sha256(payloads["ledger"]).hexdigest(),
        "3" * 40,
        engine_sha256=hashlib.sha256(payloads["engine"]).hexdigest(),
        checkpoint_sha256=hashlib.sha256(payloads["checkpoint"]).hexdigest(),
        onnx_sha256=hashlib.sha256(payloads["onnx"]).hexdigest(),
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
    alternate_engine = tmp_path / "alternate.engine"
    alternate_engine.write_bytes(payloads["engine"])
    runner_constructions = 0

    class ForbiddenRunner:
        def __init__(self, **_: object) -> None:
            nonlocal runner_constructions
            runner_constructions += 1

    monkeypatch.setattr(
        "benchmarks.m7.run_measurement.load_strict_input_ledger", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        "benchmarks.m7.run_measurement.verify_external_asset", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "benchmarks.m7.run_measurement.CanonicalM7SourceAdapter",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr("benchmarks.m7.run_measurement.M7CorpusRunner", ForbiddenRunner)

    def wrong_builder(
        actual_artifacts: RuntimeArtifacts, actual_expected: ExecutionIdentity
    ) -> _IdentityDetector:
        return _IdentityDetector(
            _detector_identity(
                actual_artifacts,
                actual_expected,
                engine_path=alternate_engine,
            )
        )

    with pytest.raises(ProtocolViolation, match="runtime identity mismatch: engine path"):
        _run_measurement_internal(
            authorization_path,
            expected,
            artifacts,
            dataset_root=tmp_path / "dataset",
            m6b_input_asset=tmp_path / "m6-input.json",
            m6b_result_asset=tmp_path / "m6-result.json",
            checkpoint_root=tmp_path / "checkpoints",
            output_root=tmp_path / "output",
            _detector_builder=wrong_builder,
        )
    assert runner_constructions == 0


def test_canonical_detector_constructs_backend_from_verified_runtime_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = RuntimeArtifacts(
        input_ledger=tmp_path / "ledger.json",
        engine=tmp_path / "verified.engine",
        checkpoint=tmp_path / "verified.pth",
        onnx=tmp_path / "verified.onnx",
        evaluator_identity=ExecutionIdentity("1" * 40, "2" * 64, "3" * 40).evaluator_identity,
    )
    expected = ExecutionIdentity("1" * 40, "2" * 64, "3" * 40)
    model_root = tmp_path / "mmdetection3d-v1.4.0"
    deploy_root = tmp_path / "mmdeploy-v1.3.1"
    model_config = model_root / M6B_MODEL_CONFIG_RELATIVE
    deploy_config = deploy_root / M6B_DEPLOY_CONFIG_RELATIVE
    model_config.parent.mkdir(parents=True)
    deploy_config.parent.mkdir(parents=True)
    model_config.write_text("model", encoding="utf-8")
    deploy_config.write_text("deploy", encoding="utf-8")
    constructed: dict[str, object] = {}

    class FakeBackend:
        def __init__(
            self,
            config_path: str | Path,
            checkpoint_path: str | Path,
            deploy_config_path: str | Path,
            *,
            checkpoint_sha256: str,
            device: str,
            voxelization_mode: str,
        ) -> None:
            self.config_path = Path(config_path)
            self.checkpoint_path = Path(checkpoint_path)
            self.deploy_config_path = Path(deploy_config_path)
            self.device = device
            self.voxelization_mode = voxelization_mode
            constructed.update(
                {
                    "config_path": self.config_path,
                    "checkpoint_path": self.checkpoint_path,
                    "deploy_config_path": self.deploy_config_path,
                    "checkpoint_sha256": checkpoint_sha256,
                    "device": device,
                    "voxelization_mode": voxelization_mode,
                }
            )

        def initialize(self) -> None:
            constructed["initialized"] = True

        def _backend_model(self, engine_path: str | Path) -> SimpleNamespace:
            constructed["engine_path"] = Path(engine_path)
            return SimpleNamespace(device=M6B_DEVICE)

    m6b_manifest = {
        "frozen_detector": {
            "source_manifest": M6B_DEPLOYMENT_MANIFEST_RELATIVE,
            "checkpoint_sha256": expected.checkpoint_sha256,
            "onnx_sha256": expected.onnx_sha256,
            "tensorrt_engine_sha256": expected.engine_sha256,
            "precision": M6B_PRECISION,
            "device": M6B_DEVICE,
            "voxelization_mode": M6B_VOXELIZATION_MODE,
            "provenance_mode": M6B_PROVENANCE_MODE,
            "score_threshold": M6B_EVALUATION_SCORE_THRESHOLD,
        }
    }
    m1_manifest = {
        "backend": {"commit": M6B_MODEL_REPOSITORY_COMMIT},
        "model": {
            "upstream_config": M6B_MODEL_CONFIG_RELATIVE,
            "checkpoint": {"sha256": expected.checkpoint_sha256},
        },
    }
    deployment_manifest = {
        "source_model": {"checkpoint_sha256": expected.checkpoint_sha256},
        "deployment": {
            "official_deployment_config": M6B_DEPLOY_CONFIG_RELATIVE,
            "exporter_commit": M6B_DEPLOY_REPOSITORY_COMMIT,
            "device": M6B_DEVICE,
            "precision": M6B_PRECISION,
        },
        "artifacts": {
            "onnx": {"sha256": expected.onnx_sha256},
            "engine": {"sha256": expected.engine_sha256},
        },
    }

    def load_manifest(path: Path) -> dict[str, object]:
        if path.as_posix().endswith(M6B_CONFIG_RELATIVE):
            return m6b_manifest
        if path.as_posix().endswith(M1_MANIFEST_RELATIVE):
            return m1_manifest
        return deployment_manifest

    def frozen_sha(path: str | Path) -> str:
        normalized = Path(path).as_posix()
        if normalized.endswith(M6B_MODEL_CONFIG_RELATIVE):
            return M6B_MODEL_CONFIG_SHA256
        if normalized.endswith(M6B_DEPLOY_CONFIG_RELATIVE):
            return M6B_DEPLOY_CONFIG_SHA256
        return "a" * 64

    monkeypatch.setattr(RuntimeArtifacts, "verify_runtime", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("benchmarks.m7.detector._load_yaml_mapping", load_manifest)
    monkeypatch.setattr(
        "benchmarks.m7.detector.resolve_m1_asset_paths",
        lambda _manifest: SimpleNamespace(mmdet3d_root=model_root),
    )
    monkeypatch.setattr(
        "benchmarks.m7.detector.resolve_m2_asset_paths",
        lambda _manifest: SimpleNamespace(mmdeploy_root=deploy_root),
    )
    monkeypatch.setattr("benchmarks.m7.detector.sha256_file", frozen_sha)
    monkeypatch.setattr("benchmarks.m7.detector.M2Backend", FakeBackend)

    detector = CanonicalM7Detector.from_verified_artifacts(
        artifacts,
        expected,
        repository_root=tmp_path,
    )

    assert constructed == {
        "config_path": model_config.resolve(),
        "checkpoint_path": artifacts.checkpoint.resolve(),
        "deploy_config_path": deploy_config.resolve(),
        "checkpoint_sha256": expected.checkpoint_sha256,
        "device": M6B_DEVICE,
        "voxelization_mode": M6B_VOXELIZATION_MODE,
        "initialized": True,
        "engine_path": artifacts.engine.resolve(),
    }
    assert detector.identity.checkpoint_path == artifacts.checkpoint.resolve()
    assert detector.identity.engine_path == artifacts.engine.resolve()
    assert detector.identity.onnx_path == artifacts.onnx.resolve()
    assert detector.identity.onnx_role == "verified_provenance_only_not_runtime_consumed"


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
    identity = CheckpointIdentity("1" * 40, "2" * 64, "3" * 40)
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
            CheckpointIdentity("9" * 40, "2" * 64, "3" * 40),
            conditions,
            _allow_synthetic_fixture=True,
        )
    with pytest.raises(ProtocolViolation, match="identity differs"):
        M7CheckpointStore(
            tmp_path,
            CheckpointIdentity("1" * 40, "2" * 64, "4" * 40),
            conditions,
            _allow_synthetic_fixture=True,
        )
    with pytest.raises(ProtocolViolation, match="shorten or reorder"):
        M7CheckpointStore(tmp_path / "canonical", identity, conditions)


def test_checkpoint_rejects_malformed_payload_and_never_deletes(tmp_path: Path) -> None:
    condition = "2011_09_26_drive_0001/0000000010|H10_LAG_COMPRESSED"
    identity = CheckpointIdentity("1" * 40, "2" * 64, "3" * 40)
    store = M7CheckpointStore(tmp_path, identity, (condition,), _allow_synthetic_fixture=True)
    store.save_complete(condition, input_sha256="3" * 64, hashes=_hashes(), payload={})
    path = next((tmp_path / "conditions").rglob("*.json"))
    record = json.loads(path.read_text(encoding="utf-8"))
    record["payload"] = {"tampered": True}
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ProtocolViolation, match="payload hash"):
        store.load_complete(condition, expected_input_sha256="3" * 64)
    assert path.exists()


def test_authorized_ledger_wrong_actual_input_stops_before_detector_or_checkpoint(
    tmp_path: Path,
) -> None:
    points_a = np.zeros((1, 4), dtype=np.float32)
    points_b = points_a.copy()
    points_b[0, 0] = np.nextafter(np.float32(0.0), np.float32(1.0))
    points_a.setflags(write=False)
    points_b.setflags(write=False)
    conditions = canonical_condition_ids()
    condition = conditions[0]
    authorized = _condition_record(condition, model_ready_sha256(points_a))
    ledger = StrictInputLedger({}, tuple(authorized for _ in conditions))
    detector = _Detector()
    store = M7CheckpointStore(tmp_path, CheckpointIdentity("1" * 40, "2" * 64, "3" * 40))
    runner = M7CorpusRunner(
        ledger=ledger,
        source_adapter=object(),  # type: ignore[arg-type]
        detector=detector,
        checkpoint_store=store,
        implementation_commit="1" * 40,
    )
    wrong = GeneratedCondition(Arm.B, points_b, authorized)

    with pytest.raises(ProtocolViolation, match="actual detector input SHA256 mismatch"):
        runner._invoke(condition, wrong, repeatability=False)

    assert detector.calls == []
    assert not (tmp_path / "conditions").exists()

    correct = GeneratedCondition(Arm.B, points_a, authorized)
    runner._invoke(condition, correct, repeatability=False)
    assert detector.calls == [condition]
    assert detector.point_ids == [id(points_a)]


def test_integrated_repeatability_reuses_repeat_one_without_eleventh_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    points = np.zeros((1, 4), dtype=np.float32)
    points.setflags(write=False)
    input_sha = model_ready_sha256(points)
    condition_ids = canonical_condition_ids()
    records = tuple(_condition_record(condition, input_sha) for condition in condition_ids)
    by_condition = {str(record["condition_id"]): record for record in records}
    ledger = StrictInputLedger({}, records)

    def generate(_adapter: object, frame_id: str, *, implementation_commit: str) -> GeneratedFrame:
        assert implementation_commit == "1" * 40
        values = tuple(
            GeneratedCondition(arm, points, by_condition[f"{frame_id}|{arm.value}"])
            for arm in (Arm.B, Arm.C, Arm.D, Arm.F)
        )
        return GeneratedFrame(frame_id, values)

    monkeypatch.setattr("benchmarks.m7.run_measurement.generate_canonical_frame", generate)
    detector = _Detector()
    store = M7CheckpointStore(tmp_path, CheckpointIdentity("1" * 40, "2" * 64, "3" * 40))
    runner = M7CorpusRunner(
        ledger=ledger,
        source_adapter=object(),  # type: ignore[arg-type]
        detector=detector,
        checkpoint_store=store,
        implementation_commit="1" * 40,
    )

    summary = runner.run()

    sentinel_conditions = {
        f"{frame_id}|{arm.value}"
        for frame_id in (
            "2011_09_26_drive_0001/0000000010",
            "2011_09_26_drive_0001/0000000083",
            "2011_09_26_drive_0001/0000000011",
            "2011_09_26_drive_0001/0000000015",
            "2011_09_26_drive_0091/0000000010",
        )
        for arm in (Arm.B, Arm.C, Arm.D, Arm.F)
    }
    assert all(detector.calls.count(condition) == 10 for condition in sentinel_conditions)
    assert detector.calls.count(condition_ids[100]) == (
        10 if condition_ids[100] in sentinel_conditions else 1
    )
    assert summary.repeatability_call_count == 200
    assert summary.inference_call_count == 1_892
    assert len(tuple((tmp_path / "conditions").rglob("*.json"))) == 1_712


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
