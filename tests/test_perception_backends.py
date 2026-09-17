"""CPU-only tests for P1 backend discovery, gates, adapters, and pipeline."""

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from laserperception.cli import main
from laserperception.detection.m8_input import M8PointCloud
from laserperception.detection.ros2_contract import ModelReadyPointCloud
from laserperception.detection.types import DetectionFrame
from laserperception.perception import (
    ExecutionAuthorization,
    ExecutionContext,
    ExecutionPlan,
    InputDescription,
    ModelRegistry,
    ModelResources,
    PerceptionInput,
    Precision,
    RuntimeTarget,
    load_model,
    registry,
)
from laserperception.perception.backends import factory as factory_module
from laserperception.perception.backends.base import BackendUnavailableError, PreparedInput
from laserperception.perception.backends.dsvt import (
    M8_ACCOUNTING_IDENTITY,
    AccountedSessionBinding,
    DsvtAdapter,
)
from laserperception.perception.backends.fake import FakeDetectionBackend
from laserperception.perception.backends.pointpillars import PointPillarsAdapter
from laserperception.perception.execution import generic_prediction_authorization_fields
from laserperception.perception.pipeline import DetectionPipeline


def _description(manifest, payload_kind, *, sample_id="sample"):
    return InputDescription(
        "1.0",
        sample_id,
        "lidar",
        payload_kind,
        manifest.input_features,
        manifest.coordinates,
        manifest.temporal,
        "synthetic CPU fixture",
    )


def _context(*, authorization=None, target=RuntimeTarget.EXTERNAL_CUDA, precision=Precision.FP32):
    return ExecutionContext(
        target,
        precision,
        "cuda:0" if target is RuntimeTarget.EXTERNAL_CUDA else "cpu",
        "external-fixture-runtime",
        "fixture-task",
        "a" * 40,
        True,
        authorization,
    )


def _pointpillars_fixture(tmp_path):
    base = registry.get("pointpillars-nuscenes-v0.3")
    config_bytes = b"# inert config fixture\n"
    checkpoint_bytes = b"inert checkpoint fixture"
    config = tmp_path / "model.py"
    checkpoint = tmp_path / "model.pth"
    config.write_bytes(config_bytes)
    checkpoint.write_bytes(checkpoint_bytes)
    manifest = base
    cloud = ModelReadyPointCloud(np.zeros((2, 4), dtype=np.float32))
    description = _description(manifest, "model_ready_xyzt")
    resources = ModelResources(config, checkpoint, upstream_root=tmp_path)
    unsigned = _context()
    payload = generic_prediction_authorization_fields(
        unsigned,
        manifest,
        description,
        input_sha256=cloud.sha256,
    )
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    authorization = ExecutionAuthorization(
        "generic_prediction",
        authorization_path,
        hashlib.sha256(authorization_path.read_bytes()).hexdigest(),
    )
    context = replace(unsigned, authorization=authorization)
    return manifest, cloud, description, context, resources


def _m8_fixture(tmp_path):
    manifest = registry.get("dsvt-pillar-transfusion-m8")
    candidate = tmp_path / "candidate.json"
    candidate.write_bytes(b"candidate fixture")
    candidate_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
    policy = tmp_path / "policy.json"
    policy.write_bytes(b"policy fixture")
    policy_sha256 = hashlib.sha256(policy.read_bytes()).hexdigest()
    authorization_path = tmp_path / "m8-authorization.json"
    authorization_path.write_bytes(b"authorization fixture")
    authorization_sha256 = hashlib.sha256(authorization_path.read_bytes()).hexdigest()
    authorization = ExecutionAuthorization(
        "m8_s1",
        authorization_path,
        authorization_sha256,
        "stage-r",
        "stage-r-1",
        policy,
    )
    context = _context(authorization=authorization)
    resources = ModelResources(candidate_manifest_path=candidate)
    cloud = M8PointCloud(np.zeros((2, 5), dtype=np.float32))
    value = PerceptionInput(_description(manifest, "m8_xyzit"), cloud)
    return (
        manifest,
        value,
        context,
        resources,
        candidate_sha256,
        policy_sha256,
        authorization_sha256,
    )


def test_input_envelope_retains_existing_payload_reference():
    manifest = registry.get("pointpillars-nuscenes-v0.3")
    cloud = ModelReadyPointCloud(np.zeros((1, 4), dtype=np.float32))
    value = PerceptionInput(_description(manifest, "model_ready_xyzt"), cloud)
    assert value.payload is cloud


def test_model_loading_validation_and_planning_are_metadata_only(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry run attempted backend import or file hashing")

    monkeypatch.setattr(factory_module, "_import_adapter", forbidden)
    import laserperception.perception.execution as execution_module

    monkeypatch.setattr(execution_module, "file_sha256", forbidden)
    manifest = registry.get("pointpillars-nuscenes-v0.3")
    description = _description(manifest, "model_ready_xyzt")
    authorization = ExecutionAuthorization("generic_prediction", tmp_path / "auth", "b" * 64)
    model = load_model(manifest.model_id)
    assert model.describe() == manifest
    assert model.validate(description).valid
    plan = model.plan(
        description,
        _context(authorization=authorization),
        ModelResources(tmp_path / "config", tmp_path / "checkpoint", upstream_root=tmp_path),
    )
    assert plan.ready_for_guarded_initialization
    assert not plan.executable
    assert not plan.authorization_verified
    assert not plan.hardware_probed
    assert plan.dependency_state == "not_imported_not_verified"
    assert ExecutionPlan.from_json(plan.to_json()) == plan
    assert (
        plan.to_json()
        == model.plan(
            description,
            _context(authorization=authorization),
            ModelResources(tmp_path / "config", tmp_path / "checkpoint", upstream_root=tmp_path),
        ).to_json()
    )


def test_dsvt_plan_is_always_blocked_from_generic_execution(tmp_path):
    manifest = registry.get("dsvt-pillar-transfusion-m8")
    reference = ExecutionAuthorization(
        "m8_s1",
        tmp_path / "auth",
        "b" * 64,
        "stage-r",
        "stage-r-1",
        tmp_path / "policy",
    )
    plan = load_model(manifest.model_id).plan(
        _description(manifest, "m8_xyzit"),
        _context(authorization=reference),
        ModelResources(candidate_manifest_path=tmp_path / "candidate"),
    )
    assert not plan.ready_for_guarded_initialization
    assert not plan.executable
    assert any("accounting-bound session" in error for error in plan.context_errors)


@pytest.mark.parametrize("case", ["runtime", "precision", "features", "coordinates", "temporal"])
def test_plan_rejects_incompatible_static_contracts(tmp_path, case):
    manifest = registry.get("pointpillars-nuscenes-v0.3")
    description = _description(manifest, "model_ready_xyzt")
    context = _context(
        authorization=ExecutionAuthorization("generic_prediction", tmp_path / "auth", "b" * 64)
    )
    if case == "runtime":
        context = _context(
            target=RuntimeTarget.CPU_TEST,
            authorization=context.authorization,
        )
    elif case == "precision":
        context = _context(precision=Precision.FP16, authorization=context.authorization)
    elif case == "features":
        description = replace(description, features=description.features[:-1])
    elif case == "coordinates":
        description = replace(
            description,
            coordinates=replace(description.coordinates, handedness="left"),
        )
    else:
        description = replace(
            description,
            temporal=replace(description.temporal, max_history=9),
        )
    plan = load_model(manifest.model_id).plan(
        description,
        context,
        ModelResources(tmp_path / "config", tmp_path / "checkpoint", upstream_root=tmp_path),
    )
    assert not plan.ready_for_guarded_initialization
    assert not plan.executable


def test_feature_order_is_explicitly_rejected():
    manifest = registry.get("pointpillars-nuscenes-v0.3")
    with pytest.raises(ValueError, match="ordered"):
        replace(
            _description(manifest, "model_ready_xyzt"),
            features=tuple(reversed(manifest.input_features)),
        )


def test_fake_pipeline_success_and_idempotent_close():
    base = registry.get("pointpillars-nuscenes-v0.3")
    manifest = replace(
        base,
        model_id="fake-test-only",
        display_name="Fake test backend",
        status="research",
        capabilities=replace(base.capabilities, runtime_targets=("cpu_test",)),
        scientific_status="non-scientific synthetic test only",
        authorization_required=False,
    )
    payload = object()
    value = PerceptionInput(_description(manifest, "point_cloud"), payload)
    backend = FakeDetectionBackend(manifest)
    pipeline = DetectionPipeline(backend)
    frame = pipeline.predict(value)
    assert frame.detections == ()
    assert frame.sample_id == value.description.sample_id
    assert frame.coordinate_frame == value.description.frame_id
    assert frame.metadata == {"backend": "fake", "non_scientific": True}
    assert value.payload is payload
    pipeline.close()
    pipeline.close()
    assert backend.events == ("prepare", "predict", "close")
    with pytest.raises(RuntimeError, match="closed"):
        pipeline.predict(value)


def test_pipeline_rejects_non_detection_output_and_closes():
    base = registry.get("pointpillars-nuscenes-v0.3")
    manifest = replace(base, model_id="bad-output-test", status="research")

    class BadOutputBackend(FakeDetectionBackend):
        def predict(self, value):
            super().predict(value)
            return object()

    value = PerceptionInput(_description(manifest, "point_cloud"), object())
    backend = BadOutputBackend(manifest)
    pipeline = DetectionPipeline(backend)
    with pytest.raises(TypeError, match="DetectionFrame"):
        pipeline.predict(value)
    assert pipeline.closed
    assert backend.events[-1] == "close"


def test_pointpillars_adapter_delegates_exact_path_and_frame(monkeypatch, tmp_path):
    manifest, cloud, description, context, resources = _pointpillars_fixture(tmp_path)
    value = PerceptionInput(description, cloud)
    frame = DetectionFrame((), "delegate-sample", "delegate-frame", {"untouched": True})
    calls = []

    class Delegate:
        def prepare_model_ready_points(self, points, *, sample_id, coordinate_frame):
            calls.append(("prepare", points, sample_id, coordinate_frame))
            return "prepared-sentinel"

        def run_prepared(self, sample):
            calls.append(("predict", sample))
            return frame

        def close(self):
            calls.append(("close",))

    def load(actual_context, actual_resources):
        calls.append(("load", actual_context, actual_resources))
        return Delegate()

    import laserperception.perception.backends.pointpillars as module

    monkeypatch.setattr(module, "verify_generic_prediction_authorization", lambda *a, **k: {})
    monkeypatch.setattr(module, "_load_existing_backend", load)
    adapter = PointPillarsAdapter(manifest, context, resources)
    prepared = adapter.prepare(value)
    assert adapter.predict(prepared) is frame
    assert calls[0][0] == "load"
    assert calls[1] == ("prepare", cloud, description.sample_id, description.frame_id)
    assert calls[2] == ("predict", "prepared-sentinel")
    forged = PreparedInput(manifest.model_id, value, prepared.payload, object())
    with pytest.raises(ValueError, match="not created"):
        adapter.predict(forged)
    adapter.close()
    adapter.close()
    assert calls.count(("close",)) == 1


def test_pointpillars_bad_payload_fails_before_authorization_or_loader(monkeypatch, tmp_path):
    manifest = registry.get("pointpillars-nuscenes-v0.3")
    value = PerceptionInput(_description(manifest, "model_ready_xyzt"), object())
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("loader must not run")

    import laserperception.perception.backends.pointpillars as module

    monkeypatch.setattr(module, "_load_existing_backend", forbidden)
    adapter = PointPillarsAdapter(
        manifest,
        _context(),
        ModelResources(tmp_path / "config", tmp_path / "checkpoint", upstream_root=tmp_path),
    )
    with pytest.raises(TypeError, match="ModelReadyPointCloud"):
        adapter.prepare(value)
    assert calls == []


def test_factory_verifies_authorization_before_adapter_import(monkeypatch, tmp_path):
    manifest, cloud, description, context, resources = _pointpillars_fixture(tmp_path)
    value = PerceptionInput(description, cloud)
    calls = []

    def importer(name):
        calls.append(name)
        return SimpleNamespace(
            PointPillarsAdapter=lambda actual_manifest, actual_context, actual_resources: (
                FakeDetectionBackend(actual_manifest, payload_kind="model_ready_xyzt")
            )
        )

    monkeypatch.setattr(
        factory_module,
        "verify_generic_prediction_authorization",
        lambda *a, **k: calls.append("authorize"),
    )
    monkeypatch.setattr(factory_module, "_import_adapter", importer)
    backend = factory_module.backend_for(
        ModelRegistry((manifest,)),
        manifest.model_id,
        value,
        context,
        resources,
    )
    assert isinstance(backend, FakeDetectionBackend)
    assert calls == ["authorize", "laserperception.perception.backends.pointpillars"]


def test_factory_missing_adapter_is_actionable(monkeypatch, tmp_path):
    manifest, cloud, description, context, resources = _pointpillars_fixture(tmp_path)

    def missing(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(
        factory_module, "verify_generic_prediction_authorization", lambda *a, **k: {}
    )
    monkeypatch.setattr(factory_module, "_import_adapter", missing)
    with pytest.raises(BackendUnavailableError, match="unavailable"):
        factory_module.backend_for(
            ModelRegistry((manifest,)),
            manifest.model_id,
            PerceptionInput(description, cloud),
            context,
            resources,
        )


def test_generic_authorization_duplicate_key_is_rejected_before_import(monkeypatch, tmp_path):
    manifest, cloud, description, context, resources = _pointpillars_fixture(tmp_path)
    assert context.authorization is not None
    path = context.authorization.path
    path.write_text('{"model_id":"duplicate",' + path.read_text()[1:], encoding="utf-8")
    reference = replace(context.authorization, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    bad_context = replace(context, authorization=reference)
    calls = []

    def importer(name):
        calls.append(name)
        raise AssertionError("adapter import must not run")

    monkeypatch.setattr(factory_module, "_import_adapter", importer)
    with pytest.raises(ValueError, match="duplicate"):
        factory_module.backend_for(
            ModelRegistry((manifest,)),
            manifest.model_id,
            PerceptionInput(description, cloud),
            bad_context,
            resources,
        )
    assert calls == []


def _fake_m8_runtime(events, candidate_sha256, policy_sha256):
    def authorizer(mode, logical_pass_id, path, identity):
        events.append(("authorize", mode, logical_pass_id, path, identity))
        return {"runtime_policy_binding_sha256": policy_sha256}

    return SimpleNamespace(
        CANDIDATE_MANIFEST_SHA256=candidate_sha256,
        AuthorizationIdentity=lambda commit: ("identity", commit),
        require_scientific_authorization=authorizer,
    )


def _session_binding(context, candidate_sha256, policy_sha256, authorization_sha256):
    assert context.authorization is not None
    assert context.authorization.mode is not None
    assert context.authorization.logical_pass_id is not None
    return AccountedSessionBinding(
        M8_ACCOUNTING_IDENTITY,
        "dsvt-pillar-transfusion-m8",
        context.runtime_id,
        context.task_id,
        context.execution_commit,
        context.authorization.mode,
        context.authorization.logical_pass_id,
        authorization_sha256,
        policy_sha256,
        candidate_sha256,
    )


def test_dsvt_unauthorized_fails_before_runtime_or_session(monkeypatch, tmp_path):
    manifest = registry.get("dsvt-pillar-transfusion-m8")
    value = PerceptionInput(
        _description(manifest, "m8_xyzit"),
        M8PointCloud(np.zeros((1, 5), dtype=np.float32)),
    )
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("runtime/session must not open")

    import laserperception.perception.backends.dsvt as module

    monkeypatch.setattr(module, "_import_m8_runtime", forbidden)
    monkeypatch.setattr(module, "_open_accounted_session", forbidden)
    adapter = DsvtAdapter(
        manifest,
        _context(),
        ModelResources(candidate_manifest_path=tmp_path / "candidate"),
    )
    with pytest.raises(ValueError, match="authorization"):
        adapter.prepare(value)
    assert calls == []


def test_dsvt_adapter_delegates_only_to_exact_accounted_session(monkeypatch, tmp_path):
    (
        manifest,
        value,
        context,
        resources,
        candidate_sha256,
        policy_sha256,
        authorization_sha256,
    ) = _m8_fixture(tmp_path)
    frame = DetectionFrame((), "m8-output", "lidar", {"untouched": True})
    events = []

    class Session:
        binding = _session_binding(
            context,
            candidate_sha256,
            policy_sha256,
            authorization_sha256,
        )

        def predict_accounted(self, actual):
            events.append(("predict_accounted", actual))
            return frame

        def close(self):
            events.append(("close",))

    import laserperception.perception.backends.dsvt as module

    monkeypatch.setattr(
        module,
        "_import_m8_runtime",
        lambda: _fake_m8_runtime(events, candidate_sha256, policy_sha256),
    )

    def open_session(actual_context, actual_resources, authorization):
        events.append(("open_accounted", actual_context, actual_resources, authorization))
        return Session()

    monkeypatch.setattr(module, "_open_accounted_session", open_session)
    adapter = DsvtAdapter(manifest, context, resources)
    prepared = adapter.prepare(value)
    assert events[0][0] == "authorize"
    assert events[1][0] == "open_accounted"
    assert adapter.predict(prepared) is frame
    assert events[2] == ("predict_accounted", value)
    adapter.close()
    adapter.close()
    assert events.count(("close",)) == 1


def test_dsvt_policy_or_session_binding_mismatch_fails_closed(monkeypatch, tmp_path):
    (
        manifest,
        value,
        context,
        resources,
        candidate_sha256,
        policy_sha256,
        authorization_sha256,
    ) = _m8_fixture(tmp_path)
    events = []
    import laserperception.perception.backends.dsvt as module

    monkeypatch.setattr(
        module,
        "_import_m8_runtime",
        lambda: _fake_m8_runtime(events, candidate_sha256, policy_sha256),
    )
    assert context.authorization is not None
    assert context.authorization.runtime_policy_path is not None
    context.authorization.runtime_policy_path.write_bytes(b"changed")
    monkeypatch.setattr(
        module,
        "_open_accounted_session",
        lambda *args: pytest.fail("session opened after policy mismatch"),
    )
    with pytest.raises(ValueError, match="policy"):
        DsvtAdapter(manifest, context, resources).prepare(value)

    context.authorization.runtime_policy_path.write_bytes(b"policy fixture")

    class WrongSession:
        binding = replace(
            _session_binding(
                context,
                candidate_sha256,
                policy_sha256,
                authorization_sha256,
            ),
            runtime_id="wrong",
        )

        def predict_accounted(self, actual):
            pytest.fail("wrong session predicted")

        def close(self):
            events.append(("wrong_close",))

    monkeypatch.setattr(module, "_open_accounted_session", lambda *args: WrongSession())
    with pytest.raises(ValueError, match="bindings"):
        DsvtAdapter(manifest, context, resources).prepare(value)
    assert ("wrong_close",) in events


def test_dsvt_default_production_bridge_fails_before_backend_import(monkeypatch, tmp_path):
    (
        manifest,
        value,
        context,
        resources,
        candidate_sha256,
        policy_sha256,
        _,
    ) = _m8_fixture(tmp_path)
    events = []
    import laserperception.perception.backends.dsvt as module

    monkeypatch.setattr(
        module,
        "_import_m8_runtime",
        lambda: _fake_m8_runtime(events, candidate_sha256, policy_sha256),
    )
    with pytest.raises(BackendUnavailableError, match="frozen S1 runner"):
        DsvtAdapter(manifest, context, resources).prepare(value)
    assert events[0][0] == "authorize"


def test_cli_dry_run_reports_structured_blockers_without_execution(monkeypatch, tmp_path, capsys):
    manifest = registry.get("pointpillars-nuscenes-v0.3")
    input_path = tmp_path / "input.json"
    input_path.write_text(_description(manifest, "model_ready_xyzt").to_json(), encoding="utf-8")
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("dry run attempted execution")

    monkeypatch.setattr(factory_module, "_import_adapter", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    assert (
        main(
            [
                "predict",
                "--model",
                manifest.model_id,
                "--input",
                str(input_path),
                "--runtime-target",
                "external_cuda",
                "--precision",
                "fp32",
                "--device",
                "cuda:0",
                "--runtime-id",
                "planned-worker",
                "--task-id",
                "planned-task",
                "--execution-commit",
                "a" * 40,
                "--deterministic",
                "--dry-run",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["model_id"] == manifest.model_id
    assert output["hardware_probed"] is False
    assert output["executable"] is False
    assert output["authorization_state"] == "missing_generic_prediction_authorization"
    assert output["dependency_state"] == "not_imported_not_verified"
    assert calls == []


def test_clean_process_discovery_plan_blocks_optional_imports():
    root = Path(__file__).resolve().parents[1]
    script = r"""
import importlib.abc
import sys

forbidden = {
    "torch", "mmdet", "mmdet3d", "mmengine", "mmdeploy", "tensorrt",
    "pcdet", "spconv", "torch_scatter", "pycuda", "cupy", "rclpy",
}
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in forbidden:
            raise AssertionError(fullname)
        return None
sys.meta_path.insert(0, Guard())
from pathlib import Path
from laserperception.perception import (
    ExecutionContext, InputDescription, ModelResources, Precision, RuntimeTarget,
    load_model, registry,
)
manifest = registry.get("pointpillars-nuscenes-v0.3")
description = InputDescription(
    "1.0", "sample", "lidar", "model_ready_xyzt", manifest.input_features,
    manifest.coordinates, manifest.temporal, "isolated import fixture",
)
model = load_model(manifest.model_id)
model.validate(description)
model.plan(
    description,
    ExecutionContext(
        RuntimeTarget.EXTERNAL_CUDA, Precision.FP32, "cuda:0", "runtime", "task",
        "a" * 40, True,
    ),
    ModelResources(Path("config"), Path("checkpoint")),
)
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "failure", ["commit", "dirty", "path", "git_missing", "nested_root", "hidden_index"]
)
def test_pointpillars_checkout_rejects_changed_identity(monkeypatch, tmp_path, failure):
    import laserperception.perception.execution as module

    config = tmp_path / module.POINTPILLARS_CONFIG_RELATIVE
    resources = ModelResources(config_path=config, upstream_root=tmp_path)
    calls = []

    def git(command, **kwargs):
        calls.append(command)
        assert kwargs["cwd"] == tmp_path.resolve()
        if failure == "git_missing":
            raise OSError("mock missing Git")
        output = ""
        if command[-1] == "HEAD":
            output = module.POINTPILLARS_UPSTREAM_COMMIT
        elif command[-1] == "--show-toplevel":
            output = str(tmp_path.parent if failure == "nested_root" else tmp_path)
        elif command[1] == "ls-files":
            output = "h configs/base.py" if failure == "hidden_index" else "H configs/base.py"
        if failure == "commit" and command[-1] == "HEAD":
            output = "b" * 40
        if failure == "dirty" and command[1] == "status":
            output = " M configs/base.py"
        return SimpleNamespace(returncode=0, stdout=output)

    monkeypatch.setattr(module.subprocess, "run", git)
    if failure == "path":
        resources = replace(resources, config_path=tmp_path / "wrong.py")
    with pytest.raises(ValueError, match="checkout|pinned|top level"):
        module._verify_pointpillars_checkout(resources)
    if failure == "path":
        assert calls == []


def test_pointpillars_checkout_accepts_exact_clean_pinned_identity(monkeypatch, tmp_path):
    import laserperception.perception.execution as module

    calls = []

    def git(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout=(
                module.POINTPILLARS_UPSTREAM_COMMIT
                if command[-1] == "HEAD"
                else str(tmp_path)
                if command[-1] == "--show-toplevel"
                else "H configs/base.py"
                if command[1] == "ls-files"
                else ""
            ),
        )

    monkeypatch.setattr(module.subprocess, "run", git)
    module._verify_pointpillars_checkout(
        ModelResources(
            config_path=tmp_path / module.POINTPILLARS_CONFIG_RELATIVE, upstream_root=tmp_path
        )
    )
    assert len(calls) == 4


@pytest.mark.parametrize("failure", ["authorization_hash", "input", "config", "checkpoint"])
def test_generic_authorization_rejects_binding_or_artifact_changes(monkeypatch, tmp_path, failure):
    import laserperception.perception.execution as module

    manifest, cloud, description, context, resources = _pointpillars_fixture(tmp_path)
    monkeypatch.setattr(module, "_verify_pointpillars_checkout", lambda resources: None)
    original_hash = module.file_sha256

    def hashes(path):
        if path == resources.config_path and failure != "config":
            return module.POINTPILLARS_CONFIG_SHA256
        if path == resources.checkpoint_path and failure != "checkpoint":
            return module.POINTPILLARS_CHECKPOINT_SHA256
        return original_hash(path)

    monkeypatch.setattr(module, "file_sha256", hashes)
    if failure == "authorization_hash":
        context = replace(context, authorization=replace(context.authorization, sha256="b" * 64))
    input_hash = "b" * 64 if failure == "input" else cloud.sha256
    with pytest.raises(ValueError, match="identity|scope|SHA256"):
        module.verify_generic_prediction_authorization(
            context,
            manifest,
            resources,
            description,
            input_sha256=input_hash,
        )


def test_generic_authorization_accepts_exact_mocked_file_bindings(monkeypatch, tmp_path):
    import laserperception.perception.execution as module

    manifest, cloud, description, context, resources = _pointpillars_fixture(tmp_path)
    events = []
    monkeypatch.setattr(
        module, "_verify_pointpillars_checkout", lambda r: events.append("checkout")
    )
    original_hash = module.file_sha256

    def hashes(path):
        if path == resources.config_path:
            return module.POINTPILLARS_CONFIG_SHA256
        if path == resources.checkpoint_path:
            return module.POINTPILLARS_CHECKPOINT_SHA256
        return original_hash(path)

    monkeypatch.setattr(module, "file_sha256", hashes)
    payload = module.verify_generic_prediction_authorization(
        context,
        manifest,
        resources,
        description,
        input_sha256=cloud.sha256,
    )
    assert payload["upstream_commit"] == module.POINTPILLARS_UPSTREAM_COMMIT
    assert events == ["checkout"]


def test_custom_registry_cannot_change_builtin_execution_identity(monkeypatch, tmp_path):
    manifest, cloud, description, context, resources = _pointpillars_fixture(tmp_path)
    changed = replace(
        manifest,
        artifacts=tuple(
            replace(artifact, sha256="b" * 64) if artifact.role == "checkpoint" else artifact
            for artifact in manifest.artifacts
        ),
    )
    monkeypatch.setattr(
        factory_module, "_import_adapter", lambda *a: pytest.fail("adapter imported")
    )
    with pytest.raises(ValueError, match="canonical"):
        factory_module.backend_for(
            ModelRegistry((changed,)),
            changed.model_id,
            PerceptionInput(description, cloud),
            context,
            resources,
        )
    with pytest.raises(ValueError, match="canonical"):
        PointPillarsAdapter(changed, context, resources).prepare(
            PerceptionInput(description, cloud)
        )


def test_loaded_model_predict_lifecycle_and_execution_binding(monkeypatch, tmp_path):
    manifest, cloud, description, context, resources = _pointpillars_fixture(tmp_path)
    backend = FakeDetectionBackend(manifest, payload_kind="model_ready_xyzt")
    constructions = []

    def factory(*args):
        constructions.append(args)
        return backend

    monkeypatch.setitem(load_model.__globals__, "backend_for", factory)
    model = load_model(manifest.model_id)
    value = PerceptionInput(description, cloud)
    assert (
        model.predict(value, context=context, resources=resources).sample_id
        == description.sample_id
    )
    model.predict(value, context=context, resources=resources)
    assert len(constructions) == 1
    with pytest.raises(ValueError, match="cannot change"):
        model.predict(value, context=replace(context, task_id="changed"), resources=resources)
    model.close()
    model.close()
    assert backend.events.count("close") == 1
    with pytest.raises(RuntimeError, match="closed"):
        model.predict(value, context=context, resources=resources)


def test_loaded_model_closes_after_backend_failure(monkeypatch, tmp_path):
    manifest, cloud, description, context, resources = _pointpillars_fixture(tmp_path)

    class FailedBackend(FakeDetectionBackend):
        def prepare(self, value):
            raise ValueError("mock prepare failure")

    backend = FailedBackend(manifest, payload_kind="model_ready_xyzt")
    monkeypatch.setitem(load_model.__globals__, "backend_for", lambda *a: backend)
    model = load_model(manifest.model_id)
    value = PerceptionInput(description, cloud)
    with pytest.raises(ValueError, match="prepare failure"):
        model.predict(value, context=context, resources=resources)
    assert backend.events == ("close",)
    with pytest.raises(RuntimeError, match="closed"):
        model.predict(value, context=context, resources=resources)


def test_cli_complete_references_preserve_task_binding_without_file_checks(
    monkeypatch, tmp_path, capsys
):
    manifest, cloud, description, context, resources = _pointpillars_fixture(tmp_path)
    path = tmp_path / "input.json"
    path.write_text(description.to_json(), encoding="utf-8")
    monkeypatch.setattr(
        factory_module, "_import_adapter", lambda *a: pytest.fail("adapter imported")
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("subprocess started"))
    assert (
        main(
            [
                "predict",
                "--model",
                manifest.model_id,
                "--input",
                str(path),
                "--runtime-target",
                "external_cuda",
                "--precision",
                "fp32",
                "--device",
                "cuda:0",
                "--runtime-id",
                context.runtime_id,
                "--task-id",
                "explicit-cli-task",
                "--execution-commit",
                context.execution_commit,
                "--deterministic",
                "--dry-run",
                "--authorization-kind",
                "generic_prediction",
                "--authorization",
                str(context.authorization.path),
                "--authorization-sha256",
                context.authorization.sha256,
                "--config",
                str(resources.config_path),
                "--checkpoint",
                str(resources.checkpoint_path),
                "--upstream-root",
                str(resources.upstream_root),
            ]
        )
        == 0
    )
    plan = ExecutionPlan.from_json(capsys.readouterr().out)
    assert plan.ready_for_guarded_initialization
    assert plan.task_id == "explicit-cli-task"
    assert not plan.authorization_verified
    assert not plan.executable


def test_pipeline_preserves_prediction_error_if_cleanup_also_fails():
    manifest = registry.get("pointpillars-nuscenes-v0.3")

    class FailedBackend(FakeDetectionBackend):
        def predict(self, value):
            raise ValueError("original prediction error")

        def close(self):
            super().close()
            raise RuntimeError("cleanup error")

    pipeline = DetectionPipeline(FailedBackend(manifest))
    with pytest.raises(ValueError, match="original prediction error") as caught:
        pipeline.predict(PerceptionInput(_description(manifest, "point_cloud"), object()))
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert pipeline.closed
    pipeline.close()


def test_loaded_model_context_manager_returns_exact_frame_and_closes(monkeypatch, tmp_path):
    manifest, cloud, description, context, resources = _pointpillars_fixture(tmp_path)
    frame = DetectionFrame((), "exact-delegate-sample", "exact-frame", {"unchanged": True})

    class ExactBackend(FakeDetectionBackend):
        def predict(self, value):
            super().predict(value)
            return frame

    backend = ExactBackend(manifest, payload_kind="model_ready_xyzt")
    monkeypatch.setitem(load_model.__globals__, "backend_for", lambda *a: backend)
    with load_model(manifest.model_id) as model:
        assert (
            model.predict(PerceptionInput(description, cloud), context=context, resources=resources)
            is frame
        )
    assert backend.events[-1] == "close"


def test_generic_authorization_rejects_numeric_deterministic_boolean(monkeypatch, tmp_path):
    import laserperception.perception.execution as module

    manifest, cloud, description, context, resources = _pointpillars_fixture(tmp_path)
    path = context.authorization.path
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["deterministic"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    context = replace(
        context,
        authorization=replace(
            context.authorization, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        ),
    )
    monkeypatch.setattr(
        module, "_verify_pointpillars_checkout", lambda *a: pytest.fail("checkout checked")
    )
    with pytest.raises(ValueError, match="scope or schema"):
        module.verify_generic_prediction_authorization(
            context,
            manifest,
            resources,
            description,
            input_sha256=cloud.sha256,
        )
