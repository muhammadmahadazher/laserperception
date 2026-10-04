"""Zero-forward initialization and prospective authorization gates, using CPU doubles."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from laserperception.detection import m8_s2_readiness as readiness
from laserperception.detection import m8_s2_runtime as runtime
from laserperception.detection.m8_s1_runtime import CANDIDATE_MANIFEST_PATH, atomic_write_json
from laserperception.detection.m8_s2_bootstrap import REQUIRED_PACKAGES

ROOT = Path(__file__).resolve().parents[1]
COMMIT, INPUT, POLICY, QUAL, READY = "a" * 40, "b" * 64, "c" * 64, "d" * 64, "e" * 64
CANDIDATE = json.loads((ROOT / CANDIDATE_MANIFEST_PATH).read_text())


@pytest.fixture
def bindings() -> dict:
    policy = {
        "schema_version": runtime.RUNTIME_POLICY_SCHEMA,
        "repository_execution_commit": COMMIT,
        "candidate_identity": {
            "architecture": CANDIDATE["architecture"],
            "candidate_manifest_sha256": runtime.CANDIDATE_MANIFEST_SHA256,
            "upstream_commit": CANDIDATE["upstream"]["commit"],
            "config_sha256": CANDIDATE["upstream"]["config_sha256"],
            "checkpoint_sha256": CANDIDATE["checkpoint"]["sha256"],
        },
        "s2_protocol_sha256": runtime.PROTOCOL_SHA256,
        "s2_partitions_sha256": runtime.PARTITIONS_SHA256,
        "s2_input_freeze_sha256": runtime.INPUT_FREEZE_SHA256,
        "s2_full_ledger_sha256": runtime.FULL_LEDGER_SHA256,
        "s2_compact_manifest_sha256": runtime.COMPACT_MANIFEST_SHA256,
        "worker_hostname": "synthetic",
        "gpu_name": "synthetic",
        "gpu_uuid": "GPU-synthetic",
        "nvidia_driver": "synthetic",
        "gpu_vram_bytes": 48,
        "python_exact_version": sys.version,
        "pytorch_exact_version": "2.1.0+cu118",
        "torchvision": "0.16.0+cu118",
        "cuda_runtime": "11.8",
        "numpy": "1.23.5",
        "spconv": "2.3.8",
        "torch_scatter": "2.1.2+pt21cu118",
        "upstream_native_extensions": {"pcdet/ops/fixture.so": "f" * 64},
        "candidate_import_packages": dict(REQUIRED_PACKAGES),
    }
    return dict(
        execution_commit=COMMIT,
        input_gate_receipt_sha256=INPUT,
        runtime_policy_sha256=POLICY,
        qualification_receipt_sha256=QUAL,
        policy=policy,
        candidate=CANDIDATE,
    )


@pytest.mark.parametrize("version", [None, "0.16.0", "0.16.0+cpu", "0.16.0+cu121", "0.17.0+cu118"])
def test_policy_requires_exact_torchvision(bindings: dict, version: object) -> None:
    bindings["policy"]["torchvision"] = version
    with pytest.raises(runtime.M8S2ProtocolViolation):
        readiness.readiness_document(**bindings)


@pytest.mark.parametrize("package", list(REQUIRED_PACKAGES))
def test_preimport_missing_dependency_fails(
    bindings: dict, monkeypatch: pytest.MonkeyPatch, package: str
) -> None:
    def version(name: str) -> str:
        if name == package:
            raise readiness.metadata.PackageNotFoundError(name)
        return REQUIRED_PACKAGES[name]

    monkeypatch.setattr(readiness.metadata, "version", version)
    monkeypatch.setattr(
        readiness, "importlib", SimpleNamespace(import_module=lambda _: pytest.fail("heavy import"))
    )
    with pytest.raises(runtime.M8S2ProtocolViolation, match="package absent"):
        readiness.verify_preimport_runtime(bindings["policy"], ROOT)


def test_preimport_identity_and_inventory(bindings: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(readiness.metadata, "version", lambda name: REQUIRED_PACKAGES[name])
    monkeypatch.setattr(readiness, "native_extension_hashes", lambda _: {"different.so": "0" * 64})
    with pytest.raises(runtime.M8S2ProtocolViolation, match="native inventory"):
        readiness.verify_preimport_runtime(bindings["policy"], ROOT)
    monkeypatch.setattr(
        readiness,
        "native_extension_hashes",
        lambda _: bindings["policy"]["upstream_native_extensions"],
    )
    readiness.verify_preimport_runtime(bindings["policy"], ROOT)
    bindings["policy"]["python_exact_version"] = "different"
    with pytest.raises(runtime.M8S2ProtocolViolation, match="Python"):
        readiness.verify_preimport_runtime(bindings["policy"], ROOT)


@pytest.mark.parametrize("bad_state", ["cpu", "train", "empty", "sync", "valid"])
def test_backend_state_requires_all_cuda0_eval_and_sync(bad_state: str) -> None:
    from laserperception.detection.m8_backend import DsvtBackend

    backend = object.__new__(DsvtBackend)
    events = []
    backend._model = SimpleNamespace(
        parameters=lambda: [] if bad_state == "empty" else [SimpleNamespace(device="cuda:0")],
        buffers=lambda: (
            []
            if bad_state == "empty"
            else [SimpleNamespace(device="cpu" if bad_state == "cpu" else "cuda:0")]
        ),
        modules=lambda: [SimpleNamespace(training=bad_state == "train")],
    )

    def sync(index: int) -> None:
        assert index == 0
        if bad_state == "sync":
            raise RuntimeError("synthetic synchronization failure")
        events.append("sync")

    backend._torch = SimpleNamespace(
        device=lambda name: name, cuda=SimpleNamespace(synchronize=sync)
    )
    if bad_state == "valid":
        assert backend.initialization_state()["initialization_complete"] is True
        assert events == ["sync"]
    else:
        with pytest.raises(RuntimeError):
            backend.initialization_state()


@pytest.mark.parametrize(
    "field",
    [
        "execution_commit",
        "input_gate_receipt_sha256",
        "runtime_policy_sha256",
        "qualification_receipt_sha256",
        "worker_hostname",
        "gpu_uuid",
        "checkpoint_sha256",
        "model_device",
        "model_eval",
        "initialization_complete",
        "inference_calls",
        "GT_loaded",
        "evaluator_loaded",
        "generated_version_sha256",
        "native_extension_inventory_sha256",
    ],
)
def test_receipt_rejects_binding_or_semantic_changes(
    tmp_path: Path, bindings: dict, field: str
) -> None:
    receipt = readiness.readiness_document(**bindings)
    path = tmp_path / "receipt.json"
    atomic_write_json(path, receipt)
    assert len(readiness.verify_readiness_receipt(path, **bindings)) == 64
    receipt[field] = "changed"
    atomic_write_json(path, receipt)
    with pytest.raises(runtime.M8S2ProtocolViolation, match="binding differs"):
        readiness.verify_readiness_receipt(path, **bindings)


@pytest.mark.parametrize("failure", [False, True])
def test_exact_backend_factory_zero_forward(
    bindings: dict, monkeypatch: pytest.MonkeyPatch, failure: bool
) -> None:
    expected = readiness.readiness_document(**bindings)
    identity = {
        k: expected[k]
        for k in ("architecture", "upstream_commit", "config_sha256", "checkpoint_sha256")
    }
    identity.update(
        torch="2.1.0+cu118",
        cuda="11.8",
        spconv="2.3.8",
        torch_scatter="2.1.2+pt21cu118",
        device="cuda:0",
    )
    calls = []

    def factory(*, manifest_path: Path) -> object:
        calls.append(manifest_path)
        if failure:
            raise RuntimeError("synthetic missing dependency")
        return SimpleNamespace(
            identity=identity,
            initialization_state=lambda: {
                "model_device": "cuda:0",
                "model_eval": True,
                "initialization_complete": True,
            },
        )  # No inference methods: any forward/smoke/timing call fails this double.

    def load(name: str) -> object:
        assert name == "laserperception.detection.m8_backend"
        return SimpleNamespace(DsvtBackend=SimpleNamespace(from_environment=factory))

    monkeypatch.setattr(readiness, "importlib", SimpleNamespace(import_module=load))
    if failure:
        with pytest.raises(RuntimeError, match="missing dependency"):
            readiness.initialize_candidate(ROOT, expected)
    else:
        assert readiness.initialize_candidate(ROOT, expected) == expected
        assert expected["inference_calls"] == 0 and expected["GT_loaded"] is False
    assert calls == [ROOT / CANDIDATE_MANIFEST_PATH]


@pytest.mark.parametrize(
    "scope,logical",
    [
        ("sizing-only", "s2-sizing-01"),
        ("repeatability-only", "s2-repeatability-01"),
        ("full-pass-only", "s2-pass-1"),
    ],
)
def test_all_execution_grants_require_readiness(tmp_path: Path, scope: str, logical: str) -> None:
    payload = dict(
        schema_version=runtime.AUTHORIZATION_SCHEMA,
        authorized=True,
        owner_approval=True,
        scope=scope,
        authorization_id="synthetic",
        authorization_timestamp_utc="synthetic",
        authorization_provenance="synthetic fixture only",
        authorized_gpu_uuid=None,
        authorized_worker_hostname=None,
        authorized_campaign_root=str(tmp_path.resolve()),
        execution_commit=COMMIT,
        protocol_sha256=runtime.PROTOCOL_SHA256,
        partitions_sha256=runtime.PARTITIONS_SHA256,
        input_freeze_sha256=runtime.INPUT_FREEZE_SHA256,
        full_ledger_sha256=runtime.FULL_LEDGER_SHA256,
        compact_manifest_sha256=runtime.COMPACT_MANIFEST_SHA256,
        logical_pass_ids=[logical],
        runtime_policy_binding_sha256=POLICY,
        input_gate_receipt_sha256=INPUT,
        qualification_receipt_sha256=QUAL,
        candidate_readiness_receipt_sha256=READY,
        repeatability_review_sha256="f" * 64 if scope == "full-pass-only" else None,
        repeatability_owner_attestation_sha256="0" * 64 if scope == "full-pass-only" else None,
    )
    expected = dict(
        scope=scope,
        execution_commit=COMMIT,
        logical_pass_id=logical,
        campaign_root=tmp_path,
        runtime_policy_sha256=POLICY,
        input_gate_receipt_sha256=INPUT,
        qualification_receipt_sha256=QUAL,
        candidate_readiness_receipt_sha256=READY,
        repeatability_review_sha256=payload["repeatability_review_sha256"],
        repeatability_owner_attestation_sha256=payload["repeatability_owner_attestation_sha256"],
    )
    if scope == "sizing-only":
        payload["sizing_plan_sha256"] = runtime.SIZING_PLAN_SHA256
        expected["sizing_plan_sha256"] = runtime.SIZING_PLAN_SHA256
    runtime.verify_authorization(payload, **expected)
    for value in (None, "1" * 64):
        payload["candidate_readiness_receipt_sha256"] = value
        with pytest.raises(runtime.M8S2ProtocolViolation):
            runtime.verify_authorization(payload, **expected)
    payload.pop("candidate_readiness_receipt_sha256")
    with pytest.raises(runtime.M8S2ProtocolViolation):
        runtime.verify_authorization(payload, **expected)


@pytest.mark.parametrize(
    "failure",
    [
        "authorization",
        "static",
        "input",
        "qualification",
        "policy",
        "worker",
        "native",
        "init",
        "success",
    ],
)
def test_readiness_cli_order_and_receipt_boundary(
    tmp_path: Path, bindings: dict, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    main = runpy.run_path(str(ROOT / "scripts/detection/run_m8_s2.py"))["main"]
    glob = main.__globals__
    events = []

    def gate(name: str, value: object = None):
        def check(*a: object, **kw: object) -> object:
            events.append(name)
            if name == failure:
                raise runtime.M8S2ProtocolViolation("synthetic " + name)
            return value

        return check

    monkeypatch.setitem(glob, "verify_static_bindings", gate("static"))
    monkeypatch.setitem(
        glob,
        "require_authorization",
        gate(
            "authorization",
            {"authorized_worker_hostname": "synthetic", "authorized_gpu_uuid": "GPU-synthetic"},
        ),
    )
    monkeypatch.setitem(glob, "verify_input_gate_receipt", gate("input", INPUT))
    monkeypatch.setitem(glob, "verify_qualification_receipt", gate("qualification", QUAL))
    monkeypatch.setitem(glob, "_external_candidate", gate("candidate"))
    monkeypatch.setitem(glob, "verify_preimport_runtime", gate("native"))
    monkeypatch.setitem(glob, "verify_qualification_worker", gate("worker"))
    monkeypatch.setitem(glob, "sha256_file", lambda _: POLICY)

    def capture(*a: object) -> dict:
        assert events == [
            "static",
            "authorization",
            "input",
            "qualification",
            "candidate",
            "native",
            "worker",
        ]
        events.append("heavy")
        return bindings["policy"]

    monkeypatch.setitem(glob, "_capture_bound_policy", capture)
    monkeypatch.setitem(glob, "verify_runtime_policy", gate("live"))
    expected = readiness.readiness_document(**bindings)
    monkeypatch.setitem(glob, "initialize_candidate", gate("init", expected))
    policy = tmp_path / "policy.json"
    if failure == "policy":
        bindings["policy"]["torchvision"] = "0.16.0"
    atomic_write_json(policy, bindings["policy"])
    output = tmp_path / "output.json"
    args = ["candidate-readiness", "--external-worker", "--execution-commit", COMMIT]
    for option, path in {
        "repository-root": ROOT,
        "qualification-authorization": tmp_path / "grant",
        "full-ledger": tmp_path / "ledger",
        "input-gate-receipt": tmp_path / "input",
        "runtime-policy-binding": policy,
        "qualification-receipt": tmp_path / "qual",
        "upstream-root": tmp_path / "dsvt",
        "checkpoint": tmp_path / "checkpoint",
        "output": output,
    }.items():
        args += ["--" + option, str(path)]
    before = set(sys.modules)
    if failure == "success":
        assert main(args) == 0
        assert json.loads(output.read_text()) == expected
    else:
        with pytest.raises(runtime.M8S2ProtocolViolation):
            main(args)
        assert not output.exists()
        if failure != "init":
            assert "heavy" not in events
    assert not any(
        name == "torch" or name.startswith(("torch.", "pcdet.", "laserperception.evaluation"))
        for name in set(sys.modules) - before
    )
