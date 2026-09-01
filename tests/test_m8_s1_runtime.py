from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import laserperception.detection.m8_s1_runtime as runtime
from laserperception.detection.m8_s1_runtime import (
    AtomicAttempt,
    AttemptIdentity,
    AuthorizationIdentity,
    M8S1ProtocolViolation,
    authorize_then_construct,
    canonical_condition_ids,
    canonical_frame_ids,
    require_scientific_authorization,
    stage_r_condition_ids,
    verify_cross_mode_process_separation,
    verify_runtime_policy_binding,
    verify_scientific_authorization,
    verify_static_bindings,
    verify_three_process_realizations,
    zero_intensity_copy,
)
from laserperception.detection.m8_s1_runtime_policy import capture_runtime_policy

ROOT = Path(__file__).resolve().parents[1]


def _runtime_policy(commit: str = "a" * 40) -> dict[str, object]:
    return {
        "schema_version": runtime.RUNTIME_POLICY_SCHEMA,
        "repository_execution_commit": commit,
        "python_exact_version": "3.10.12 exact",
        "pytorch_exact_version": "2.1.0+cu118",
        "cuda_runtime": "11.8",
        "nvidia_driver": "610.88",
        "gpu_name": "NVIDIA GeForce RTX 4060 Laptop GPU",
        "gpu_uuid": "GPU-test",
        "spconv": "2.3.8",
        "torch_scatter": "2.1.2+pt21cu118",
        "numpy": "1.23.5",
        "tf32": {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": True},
        "cudnn_benchmark": False,
        "cudnn_deterministic": False,
        "torch_deterministic_algorithms": False,
        "PYTORCH_CUDA_ALLOC_CONF": None,
        "CUDA_MODULE_LOADING": "LAZY",
        "point_order_policy": runtime.POINT_ORDER_POLICY,
        "candidate_identity": {
            "architecture": "DSVT-Pillar with TransFusion head",
            "candidate_manifest_sha256": runtime.CANDIDATE_MANIFEST_SHA256,
            "upstream_commit": runtime.UPSTREAM_COMMIT,
            "config_sha256": runtime.CONFIG_SHA256,
            "checkpoint_sha256": runtime.CHECKPOINT_SHA256,
        },
        "random_policy": {
            "python": runtime.RANDOM_POLICY,
            "numpy": runtime.RANDOM_POLICY,
            "torch": runtime.RANDOM_POLICY,
            "process_rng_state_or_seed_bound": False,
        },
        "operational_constraints": dict(runtime.OPERATIONAL_CONSTRAINTS),
    }


def _runtime_policy_file(
    tmp_path: Path, policy: dict[str, object] | None = None
) -> tuple[Path, dict[str, object], str]:
    record = _runtime_policy() if policy is None else policy
    path = tmp_path / "runtime-policy.json"
    runtime.atomic_write_json(path, record)
    return path, record, runtime.sha256_file(path)


def _authorization(
    expected: AuthorizationIdentity,
    *,
    modes: list[str] | None = None,
    logical_pass_ids: list[str] | None = None,
    runtime_policy_sha256: str = "0" * 64,
) -> dict[str, object]:
    return {
        "schema_version": runtime.AUTHORIZATION_SCHEMA,
        "scientific_inference_authorized": True,
        "authorization_id": "owner-stage-r-authorization",
        "authorization_role": "owner_scientific_inference_authorization",
        "owner_approval": True,
        "authorization_timestamp_utc": "2099-01-01T00:00:00Z",
        "authorization_provenance": "future owner decision",
        "runtime_policy_binding_sha256": runtime_policy_sha256,
        "authorized_modes": ["stage-r"] if modes is None else modes,
        "authorized_logical_pass_ids": (
            list(runtime.LOGICAL_PASS_IDS_BY_MODE["stage-r"])
            if logical_pass_ids is None
            else logical_pass_ids
        ),
        **expected.to_dict(),
    }


@pytest.mark.parametrize(
    ("mode", "logical_pass_id"),
    [
        ("stage-r", "stage-r-1"),
        ("primary-pass", "primary-pass-1"),
        ("zero-intensity-pass", "zero-intensity-pass-1"),
    ],
)
def test_scientific_modes_refuse_before_backend_initialization(
    mode: str, logical_pass_id: str, tmp_path: Path
) -> None:
    initialized = False

    def factory() -> object:
        nonlocal initialized
        initialized = True
        return object()

    policy_path, policy, _ = _runtime_policy_file(tmp_path)
    expected = AuthorizationIdentity("a" * 40)
    with pytest.raises(M8S1ProtocolViolation, match="normally disabled"):
        authorize_then_construct(
            mode,
            logical_pass_id,
            None,
            expected,
            policy_path,
            policy,
            factory,
        )
    assert initialized is False


def test_future_authorization_binds_runtime_implementation(tmp_path: Path) -> None:
    expected = AuthorizationIdentity("a" * 40)
    authorization = _authorization(expected)
    verify_scientific_authorization(authorization, expected, "stage-r", "stage-r-1")
    authorization["measurement_runtime_execution_commit"] = "b" * 40
    with pytest.raises(M8S1ProtocolViolation, match="measurement_runtime_execution_commit"):
        verify_scientific_authorization(authorization, expected, "stage-r", "stage-r-1")
    assert not (tmp_path / "m8_s1_inference_authorization.json").exists()


def test_authorization_file_is_exact_schema_and_has_no_bypass(tmp_path: Path) -> None:
    expected = AuthorizationIdentity("a" * 40)
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps({**_authorization(expected), "skip_auth": True}), encoding="utf-8")
    with pytest.raises(M8S1ProtocolViolation, match="schema fields differ"):
        require_scientific_authorization("stage-r", "stage-r-1", path, expected)


@pytest.mark.parametrize("logical_pass_id", ["stage-r-1", "stage-r-10"])
def test_stage_r_only_authorization_accepts_first_and_last_repeat(
    logical_pass_id: str,
) -> None:
    expected = AuthorizationIdentity("a" * 40)
    verify_scientific_authorization(
        _authorization(expected),
        expected,
        "stage-r",
        logical_pass_id,
    )


@pytest.mark.parametrize(
    ("mode", "logical_pass_id"),
    [
        ("primary-pass", "primary-pass-1"),
        ("zero-intensity-pass", "zero-intensity-pass-1"),
        ("stage-r", "stage-r-11"),
        ("stage-r", "arbitrary-name"),
    ],
)
def test_stage_r_only_authorization_rejects_other_science_or_unknown_passes(
    mode: str, logical_pass_id: str
) -> None:
    expected = AuthorizationIdentity("a" * 40)
    with pytest.raises(M8S1ProtocolViolation, match="not authorized"):
        verify_scientific_authorization(
            _authorization(expected),
            expected,
            mode,
            logical_pass_id,
        )


def test_corpus_authorization_does_not_authorize_stage_r() -> None:
    expected = AuthorizationIdentity("a" * 40)
    corpus = _authorization(
        expected,
        modes=["primary-pass", "zero-intensity-pass"],
        logical_pass_ids=[
            *runtime.LOGICAL_PASS_IDS_BY_MODE["primary-pass"],
            *runtime.LOGICAL_PASS_IDS_BY_MODE["zero-intensity-pass"],
        ],
    )
    verify_scientific_authorization(corpus, expected, "primary-pass", "primary-pass-1")
    with pytest.raises(M8S1ProtocolViolation, match="mode is not authorized"):
        verify_scientific_authorization(corpus, expected, "stage-r", "stage-r-1")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"schema_version": "laserperception.m8.s1.authorization.v1"}, "schema is invalid"),
        ({"authorized_modes": []}, "authorized_modes is invalid"),
        ({"authorized_modes": ["all"]}, "authorized_modes is invalid"),
        ({"authorized_modes": ["unknown"]}, "authorized_modes is invalid"),
    ],
)
def test_authorization_v1_empty_wildcard_and_unknown_modes_fail(
    mutation: dict[str, object], message: str
) -> None:
    expected = AuthorizationIdentity("a" * 40)
    authorization = {**_authorization(expected), **mutation}
    with pytest.raises(M8S1ProtocolViolation, match=message):
        verify_scientific_authorization(authorization, expected, "stage-r", "stage-r-1")


def test_missing_authorized_modes_fails_closed() -> None:
    expected = AuthorizationIdentity("a" * 40)
    authorization = _authorization(expected)
    del authorization["authorized_modes"]
    with pytest.raises(M8S1ProtocolViolation, match="schema fields differ"):
        verify_scientific_authorization(authorization, expected, "stage-r", "stage-r-1")


def test_invalid_runtime_policy_sha_in_authorization_fails() -> None:
    expected = AuthorizationIdentity("a" * 40)
    authorization = _authorization(expected, runtime_policy_sha256="caller-invented-value")
    with pytest.raises(M8S1ProtocolViolation, match="binding identity is invalid"):
        verify_scientific_authorization(authorization, expected, "stage-r", "stage-r-1")


def test_owner_memory_resolution_is_additive_and_manifest_bound() -> None:
    owner_path = ROOT / "benchmarks/m8/diagnostics/m8_s1_memory_margin_owner_review.json"
    capacity_path = ROOT / "benchmarks/m8/diagnostics/m8_s1_max_pillar_capacity.json"
    manifest_path = ROOT / "benchmarks/m8/preregistration/m8_s1_measurement_runtime.json"
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    capacity = json.loads(capacity_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert owner["decision"] == "ACCEPTED_FOR_S1_RUNTIME"
    assert capacity["capacity_review"]["classification"] == ("OWNER MEMORY-MARGIN REVIEW REQUIRED")
    assert owner["capacity_evidence"]["artifact_sha256"] == runtime.sha256_file(capacity_path)
    assert manifest["capacity_review_owner_resolution"] == {
        "artifact": "benchmarks/m8/diagnostics/m8_s1_memory_margin_owner_review.json",
        "artifact_bytes": owner_path.stat().st_size,
        "artifact_sha256": runtime.sha256_file(owner_path),
        "capacity_evidence_commit": "2655c26ce3f4298438a80f265fb3884c7046e40c",
        "decision": "ACCEPTED_FOR_S1_RUNTIME",
        "historical_capacity_classification_preserved": True,
    }
    assert owner["decision_basis"]["post_quantile_reserved_equals_post_maximum_reserved"] is True
    assert owner["operational_constraints"] == runtime.OPERATIONAL_CONSTRAINTS
    assert manifest["operational_constraints"] == runtime.OPERATIONAL_CONSTRAINTS


def test_runtime_policy_binding_uses_atomic_file_sha_identity(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    policy = _runtime_policy()
    runtime.atomic_write_json(first, policy)
    runtime.atomic_write_json(second, policy)
    assert runtime.sha256_file(first) == runtime.sha256_file(second)
    assert verify_runtime_policy_binding(first, runtime.sha256_file(first), policy) == policy


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tf32", {"cuda_matmul_allow_tf32": True, "cudnn_allow_tf32": True}),
        ("cudnn_deterministic", True),
        ("torch_deterministic_algorithms", True),
        (
            "candidate_identity",
            {
                "architecture": "DSVT-Pillar with TransFusion head",
                "candidate_manifest_sha256": runtime.CANDIDATE_MANIFEST_SHA256,
                "upstream_commit": runtime.UPSTREAM_COMMIT,
                "config_sha256": runtime.CONFIG_SHA256,
                "checkpoint_sha256": "changed",
            },
        ),
    ],
)
def test_live_runtime_policy_mismatch_rejects_before_science(
    field: str, value: object, tmp_path: Path
) -> None:
    path, policy, identity = _runtime_policy_file(tmp_path)
    live = deepcopy(policy)
    live[field] = value
    with pytest.raises(M8S1ProtocolViolation, match=f"(mismatch: {field}|policy changed)"):
        verify_runtime_policy_binding(path, identity, live)


def test_live_allocator_configuration_mismatch_rejects(tmp_path: Path) -> None:
    path, policy, identity = _runtime_policy_file(tmp_path)
    live = deepcopy(policy)
    live["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    with pytest.raises(M8S1ProtocolViolation, match="PYTORCH_CUDA_ALLOC_CONF"):
        verify_runtime_policy_binding(path, identity, live)


def test_runtime_policy_mismatch_precedes_backend_construction(tmp_path: Path) -> None:
    path, policy, identity = _runtime_policy_file(tmp_path)
    expected = AuthorizationIdentity("a" * 40)
    authorization_path = tmp_path / "authorization.json"
    runtime.atomic_write_json(
        authorization_path,
        _authorization(expected, runtime_policy_sha256=identity),
    )
    live = deepcopy(policy)
    live["tf32"] = {"cuda_matmul_allow_tf32": True, "cudnn_allow_tf32": True}
    initialized = False

    def factory() -> object:
        nonlocal initialized
        initialized = True
        return object()

    with pytest.raises(M8S1ProtocolViolation, match="mismatch: tf32"):
        authorize_then_construct(
            "stage-r",
            "stage-r-1",
            authorization_path,
            expected,
            path,
            live,
            factory,
        )
    assert initialized is False


@pytest.mark.parametrize(
    ("mode", "logical_pass_id"),
    [
        ("primary-pass", "primary-pass-1"),
        ("zero-intensity-pass", "zero-intensity-pass-1"),
    ],
)
def test_stage_r_scope_failure_precedes_backend_construction(
    mode: str, logical_pass_id: str, tmp_path: Path
) -> None:
    policy_path, policy, identity = _runtime_policy_file(tmp_path)
    expected = AuthorizationIdentity("a" * 40)
    authorization_path = tmp_path / "authorization.json"
    runtime.atomic_write_json(
        authorization_path,
        _authorization(expected, runtime_policy_sha256=identity),
    )
    initialized = False

    def factory() -> object:
        nonlocal initialized
        initialized = True
        return object()

    with pytest.raises(M8S1ProtocolViolation, match="mode is not authorized"):
        authorize_then_construct(
            mode,
            logical_pass_id,
            authorization_path,
            expected,
            policy_path,
            policy,
            factory,
        )
    assert initialized is False


def test_runtime_binding_capture_is_gt_blind_and_dsvt_blind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_modules: list[str] = []
    fake_torch = SimpleNamespace(
        __version__="2.1.0+cu118",
        version=SimpleNamespace(cuda="11.8"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_name=lambda _: "NVIDIA GeForce RTX 4060 Laptop GPU",
        ),
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
            cudnn=SimpleNamespace(allow_tf32=True, benchmark=False, deterministic=False),
        ),
        are_deterministic_algorithms_enabled=lambda: False,
    )
    modules = {
        "torch": fake_torch,
        "spconv": SimpleNamespace(__version__="2.3.8"),
        "torch_scatter": SimpleNamespace(__version__="2.1.2+pt21cu118"),
        "numpy": SimpleNamespace(__version__="1.23.5"),
    }

    def module_loader(name: str) -> object:
        requested_modules.append(name)
        return modules[name]

    def command_runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="610.88, GPU-test\n",
            stderr="",
        )

    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setenv("CUDA_MODULE_LOADING", "LAZY")
    manifest = json.loads((ROOT / runtime.CANDIDATE_MANIFEST_PATH).read_text(encoding="utf-8"))
    result = capture_runtime_policy(
        "a" * 40,
        manifest,
        module_loader=module_loader,
        command_runner=command_runner,
    )
    assert requested_modules == ["torch", "spconv", "torch_scatter", "numpy"]
    assert set(result) == runtime.RUNTIME_POLICY_FIELDS
    assert result["schema_version"] == runtime.RUNTIME_POLICY_SCHEMA
    assert result["random_policy"] == {
        "python": runtime.RANDOM_POLICY,
        "numpy": runtime.RANDOM_POLICY,
        "torch": runtime.RANDOM_POLICY,
        "process_rng_state_or_seed_bound": False,
    }
    assert not any(name.startswith(("pcdet", "laserperception.evaluation")) for name in modules)


def test_cli_removes_free_form_binding_and_orchestrator_stays_sequential() -> None:
    runner = (ROOT / "scripts/detection/run_m8_s1.py").read_text(encoding="utf-8")
    orchestrator = (ROOT / "scripts/detection/orchestrate_m8_s1.py").read_text(encoding="utf-8")
    assert "runtime-binding-identity" not in runner
    assert "runtime-binding-identity" not in orchestrator
    assert '"runtime-binding"' in runner
    assert '"--runtime-policy-binding"' in runner
    assert '"--runtime-policy-binding"' in orchestrator
    assert "subprocess.run" in orchestrator
    assert "Popen" not in orchestrator
    authorization_position = runner.index("require_scientific_authorization(")
    live_capture_position = runner.index("live_policy = capture_runtime_policy(")
    live_verify_position = runner.index("verify_runtime_policy_binding(")
    external_position = runner.index("upstream, checkpoint = _external_runtime_paths(root)")
    science_import_position = runner.index(
        "from laserperception.evaluation.m8_s1_science import run_scientific_attempt"
    )
    assert authorization_position < live_capture_position < live_verify_position
    assert live_verify_position < external_position < science_import_position


def _static_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for relative in (
        runtime.PROTOCOL_MARKDOWN_PATH,
        runtime.PROTOCOL_JSON_PATH,
        runtime.CANDIDATE_MANIFEST_PATH,
        runtime.INPUT_LEDGER_PATH,
        runtime.INPUT_REVALIDATION_PATH,
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    monkeypatch.setattr(runtime, "_require_git_objects", lambda _: None)
    monkeypatch.setattr(runtime, "_repository_head", lambda _: "c" * 40)
    return tmp_path


def test_static_bindings_accept_frozen_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _static_fixture(tmp_path, monkeypatch)
    result = verify_static_bindings(root)
    assert result.repository_head == "c" * 40


@pytest.mark.parametrize(
    "relative",
    [
        runtime.PROTOCOL_MARKDOWN_PATH,
        runtime.CANDIDATE_MANIFEST_PATH,
        runtime.INPUT_LEDGER_PATH,
    ],
)
def test_static_binding_hash_mismatch_refuses(
    relative: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _static_fixture(tmp_path, monkeypatch)
    with (root / relative).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(M8S1ProtocolViolation, match="(byte count|SHA256) changed"):
        verify_static_bindings(root)


def test_checkpoint_identity_mismatch_refuses(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"not-the-frozen-checkpoint")
    with pytest.raises(M8S1ProtocolViolation, match="byte count changed"):
        runtime._require_identity(checkpoint, 28_665_215, runtime.CHECKPOINT_SHA256)


def test_frozen_order_is_428_frames_h10_then_h5() -> None:
    frames = canonical_frame_ids()
    conditions = canonical_condition_ids()
    assert len(frames) == 428
    assert len(conditions) == 856
    assert conditions[:4] == (
        f"{frames[0]}/H10",
        f"{frames[0]}/H5",
        f"{frames[1]}/H10",
        f"{frames[1]}/H5",
    )
    assert hashlib.sha256(("\n".join(frames) + "\n").encode()).hexdigest() == (
        runtime.ORDERED_FRAME_SHA256
    )


def test_stage_r_order_contains_seven_sentinels_and_14_calls() -> None:
    conditions = stage_r_condition_ids()
    assert len(runtime.STAGE_R_FRAMES) == 7
    assert len(conditions) == 14
    assert all(
        conditions[index * 2 : index * 2 + 2] == (f"{frame}/H10", f"{frame}/H5")
        for index, frame in enumerate(runtime.STAGE_R_FRAMES)
    )


def test_zero_intensity_is_positive_zero_and_other_bytes_exact() -> None:
    points = np.asarray(
        [[1.0, 2.0, 3.0, -0.0, 0.0], [4.0, 5.0, 6.0, 0.75, 0.2]],
        dtype=np.float32,
    )
    result = zero_intensity_copy(points)
    assert result.shape == points.shape
    assert np.array_equal(
        result[:, [0, 1, 2, 4]].view(np.uint32), points[:, [0, 1, 2, 4]].view(np.uint32)
    )
    assert np.array_equal(result[:, 3].view(np.uint32), np.zeros(2, dtype=np.uint32))
    assert np.array_equal(
        points[:, 3].view(np.uint32), np.asarray([0x80000000, 0x3F400000], dtype=np.uint32)
    )


def _identity(mode: str, process_uuid: str = "process-1") -> AttemptIdentity:
    return AttemptIdentity(mode, "logical-1", "attempt-1", process_uuid, 123, "a" * 40)


def test_complete_856_condition_attempt_finalizes(tmp_path: Path) -> None:
    attempt = AtomicAttempt(tmp_path / "complete", _identity("primary-pass"))
    for condition in canonical_condition_ids():
        attempt.record(condition, {})
    final = attempt.finalize()
    assert final["status"] == "COMPLETE"
    assert final["accepted_canonical_calls"] == 856
    assert (tmp_path / "complete/final_pass_manifest.json").is_file()


def test_855_condition_attempt_cannot_finalize(tmp_path: Path) -> None:
    attempt = AtomicAttempt(tmp_path / "short", _identity("primary-pass"))
    for condition in canonical_condition_ids()[:-1]:
        attempt.record(condition, {})
    with pytest.raises(M8S1ProtocolViolation, match="855/856"):
        attempt.finalize()


def test_failed_attempt_is_incomplete_and_replacement_restarts_at_one(tmp_path: Path) -> None:
    first = AtomicAttempt(tmp_path / "first", _identity("primary-pass"))
    first.record(canonical_condition_ids()[0], {})
    failed = first.fail("synthetic failure")
    assert failed["status"] == "INCOMPLETE"
    assert failed["accepted_canonical_calls"] == 0
    with pytest.raises(M8S1ProtocolViolation, match="already exists"):
        AtomicAttempt(tmp_path / "first", _identity("primary-pass", "process-2"))
    replacement = AtomicAttempt(tmp_path / "replacement", _identity("primary-pass", "process-2"))
    assert (
        json.loads((tmp_path / "replacement/attempt_manifest.json").read_text())[
            "next_condition_id"
        ]
        == canonical_condition_ids()[0]
    )
    with pytest.raises(M8S1ProtocolViolation, match="condition order changed"):
        replacement.record(canonical_condition_ids()[1], {})


def test_stage_r_finalizes_only_at_14_and_failed_repeat_cannot_splice(tmp_path: Path) -> None:
    attempt = AtomicAttempt(tmp_path / "stage", _identity("stage-r"))
    for condition in stage_r_condition_ids()[:-1]:
        attempt.record(condition, {})
    with pytest.raises(M8S1ProtocolViolation, match="13/14"):
        attempt.finalize()
    attempt.fail("synthetic")
    with pytest.raises(M8S1ProtocolViolation):
        AtomicAttempt(tmp_path / "stage", _identity("stage-r", "process-2"))


def test_three_passes_require_distinct_processes_and_cross_mode_separation() -> None:
    primary = [{"mode": "primary-pass", "process_uuid": f"p{index}"} for index in range(3)]
    zero = [{"mode": "zero-intensity-pass", "process_uuid": f"z{index}"} for index in range(3)]
    verify_three_process_realizations(primary)
    verify_three_process_realizations(zero)
    verify_cross_mode_process_separation(primary, zero)
    zero[0]["process_uuid"] = "p0"
    with pytest.raises(M8S1ProtocolViolation, match="cannot share"):
        verify_cross_mode_process_separation(primary, zero)


def test_atomic_json_never_leaves_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    runtime.atomic_write_json(path, {"status": "COMPLETE"})
    assert json.loads(path.read_text()) == {"status": "COMPLETE"}
    assert list(tmp_path.glob("*.tmp")) == []
