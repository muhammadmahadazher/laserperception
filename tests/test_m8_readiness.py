"""Synthetic externally reported records; no GPU discovery or scientific execution."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from laserperception.cli import main
from laserperception.worker.bootstrap import FrameworkVersion, GPURecord, QualificationRecord
from laserperception.worker.guards import require_external_worker
from laserperception.worker.m8_readiness import (
    CANDIDATE_SHA256,
    CHECKPOINT_SHA256,
    CONFIG_SHA256,
    GATES,
    INPUT_SHA256,
    RETIRED_AUTHORIZATIONS,
    RETIRED_POLICY_SHA256,
    UPSTREAM_COMMIT,
    M8CapacityRecord,
    M8GateEvidence,
    M8QualificationProgress,
    M8QualificationRecord,
    M8ReadinessRequest,
    plan_m8_qualification,
    required_m8_artifacts,
)
from laserperception.worker.manifests import AuthorizationReference, TaskType

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40
RUNTIME = "synthetic-external-001"
POLICY = "b" * 64


def request():
    return M8ReadinessRequest(COMMIT, "synthetic-qualification")


def report(capacity="passed"):
    environment = QualificationRecord(
        "1.0",
        request().task_id,
        RUNTIME,
        COMMIT,
        (GPURecord("synthetic GPU", 24576, 22000, "synthetic driver"),),
        "11.8",
        "2.1.0+cu118",
        (FrameworkVersion("spconv", "2.3.8"),),
        required_m8_artifacts(),
        "recorded_for_owner_review_not_scientific_authorization",
    )
    memory = M8CapacityRecord(
        capacity,
        "c" * 64 if capacity != "pending" else None,
        1_000_000 if capacity != "pending" else None,
        2_000_000 if capacity != "pending" else None,
        "synthetic external report",
    )
    return M8QualificationRecord(
        request(),
        environment,
        "3.10.12",
        "synthetic Linux",
        UPSTREAM_COMMIT,
        CANDIDATE_SHA256,
        CONFIG_SHA256,
        CHECKPOINT_SHA256,
        INPUT_SHA256,
        memory,
        None,
    )


def evidence(state):
    index = GATES.index(state)
    role = (
        TaskType.STAGE_R
        if state == "stage_r_authorized"
        else TaskType.PRIMARY
        if state == "primary_authorized"
        else None
    )
    auth = (
        AuthorizationReference(
            role, "authorizations/" + role.value + ".json", "d" * 64, RUNTIME, COMMIT
        )
        if role
        else None
    )
    return M8GateEvidence(
        RUNTIME,
        COMMIT,
        "e" * 64,
        "_CLOUD_WORK/synthetic/evidence.json",
        state in ("owner_reviewed", "stage_r_reviewed"),
        auth,
        POLICY if index >= 4 else None,
        140 if state == "stage_r_reviewed" else None,
        report() if state in ("environment_recorded", "capacity_reviewed") else None,
    )


def progress_to(state):
    progress = M8QualificationProgress(request(), RUNTIME)
    for name in GATES[1 : GATES.index(state) + 1]:
        progress = progress.advance(name, evidence(name))
    return progress


def test_plan_determinism_and_non_executability():
    plan = plan_m8_qualification(request(), expected_repository_commit=COMMIT)
    assert (
        plan.to_json()
        == plan_m8_qualification(request(), expected_repository_commit=COMMIT).to_json()
    )
    assert not plan.executes and not plan.hardware_probed and not plan.provider_selected
    assert plan.scientific_calls == 0 and plan.authorization_state.startswith("missing")
    assert plan.minimum_vram_gib == 16 and plan.preferred_vram_gib == 24
    assert plan.capsule_target == "lpdrive:_CLOUD_WORK/synthetic-qualification"
    assert type(plan).from_json(plan.to_json()) == plan


@pytest.mark.parametrize(
    "field",
    [
        "candidate_sha256",
        "config_sha256",
        "checkpoint_sha256",
        "upstream_commit",
        "ordered_input_sha256",
    ],
)
def test_altered_frozen_identity_rejected(field):
    with pytest.raises(ValueError, match="identity mismatch"):
        replace(request(), **{field: "0" * (40 if field == "upstream_commit" else 64)})


def test_altered_repository_and_bad_commit_rejected():
    with pytest.raises(ValueError, match="authoritative"):
        plan_m8_qualification(request(), expected_repository_commit="f" * 40)
    with pytest.raises(ValueError):
        replace(request(), execution_commit="short")


@pytest.mark.parametrize(
    "task", [TaskType.ENGINEERING, TaskType.STAGE_R, TaskType.PRIMARY, TaskType.TRAINING]
)
def test_wrong_task_type_rejected(task):
    with pytest.raises(ValueError, match="qualification only"):
        replace(request(), task_type=task)


def test_exact_artifacts_match_authoritative_candidate_and_git_blobs():
    candidate = json.loads(
        (ROOT / "configs/m8/dsvt_nuscenes_pillar.json").read_text(encoding="utf-8")
    )
    assert candidate["checkpoint"]["sha256"] == CHECKPOINT_SHA256
    assert candidate["upstream"]["config_sha256"] == CONFIG_SHA256
    assert candidate["upstream"]["commit"] == UPSTREAM_COMMIT
    for artifact in required_m8_artifacts():
        if artifact.relative_path.startswith("inputs/"):
            continue
        raw = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"HEAD:{artifact.relative_path}"]
        )
        assert len(raw) == artifact.byte_size
        assert hashlib.sha256(raw).hexdigest() == artifact.sha256


def test_report_roundtrip_and_identity_validation():
    record = report()
    assert type(record).from_json(record.to_json()) == record
    assert "not_permission" in record.status
    with pytest.raises(ValueError):
        replace(record, checkpoint_sha256="0" * 64)
    with pytest.raises(ValueError):
        replace(record, environment=replace(record.environment, execution_commit="f" * 40))
    with pytest.raises(ValueError):
        replace(record, environment=replace(record.environment, verified_inputs=()))
    with pytest.raises(ValueError):
        replace(record, machine_policy_sha256=RETIRED_POLICY_SHA256)


def test_capacity_pass_requires_complete_plausible_measurements():
    with pytest.raises(ValueError):
        M8CapacityRecord("passed", "c" * 64, None, None, "synthetic")
    with pytest.raises(ValueError):
        replace(report(), capacity=replace(report().capacity, peak_reserved_bytes=10**15))


def test_complete_progress_index_remains_non_executing():
    progress = progress_to("primary_authorized")
    assert len(progress.gates) == 8 and not progress.executes
    assert type(progress).from_json(progress.to_json()) == progress
    assert progress_to("planned").gates == ()


def test_skipped_mixed_runtime_commit_policy_gates_rejected():
    progress = progress_to("planned")
    with pytest.raises(ValueError, match="skipped"):
        progress.advance("capacity_reviewed", evidence("capacity_reviewed"))
    for changes in ({"runtime_id": "different"}, {"execution_commit": "f" * 40}):
        with pytest.raises(ValueError):
            progress.advance(
                "artifacts_verified", replace(evidence("artifacts_verified"), **changes)
            )
    with pytest.raises(ValueError):
        progress_to("policy_bound").advance(
            "owner_reviewed", replace(evidence("owner_reviewed"), machine_policy_sha256="f" * 64)
        )


@pytest.mark.parametrize("state", ["environment_recorded", "capacity_reviewed"])
def test_rich_qualification_record_required(state):
    prior = "artifacts_verified" if state == "environment_recorded" else "environment_recorded"
    with pytest.raises(ValueError):
        progress_to(prior).advance(state, replace(evidence(state), qualification=None))


@pytest.mark.parametrize(
    "field",
    ["gpu", "vram", "driver", "cuda", "pytorch", "frameworks", "python", "operating_system"],
)
def test_capacity_rejects_changed_stable_environment(field):
    original = report()
    environment = original.environment
    gpu = environment.gpus[0]
    if field in ("gpu", "vram", "driver"):
        changes = {
            "gpu": {"model": "different GPU"},
            "vram": {"vram_mib": 32768},
            "driver": {"driver": "different driver"},
        }
        changed = replace(
            original, environment=replace(environment, gpus=(replace(gpu, **changes[field]),))
        )
    elif field in ("cuda", "pytorch", "frameworks"):
        value = (FrameworkVersion("spconv", "different"),) if field == "frameworks" else "different"
        changed = replace(original, environment=replace(environment, **{field: value}))
    else:
        changed = replace(original, **{field: "different"})
    with pytest.raises(ValueError, match="identity changed"):
        progress_to("environment_recorded").advance(
            "capacity_reviewed", replace(evidence("capacity_reviewed"), qualification=changed)
        )


def test_capacity_allows_evolving_free_memory_measurements_and_policy():
    original = report()
    changed = replace(
        original,
        environment=replace(
            original.environment, gpus=(replace(original.environment.gpus[0], free_mib=21000),)
        ),
        capacity=replace(original.capacity, peak_allocated_bytes=1_100_000),
        machine_policy_sha256=POLICY,
    )
    progress = progress_to("environment_recorded").advance(
        "capacity_reviewed", replace(evidence("capacity_reviewed"), qualification=changed)
    )
    assert progress.state == "capacity_reviewed" and not progress.executes


@pytest.mark.parametrize("status", ["pending", "failed"])
def test_capacity_not_passed_rejected(status):
    with pytest.raises(ValueError):
        progress_to("environment_recorded").advance(
            "capacity_reviewed",
            replace(evidence("capacity_reviewed"), qualification=report(status)),
        )


@pytest.mark.parametrize("state", ["stage_r_authorized", "primary_authorized"])
def test_missing_fresh_authorization_rejected(state):
    prior = "owner_reviewed" if state == "stage_r_authorized" else "stage_r_reviewed"
    with pytest.raises(ValueError, match="authorization required"):
        progress_to(prior).advance(state, replace(evidence(state), authorization=None))


@pytest.mark.parametrize("sha", RETIRED_AUTHORIZATIONS)
def test_historical_authorization_rejected_as_portable(sha):
    auth = replace(evidence("stage_r_authorized").authorization, sha256=sha)
    with pytest.raises(ValueError, match="historical"):
        replace(evidence("stage_r_authorized"), authorization=auth)


def test_fresh_stage_r_review_requires_owner_and_exact_calls():
    for changes in (
        {"owner_reviewed": False},
        {"accepted_stage_r_calls": 0},
        {"accepted_stage_r_calls": 139},
    ):
        with pytest.raises(ValueError):
            progress_to("stage_r_authorized").advance(
                "stage_r_reviewed", replace(evidence("stage_r_reviewed"), **changes)
            )


@pytest.mark.parametrize("value", [False, None, 0, 1, "yes"])
def test_external_guard_fails_closed(value):
    with pytest.raises(ValueError):
        require_external_worker(value)


GPU_SCRIPTS = (
    "run_m8_s1.py",
    "orchestrate_m8_s1.py",
    "run_m8_s1_preflight.py",
    "run_m8_s1_capacity.py",
    "run_m8_source_smoke.py",
    "run_m8_h10_capacity_smoke.py",
    "analyze_m8_h10_capacity.py",
    "audit_m8_dsvt_deployment.py",
    "audit_m8_dsvt_h10_deployment.py",
)


@pytest.mark.parametrize("name", GPU_SCRIPTS)
def test_every_m8_gpu_cli_rejects_before_process_or_execution(name, monkeypatch):
    path = ROOT / "scripts/detection" / name
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("synthetic_guard_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = SimpleNamespace(external_worker=False, mode="preflight")
    if hasattr(module, "_parse_args"):
        monkeypatch.setattr(module, "_parse_args", lambda: args)
    if hasattr(module, "_parser"):
        monkeypatch.setattr(module, "_parser", lambda: SimpleNamespace(parse_args=lambda: args))
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda *a, **k: args)

    def forbidden(*args, **kwargs):
        raise AssertionError("guard must run before GPU child/execution")

    monkeypatch.setattr(subprocess, "run", forbidden)
    if hasattr(module, "DsvtBackend"):
        monkeypatch.setattr(module, "DsvtBackend", forbidden)
    if hasattr(module, "capture_runtime_policy"):
        monkeypatch.setattr(module, "capture_runtime_policy", forbidden)
    with pytest.raises(ValueError, match="external-worker"):
        module.main()


def test_cli_m8_plan_bound_to_current_git_without_probing(capsys):
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    assert (
        main(
            [
                "worker",
                "plan",
                "--task",
                "m8-qualification",
                "--execution-commit",
                head,
                "--repository-root",
                str(ROOT),
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert not plan["executes"] and not plan["hardware_probed"]
    with pytest.raises(SystemExit):
        main(
            [
                "worker",
                "plan",
                "--task",
                "m8-qualification",
                "--execution-commit",
                "0" * 40,
                "--repository-root",
                str(ROOT),
            ]
        )


def test_deserialized_plan_changed_checkpoint_and_capsule_rejected():
    plan = plan_m8_qualification(request(), expected_repository_commit=COMMIT)
    altered = tuple(
        replace(a, sha256="0" * 64) if a.name == "checkpoint" else a for a in plan.artifacts
    )
    with pytest.raises(ValueError, match="artifact identities"):
        replace(plan, artifacts=altered)
    with pytest.raises(ValueError, match="capsule"):
        replace(plan, capsule_target="lpdrive:_CLOUD_WORK/wrong")


def test_programmatic_sizing_and_smoke_guard_before_input_read(monkeypatch):
    for filename, function, kwargs in (
        (
            "analyze_m8_h10_capacity.py",
            "build_capacity_census",
            dict(
                full_ledger=Path("missing"),
                accepted_ledger=Path("missing"),
                date_root=Path("missing"),
                manifest_path=Path("missing"),
                coordinate_device="cuda:0",
            ),
        ),
        ("run_m8_h10_capacity_smoke.py", "run_smoke", None),
    ):
        path = ROOT / "scripts/detection" / filename
        spec = importlib.util.spec_from_file_location("direct_guard_" + path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        def forbidden(*args, **kwargs):
            raise AssertionError("guard must precede any input read")

        monkeypatch.setattr(module, "_load_mapping", forbidden)
        with pytest.raises(ValueError, match="external-worker"):
            if kwargs:
                getattr(module, function)(**kwargs)
            else:
                getattr(module, function)(SimpleNamespace())
