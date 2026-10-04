"""Sizing scope, timing, claims and discard boundaries using CPU-only doubles."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from laserperception.detection import m8_s2_sizing as sizing
from laserperception.detection.m8_s1_runtime import atomic_write_json
from laserperception.detection.m8_s2_runtime import (
    AUTHORIZATION_SCHEMA,
    COMPACT_MANIFEST_PATH,
    COMPACT_MANIFEST_SHA256,
    FULL_LEDGER_SHA256,
    INPUT_FREEZE_SHA256,
    PARTITIONS_SHA256,
    PROTOCOL_SHA256,
    SIZING_PLAN_SHA256,
    SIZING_PROCESS_IDS,
    M8S2ProtocolViolation,
    verify_authorization,
)

ROOT = Path(__file__).resolve().parents[1]
COMMIT, POLICY, INPUT, QUAL = "a" * 40, "b" * 64, "c" * 64, "d" * 64


def grant(root: Path) -> dict:
    return {
        "schema_version": AUTHORIZATION_SCHEMA,
        "authorized": True,
        "scope": "sizing-only",
        "owner_approval": True,
        "authorization_id": "synthetic-sizing-only",
        "authorization_timestamp_utc": "synthetic",
        "authorization_provenance": "synthetic fixture only; no real grant",
        "authorized_gpu_uuid": None,
        "authorized_worker_hostname": None,
        "authorized_campaign_root": str(root.resolve()),
        "execution_commit": COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "partitions_sha256": PARTITIONS_SHA256,
        "input_freeze_sha256": INPUT_FREEZE_SHA256,
        "full_ledger_sha256": FULL_LEDGER_SHA256,
        "compact_manifest_sha256": COMPACT_MANIFEST_SHA256,
        "logical_pass_ids": list(SIZING_PROCESS_IDS),
        "runtime_policy_binding_sha256": POLICY,
        "input_gate_receipt_sha256": INPUT,
        "qualification_receipt_sha256": QUAL,
        "candidate_readiness_receipt_sha256": "f" * 64,
        "repeatability_review_sha256": None,
        "repeatability_owner_attestation_sha256": None,
        "sizing_plan_sha256": SIZING_PLAN_SHA256,
    }


def expected(root: Path) -> dict:
    return dict(
        scope="sizing-only",
        execution_commit=COMMIT,
        logical_pass_id=SIZING_PROCESS_IDS[0],
        runtime_policy_sha256=POLICY,
        input_gate_receipt_sha256=INPUT,
        qualification_receipt_sha256=QUAL,
        candidate_readiness_receipt_sha256="f" * 64,
        campaign_root=root,
        sizing_plan_sha256=SIZING_PLAN_SHA256,
    )


@pytest.mark.parametrize("scope", ["qualification-only", "repeatability-only", "full-pass-only"])
def test_scopes_never_imply_sizing(tmp_path: Path, scope: str) -> None:
    g = grant(tmp_path)
    verify_authorization(g, **expected(tmp_path))
    g["scope"] = scope
    with pytest.raises(M8S2ProtocolViolation):
        verify_authorization(g, **expected(tmp_path))
    with pytest.raises(M8S2ProtocolViolation):
        verify_authorization(grant(tmp_path), **{**expected(tmp_path), "scope": scope})


@pytest.mark.parametrize(
    "key",
    [
        "runtime_policy_binding_sha256",
        "input_gate_receipt_sha256",
        "qualification_receipt_sha256",
        "sizing_plan_sha256",
        "candidate_readiness_receipt_sha256",
    ],
)
@pytest.mark.parametrize("value", [None, "0" * 64])
def test_sizing_requires_all_receipt_and_plan_bindings(
    tmp_path: Path, key: str, value: object
) -> None:
    g = grant(tmp_path)
    g[key] = value
    with pytest.raises(M8S2ProtocolViolation):
        verify_authorization(g, **expected(tmp_path))


def consumed(condition: str) -> dict:
    compact = json.loads((ROOT / COMPACT_MANIFEST_PATH).read_text())
    row = next(r for r in compact["conditions"] if r["condition_id"] == condition)
    return {
        "condition_id": condition,
        "arm": row["arm"],
        "input_sha256": row["full_XYZIT_sha256"],
        "M7_XYZT_sha256": row["M7_expected_XYZT_sha256"],
        "selected_global_row_sha256": row["selected_global_row_sha256"],
        "point_count": row["point_count"],
        "cpu_analytic_candidate_pillar_count": row["cpu_analytic_candidate_pillar_count"],
    }


def worker(**kwargs: object) -> float:
    plan, process, record = kwargs["plan"], kwargs["process_uuid"], kwargs["record"]
    for i, condition in enumerate(sizing.planned_ids(plan)):
        record(
            {
                **consumed(condition),
                "sequence_index": i,
                "warmup": i < 2,
                "elapsed_seconds": 999.0 if i < 2 else float(i),
                "process_uuid": process,
                "memory": dict(
                    torch_allocated_before_bytes=10,
                    torch_peak_allocated_bytes=20,
                    torch_reserved_after_bytes=30,
                ),
            }
        )
    return 4.0


def attempt(
    root: Path,
    logical: str,
    monkeypatch: pytest.MonkeyPatch,
    pid: int,
    name: str | None = None,
    work=worker,
) -> Path:
    monkeypatch.setattr(sizing.os, "getpid", lambda: pid)
    auth = root.parent / "fixture-grant.json"
    atomic_write_json(auth, grant(root))
    path = root / (name or logical)
    sizing.run_sizing_attempt(
        repository_root=ROOT,
        date_root=root / "data-not-opened",
        m6_ledger=root / "not-opened",
        campaign_root=root,
        attempt_root=path,
        logical_process_id=logical,
        execution_commit=COMMIT,
        runtime_policy_sha256=POLICY,
        input_gate_receipt_sha256=INPUT,
        qualification_receipt_sha256=QUAL,
        candidate_readiness_receipt_sha256="f" * 64,
        authorization_path=auth,
        worker=work,
    )
    return path


def aggregate(paths: list[Path]) -> dict:
    return sizing.aggregate_sizing(
        paths, repository_root=ROOT, execution_commit=COMMIT, usd_per_hour=0.49
    )


def test_exact_plan_two_processes_and_estimator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = sizing.sizing_plan(ROOT)
    ids = sizing.planned_ids(plan)
    assert len(ids) == 18 and len(set(ids[2:])) == 16
    assert ids[:2] == ("2011_09_26_drive_0091/0000000322/B2", "2011_09_26_drive_0091/0000000302/F2")
    assert ids[2:] == tuple(r["condition_id"] for r in plan["measured_conditions_per_process"])
    paths = [
        attempt(tmp_path, logical, monkeypatch, 100 + i)
        for i, logical in enumerate(SIZING_PROCESS_IDS)
    ]
    original = sizing.estimate_future_cost
    observed = []

    def estimate(**kw: object) -> dict:
        observed.append(kw)
        return original(**kw)

    monkeypatch.setattr(sizing, "estimate_future_cost", estimate)
    result = aggregate(paths)
    assert result == aggregate(list(reversed(paths)))
    assert result["measured_call_count"] == 32 and result["warmup_call_count"] == 4
    assert result["engineering_calls"] == 36 and result["scientific_calls"] == 0
    assert result["initialization_seconds"] == [4.0, 4.0]
    assert 999.0 not in result["measured_call_seconds"]
    assert len(observed[0]["measured_call_seconds"]) == 32
    assert len(observed[0]["initialization_seconds"]) == 2
    assert result["repeatability_authorized"] is False
    assert result["budget_decision"]["total_accepted_science"]["median"] > 0


def test_timing_synchronization_same_array_and_no_semantic_access() -> None:
    events, records = [], []
    points = np.zeros((2, 5), dtype=np.float32)

    class Poison:
        def __getattribute__(self, name: str) -> object:
            raise AssertionError("semantic output inspected")

    class Backend:
        def run_gt_blind_timing_call(self, value: np.ndarray) -> object:
            assert value is points
            events.append("infer")
            return Poison()

    runtime = SimpleNamespace(
        synchronize=lambda: events.append("sync"),
        reset_peak_memory_stats=lambda: events.append("reset"),
        memory_allocated=lambda: 10,
        max_memory_allocated=lambda: 20,
        memory_reserved=lambda: 30,
    )

    def load(condition: str) -> tuple:
        events.append("reconstruct+verify")
        return points, consumed(condition)

    def clock() -> float:
        events.append("clock")
        return float(events.count("clock"))

    start = sizing.measure_calls(
        plan=sizing.sizing_plan(ROOT),
        process_uuid="synthetic",
        backend_factory=lambda: Backend(),
        runtime=runtime,
        load=load,
        record=records.append,
        clock=clock,
    )
    assert start == 1 and len(records) == 18
    assert sum(r["warmup"] for r in records) == 2
    for i, e in enumerate(events):
        if e == "infer":
            assert events[i - 1] == "clock" and events[i + 1 : i + 3] == ["sync", "clock"]
    assert events[:3] == ["sync", "clock", "sync"]
    assert all(set(r) == sizing.CALL_KEYS for r in records)


def test_incomplete_retry_cannot_splice_or_rerun_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(**kwargs: object) -> float:
        def stop(record: dict) -> None:
            kwargs["record"](record)
            raise RuntimeError("synthetic interruption")

        worker(**{**kwargs, "record": stop})
        return 0

    with pytest.raises(RuntimeError):
        attempt(tmp_path, SIZING_PROCESS_IDS[0], monkeypatch, 200, "failed", failing)
    manifest = json.loads((tmp_path / "failed/manifest.json").read_text())
    assert manifest["status"] == "INCOMPLETE" and manifest["accepted_engineering_calls"] == 0
    assert not (tmp_path / ".sizing.lock").exists()
    with pytest.raises(M8S2ProtocolViolation, match="fresh OS process"):
        attempt(tmp_path, SIZING_PROCESS_IDS[0], monkeypatch, 200, "same-failed-process")
    p1 = attempt(tmp_path, SIZING_PROCESS_IDS[0], monkeypatch, 201, "retry")
    with pytest.raises(M8S2ProtocolViolation, match="fresh OS process"):
        attempt(tmp_path, SIZING_PROCESS_IDS[1], monkeypatch, 201, "same-process")
    p2 = attempt(tmp_path, SIZING_PROCESS_IDS[1], monkeypatch, 202)
    assert aggregate([p1, p2])["engineering_calls"] == 36
    with pytest.raises(M8S2ProtocolViolation):
        aggregate([tmp_path / "failed", p2])
    with pytest.raises(M8S2ProtocolViolation, match="cannot be rerun"):
        attempt(tmp_path, SIZING_PROCESS_IDS[0], monkeypatch, 203, "rerun")
    # Even a re-sealed history cannot bless a completed retry using a failed PID.
    completed = json.loads((p1 / "manifest.json").read_text())
    completed["process_id"] = 200
    atomic_write_json(p1 / "manifest.json", completed)
    claims = json.loads((tmp_path / "sizing-claims.json").read_text())
    entry = next(e for e in claims["attempts"] if e["attempt"] == p1.name)
    entry.update(process_id=200, manifest_sha256=sizing.sha256_file(p1 / "manifest.json"))
    atomic_write_json(tmp_path / "sizing-claims.json", claims)
    with pytest.raises(M8S2ProtocolViolation, match="history reused"):
        aggregate([p1, p2])


@pytest.mark.parametrize("mutate", ["semantic", "reorder", "truncate", "extra-file", "same-uuid"])
def test_sealed_evidence_rejects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate: str
) -> None:
    paths = [
        attempt(tmp_path, logical, monkeypatch, 300 + i)
        for i, logical in enumerate(SIZING_PROCESS_IDS)
    ]
    if mutate == "extra-file":
        (paths[0] / "predictions.json").write_text("[]")
    elif mutate == "same-uuid":
        m = json.loads((paths[1] / "manifest.json").read_text())
        m["process_uuid"] = json.loads((paths[0] / "manifest.json").read_text())["process_uuid"]
        atomic_write_json(paths[1] / "manifest.json", m)
    else:
        doc = json.loads((paths[0] / "calls.json").read_text())
        if mutate == "semantic":
            doc["calls"][0]["boxes"] = []
        elif mutate == "reorder":
            doc["calls"].reverse()
        else:
            doc["calls"].pop()
        atomic_write_json(paths[0] / "calls.json", doc)
    with pytest.raises(M8S2ProtocolViolation):
        aggregate(paths)


def test_sizing_import_is_cpu_only() -> None:
    assert "torch" not in sys.modules
    assert "laserperception.evaluation.m8_s2_science" not in sys.modules


@pytest.mark.parametrize("fail_at", ["authorization", "worker", "plan", "success"])
def test_cli_rejects_before_accelerator_or_scoring_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_at: str
) -> None:
    module = runpy.run_path(str(ROOT / "scripts/detection/run_m8_s2.py"))
    main = module["main"]
    glob = main.__globals__
    for name in ("verify_static_bindings", "verify_runtime_policy_document", "_external_candidate"):
        monkeypatch.setitem(glob, name, lambda *a, **k: None)
    monkeypatch.setitem(glob, "verify_input_gate_receipt", lambda *a, **k: INPUT)
    monkeypatch.setitem(glob, "verify_qualification_receipt", lambda *a, **k: QUAL)
    monkeypatch.setitem(glob, "verify_readiness_receipt", lambda *a, **k: "f" * 64)
    monkeypatch.setitem(glob, "verify_preimport_runtime", lambda *a, **k: None)
    monkeypatch.setitem(glob, "sha256_file", lambda *a: POLICY)

    def authorization(*args: object, **kw: object) -> dict:
        if fail_at == "authorization":
            raise M8S2ProtocolViolation("synthetic authorization rejection")
        return {
            **grant(tmp_path),
            "authorized_worker_hostname": "fixture",
            "authorized_gpu_uuid": "GPU-fixture",
        }

    def worker_check(*args: object) -> None:
        if fail_at == "worker":
            raise M8S2ProtocolViolation("synthetic worker rejection")

    monkeypatch.setitem(glob, "require_authorization", authorization)
    monkeypatch.setitem(glob, "verify_qualification_worker", worker_check)
    monkeypatch.setitem(
        glob,
        "_capture_bound_policy",
        lambda *a: {} if fail_at == "success" else pytest.fail("accelerator import reached"),
    )
    monkeypatch.setitem(glob, "verify_runtime_policy", lambda *a: None)
    monkeypatch.setitem(glob, "verify_frozen_gt_assets", lambda *a: pytest.fail("GT path reached"))
    if fail_at == "plan":
        monkeypatch.setattr(
            sizing,
            "sizing_plan",
            lambda *a: (_ for _ in ()).throw(M8S2ProtocolViolation("synthetic plan rejection")),
        )
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"worker_hostname": "fixture", "gpu_uuid": "GPU-fixture"}))
    args = [
        "sizing",
        "--external-worker",
        "--execution-commit",
        COMMIT,
        "--logical-pass-id",
        SIZING_PROCESS_IDS[0],
        "--attempt-id",
        "fixture",
    ]
    for option, value in {
        "repository-root": ROOT,
        "campaign-root": tmp_path,
        "attempt-root": tmp_path / "attempt",
        "full-ledger": tmp_path / "ledger",
        "input-gate-receipt": tmp_path / "input",
        "runtime-policy-binding": policy,
        "qualification-receipt": tmp_path / "qual",
        "candidate-readiness-receipt": tmp_path / "readiness",
        "qualification-authorization": tmp_path / "qual-grant",
        "authorization": tmp_path / "grant",
        "upstream-root": tmp_path / "dsvt",
        "checkpoint": tmp_path / "checkpoint",
        "date-root": tmp_path / "date",
        "m6-ledger": tmp_path / "m6",
    }.items():
        args += ["--" + option, str(value)]
    if fail_at == "success":
        called = []

        def import_only_sizing(name: str) -> object:
            assert name == "laserperception.detection.m8_s2_sizing"
            return SimpleNamespace(run_sizing_attempt=lambda **kw: called.append(kw))

        monkeypatch.setitem(glob, "importlib", SimpleNamespace(import_module=import_only_sizing))
        assert main(args) == 0 and len(called) == 1
    else:
        with pytest.raises(M8S2ProtocolViolation, match="synthetic"):
            main(args)
    assert "torch" not in sys.modules


def test_live_array_identity_and_mismatch_before_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    from laserperception.detection import m8_s2_reconstruction as reconstruction
    from laserperception.detection.m8_s2_input import array_sha256, canonical_xyzt

    points = np.zeros((2, 5), dtype=np.float32)
    condition = "synthetic/0000000000/B2"
    row = {
        "condition_id": condition,
        "arm": "B2",
        "point_count": 2,
        "full_XYZIT_sha256": array_sha256(points),
        "XYZT_projection_sha256": array_sha256(canonical_xyzt(points)),
        "M7_expected_XYZT_sha256": array_sha256(canonical_xyzt(points)),
        "selected_global_row_sha256": "a" * 64,
        "cpu_analytic_candidate_pillar_count": 1,
    }
    monkeypatch.setattr(reconstruction, "sources", lambda **kw: (None, None, {condition: row}))
    monkeypatch.setattr(
        reconstruction,
        "frame_inputs",
        lambda *a: (
            None,
            (SimpleNamespace(arm="B2", points=points, selected_row_sha256="a" * 64),),
        ),
    )
    load = sizing.verified_condition_loader(ROOT, Path("unused"), Path("unused"))
    actual, evidence = load(condition)
    assert actual is points and evidence["point_count"] == 2
    points[0, 0] = 1
    with pytest.raises(M8S2ProtocolViolation, match="consumed input differs"):
        load(condition)


@pytest.mark.parametrize(
    "state",
    [
        "live",
        "different-worker",
        "dead",
        "wrong-commit",
        "owned-temporaries",
        "foreign-temporary",
        "complete-with-temporary",
    ],
)
def test_interrupted_recovery_requires_dead_original_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    path = attempt(tmp_path, SIZING_PROCESS_IDS[0], monkeypatch, 999)
    manifest = json.loads((path / "manifest.json").read_text())
    if state != "complete-with-temporary":
        manifest.update(status="RUNNING", accepted_engineering_calls=0, files={})
    atomic_write_json(path / "manifest.json", manifest)
    claims = json.loads((tmp_path / "sizing-claims.json").read_text())
    claims["attempts"][0].update(status="RUNNING", manifest_sha256=None)
    atomic_write_json(tmp_path / "sizing-claims.json", claims)
    atomic_write_json(
        tmp_path / ".sizing.lock",
        {
            "process_id": 999,
            "attempt": path.name,
            "hostname": "other" if state == "different-worker" else sizing.socket.gethostname(),
        },
    )

    def dead(*args: object) -> None:
        if state != "live":
            raise ProcessLookupError

    monkeypatch.setattr(sizing.os, "kill", dead)
    monkeypatch.setattr(sizing.os, "getpid", lambda: 1002)
    if state in {"owned-temporaries", "complete-with-temporary"}:
        (path / ".calls.json.999.tmp").write_bytes(b'{"calls":[')
        (path / ".manifest.json.999.tmp").write_bytes(b'{"status":')
        (tmp_path / ".sizing-claims.json.999.tmp").write_bytes(b'{"attempts":[')
    if state == "foreign-temporary":
        (path / ".calls.json.998.tmp").write_bytes(b'{"calls":[')
    if state in {"dead", "owned-temporaries", "complete-with-temporary"}:
        receipt = sizing.seal_interrupted_sizing(tmp_path, path, execution_commit=COMMIT)
        if state == "complete-with-temporary":
            assert receipt["status"] == "COMPLETE" and receipt["accepted_engineering_calls"] == 18
        else:
            assert receipt["status"] == "INCOMPLETE" and receipt["accepted_engineering_calls"] == 0
        if state != "dead":
            assert len(receipt["files"]) == 5
            assert (path / ".calls.json.999.tmp").read_bytes() == b'{"calls":['
            assert (path / ".sizing-claims.json.999.tmp").read_bytes() == b'{"attempts":['
            assert (tmp_path / ".sizing-claims.json.999.tmp").exists()
            sizing._verify_files(path, receipt)
            p2 = attempt(tmp_path, SIZING_PROCESS_IDS[1], monkeypatch, 1000)
            if state == "owned-temporaries":
                p1 = attempt(tmp_path, SIZING_PROCESS_IDS[0], monkeypatch, 1001, "retry")
            else:
                p1 = path
            assert aggregate([p1, p2])["engineering_calls"] == 36
        assert not (tmp_path / ".sizing.lock").exists()
    else:
        with pytest.raises(M8S2ProtocolViolation):
            sizing.seal_interrupted_sizing(
                tmp_path, path, execution_commit="f" * 40 if state == "wrong-commit" else COMMIT
            )
        assert (tmp_path / ".sizing.lock").exists()


@pytest.mark.parametrize(
    "missing", ["ledger", "manifest", "calls", "authorization", "directory", "executed"]
)
def test_initial_atomic_writes_recover_only_zero_call_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    snapshots = []

    def capture(**kwargs: object) -> float:
        snapshots.append(json.loads((tmp_path / ".sizing.lock").read_text()))
        return worker(**kwargs)

    path = attempt(tmp_path, SIZING_PROCESS_IDS[0], monkeypatch, 999, work=capture)
    lock = snapshots[0]
    initial = lock["bootstrap"]["manifest"]
    atomic_write_json(path / "manifest.json", initial)
    atomic_write_json(path / "calls.json", {"calls": []})
    claims = json.loads((tmp_path / "sizing-claims.json").read_text())
    claims["attempts"][0].update(status="RUNNING", manifest_sha256=None)
    atomic_write_json(tmp_path / "sizing-claims.json", claims)
    atomic_write_json(tmp_path / ".sizing.lock", lock)
    if missing == "ledger":
        (tmp_path / "sizing-claims.json").unlink()
        (tmp_path / ".sizing-claims.json.999.tmp").write_bytes(b'{"attempts":[')
    elif missing in {"manifest", "calls", "authorization", "executed"}:
        name = "calls" if missing == "executed" else missing
        (path / f"{name}.json").unlink()
        (path / f".{name}.json.999.tmp").write_bytes(b'{"partial":')
        if missing == "executed":
            atomic_write_json(path / "manifest.json", {**initial, "engineering_calls_started": 1})
    else:
        for name in ("manifest.json", "calls.json", "authorization.json"):
            (path / name).unlink()
        path.rmdir()
        claims["attempts"] = []
        atomic_write_json(tmp_path / "sizing-claims.json", claims)
    monkeypatch.setattr(sizing.os, "getpid", lambda: 1002)
    monkeypatch.setattr(sizing.os, "kill", lambda *a: (_ for _ in ()).throw(ProcessLookupError()))
    if missing == "executed":
        with pytest.raises(M8S2ProtocolViolation):
            sizing.seal_interrupted_sizing(tmp_path, path, execution_commit=COMMIT)
        assert (tmp_path / ".sizing.lock").exists()
        return
    receipt = sizing.seal_interrupted_sizing(tmp_path, path, execution_commit=COMMIT)
    assert receipt["status"] == "INCOMPLETE" and receipt["engineering_calls_completed"] == 0
    assert receipt["accepted_engineering_calls"] == 0
    sizing._verify_files(path, receipt)
    p1 = attempt(tmp_path, SIZING_PROCESS_IDS[0], monkeypatch, 1003, "retry")
    p2 = attempt(tmp_path, SIZING_PROCESS_IDS[1], monkeypatch, 1004)
    assert aggregate([p1, p2])["engineering_calls"] == 36


@pytest.mark.parametrize(
    "target",
    ["calls", "manifest", "claims", "attempt-extra", "failed-extra", "lock", "existing", "safe"],
)
def test_result_output_cannot_damage_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    paths = [
        attempt(tmp_path, logical, monkeypatch, 800 + i)
        for i, logical in enumerate(SIZING_PROCESS_IDS)
    ]
    failed = tmp_path / "failed-attempt"
    failed.mkdir()
    existing = tmp_path.parent / "existing.json"
    existing.write_text("preserved")
    output = {
        "calls": paths[0] / "calls.json",
        "manifest": paths[1] / "manifest.json",
        "claims": tmp_path / "sizing-claims.json",
        "attempt-extra": paths[0] / "extra.json",
        "failed-extra": failed / "extra.json",
        "lock": tmp_path / ".sizing.lock",
        "existing": existing,
        "safe": tmp_path / "sizing-result.json",
    }[target]
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = aggregate(paths)
    if target == "safe":
        sizing.write_sizing_result(output, paths, result)
        assert json.loads(output.read_text())["engineering_calls"] == 36
    else:
        with pytest.raises(M8S2ProtocolViolation):
            sizing.write_sizing_result(output, paths, result)
        assert all(p.read_bytes() == data for p, data in before.items())
        assert existing.read_text() == "preserved"


@pytest.mark.parametrize(
    "stage", ["manifest", "ledger", "bootstrap", "retained-copy", "live-recovery"]
)
def test_interrupted_recovery_can_itself_be_recovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    snapshots = []

    def capture(**kwargs: object) -> float:
        snapshots.append(json.loads((tmp_path / ".sizing.lock").read_text()))
        return worker(**kwargs)

    path = attempt(tmp_path, SIZING_PROCESS_IDS[0], monkeypatch, 999, work=capture)
    lock = snapshots[0]
    manifest = json.loads((path / "manifest.json").read_text())
    manifest.update(status="RUNNING", accepted_engineering_calls=0, files={})
    if stage == "bootstrap":
        atomic_write_json(path / "calls.json", {"calls": []})
        (path / "manifest.json").unlink()
    else:
        atomic_write_json(path / "manifest.json", manifest)
    claims = json.loads((tmp_path / "sizing-claims.json").read_text())
    claims["attempts"][0].update(status="RUNNING", manifest_sha256=None)
    atomic_write_json(tmp_path / "sizing-claims.json", claims)
    atomic_write_json(tmp_path / ".sizing.lock", lock)
    if stage == "retained-copy":
        (tmp_path / ".sizing-claims.json.999.tmp").write_bytes(b'{"original":')
    monkeypatch.setattr(sizing.os, "getpid", lambda: 1002)
    monkeypatch.setattr(sizing.os, "kill", lambda *a: (_ for _ in ()).throw(ProcessLookupError()))
    target = tmp_path / "sizing-claims.json" if stage == "ledger" else path / "manifest.json"

    def interrupted_write(destination: Path, payload: dict) -> None:
        if destination == target:
            destination.with_name(f".{destination.name}.1002.tmp").write_bytes(b'{"interrupted":')
            raise SystemExit("synthetic interrupted recovery write")
        atomic_write_json(destination, payload)

    monkeypatch.setattr(sizing, "atomic_write_json", interrupted_write)
    original_replace = sizing.os.replace
    if stage == "retained-copy":
        monkeypatch.setattr(sizing, "atomic_write_json", atomic_write_json)

        def interrupted_copy(source: Path, destination: Path) -> None:
            if destination == path / ".sizing-claims.json.999.tmp":
                Path(source).write_bytes(b'{"partial-copy":')
                raise SystemExit("synthetic interrupted retained byte copy")
            original_replace(source, destination)

        monkeypatch.setattr(sizing.os, "replace", interrupted_copy)
    with pytest.raises(SystemExit):
        sizing.seal_interrupted_sizing(tmp_path, path, execution_commit=COMMIT)
    assert json.loads((tmp_path / ".sizing.lock").read_text())["recovery_process_ids"] == [1002]
    monkeypatch.setattr(sizing, "atomic_write_json", atomic_write_json)
    monkeypatch.setattr(sizing.os, "replace", original_replace)
    monkeypatch.setattr(sizing.os, "getpid", lambda: 1003)
    if stage == "live-recovery":

        def live_recovery(pid: int, signal: int) -> None:
            if pid != 1002:
                raise ProcessLookupError

        monkeypatch.setattr(sizing.os, "kill", live_recovery)
        with pytest.raises(M8S2ProtocolViolation):
            sizing.seal_interrupted_sizing(tmp_path, path, execution_commit=COMMIT)
        return
    receipt = sizing.seal_interrupted_sizing(tmp_path, path, execution_commit=COMMIT)
    assert receipt["status"] == "INCOMPLETE"
    assert receipt["recovery_process_ids"] == [1002, 1003]
    assert any(".1002.tmp" in name for name in receipt["files"])
    sizing._verify_files(path, receipt)
    p1 = attempt(tmp_path, SIZING_PROCESS_IDS[0], monkeypatch, 1004, "retry")
    p2 = attempt(tmp_path, SIZING_PROCESS_IDS[1], monkeypatch, 1005)
    assert aggregate([p1, p2])["engineering_calls"] == 36


@pytest.mark.parametrize("target", ["calls", "claims", "attempt-extra", "safe"])
def test_recovery_cli_output_uses_evidence_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    path = tmp_path / "attempt"
    path.mkdir()
    (path / "calls.json").write_text("preserved calls")
    (tmp_path / "sizing-claims.json").write_text("preserved claims")
    output = {
        "calls": path / "calls.json",
        "claims": tmp_path / "sizing-claims.json",
        "attempt-extra": path / "extra.json",
        "safe": tmp_path / "recovery-receipt.json",
    }[target]
    main = runpy.run_path(str(ROOT / "scripts/detection/run_m8_s2.py"))["main"]
    monkeypatch.setitem(main.__globals__, "verify_static_bindings", lambda *a: None)
    monkeypatch.setattr(sizing, "seal_interrupted_sizing", lambda *a, **k: {"status": "INCOMPLETE"})
    args = [
        "sizing-seal-interrupted",
        "--external-worker",
        "--repository-root",
        str(ROOT),
        "--execution-commit",
        COMMIT,
        "--campaign-root",
        str(tmp_path),
        "--attempt-root",
        str(path),
        "--output",
        str(output),
    ]
    if target == "safe":
        assert main(args) == 0
        assert json.loads(output.read_text()) == {"status": "INCOMPLETE"}
    else:
        with pytest.raises(M8S2ProtocolViolation):
            main(args)
        assert (path / "calls.json").read_text() == "preserved calls"
        assert (tmp_path / "sizing-claims.json").read_text() == "preserved claims"


@pytest.mark.parametrize(
    "failure",
    ["call-before-replace", "call-after-replace", "manifest", "initial-auth", "initial-ledger"],
)
def test_caught_write_interrupt_seals_only_persisted_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    fired = False
    path = tmp_path / SIZING_PROCESS_IDS[0]

    def write(destination: Path, payload: dict) -> None:
        nonlocal fired
        selected = (
            (
                destination == path / "calls.json"
                and payload["calls"]
                and failure.startswith("call-")
            )
            or (
                destination == path / "manifest.json"
                and payload.get("initialization_seconds") == 4.0
                and failure == "manifest"
            )
            or (destination == tmp_path / "sizing-claims.json" and failure == "initial-ledger")
        )
        if selected and not fired:
            fired = True
            if failure == "call-after-replace":
                atomic_write_json(destination, payload)
            else:
                destination.with_name(f".{destination.name}.999.tmp").write_bytes(b'{"partial":')
            raise KeyboardInterrupt("synthetic caught persistence interruption")
        atomic_write_json(destination, payload)

    def work(**kwargs: object) -> float:
        kwargs["on_initialized"](4.0)
        record = kwargs["record"]

        def started_record(call: dict) -> None:
            kwargs["on_call_start"]()
            record(call)

        kwargs["record"] = started_record
        return worker(**kwargs)

    monkeypatch.setattr(sizing, "atomic_write_json", write)
    original_replace = sizing.os.replace

    def replace(source: Path, destination: Path) -> None:
        if destination == path / "authorization.json" and failure == "initial-auth":
            Path(source).write_bytes(b'{"partial-authorization":')
            raise KeyboardInterrupt("synthetic initial authorization interruption")
        original_replace(source, destination)

    monkeypatch.setattr(sizing.os, "replace", replace)
    with pytest.raises(KeyboardInterrupt):
        attempt(tmp_path, SIZING_PROCESS_IDS[0], monkeypatch, 999, work=work)
    monkeypatch.setattr(sizing, "atomic_write_json", atomic_write_json)
    monkeypatch.setattr(sizing.os, "replace", original_replace)
    if failure.startswith("initial-"):
        assert (tmp_path / ".sizing.lock").exists()
        monkeypatch.setattr(sizing.os, "getpid", lambda: 1000)
        monkeypatch.setattr(
            sizing.os, "kill", lambda *a: (_ for _ in ()).throw(ProcessLookupError())
        )
        manifest = sizing.seal_interrupted_sizing(tmp_path, path, execution_commit=COMMIT)
    else:
        assert not (tmp_path / ".sizing.lock").exists()
        manifest = json.loads((path / "manifest.json").read_text())
    expected_count = 1 if failure == "call-after-replace" else 0
    assert manifest["status"] == "INCOMPLETE" and manifest["accepted_engineering_calls"] == 0
    assert manifest["engineering_calls_completed"] == expected_count
    assert len(json.loads((path / "calls.json").read_text())["calls"]) == expected_count
    sizing._verify_files(path, manifest)
    p1 = attempt(tmp_path, SIZING_PROCESS_IDS[0], monkeypatch, 1001, "retry")
    p2 = attempt(tmp_path, SIZING_PROCESS_IDS[1], monkeypatch, 1002)
    assert aggregate([p1, p2])["engineering_calls"] == 36


def test_recovery_guard_excludes_another_os_process(tmp_path: Path) -> None:
    script = """
import sys
from pathlib import Path
from laserperception.detection.m8_s2_sizing import _recovery_guard, M8S2ProtocolViolation
try:
    with _recovery_guard(Path(sys.argv[1])):
        sys.exit(3)
except M8S2ProtocolViolation:
    print("blocked")
"""
    with sizing._recovery_guard(tmp_path):
        child = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert child.returncode == 0 and child.stdout.strip() == "blocked", child.stderr
    with sizing._recovery_guard(tmp_path):
        assert (tmp_path / ".sizing-recovery-guard").exists()


def test_recovery_guard_releases_when_owner_is_killed(tmp_path: Path) -> None:
    script = """
import sys
from pathlib import Path
from laserperception.detection.m8_s2_sizing import _recovery_guard
with _recovery_guard(Path(sys.argv[1])):
    print("owned", flush=True)
    sys.stdin.read()
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None and child.stdout.readline().strip() == "owned"
        with pytest.raises(M8S2ProtocolViolation):
            with sizing._recovery_guard(tmp_path):
                pytest.fail("concurrent recovery acquired ownership")
    finally:
        child.kill()
        child.communicate(timeout=20)
    with sizing._recovery_guard(tmp_path):
        assert (tmp_path / ".sizing-recovery-guard").exists()
