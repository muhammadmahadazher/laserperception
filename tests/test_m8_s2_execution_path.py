"""CPU-only contracts for the future S2 execution path; no science imports."""

from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
import sys
import uuid
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from laserperception.detection.m8_s2_input_gate import stream_full_ledger
from laserperception.detection.m8_s2_planning import estimate_future_cost, qualification_plan
from laserperception.detection.m8_s2_preflight import (
    check_sentinel_coordinates,
    make_qualification_receipt,
)
from laserperception.detection.m8_s2_runtime import (
    FULL_LEDGER_SHA256,
    PROTOCOL_SHA256,
    AtomicAttempt,
    AttemptIdentity,
    M8S2ProtocolViolation,
    claim_logical_pass,
    condition_ids,
    seal_interrupted_attempt,
    verify_authorization,
    verify_candidate_environment,
    verify_clean_tracked_tree,
    verify_frozen_gt_assets,
    verify_qualification_worker,
)
from laserperception.evaluation.m8_s2_aggregation import (
    aggregate_three_full_passes,
    factorial_contrasts,
    load_completed_attempt,
    review_repeatability,
    validate_attempts,
)

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40
POLICY = "b" * 64
RECEIPT = "c" * 64
QUALIFICATION = "d" * 64
AUTHORIZATION = "e" * 64
AUTHORIZATION_ID = "synthetic-test-grant"


def _pass(mode: str, index: int) -> dict[str, object]:
    logical = f"s2-repeatability-{index:02d}" if mode == "repeatability" else f"s2-pass-{index}"
    order = condition_ids(mode)
    return {
        "status": "COMPLETE",
        "mode": mode,
        "logical_pass_id": logical,
        "process_uuid": str(uuid.UUID(int=index)),
        "attempt_id": f"attempt-{index}",
        "result_sha256": f"{index:064x}",
        "execution_commit": COMMIT,
        "runtime_policy_sha256": POLICY,
        "input_gate_receipt_sha256": RECEIPT,
        "qualification_receipt_sha256": QUALIFICATION,
        "authorization_id": AUTHORIZATION_ID,
        "authorization_sha256": AUTHORIZATION,
        "full_ledger_sha256": FULL_LEDGER_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "expected_calls": len(order),
        "completed_calls": len(order),
        "accepted_canonical_calls": len(order),
        "failed_calls": 0,
        "conditions": [{"condition_id": item} for item in order],
    }


def test_canonical_orders_and_no_splicing() -> None:
    repeat = condition_ids("repeatability")
    full = condition_ids("full-pass")
    assert len(repeat) == len(set(repeat)) == 28
    assert len(full) == len(set(full)) == 1712
    assert repeat[:4] == tuple(
        f"{repeat[0].rsplit('/', 1)[0]}/{arm}" for arm in ("B2", "C2", "D2", "F2")
    )
    assert full[:4] == tuple(
        f"{full[0].rsplit('/', 1)[0]}/{arm}" for arm in ("B2", "C2", "D2", "F2")
    )
    passes = [_pass("full-pass", index) for index in (1, 2, 3)]
    validate_attempts(passes, mode="full-pass")
    passes[1]["process_uuid"] = passes[0]["process_uuid"]
    with pytest.raises(M8S2ProtocolViolation, match="reused"):
        validate_attempts(passes, mode="full-pass")
    passes[1]["process_uuid"] = str(uuid.UUID(int=2))
    passes[2]["conditions"] = list(passes[1]["conditions"])[1:]
    with pytest.raises(M8S2ProtocolViolation, match="incomplete or reordered"):
        validate_attempts(passes, mode="full-pass")


def test_atomic_attempt_failure_is_never_canonical(tmp_path: Path) -> None:
    identity = AttemptIdentity(
        "repeatability",
        "s2-repeatability-01",
        "attempt-1",
        str(uuid.uuid4()),
        123,
        COMMIT,
        POLICY,
        RECEIPT,
        QUALIFICATION,
        AUTHORIZATION_ID,
        AUTHORIZATION,
    )
    attempt = AtomicAttempt(tmp_path / "attempt", identity)
    with pytest.raises(M8S2ProtocolViolation, match="order"):
        attempt.record(condition_ids("repeatability")[1], {})
    attempt.record(
        condition_ids("repeatability")[0], {"condition_id": condition_ids("repeatability")[0]}
    )
    failed = attempt.fail("synthetic failure")
    assert failed["accepted_canonical_calls"] == 0
    assert failed["qualification_receipt_sha256"] == QUALIFICATION
    assert failed["authorization_sha256"] == AUTHORIZATION
    assert failed["completed_calls"] == 1
    with pytest.raises(M8S2ProtocolViolation, match="incomplete"):
        attempt.finalize()
    assert not (tmp_path / "attempt" / "final_pass_manifest.json").exists()


def test_completed_attempt_loader_retains_authorization_chain(tmp_path: Path) -> None:
    identity = AttemptIdentity(
        "repeatability",
        "s2-repeatability-01",
        "attempt-complete",
        str(uuid.uuid4()),
        123,
        COMMIT,
        POLICY,
        RECEIPT,
        QUALIFICATION,
        AUTHORIZATION_ID,
        AUTHORIZATION,
    )
    root = tmp_path / "complete"
    attempt = AtomicAttempt(root, identity)
    for condition_id in condition_ids("repeatability"):
        attempt.record(condition_id, {"condition_id": condition_id})
    attempt.finalize()
    loaded = load_completed_attempt(root, mode="repeatability")
    assert loaded["accepted_canonical_calls"] == 28
    assert loaded["authorization_id"] == AUTHORIZATION_ID
    assert loaded["authorization_sha256"] == AUTHORIZATION
    assert loaded["qualification_receipt_sha256"] == QUALIFICATION


def test_repeatability_requires_ten_exact_processes() -> None:
    passes = [_pass("repeatability", index) for index in range(1, 11)]
    for row in passes:
        for condition in row["conditions"]:
            evidence = {
                "thresholded_prediction_count": 0,
                "thresholds": {
                    threshold: {
                        "true_positives": 0,
                        "false_positives": 0,
                        "false_negatives": 0,
                        "ignored_predictions": 0,
                        "matched_gt_identity_set": [],
                    }
                    for threshold in ("0.30", "0.50", "0.70")
                },
            }
            condition["classes"] = {"car": deepcopy(evidence), "pedestrian": deepcopy(evidence)}
    result = review_repeatability(passes)
    assert result["status"] == "ACCEPTED"
    assert result["accepted_calls"] == 280
    assert result["owner_reviewed"] is False
    assert len(result["source_attempts"]) == 10
    assert result["aggregation_commit"] == COMMIT
    assert result["source_attempts"][0]["authorization_sha256"] == AUTHORIZATION
    assert result["qualification_receipt_sha256"] == QUALIFICATION
    with pytest.raises(M8S2ProtocolViolation, match="aggregation checkout differs"):
        review_repeatability(passes, aggregation_commit="d" * 40)
    passes[1]["conditions"][0]["classes"]["car"]["thresholds"]["0.50"]["true_positives"] = 1
    with pytest.raises(M8S2ProtocolViolation, match="repeatability differs"):
        review_repeatability(passes)
    with pytest.raises(M8S2ProtocolViolation, match="process count"):
        review_repeatability(passes[:9])


def test_authorization_scopes_fail_closed(tmp_path: Path) -> None:
    from laserperception.detection.m8_s2_runtime import (
        COMPACT_MANIFEST_SHA256,
        INPUT_FREEZE_SHA256,
        PARTITIONS_SHA256,
        PROTOCOL_SHA256,
    )

    payload = {
        "schema_version": "laserperception.m8.s2.authorization.v1",
        "authorized": True,
        "scope": "qualification-only",
        "owner_approval": True,
        "authorization_id": "synthetic-test-only",
        "authorization_timestamp_utc": "2026-01-01T00:00:00Z",
        "authorization_provenance": "synthetic-test-only",
        "authorized_gpu_uuid": "GPU-synthetic-test-only",
        "authorized_worker_hostname": "synthetic-host",
        "authorized_campaign_root": None,
        "execution_commit": COMMIT,
        "protocol_sha256": PROTOCOL_SHA256,
        "partitions_sha256": PARTITIONS_SHA256,
        "input_freeze_sha256": INPUT_FREEZE_SHA256,
        "full_ledger_sha256": FULL_LEDGER_SHA256,
        "compact_manifest_sha256": COMPACT_MANIFEST_SHA256,
        "logical_pass_ids": [],
        "runtime_policy_binding_sha256": None,
        "input_gate_receipt_sha256": None,
        "qualification_receipt_sha256": None,
        "repeatability_review_sha256": None,
    }
    verify_authorization(
        payload, scope="qualification-only", execution_commit=COMMIT, logical_pass_id=None
    )
    with pytest.raises(M8S2ProtocolViolation, match="scope differs"):
        verify_authorization(
            payload,
            scope="repeatability-only",
            execution_commit=COMMIT,
            logical_pass_id="s2-repeatability-01",
        )
    payload["protocol_sha256"] = "0" * 64
    with pytest.raises(M8S2ProtocolViolation, match="binding differs"):
        verify_authorization(
            payload, scope="qualification-only", execution_commit=COMMIT, logical_pass_id=None
        )
    payload.update(
        scope="repeatability-only",
        protocol_sha256=PROTOCOL_SHA256,
        authorized_gpu_uuid=None,
        authorized_worker_hostname=None,
        authorized_campaign_root=str(tmp_path.resolve()),
        logical_pass_ids=["s2-repeatability-01"],
        runtime_policy_binding_sha256=POLICY,
        input_gate_receipt_sha256=RECEIPT,
        qualification_receipt_sha256=RECEIPT,
    )
    expected = {
        "scope": "repeatability-only",
        "execution_commit": COMMIT,
        "logical_pass_id": "s2-repeatability-01",
        "runtime_policy_sha256": POLICY,
        "input_gate_receipt_sha256": RECEIPT,
        "qualification_receipt_sha256": RECEIPT,
        "campaign_root": tmp_path,
    }
    verify_authorization(payload, **expected)
    with pytest.raises(M8S2ProtocolViolation, match="campaign root differs"):
        verify_authorization(payload, **{**expected, "campaign_root": tmp_path / "other"})


def test_qualification_grant_matches_live_external_identity_only_with_mock() -> None:
    grant = {
        "authorized_gpu_uuid": "GPU-synthetic-123",
        "authorized_worker_hostname": "synthetic-host",
    }

    def matching_runner(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, "GPU-synthetic-123\n", "")

    def different_runner(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, "GPU-other\n", "")

    def multiple_runner(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, "GPU-synthetic-123\nGPU-other\n", "")

    assert (
        verify_qualification_worker(
            grant, command_runner=matching_runner, hostname_provider=lambda: "synthetic-host"
        )
        == "GPU-synthetic-123"
    )
    with pytest.raises(M8S2ProtocolViolation, match="GPU UUID differs"):
        verify_qualification_worker(
            grant, command_runner=different_runner, hostname_provider=lambda: "synthetic-host"
        )
    with pytest.raises(M8S2ProtocolViolation, match="hostname differs"):
        verify_qualification_worker(
            grant, command_runner=matching_runner, hostname_provider=lambda: "different-host"
        )
    with pytest.raises(M8S2ProtocolViolation, match="exactly one"):
        verify_qualification_worker(
            grant, command_runner=multiple_runner, hostname_provider=lambda: "synthetic-host"
        )


def test_policy_rejects_a_different_cuda_visible_gpu_without_importing_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from laserperception.detection import m8_s2_runtime_policy as policy_module

    monkeypatch.setattr(policy_module, "single_visible_gpu_uuid", lambda: "GPU-synthetic-123")
    monkeypatch.setattr(
        policy_module,
        "capture_s1_policy",
        lambda *_: {"gpu_uuid": "GPU-synthetic-123"},
    )
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2))
    monkeypatch.setattr(
        policy_module, "importlib", SimpleNamespace(import_module=lambda _: fake_torch)
    )
    with pytest.raises(M8S2ProtocolViolation, match="ambiguous"):
        policy_module.capture_runtime_policy(COMMIT, {})
    fake_torch.cuda.device_count = lambda: 1
    monkeypatch.setattr(policy_module, "capture_s1_policy", lambda *_: {"gpu_uuid": "GPU-other"})
    with pytest.raises(M8S2ProtocolViolation, match="ambiguous"):
        policy_module.capture_runtime_policy(COMMIT, {})
    fake_torch.cuda.get_device_properties = lambda _: SimpleNamespace(total_memory=1024)
    monkeypatch.setattr(policy_module.socket, "gethostname", lambda: "synthetic-host")
    monkeypatch.setattr(
        policy_module, "capture_s1_policy", lambda *_: {"gpu_uuid": "GPU-synthetic-123"}
    )
    assert policy_module.capture_runtime_policy(COMMIT, {})["worker_hostname"] == "synthetic-host"


def test_logical_pass_claim_allows_incomplete_retry_but_consumes_success(tmp_path: Path) -> None:
    common = {
        "campaign_root": tmp_path,
        "mode": "repeatability",
        "logical_pass_id": "s2-repeatability-01",
        "execution_commit": COMMIT,
        "runtime_policy_sha256": POLICY,
        "input_gate_receipt_sha256": RECEIPT,
        "qualification_receipt_sha256": QUALIFICATION,
        "authorization_id": AUTHORIZATION_ID,
        "authorization_sha256": AUTHORIZATION,
    }
    first = tmp_path / "attempt-1"
    with claim_logical_pass(attempt_root=first, attempt_id="attempt-1", **common):
        first.mkdir()
        (first / "attempt_manifest.json").write_text(
            json.dumps(
                {
                    "status": "INCOMPLETE",
                    "mode": "repeatability",
                    "logical_pass_id": "s2-repeatability-01",
                    "attempt_id": "attempt-1",
                    "execution_commit": COMMIT,
                    "runtime_policy_sha256": POLICY,
                    "input_gate_receipt_sha256": RECEIPT,
                    "qualification_receipt_sha256": QUALIFICATION,
                    "authorization_id": AUTHORIZATION_ID,
                    "authorization_sha256": AUTHORIZATION,
                }
            ),
            encoding="utf-8",
        )
    second = tmp_path / "attempt-2"
    with claim_logical_pass(attempt_root=second, attempt_id="attempt-2", **common):
        second.mkdir()
        (second / "attempt_manifest.json").write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "mode": "repeatability",
                    "logical_pass_id": "s2-repeatability-01",
                    "attempt_id": "attempt-2",
                    "execution_commit": COMMIT,
                    "runtime_policy_sha256": POLICY,
                    "input_gate_receipt_sha256": RECEIPT,
                    "qualification_receipt_sha256": QUALIFICATION,
                    "authorization_id": AUTHORIZATION_ID,
                    "authorization_sha256": AUTHORIZATION,
                }
            ),
            encoding="utf-8",
        )
    with pytest.raises(M8S2ProtocolViolation, match="already completed"):
        with claim_logical_pass(
            attempt_root=tmp_path / "attempt-3", attempt_id="attempt-3", **common
        ):
            pass


def test_tracked_tree_changes_are_rejected_before_binding(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init")
    git("config", "user.name", "S2 Test")
    git("config", "user.email", "s2-test@example.invalid")
    tracked = tmp_path / "runner.py"
    tracked.write_text("value = 1\n", encoding="utf-8")
    git("add", "runner.py")
    git("commit", "-m", "test fixture")
    verify_clean_tracked_tree(tmp_path)
    tracked.write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(M8S2ProtocolViolation, match="tracked execution tree"):
        verify_clean_tracked_tree(tmp_path)


def test_backend_environment_must_match_checked_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from laserperception.detection.m8_s2_runtime import CANDIDATE_MANIFEST_PATH

    manifest = tmp_path / CANDIDATE_MANIFEST_PATH
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "environment": {
                    "upstream_root_variable": "S2_TEST_UPSTREAM",
                    "checkpoint_variable": "S2_TEST_CHECKPOINT",
                }
            }
        ),
        encoding="utf-8",
    )
    upstream = tmp_path / "upstream"
    checkpoint = tmp_path / "checkpoint.pth"
    monkeypatch.setenv("S2_TEST_UPSTREAM", str(upstream))
    monkeypatch.setenv("S2_TEST_CHECKPOINT", str(checkpoint))
    verify_candidate_environment(tmp_path, upstream, checkpoint)
    monkeypatch.setenv("S2_TEST_UPSTREAM", str(tmp_path / "different"))
    with pytest.raises(M8S2ProtocolViolation, match="backend environment differs"):
        verify_candidate_environment(tmp_path, upstream, checkpoint)


def test_interrupted_lock_can_be_sealed_without_scientific_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from laserperception.detection import m8_s2_runtime

    common = {
        "campaign_root": tmp_path,
        "mode": "repeatability",
        "logical_pass_id": "s2-repeatability-01",
        "execution_commit": COMMIT,
        "runtime_policy_sha256": POLICY,
        "input_gate_receipt_sha256": RECEIPT,
        "qualification_receipt_sha256": QUALIFICATION,
        "authorization_id": AUTHORIZATION_ID,
        "authorization_sha256": AUTHORIZATION,
    }
    first = tmp_path / "attempt-1"
    with claim_logical_pass(attempt_root=first, attempt_id="attempt-1", **common):
        first.mkdir()
        (first / "attempt_manifest.json").write_text(
            json.dumps(
                {
                    "status": "IN_PROGRESS",
                    "mode": "repeatability",
                    "logical_pass_id": "s2-repeatability-01",
                    "attempt_id": "attempt-1",
                    "execution_commit": COMMIT,
                    "runtime_policy_sha256": POLICY,
                    "input_gate_receipt_sha256": RECEIPT,
                    "qualification_receipt_sha256": QUALIFICATION,
                    "authorization_id": AUTHORIZATION_ID,
                    "authorization_sha256": AUTHORIZATION,
                }
            ),
            encoding="utf-8",
        )
    lock = tmp_path / ".s2_pass_claims" / "repeatability-s2-repeatability-01.lock"
    assert lock.exists()
    original_lock_bytes = lock.read_bytes()
    recovery_kwargs = dict(
        campaign_root=tmp_path,
        attempt_root=first,
        mode="repeatability",
        logical_pass_id="s2-repeatability-01",
        attempt_id="attempt-1",
        execution_commit=COMMIT,
        recovery_note="synthetic interrupted process",
        process_alive=lambda _: False,
    )
    original_write = m8_s2_runtime.atomic_write_json

    def interrupted_write(path: Path, payload: dict[str, object]) -> None:
        if path.name == "interrupted_recovery.json":
            raise RuntimeError("synthetic recovery interruption")
        original_write(path, payload)

    monkeypatch.setattr(m8_s2_runtime, "atomic_write_json", interrupted_write)
    with pytest.raises(RuntimeError, match="synthetic recovery interruption"):
        seal_interrupted_attempt(**recovery_kwargs)
    assert lock.exists()
    assert (first / "attempt_manifest_before_recovery.json").is_file()
    monkeypatch.setattr(m8_s2_runtime, "atomic_write_json", original_write)
    receipt = seal_interrupted_attempt(**recovery_kwargs)
    assert receipt["status"] == "SEALED_INCOMPLETE"
    lock.write_bytes(original_lock_bytes)
    assert seal_interrupted_attempt(**recovery_kwargs) == receipt
    assert seal_interrupted_attempt(**recovery_kwargs) == receipt
    assert not lock.exists()
    assert (first / "attempt_manifest_before_recovery.json").is_file()
    assert (
        json.loads((first / "attempt_manifest.json").read_text())["accepted_canonical_calls"] == 0
    )
    retry = tmp_path / "attempt-2"
    with claim_logical_pass(attempt_root=retry, attempt_id="attempt-2", **common):
        retry.mkdir()
        (retry / "attempt_manifest.json").write_text(
            json.dumps(
                {
                    "status": "INCOMPLETE",
                    "mode": "repeatability",
                    "logical_pass_id": "s2-repeatability-01",
                    "attempt_id": "attempt-2",
                    "execution_commit": COMMIT,
                    "runtime_policy_sha256": POLICY,
                    "input_gate_receipt_sha256": RECEIPT,
                    "qualification_receipt_sha256": QUALIFICATION,
                    "authorization_id": AUTHORIZATION_ID,
                    "authorization_sha256": AUTHORIZATION,
                }
            ),
            encoding="utf-8",
        )


def test_frozen_gt_hashes_are_checked_before_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from laserperception.detection import m8_s2_runtime

    root = tmp_path / "source"
    date_root = tmp_path / "date"
    root.mkdir()
    date_root.mkdir()
    calibration = {}
    for name in ("calib_cam_to_cam.txt", "calib_velo_to_cam.txt"):
        content = f"synthetic {name}\n".encode()
        (date_root / name).write_bytes(content)
        calibration[name] = hashlib.sha256(content).hexdigest()
    tracklets = {}
    for drive in ("2011_09_26_drive_0001", "2011_09_26_drive_0091"):
        content = f"synthetic {drive}\n".encode()
        path = date_root / f"{drive}_sync" / "tracklet_labels.xml"
        path.parent.mkdir()
        path.write_bytes(content)
        tracklets[drive] = hashlib.sha256(content).hexdigest()
    protocol_path = root / "frozen_protocol.json"
    protocol_path.write_text(
        json.dumps(
            {
                "sentinel_gt_only_audit": {
                    "calibration_sha256": calibration,
                    "tracklet_sha256": tracklets,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(m8_s2_runtime, "S1_PROTOCOL_PATH", Path("frozen_protocol.json"))
    monkeypatch.setattr(
        m8_s2_runtime, "S1_PROTOCOL_SHA256", hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    )
    verify_frozen_gt_assets(root, date_root)
    (date_root / "2011_09_26_drive_0001_sync" / "tracklet_labels.xml").write_text(
        "changed geometry", encoding="utf-8"
    )
    with pytest.raises(M8S2ProtocolViolation, match="frozen GT asset differs"):
        verify_frozen_gt_assets(root, date_root)


def test_static_plan_and_cost_are_input_only() -> None:
    plan = qualification_plan(ROOT)
    rows = plan["measured_conditions_per_process"]
    assert len(rows) == len({row["condition_id"] for row in rows}) == 16
    assert {row["arm"] for row in rows} == {"B2", "C2", "D2", "F2"}
    assert plan["future_engineering_calls"] == 36
    assert plan["scientific_calls_executed"] == 0
    estimate = estimate_future_cost(
        measured_call_seconds=[1.0] * 32, initialization_seconds=[2.0, 4.0], usd_per_hour=0.5
    )
    assert estimate["workloads"]["total_accepted_science"]["calls"] == 5416
    assert "confidence_interval" not in estimate
    with pytest.raises(M8S2ProtocolViolation):
        estimate_future_cost(
            measured_call_seconds=[1.0], initialization_seconds=[2.0, 4.0], usd_per_hour=0.5
        )


def test_frozen_ledger_stream_and_factorial() -> None:
    ledger = ROOT / ".local/m8-s2/evidence/m8_s2_input_ledger.jsonl"
    if ledger.exists():
        result = stream_full_ledger(ROOT, ledger)
        assert result["sha256"] == FULL_LEDGER_SHA256
        assert result["arm_counts"] == {arm: 428 for arm in ("B2", "C2", "D2", "F2")}
    assert factorial_contrasts(31, 32, 43, 19) == {"L": 11.5, "P": 12.5, "I": -1}


def test_wrong_commit_protocol_and_ledger_fail_cpu_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from laserperception.detection import m8_s2_input_gate, m8_s2_runtime

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    with pytest.raises(M8S2ProtocolViolation, match="execution commit differs"):
        m8_s2_runtime.verify_static_bindings(ROOT, "0" * 40)
    monkeypatch.setattr(m8_s2_runtime, "verify_clean_tracked_tree", lambda _: None)
    monkeypatch.setattr(m8_s2_runtime, "PROTOCOL_SHA256", "0" * 64)
    with pytest.raises(M8S2ProtocolViolation, match="frozen S2 identity"):
        m8_s2_runtime.verify_static_bindings(ROOT, commit)
    monkeypatch.undo()
    ledger = ROOT / ".local/m8-s2/evidence/m8_s2_input_ledger.jsonl"
    if ledger.exists():
        monkeypatch.setattr(m8_s2_input_gate, "FULL_LEDGER_SHA256", "0" * 64)
        with pytest.raises(M8S2ProtocolViolation, match="count or SHA256"):
            stream_full_ledger(ROOT, ledger)


def test_no_inferential_fields_in_static_outputs() -> None:
    plan = qualification_plan(ROOT)
    assert all(term not in json.dumps(plan) for term in ("p_value", "confidence_interval"))


def test_runner_rejects_imported_code_outside_reviewed_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = runpy.run_path(str(ROOT / "scripts/detection/run_m8_s2.py"))
    verify = module["_verify_imported_checkout"]
    verify(ROOT)
    monkeypatch.setitem(
        sys.modules,
        "laserperception.stale_module",
        SimpleNamespace(__file__=str(tmp_path / "stale_module.py")),
    )
    with pytest.raises(M8S2ProtocolViolation, match="imported module differs"):
        verify(ROOT)


def test_unauthorized_cli_never_imports_science_or_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guarded = {
        "laserperception.evaluation.m8_s2_science",
        "laserperception.detection.m8_backend",
        "torch",
    }
    present_before = guarded.intersection(sys.modules)
    module = runpy.run_path(str(ROOT / "scripts/detection/run_m8_s2.py"))
    main = module["main"]
    monkeypatch.setitem(main.__globals__, "verify_static_bindings", lambda *_: None)
    monkeypatch.setitem(main.__globals__, "verify_input_gate_receipt", lambda *_, **__: RECEIPT)
    monkeypatch.setitem(main.__globals__, "verify_runtime_policy_document", lambda *_, **__: None)
    monkeypatch.setitem(main.__globals__, "verify_qualification_receipt", lambda *_, **__: "d" * 64)
    monkeypatch.setitem(main.__globals__, "sha256_file", lambda *_: POLICY)
    monkeypatch.setitem(
        main.__globals__,
        "require_authorization",
        lambda *_, **__: (_ for _ in ()).throw(M8S2ProtocolViolation("owner authorization absent")),
    )
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema_version": "laserperception.m8.s2.runtime-policy-binding.v1",
                "repository_execution_commit": COMMIT,
            }
        )
    )
    args = [
        "repeatability",
        "--external-worker",
        "--execution-commit",
        COMMIT,
        "--logical-pass-id",
        "s2-repeatability-01",
        "--full-ledger",
        str(tmp_path / "ledger"),
        "--input-gate-receipt",
        str(tmp_path / "receipt"),
        "--runtime-policy-binding",
        str(policy),
        "--qualification-receipt",
        str(tmp_path / "qualification"),
        "--authorization",
        str(tmp_path / "auth"),
    ]
    with pytest.raises(M8S2ProtocolViolation):
        main(args)
    assert guarded.intersection(sys.modules) == present_before


def test_structural_coordinate_contract_with_cpu_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from laserperception.detection import m8_s2_preflight
    from laserperception.detection.m8_s2_runtime import SENTINELS

    monkeypatch.setattr(m8_s2_preflight, "verify_consumed_input", lambda *_, **__: {})
    points = np.zeros((2, 5), dtype=np.float32)
    frames = [
        {
            "frame_id": frame,
            "A2": points,
            **{arm: (points.copy(), "synthetic") for arm in ("B2", "C2", "D2", "F2")},
        }
        for frame in SENTINELS
    ]
    compact = {f"{frame}/{arm}": {} for frame in SENTINELS for arm in ("B2", "C2", "D2", "F2")}

    def coordinate_fn(_: np.ndarray) -> np.ndarray:
        return np.asarray([[1, 2], [3, 4]], dtype=np.int32)

    result = check_sentinel_coordinates(frames, compact_by_id=compact, coordinate_fn=coordinate_fn)
    assert result["sentinel_XYZIT_exact"] == 28
    assert result["B2_A2_cuda_coordinate_identity"] == 7
    assert (
        result["sentinels"][0]["candidate_coordinate_sha256"]["A2"]
        == result["sentinels"][0]["candidate_coordinate_sha256"]["B2"]
    )
    receipt = make_qualification_receipt(
        result,
        execution_commit=COMMIT,
        runtime_policy_sha256=POLICY,
        input_gate_receipt_sha256=RECEIPT,
    )
    assert receipt["ground_truth_loaded"] is False
    frames[0]["B2"] = (np.ones((2, 5), dtype=np.float32), "synthetic")

    def different_coordinates(array: np.ndarray) -> np.ndarray:
        return (
            np.asarray([[1, 2]], dtype=np.int32)
            if array[0, 0]
            else np.asarray([[3, 4]], dtype=np.int32)
        )

    with pytest.raises(M8S2ProtocolViolation, match="A2/B2"):
        check_sentinel_coordinates(
            frames, compact_by_id=compact, coordinate_fn=different_coordinates
        )


def test_three_pass_aggregation_is_deterministic_and_uses_frozen_partitions() -> None:
    partitions = json.loads(
        (ROOT / "benchmarks/m8/preregistration/m8_s2_partitions.json").read_text()
    )
    car = partitions["classes"]["car"]
    identity_rows = [
        item for name in ("shared", "e2_only", "neither") for item in car[name]["identities"]
    ]
    detected = {
        (item["drive_id"], item["frame_index"], item["gt_track_id"])
        for name in ("shared", "e2_only")
        for item in car[name]["identities"]
    }
    passes = [_pass("full-pass", index) for index in (1, 2, 3)]
    for process in passes:
        by_id = {row["condition_id"]: row for row in process["conditions"]}
        for condition in process["conditions"]:
            empty = {
                "thresholded_prediction_count": 0,
                "neighbour_ignore_GT_count": 0,
                "target_observations": [],
                "ranked_dispositions": [],
                "thresholds": {
                    value: {
                        "true_positives": 0,
                        "false_positives": 0,
                        "false_negatives": 0,
                        "ignored_predictions": 0,
                        "matched_gt_identity_set": [],
                    }
                    for value in ("0.30", "0.50", "0.70")
                },
            }
            condition["arm"] = condition["condition_id"].rsplit("/", 1)[1]
            condition["frame_id"] = condition["condition_id"].rsplit("/", 1)[0]
            condition["outside_annotation_fov_prediction_count"] = 0
            condition["predictions"] = []
            condition["classes"] = {"car": deepcopy(empty), "pedestrian": deepcopy(empty)}
        for item in identity_rows:
            frame_id = f"{item['drive_id']}/{item['frame_index']:010d}"
            pose = (item["drive_id"], item["frame_index"], item["gt_track_id"])
            matched = pose in detected
            for arm in ("B2", "C2", "D2", "F2"):
                record = by_id[f"{frame_id}/{arm}"]["classes"]["car"]
                record["target_observations"].append(
                    {
                        "track_id": item["gt_track_id"],
                        "frame_index": item["frame_index"],
                        "gt_identity": f"{item['drive_id']}/track_{item['gt_track_id']}",
                        "range_forward_m": 10.0,
                        "range_band_metres": "0_20",
                        "matched": matched,
                    }
                )
                for threshold in ("0.30", "0.50", "0.70"):
                    summary = record["thresholds"][threshold]
                    summary["true_positives"] += int(matched)
                    summary["false_negatives"] += int(not matched)
                    if matched:
                        summary["matched_gt_identity_set"].append(
                            f"{item['drive_id']}/track_{item['gt_track_id']}"
                        )
                if matched:
                    record["thresholded_prediction_count"] += 1
                    record["ranked_dispositions"].append(
                        {
                            "score": 0.9,
                            "frame_id": frame_id,
                            "prediction_index": len(record["ranked_dispositions"]),
                            "true_positive": True,
                        }
                    )
    passes[0]["conditions"][0]["predictions"] = [
        {"class_name": "car", "score": 0.9, "inside_annotation_fov": False},
        {"class_name": "car", "score": 0.1, "inside_annotation_fov": False},
    ]
    passes[0]["conditions"][0]["outside_annotation_fov_prediction_count"] = 2
    neither = car["neither"]["identities"][0]
    frame_id = f"{neither['drive_id']}/{neither['frame_index']:010d}"
    identity = f"{neither['drive_id']}/track_{neither['gt_track_id']}"
    lower_iou = next(
        row for row in passes[0]["conditions"] if row["condition_id"] == f"{frame_id}/B2"
    )["classes"]["car"]["thresholds"]["0.30"]
    lower_iou["true_positives"] += 1
    lower_iou["false_negatives"] -= 1
    lower_iou["matched_gt_identity_set"].append(identity)
    shared = car["shared"]["identities"][0]
    shared_frame = f"{shared['drive_id']}/{shared['frame_index']:010d}/B2"
    shared_condition = next(
        row for row in passes[0]["conditions"] if row["condition_id"] == shared_frame
    )
    shared_observation = next(
        item
        for item in shared_condition["classes"]["car"]["target_observations"]
        if item["track_id"] == shared["gt_track_id"]
    )
    shared_observation["range_forward_m"] = 55.0
    second_condition = next(
        row for row in passes[1]["conditions"] if row["condition_id"] == f"{frame_id}/B2"
    )
    second_car = second_condition["classes"]["car"]
    second_observation = next(
        item for item in second_car["target_observations"] if item["gt_identity"] == identity
    )
    second_observation["matched"] = True
    for threshold in ("0.30", "0.50", "0.70"):
        second_summary = second_car["thresholds"][threshold]
        second_summary["true_positives"] += 1
        second_summary["false_negatives"] -= 1
        second_summary["matched_gt_identity_set"].append(identity)
    second_car["thresholded_prediction_count"] += 1
    second_car["ranked_dispositions"].append(
        {
            "score": 0.9,
            "frame_id": frame_id,
            "prediction_index": len(second_car["ranked_dispositions"]),
            "true_positive": True,
        }
    )
    second_condition["predictions"].append(
        {"class_name": "car", "score": 0.9, "inside_annotation_fov": True}
    )
    first = aggregate_three_full_passes(passes, repository_root=ROOT)
    second = aggregate_three_full_passes(passes, repository_root=ROOT)
    assert first == second
    assert first["accepted_calls"] == 5136
    assert first["execution_binding"]["runtime_policy_sha256"] == POLICY
    assert [item["attempt_id"] for item in first["source_attempts"]] == [
        "attempt-1",
        "attempt-2",
        "attempt-3",
    ]
    assert first["execution_binding"]["qualification_receipt_sha256"] == QUALIFICATION
    assert first["source_attempts"][0]["authorization_sha256"] == AUTHORIZATION
    assert first["car_recovery_by_pass"][0]["B2"]["G_car"] == 1.0
    assert first["car_recovery_by_pass"][0]["B2"]["R_gain"] == 1.0
    assert first["car_recovery_by_pass"][0]["B2"]["R_Aonly"] is None
    assert first["interpretation_gate_all_three"]["B2"] is True
    assert (
        first["passes"][0]["B2"]["classes"]["car"]["outside_annotation_fov_prediction_count"] == 1
    )
    assert (
        first["passes"][0]["B2"]["classes"]["pedestrian"]["outside_annotation_fov_prediction_count"]
        == 0
    )
    b2_car = first["passes"][0]["B2"]["classes"]["car"]
    assert b2_car["range"]["0.30"]["0_20"]["matched"] == (
        b2_car["range"]["0.50"]["0_20"]["matched"] + 1
    )
    assert b2_car["range"]["0.50"]["0_20"]["targets"] == len(identity_rows) - 1
    range_band = b2_car["range"]["0.50"]["0_20"]
    assert range_band["false_negatives"] == range_band["targets"] - range_band["matched"]
    assert range_band["recall"] == range_band["matched"] / range_band["targets"]
    assert b2_car["range"]["0.50"]["35_50"]["recall"] is None
    assert b2_car["track_continuity"]["0.30"][identity]["matched_poses"] == (
        b2_car["track_continuity"]["0.50"][identity]["matched_poses"] + 1
    )
    continuity_values = first["spread"]["B2"]["classes"]["car"]["track_continuity"]["0.30"][
        identity
    ]["matched_poses"]["pass_values"]
    assert continuity_values[0] == continuity_values[1] == continuity_values[2] + 1
    assert (
        first["spread"]["B2"]["classes"]["car"]["range"]["0.30"]["0_20"]["matched"]["pass_values"][
            0
        ]
        == b2_car["range"]["0.30"]["0_20"]["matched"]
    )
    assert (
        first["spread"]["B2"]["classes"]["car"]["range"]["0.50"]["0_20"]["recall"]["pass_values"][0]
        == range_band["recall"]
    )
    assert first["car_recovery_spread"]["B2"]["G_car"]["pass_values"] == [
        1.0,
        25 / 24,
        1.0,
    ]
    assert first["car_recovery_spread"]["B2"]["R_Aonly"]["median"] is None
    assert (
        first["spread"]["B2"]["classes"]["car"]["range"]["0.50"]["35_50"]["recall"]["median"]
        is None
    )
    assert first["car_factorial_by_pass"][0]["Car_TP"] == {"L": 12.0, "P": 12.0, "I": -24.0}
    assert "confidence_interval" not in json.dumps(first)
    assert "p_value" not in json.dumps(first)
