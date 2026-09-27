"""CPU-only contracts for the future S2 execution path; no science imports."""

from __future__ import annotations

import json
import runpy
import sys
import uuid
from copy import deepcopy
from pathlib import Path

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
    condition_ids,
    verify_authorization,
)
from laserperception.evaluation.m8_s2_aggregation import (
    aggregate_three_full_passes,
    factorial_contrasts,
    review_repeatability,
    validate_attempts,
)

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40
POLICY = "b" * 64
RECEIPT = "c" * 64


def _pass(mode: str, index: int) -> dict[str, object]:
    logical = f"s2-repeatability-{index:02d}" if mode == "repeatability" else f"s2-pass-{index}"
    order = condition_ids(mode)
    return {
        "status": "COMPLETE",
        "mode": mode,
        "logical_pass_id": logical,
        "process_uuid": str(uuid.UUID(int=index)),
        "attempt_id": f"attempt-{index}",
        "execution_commit": COMMIT,
        "runtime_policy_sha256": POLICY,
        "input_gate_receipt_sha256": RECEIPT,
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
    )
    attempt = AtomicAttempt(tmp_path / "attempt", identity)
    with pytest.raises(M8S2ProtocolViolation, match="order"):
        attempt.record(condition_ids("repeatability")[1], {})
    attempt.record(
        condition_ids("repeatability")[0], {"condition_id": condition_ids("repeatability")[0]}
    )
    failed = attempt.fail("synthetic failure")
    assert failed["accepted_canonical_calls"] == 0
    assert failed["completed_calls"] == 1
    with pytest.raises(M8S2ProtocolViolation, match="incomplete"):
        attempt.finalize()
    assert not (tmp_path / "attempt" / "final_pass_manifest.json").exists()


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
    passes[1]["conditions"][0]["classes"]["car"]["thresholds"]["0.50"]["true_positives"] = 1
    with pytest.raises(M8S2ProtocolViolation, match="repeatability differs"):
        review_repeatability(passes)
    with pytest.raises(M8S2ProtocolViolation, match="process count"):
        review_repeatability(passes[:9])


def test_authorization_scopes_fail_closed() -> None:
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
    first = aggregate_three_full_passes(passes, repository_root=ROOT)
    second = aggregate_three_full_passes(passes, repository_root=ROOT)
    assert first == second
    assert first["accepted_calls"] == 5136
    assert first["car_recovery_by_pass"][0]["B2"]["G_car"] == 1.0
    assert first["car_recovery_by_pass"][0]["B2"]["R_gain"] == 1.0
    assert first["car_recovery_by_pass"][0]["B2"]["R_Aonly"] is None
    assert first["interpretation_gate_all_three"]["B2"] is True
    assert first["car_factorial_by_pass"][0]["Car_TP"] == {"L": 12.0, "P": 12.0, "I": -24.0}
    assert "confidence_interval" not in json.dumps(first)
    assert "p_value" not in json.dumps(first)
