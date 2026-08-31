from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

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
    verify_scientific_authorization,
    verify_static_bindings,
    verify_three_process_realizations,
    zero_intensity_copy,
)

ROOT = Path(__file__).resolve().parents[1]


def _authorization(expected: AuthorizationIdentity) -> dict[str, object]:
    return {
        "schema_version": "laserperception.m8.s1.authorization.v1",
        "scientific_inference_authorized": True,
        "authorization_role": "owner_scientific_inference_authorization",
        "owner_approval": True,
        "authorization_timestamp_utc": "2099-01-01T00:00:00Z",
        "authorization_provenance": "future owner decision",
        **expected.to_dict(),
    }


@pytest.mark.parametrize("mode", ["stage-r", "primary-pass", "zero-intensity-pass"])
def test_scientific_modes_refuse_before_backend_initialization(mode: str) -> None:
    initialized = False

    def factory() -> object:
        nonlocal initialized
        initialized = True
        return object()

    expected = AuthorizationIdentity("a" * 40, "runtime-binding")
    with pytest.raises(M8S1ProtocolViolation, match="normally disabled"):
        authorize_then_construct(mode, None, expected, factory)
    assert initialized is False


def test_future_authorization_binds_runtime_implementation(tmp_path: Path) -> None:
    expected = AuthorizationIdentity("a" * 40, "runtime-binding")
    authorization = _authorization(expected)
    verify_scientific_authorization(authorization, expected)
    authorization["measurement_runtime_reviewed_commit"] = "b" * 40
    with pytest.raises(M8S1ProtocolViolation, match="measurement_runtime_reviewed_commit"):
        verify_scientific_authorization(authorization, expected)
    assert not (tmp_path / "m8_s1_inference_authorization.json").exists()


def test_authorization_file_is_exact_schema_and_has_no_bypass(tmp_path: Path) -> None:
    expected = AuthorizationIdentity("a" * 40, "runtime-binding")
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps({**_authorization(expected), "skip_auth": True}), encoding="utf-8")
    with pytest.raises(M8S1ProtocolViolation, match="schema fields differ"):
        require_scientific_authorization("primary-pass", path, expected)


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
