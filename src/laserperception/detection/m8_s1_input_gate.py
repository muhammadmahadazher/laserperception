"""CPU-only receipt for the complete M8 S1 frozen-input gate.

The module deliberately imports no detector, Torch, CUDA, ground-truth, or
evaluation code.  A receipt proves that the official reconstruction command
replayed every frozen H10/H5 input for one exact repository execution commit.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Protocol

from laserperception.detection.m8_s1_runtime import (
    CANDIDATE_MANIFEST_SHA256,
    INPUT_LEDGER_PATH,
    INPUT_LEDGER_SHA256,
    INPUT_REVALIDATION_SHA256,
    ORDERED_FRAME_SHA256,
    PROTOCOL_JSON_SHA256,
    M8S1ProtocolViolation,
    atomic_write_json,
    canonical_condition_ids,
    canonical_frame_ids,
    canonical_json_sha256,
    sha256_file,
)

INPUT_GATE_RECEIPT_SCHEMA = "laserperception.m8.s1.input-gate-receipt.v1"
INPUT_GATE_IMPLEMENTATION = (
    "scripts/detection/revalidate_m8_input_projection.py@repository_execution_commit"
)
EXPECTED_CONDITIONS = 856
EXPECTED_PER_HISTORY = 428
DEFAULT_PRIMARY_INPUT_REVALIDATION_WORKERS = 4
MAX_PRIMARY_INPUT_REVALIDATION_WORKERS = 8
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "repository_execution_commit",
        "gate_implementation_identity",
        "generated_at_utc",
        "protocol_json_sha256",
        "accepted_input_ledger_sha256",
        "frozen_input_revalidation_sha256",
        "full_transform_ledger_sha256",
        "ordered_frame_identity",
        "candidate_feature_contract_identity",
        "source_manifest_identity",
        "conditions_exact",
        "H10_exact",
        "H5_exact",
        "mismatch_count",
        "complete",
        "ordered_condition_evidence_identity",
        "gate_result",
        "gate_result_sha256",
        "receipt_sha256",
    }
)


def validate_primary_input_revalidation_workers(value: object) -> int:
    """Return a bounded CPU worker count for primary pre-inference validation."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise M8S1ProtocolViolation("primary input-revalidation workers must be an integer")
    if not 1 <= value <= MAX_PRIMARY_INPUT_REVALIDATION_WORKERS:
        raise M8S1ProtocolViolation(
            "primary input-revalidation workers must be between 1 and "
            f"{MAX_PRIMARY_INPUT_REVALIDATION_WORKERS}"
        )
    return value


class _PrimaryInputSource(Protocol):
    def pair(self, frame_id: str) -> tuple[tuple[object, Mapping[str, object]], ...]: ...

    def isolated(self) -> _PrimaryInputSource: ...


def _validated_primary_pair(
    source: _PrimaryInputSource, frame_id: str
) -> tuple[dict[str, object], dict[str, object]]:
    pair = source.pair(frame_id)
    expected = (f"{frame_id}/H10", f"{frame_id}/H5")
    actual = tuple(str(identity.get("condition_id")) for _, identity in pair)
    histories = tuple(str(identity.get("history")) for _, identity in pair)
    if len(pair) != 2 or actual != expected or histories != ("H10", "H5"):
        raise RuntimeError("primary pre-inference pair order changed")
    return dict(pair[0][1]), dict(pair[1][1])


def _reconstruct_primary_chunk(
    job: tuple[_PrimaryInputSource, tuple[str, ...]],
) -> list[tuple[dict[str, object], dict[str, object]]]:
    source, chunk = job
    return [_validated_primary_pair(source, frame_id) for frame_id in chunk]


def revalidate_primary_inputs(
    source: _PrimaryInputSource, *, worker_count: int
) -> dict[str, object]:
    """Verify all primary inputs before inference with worker-local CPU state."""

    workers = validate_primary_input_revalidation_workers(worker_count)
    frame_ids = canonical_frame_ids()
    if workers == 1:
        pairs = [_validated_primary_pair(source, frame_id) for frame_id in frame_ids]
    else:
        chunk_size = (len(frame_ids) + workers - 1) // workers
        frame_chunks = tuple(
            frame_ids[start : start + chunk_size] for start in range(0, len(frame_ids), chunk_size)
        )

        jobs = tuple((source.isolated(), chunk) for chunk in frame_chunks)
        with ProcessPoolExecutor(
            max_workers=len(jobs),
            mp_context=get_context("spawn"),
        ) as executor:
            chunk_results = list(executor.map(_reconstruct_primary_chunk, jobs))
        pairs = [pair for chunk in chunk_results for pair in chunk]
    records = [record for pair in pairs for record in pair]
    condition_ids = [str(record["condition_id"]) for record in records]
    histories = [str(record["history"]) for record in records]
    h10 = histories.count("H10")
    h5 = histories.count("H5")
    if (
        len(pairs),
        len(records),
        h10,
        h5,
        tuple(condition_ids),
    ) != (428, 856, 428, 428, canonical_condition_ids()):
        raise RuntimeError("primary pre-inference corpus revalidation is incomplete")
    canonical_evidence: dict[str, object] = {
        "frames_exact": len(pairs),
        "H10_exact": h10,
        "H5_exact": h5,
        "conditions_exact": len(records),
        "mismatch_count": 0,
        "condition_order_exact": True,
        "condition_ids": condition_ids,
        "records": records,
    }
    evidence: dict[str, object] = {
        "schema_version": "laserperception.m8.s1.primary-preinference-inputs.v1",
        "worker_count": workers,
        "execution_model": "serial" if workers == 1 else "spawned-processes",
        **canonical_evidence,
        "canonical_input_sha256": canonical_json_sha256(canonical_evidence),
    }
    evidence["result_sha256"] = canonical_json_sha256(evidence)
    return evidence


def _git_head(repository_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _lower_git_sha(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise M8S1ProtocolViolation(f"{name} must be a lowercase full Git SHA")
    return value


def _load_mapping(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise M8S1ProtocolViolation(f"cannot read M8 input-gate artifact: {path}") from error
    if not isinstance(payload, dict):
        raise M8S1ProtocolViolation(f"{path.name} must contain a JSON object")
    return payload


def _full_ledger_bindings(full_ledger: Path) -> tuple[str, str, str]:
    payload = _load_mapping(full_ledger)
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise M8S1ProtocolViolation("full transform ledger frames are malformed")
    frame_ids: list[str] = []
    for frame in frames:
        if not isinstance(frame, Mapping):
            raise M8S1ProtocolViolation("full transform ledger frame is malformed")
        frame_id = frame.get("frame_id")
        frozen = frame.get("frozen_sweep_transforms")
        if not isinstance(frame_id, str) or not isinstance(frozen, list):
            raise M8S1ProtocolViolation("full transform ledger identity is malformed")
        frame_ids.append(frame_id)
    if tuple(frame_ids) != canonical_frame_ids():
        raise M8S1ProtocolViolation("full transform ledger ordered frames changed")
    ordered_frame_identity = hashlib.sha256(("\n".join(frame_ids) + "\n").encode()).hexdigest()
    if ordered_frame_identity != ORDERED_FRAME_SHA256:
        raise M8S1ProtocolViolation("full transform ledger ordered-frame identity changed")
    ledger_identity = sha256_file(full_ledger)
    return ledger_identity, ledger_identity, ordered_frame_identity


def _ordered_condition_identity() -> str:
    return hashlib.sha256(("\n".join(canonical_condition_ids()) + "\n").encode()).hexdigest()


def _feature_contract_identity() -> str:
    return canonical_json_sha256(
        {
            "candidate_manifest_sha256": CANDIDATE_MANIFEST_SHA256,
            "dtype": "float32",
            "features": ["x", "y", "z", "intensity", "time_lag"],
            "point_order": "frozen source-row order",
        }
    )


def create_input_gate_receipt(
    *,
    repository_root: Path,
    full_ledger: Path,
    execution_commit: str,
    gate_result: Mapping[str, object],
) -> dict[str, object]:
    """Create a complete receipt only from a successful official 856 replay."""

    commit = _lower_git_sha(execution_commit, "execution_commit")
    if _git_head(repository_root.resolve()) != commit:
        raise M8S1ProtocolViolation("input-gate execution commit differs from repository HEAD")
    expected_result = {
        "result": "PASS",
        "detector_inference_performed": False,
        "ground_truth_loaded": False,
        "conditions_checked": EXPECTED_CONDITIONS,
        "H10_exact_count": EXPECTED_PER_HISTORY,
        "H5_exact_count": EXPECTED_PER_HISTORY,
        "full_XYZIT_exact_count": EXPECTED_CONDITIONS,
        "XYZT_projection_exact_count": EXPECTED_CONDITIONS,
        "intensity_exact_count": EXPECTED_CONDITIONS,
        "range_drop_exact_count": EXPECTED_CONDITIONS,
        "mismatch_count": 0,
        "mismatch_ids": [],
        "final_implementation_commit": commit,
    }
    for name, expected in expected_result.items():
        if gate_result.get(name) != expected:
            raise M8S1ProtocolViolation(f"input-gate result is incomplete or invalid: {name}")
    existing = gate_result.get("existing_ledger")
    source = gate_result.get("source_m6b_full_ledger")
    if not isinstance(existing, Mapping) or existing.get("sha256") != INPUT_LEDGER_SHA256:
        raise M8S1ProtocolViolation("input-gate result accepted-ledger identity changed")
    source_identity, transform_identity, ordered_frames = _full_ledger_bindings(full_ledger)
    if not isinstance(source, Mapping) or source.get("sha256") != source_identity:
        raise M8S1ProtocolViolation("input-gate result source identity changed")
    result_identity = canonical_json_sha256(gate_result)
    receipt: dict[str, object] = {
        "schema_version": INPUT_GATE_RECEIPT_SCHEMA,
        "repository_execution_commit": commit,
        "gate_implementation_identity": INPUT_GATE_IMPLEMENTATION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_json_sha256": PROTOCOL_JSON_SHA256,
        "accepted_input_ledger_sha256": INPUT_LEDGER_SHA256,
        "frozen_input_revalidation_sha256": INPUT_REVALIDATION_SHA256,
        "full_transform_ledger_sha256": transform_identity,
        "ordered_frame_identity": ordered_frames,
        "candidate_feature_contract_identity": _feature_contract_identity(),
        "source_manifest_identity": source_identity,
        "conditions_exact": EXPECTED_CONDITIONS,
        "H10_exact": EXPECTED_PER_HISTORY,
        "H5_exact": EXPECTED_PER_HISTORY,
        "mismatch_count": 0,
        "complete": True,
        "ordered_condition_evidence_identity": _ordered_condition_identity(),
        "gate_result": dict(gate_result),
        "gate_result_sha256": result_identity,
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    return receipt


def write_input_gate_receipt(path: Path, receipt: Mapping[str, object]) -> None:
    """Atomically persist a validated, deterministic JSON receipt."""

    verify_input_gate_receipt_payload(receipt)
    atomic_write_json(path, receipt)


def verify_input_gate_receipt_payload(receipt: Mapping[str, object]) -> str:
    """Validate the closed receipt schema and its self-identity."""

    if set(receipt) != _RECEIPT_FIELDS:
        raise M8S1ProtocolViolation("M8 S1 input-gate receipt fields differ")
    if receipt.get("schema_version") != INPUT_GATE_RECEIPT_SCHEMA:
        raise M8S1ProtocolViolation("M8 S1 input-gate receipt schema is invalid")
    expected = {
        "gate_implementation_identity": INPUT_GATE_IMPLEMENTATION,
        "protocol_json_sha256": PROTOCOL_JSON_SHA256,
        "accepted_input_ledger_sha256": INPUT_LEDGER_SHA256,
        "frozen_input_revalidation_sha256": INPUT_REVALIDATION_SHA256,
        "ordered_frame_identity": ORDERED_FRAME_SHA256,
        "candidate_feature_contract_identity": _feature_contract_identity(),
        "conditions_exact": EXPECTED_CONDITIONS,
        "H10_exact": EXPECTED_PER_HISTORY,
        "H5_exact": EXPECTED_PER_HISTORY,
        "mismatch_count": 0,
        "complete": True,
        "ordered_condition_evidence_identity": _ordered_condition_identity(),
    }
    for name, value in expected.items():
        if receipt.get(name) != value:
            raise M8S1ProtocolViolation(f"M8 S1 input-gate receipt binding mismatch: {name}")
    _lower_git_sha(receipt.get("repository_execution_commit"), "repository_execution_commit")
    for name in (
        "full_transform_ledger_sha256",
        "source_manifest_identity",
        "gate_result_sha256",
        "receipt_sha256",
    ):
        value = receipt.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise M8S1ProtocolViolation(f"M8 S1 input-gate receipt identity is invalid: {name}")
    generated = receipt.get("generated_at_utc")
    if not isinstance(generated, str) or not generated:
        raise M8S1ProtocolViolation("M8 S1 input-gate receipt timestamp is absent")
    gate_result = receipt.get("gate_result")
    if not isinstance(gate_result, Mapping):
        raise M8S1ProtocolViolation("M8 S1 input-gate receipt result is malformed")
    if canonical_json_sha256(gate_result) != receipt.get("gate_result_sha256"):
        raise M8S1ProtocolViolation("M8 S1 input-gate result SHA256 mismatch")
    exact_gate_result = {
        "result": "PASS",
        "detector_inference_performed": False,
        "ground_truth_loaded": False,
        "conditions_checked": EXPECTED_CONDITIONS,
        "H10_exact_count": EXPECTED_PER_HISTORY,
        "H5_exact_count": EXPECTED_PER_HISTORY,
        "full_XYZIT_exact_count": EXPECTED_CONDITIONS,
        "XYZT_projection_exact_count": EXPECTED_CONDITIONS,
        "intensity_exact_count": EXPECTED_CONDITIONS,
        "range_drop_exact_count": EXPECTED_CONDITIONS,
        "mismatch_count": 0,
        "mismatch_ids": [],
        "final_implementation_commit": receipt.get("repository_execution_commit"),
    }
    for name, value in exact_gate_result.items():
        if gate_result.get(name) != value:
            raise M8S1ProtocolViolation(f"M8 S1 input-gate result mismatch: {name}")
    unsigned = dict(receipt)
    receipt_identity = unsigned.pop("receipt_sha256")
    if canonical_json_sha256(unsigned) != receipt_identity:
        raise M8S1ProtocolViolation("M8 S1 input-gate receipt SHA256 mismatch")
    return str(receipt_identity)


def verify_input_gate_receipt(
    path: Path,
    *,
    repository_root: Path,
    full_ledger: Path,
    execution_commit: str,
) -> str:
    """Verify a receipt against live Git and frozen source bindings."""

    receipt = _load_mapping(path)
    verify_input_gate_receipt_payload(receipt)
    commit = _lower_git_sha(execution_commit, "execution_commit")
    if receipt.get("repository_execution_commit") != commit or _git_head(repository_root) != commit:
        raise M8S1ProtocolViolation("M8 S1 input-gate receipt execution commit changed")
    source_identity, transform_identity, ordered_frames = _full_ledger_bindings(full_ledger)
    live = {
        "source_manifest_identity": source_identity,
        "full_transform_ledger_sha256": transform_identity,
        "ordered_frame_identity": ordered_frames,
        "accepted_input_ledger_sha256": sha256_file(repository_root / INPUT_LEDGER_PATH),
    }
    for name, value in live.items():
        if receipt.get(name) != value:
            raise M8S1ProtocolViolation(f"M8 S1 input-gate receipt live mismatch: {name}")
    return sha256_file(path)
