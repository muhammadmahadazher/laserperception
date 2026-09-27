"""Streaming S2 full-ledger receipt and live consumed-input identity checks.

This CPU-only module reads frozen input evidence; it never imports a detector,
ground truth, Torch, or a provider API.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from laserperception.detection.m8_s1_runtime import (
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
)
from laserperception.detection.m8_s2_input import array_sha256, canonical_xyzit, canonical_xyzt
from laserperception.detection.m8_s2_runtime import (
    ARMS,
    COMPACT_MANIFEST_PATH,
    COMPACT_MANIFEST_SHA256,
    FULL_LEDGER_BYTES,
    FULL_LEDGER_SHA256,
    INPUT_FREEZE_SHA256,
    INPUT_IMPLEMENTATION_COMMIT,
    M7_MANIFEST_SHA256,
    PARTITIONS_SHA256,
    PROTOCOL_SHA256,
    M8S2ProtocolViolation,
    condition_ids,
    verify_static_bindings,
)

RECEIPT_SCHEMA = "laserperception.m8.s2.input-gate-receipt.v1"


def _compact(root: Path) -> list[dict[str, object]]:
    path = root / COMPACT_MANIFEST_PATH
    if sha256_file(path) != COMPACT_MANIFEST_SHA256:
        raise M8S2ProtocolViolation("S2 compact manifest SHA256 differs")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("conditions")
    if not isinstance(rows, list) or payload.get("condition_count") != 1712:
        raise M8S2ProtocolViolation("S2 compact manifest count differs")
    if tuple(row.get("condition_id") for row in rows) != condition_ids("full-pass"):
        raise M8S2ProtocolViolation("S2 compact manifest order differs")
    return rows


def stream_full_ledger(root: Path, path: Path) -> dict[str, object]:
    """Hash and compare one JSONL record at a time to all compact fields."""

    compact = _compact(root)
    try:
        size = path.stat().st_size
    except OSError as error:
        raise M8S2ProtocolViolation("S2 private full ledger is missing") from error
    if size != FULL_LEDGER_BYTES:
        raise M8S2ProtocolViolation("S2 full-ledger byte count differs")
    digest = hashlib.sha256()
    count = 0
    by_arm = {arm: 0 for arm in ARMS}
    with path.open("rb") as source:
        for line in source:
            digest.update(line)
            if count >= len(compact):
                raise M8S2ProtocolViolation("S2 full ledger contains an extra condition")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise M8S2ProtocolViolation("S2 full ledger JSONL is malformed") from error
            expected = compact[count]
            if not isinstance(record, dict) or any(
                record.get(key) != value for key, value in expected.items()
            ):
                raise M8S2ProtocolViolation(f"S2 full/compact record differs at index {count}")
            if record.get("implementation_commit") != INPUT_IMPLEMENTATION_COMMIT:
                raise M8S2ProtocolViolation("S2 ledger implementation identity differs")
            arm = record.get("arm")
            if arm not in by_arm:
                raise M8S2ProtocolViolation("S2 ledger arm differs")
            by_arm[str(arm)] += 1
            count += 1
    if (
        count != 1712
        or digest.hexdigest() != FULL_LEDGER_SHA256
        or any(value != 428 for value in by_arm.values())
    ):
        raise M8S2ProtocolViolation("S2 full-ledger count or SHA256 differs")
    return {
        "bytes": size,
        "sha256": digest.hexdigest(),
        "record_count": count,
        "arm_counts": by_arm,
    }


def make_input_gate_receipt(root: Path, ledger: Path, execution_commit: str) -> dict[str, object]:
    """Produce a non-authorizing receipt for one exact external-worker checkout."""

    verify_static_bindings(root, execution_commit)
    full = stream_full_ledger(root, ledger)
    receipt: dict[str, object] = {
        "schema_version": RECEIPT_SCHEMA,
        "execution_commit": execution_commit,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": PROTOCOL_SHA256,
        "partitions_sha256": PARTITIONS_SHA256,
        "input_freeze_sha256": INPUT_FREEZE_SHA256,
        "full_ledger": full,
        "compact_manifest_sha256": COMPACT_MANIFEST_SHA256,
        "input_implementation_commit": INPUT_IMPLEMENTATION_COMMIT,
        "m7_manifest_sha256": M7_MANIFEST_SHA256,
        "condition_order": "frame-major:B2,C2,D2,F2",
        "conditions_exact": 1712,
        "complete": True,
        "inference_authorized": False,
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    return receipt


def write_input_gate_receipt(path: Path, receipt: Mapping[str, object]) -> None:
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise M8S2ProtocolViolation("S2 input-gate receipt schema differs")
    atomic_write_json(path, receipt)


def verify_input_gate_receipt(
    path: Path, *, root: Path, ledger: Path, execution_commit: str
) -> str:
    """Recheck receipt, live ledger, and static identities before any science import."""

    verify_static_bindings(root, execution_commit)
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise M8S2ProtocolViolation("S2 input-gate receipt is unavailable") from error
    if not isinstance(receipt, dict):
        raise M8S2ProtocolViolation("S2 input-gate receipt is malformed")
    unsigned = dict(receipt)
    receipt_sha = unsigned.pop("receipt_sha256", None)
    if receipt_sha != canonical_json_sha256(unsigned):
        raise M8S2ProtocolViolation("S2 input-gate receipt self-identity differs")
    expected = {
        "schema_version": RECEIPT_SCHEMA,
        "execution_commit": execution_commit,
        "protocol_sha256": PROTOCOL_SHA256,
        "partitions_sha256": PARTITIONS_SHA256,
        "input_freeze_sha256": INPUT_FREEZE_SHA256,
        "compact_manifest_sha256": COMPACT_MANIFEST_SHA256,
        "input_implementation_commit": INPUT_IMPLEMENTATION_COMMIT,
        "m7_manifest_sha256": M7_MANIFEST_SHA256,
        "condition_order": "frame-major:B2,C2,D2,F2",
        "conditions_exact": 1712,
        "complete": True,
        "inference_authorized": False,
        "full_ledger": stream_full_ledger(root, ledger),
    }
    if set(receipt) != set(expected) | {"created_at_utc", "receipt_sha256"} or any(
        receipt.get(key) != value for key, value in expected.items()
    ):
        raise M8S2ProtocolViolation("S2 input-gate receipt binding differs")
    if not isinstance(receipt.get("created_at_utc"), str) or not receipt["created_at_utc"]:
        raise M8S2ProtocolViolation("S2 input-gate receipt timestamp is missing")
    return sha256_file(path)


def verify_consumed_input(
    condition_id: str,
    points: np.ndarray,
    selected_row_sha256: str,
    expected: Mapping[str, object],
) -> dict[str, object]:
    """Verify the same array that a future detector call will consume."""

    canonical = canonical_xyzit(points)
    actual = {
        "condition_id": condition_id,
        "full_XYZIT_sha256": array_sha256(canonical),
        "XYZT_projection_sha256": array_sha256(canonical_xyzt(canonical)),
        "selected_global_row_sha256": selected_row_sha256,
        "point_count": len(canonical),
    }
    for key, value in actual.items():
        if expected.get(key) != value:
            raise M8S2ProtocolViolation(f"S2 consumed input differs: {condition_id}/{key}")
    if actual["XYZT_projection_sha256"] != expected.get("M7_expected_XYZT_sha256"):
        raise M8S2ProtocolViolation("S2 consumed M7 XYZT identity differs")
    return {
        "frozen_ledger_record_identity": canonical_json_sha256(expected),
        "input_sha256": actual["full_XYZIT_sha256"],
        "M7_XYZT_sha256": actual["XYZT_projection_sha256"],
        "selected_global_row_sha256": selected_row_sha256,
    }
