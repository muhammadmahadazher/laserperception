"""CPU-safe contracts for a future, separately authorized M8 S2 runtime.

Importing this module never discovers an accelerator, imports a detector, or
loads ground truth. Scientific work is disabled without exact external records.
"""

from __future__ import annotations

import json
import re
import socket
import subprocess
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from laserperception.detection.m8_s1_runtime import (
    CANDIDATE_MANIFEST_PATH,
    CANDIDATE_MANIFEST_SHA256,
    atomic_write_json,
    canonical_frame_ids,
    canonical_json_sha256,
    sha256_file,
)

PROTOCOL_PATH = Path("docs/m8/M8_S2_PROTOCOL.md")
PROTOCOL_SHA256 = "218ef2dcc03fa4ff75562e02f368f7624f16a4c768b1758147c1d46ac1d9c53d"
PROTOCOL_JSON_PATH = Path("benchmarks/m8/preregistration/m8_s2_protocol.json")
PROTOCOL_JSON_SHA256 = "341da352f2684d7eb85f3d43112cd88bd4652009463cb59163c67ebd79748720"
PARTITIONS_PATH = Path("benchmarks/m8/preregistration/m8_s2_partitions.json")
PARTITIONS_SHA256 = "f52b0013d10d38c4fd91f52cfb42205f52b97ecf4433cd86654872dd70a59d6e"
INPUT_FREEZE_PATH = Path("benchmarks/m8/preregistration/m8_s2_input_freeze.json")
INPUT_FREEZE_SHA256 = "c08589bb9600633d5a8b675a16f8be805697a5302296ff52608e5845a3e371db"
COMPACT_MANIFEST_PATH = Path("benchmarks/m8/inputs/m8_s2_input_manifest.json")
COMPACT_MANIFEST_SHA256 = "239b563d5f850f2f20809950eca9d56f1a700677766afce8489099e940554ecc"
FULL_LEDGER_SHA256 = "a3ed54b276f77fb784035045b079573cd4e4ddfedc9d0f8eb774c1340a59396b"
FULL_LEDGER_BYTES = 7_728_782
INPUT_IMPLEMENTATION_COMMIT = "bf098b319744f1ec1df08207c1cd93853b1f31ae"
M7_MANIFEST_SHA256 = "8d4f74d783950d24956239f3a67a7a58fe10013e0e83a88d0f8b23e3139ffe90"
ARMS = ("B2", "C2", "D2", "F2")
SENTINELS = (
    "2011_09_26_drive_0001/0000000010",
    "2011_09_26_drive_0001/0000000011",
    "2011_09_26_drive_0001/0000000015",
    "2011_09_26_drive_0001/0000000083",
    "2011_09_26_drive_0091/0000000010",
    "2011_09_26_drive_0091/0000000011",
    "2011_09_26_drive_0091/0000000012",
)
AUTHORIZATION_SCHEMA = "laserperception.m8.s2.authorization.v1"
QUALIFICATION_RECEIPT_SCHEMA = "laserperception.m8.s2.qualification-receipt.v1"
RUNTIME_POLICY_SCHEMA = "laserperception.m8.s2.runtime-policy-binding.v1"
ATTEMPT_SCHEMA = "laserperception.m8.s2.attempt.v1"
CONDITION_SCHEMA = "laserperception.m8.s2.condition.v1"
REPEATABILITY_IDS = tuple(f"s2-repeatability-{index:02d}" for index in range(1, 11))
FULL_PASS_IDS = ("s2-pass-1", "s2-pass-2", "s2-pass-3")
SCOPES = frozenset({"qualification-only", "repeatability-only", "full-pass-only"})


class M8S2ProtocolViolation(ValueError):
    """A frozen S2 boundary was not satisfied."""


def _sha(value: object, label: str, *, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]+", value) is None
        or len(value) != length
    ):
        raise M8S2ProtocolViolation(f"{label} must be a lowercase {length}-character SHA")
    return value


def _mapping(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise M8S2ProtocolViolation(f"cannot read S2 artifact: {path}") from error
    if not isinstance(value, dict):
        raise M8S2ProtocolViolation(f"S2 artifact must be a JSON object: {path}")
    return value


def git_head(root: Path) -> str:
    """Read exact HEAD without relying on a branch name."""

    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _committed_bytes(root: Path, relative: Path) -> bytes:
    name = relative.as_posix()
    if subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", name], cwd=root, check=False
    ).returncode:
        raise M8S2ProtocolViolation(f"frozen S2 artifact has local changes: {name}")
    return subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=root)


def verify_static_bindings(root: Path, execution_commit: str) -> dict[str, object]:
    """Validate reviewed Git, protocol, partitions, input freeze, and candidate."""

    root = root.resolve()
    _sha(execution_commit, "execution commit", length=40)
    if git_head(root) != execution_commit:
        raise M8S2ProtocolViolation("S2 execution commit differs from repository HEAD")
    frozen = (
        (PROTOCOL_PATH, PROTOCOL_SHA256),
        (PROTOCOL_JSON_PATH, PROTOCOL_JSON_SHA256),
        (PARTITIONS_PATH, PARTITIONS_SHA256),
        (INPUT_FREEZE_PATH, INPUT_FREEZE_SHA256),
        (COMPACT_MANIFEST_PATH, COMPACT_MANIFEST_SHA256),
        (CANDIDATE_MANIFEST_PATH, CANDIDATE_MANIFEST_SHA256),
    )
    import hashlib

    for relative, expected in frozen:
        if hashlib.sha256(_committed_bytes(root, relative)).hexdigest() != expected:
            raise M8S2ProtocolViolation(f"frozen S2 identity differs: {relative}")
    freeze = _mapping(root / INPUT_FREEZE_PATH)
    if any(
        freeze.get(key) is not True
        for key in ("implementation_frozen", "input_ledger_frozen", "replay_28_exact")
    ) or any(freeze.get(key) is not False for key in ("runtime_bound", "inference_authorized")):
        raise M8S2ProtocolViolation("frozen input authorization state differs")
    if freeze.get("implementation_commit") != INPUT_IMPLEMENTATION_COMMIT:
        raise M8S2ProtocolViolation("S2 input implementation identity differs")
    ledger = freeze.get("full_ledger")
    if not isinstance(ledger, Mapping) or ledger != {
        "logical_filename": "m8_s2_input_ledger.jsonl",
        "bytes": FULL_LEDGER_BYTES,
        "sha256": FULL_LEDGER_SHA256,
        "record_count": 1712,
    }:
        raise M8S2ProtocolViolation("S2 full-ledger identity differs")
    return {"execution_commit": execution_commit, "input_freeze": freeze}


def condition_ids(mode: str) -> tuple[str, ...]:
    """Freeze frame-major B2/C2/D2/F2 order for one process."""

    frames: tuple[str, ...]
    if mode == "repeatability":
        frames = SENTINELS
    elif mode == "full-pass":
        frames = canonical_frame_ids()
    else:
        raise M8S2ProtocolViolation(f"unknown S2 scientific mode: {mode}")
    result = tuple(f"{frame}/{arm}" for frame in frames for arm in ARMS)
    if len(result) != (28 if mode == "repeatability" else 1712) or len(set(result)) != len(result):
        raise AssertionError("S2 condition order is inconsistent")
    return result


def logical_pass_ids(mode: str) -> tuple[str, ...]:
    if mode == "repeatability":
        return REPEATABILITY_IDS
    if mode == "full-pass":
        return FULL_PASS_IDS
    raise M8S2ProtocolViolation(f"unknown S2 scientific mode: {mode}")


def verify_authorization(
    payload: Mapping[str, object],
    *,
    scope: str,
    execution_commit: str,
    logical_pass_id: str | None,
    runtime_policy_sha256: str | None = None,
    input_gate_receipt_sha256: str | None = None,
    qualification_receipt_sha256: str | None = None,
    repeatability_review_sha256: str | None = None,
) -> None:
    """Require one exact owner-issued scope; scopes never imply one another."""

    if scope not in SCOPES:
        raise M8S2ProtocolViolation("unknown S2 authorization scope")
    required = {
        "schema_version",
        "authorized",
        "scope",
        "owner_approval",
        "authorization_id",
        "authorization_timestamp_utc",
        "authorization_provenance",
        "authorized_gpu_uuid",
        "authorized_worker_hostname",
        "execution_commit",
        "protocol_sha256",
        "partitions_sha256",
        "input_freeze_sha256",
        "full_ledger_sha256",
        "compact_manifest_sha256",
        "logical_pass_ids",
        "runtime_policy_binding_sha256",
        "input_gate_receipt_sha256",
        "qualification_receipt_sha256",
        "repeatability_review_sha256",
    }
    if set(payload) != required or payload.get("schema_version") != AUTHORIZATION_SCHEMA:
        raise M8S2ProtocolViolation("S2 authorization schema differs")
    if payload.get("authorized") is not True or payload.get("owner_approval") is not True:
        raise M8S2ProtocolViolation("S2 owner authorization is absent")
    if payload.get("scope") != scope:
        raise M8S2ProtocolViolation("S2 authorization scope differs")
    for key in ("authorization_id", "authorization_timestamp_utc", "authorization_provenance"):
        if not isinstance(payload.get(key), str) or not str(payload[key]).strip():
            raise M8S2ProtocolViolation(f"S2 authorization {key} is absent")
    fixed = {
        "execution_commit": execution_commit,
        "protocol_sha256": PROTOCOL_SHA256,
        "partitions_sha256": PARTITIONS_SHA256,
        "input_freeze_sha256": INPUT_FREEZE_SHA256,
        "full_ledger_sha256": FULL_LEDGER_SHA256,
        "compact_manifest_sha256": COMPACT_MANIFEST_SHA256,
        "runtime_policy_binding_sha256": runtime_policy_sha256,
        "input_gate_receipt_sha256": input_gate_receipt_sha256,
        "qualification_receipt_sha256": qualification_receipt_sha256,
        "repeatability_review_sha256": repeatability_review_sha256,
    }
    if any(payload.get(key) != value for key, value in fixed.items()):
        raise M8S2ProtocolViolation("S2 authorization binding differs")
    if scope == "qualification-only":
        worker_uuid = payload.get("authorized_gpu_uuid")
        if not isinstance(worker_uuid, str) or not worker_uuid.startswith("GPU-"):
            raise M8S2ProtocolViolation("qualification authorization lacks a GPU UUID")
        hostname = payload.get("authorized_worker_hostname")
        if not isinstance(hostname, str) or not hostname.strip():
            raise M8S2ProtocolViolation("qualification authorization lacks a worker hostname")
        if any(
            value is not None
            for value in (
                runtime_policy_sha256,
                input_gate_receipt_sha256,
                qualification_receipt_sha256,
                repeatability_review_sha256,
            )
        ):
            raise M8S2ProtocolViolation("qualification authorization cannot imply later scope")
    else:
        if (
            payload.get("authorized_gpu_uuid") is not None
            or payload.get("authorized_worker_hostname") is not None
        ):
            raise M8S2ProtocolViolation("scientific authorization must bind the policy SHA")
        mode = "repeatability" if scope == "repeatability-only" else "full-pass"
        raw_ids = payload.get("logical_pass_ids")
        if not isinstance(raw_ids, list) or any(not isinstance(item, str) for item in raw_ids):
            raise M8S2ProtocolViolation("S2 authorized logical passes are malformed")
        expected_ids = raw_ids
        if (
            not expected_ids
            or len(expected_ids) != len(set(expected_ids))
            or not set(expected_ids).issubset(logical_pass_ids(mode))
        ):
            raise M8S2ProtocolViolation("S2 authorized logical passes differ")
        if logical_pass_id not in expected_ids:
            raise M8S2ProtocolViolation("S2 logical pass is not authorized")
        if (
            runtime_policy_sha256 is None
            or input_gate_receipt_sha256 is None
            or qualification_receipt_sha256 is None
        ):
            raise M8S2ProtocolViolation("S2 runtime qualification binding is absent")
        if scope == "full-pass" and repeatability_review_sha256 is None:
            raise M8S2ProtocolViolation("accepted owner-reviewed repeatability binding is absent")
        if scope == "repeatability-only" and repeatability_review_sha256 is not None:
            raise M8S2ProtocolViolation("repeatability authorization cannot bind full-pass review")
    if scope == "qualification-only" and (
        payload.get("logical_pass_ids") != [] or logical_pass_id is not None
    ):
        raise M8S2ProtocolViolation("qualification scope cannot authorize scientific passes")


def require_authorization(path: Path | None, **expected: object) -> dict[str, object]:
    """Read a real future authorization only after an explicit path is supplied."""

    if path is None or not path.is_file():
        raise M8S2ProtocolViolation("S2 is disabled: no separate owner authorization exists")
    payload = _mapping(path)
    verify_authorization(payload, **expected)  # type: ignore[arg-type]
    return payload


def verify_qualification_worker(
    authorization: Mapping[str, object],
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    hostname_provider: Callable[[], str] = socket.gethostname,
) -> str:
    """Compare a qualification grant to live external host and GPU before Torch import."""

    expected = authorization.get("authorized_gpu_uuid")
    if not isinstance(expected, str) or not expected.startswith("GPU-"):
        raise M8S2ProtocolViolation("qualification authorization lacks a GPU UUID")
    hostname = authorization.get("authorized_worker_hostname")
    if not isinstance(hostname, str) or not hostname.strip() or hostname_provider() != hostname:
        raise M8S2ProtocolViolation("qualification authorization hostname differs from live worker")
    live_uuid = single_visible_gpu_uuid(command_runner=command_runner)
    if live_uuid != expected:
        raise M8S2ProtocolViolation("qualification authorization GPU UUID differs from live worker")
    return expected


def single_visible_gpu_uuid(
    *, command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run
) -> str:
    """Fail closed unless the external worker exposes exactly one NVIDIA GPU.

    S2 uses Torch ``cuda:0`` while the inherited policy uses NVIDIA-SMI index 0.
    A single visible GPU makes those identities unambiguous even when device
    ordering or CUDA visibility variables are configured.
    """

    try:
        result = command_runner(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise M8S2ProtocolViolation("external qualification GPU identity query failed") from error
    values = [line.strip() for line in result.stdout.splitlines()]
    if result.returncode != 0 or len(values) != 1 or not values[0].startswith("GPU-"):
        raise M8S2ProtocolViolation("S2 requires exactly one NVIDIA-SMI-visible GPU")
    return values[0]


def verify_runtime_policy(
    path: Path, expected_sha256: str, live: Mapping[str, object]
) -> dict[str, object]:
    """Bind exact policy bytes and exact live hardware/software identity."""

    _sha(expected_sha256, "runtime policy")
    if sha256_file(path) != expected_sha256:
        raise M8S2ProtocolViolation("S2 runtime-policy SHA256 differs")
    bound = _mapping(path)
    if bound.get("schema_version") != RUNTIME_POLICY_SCHEMA or bound != dict(live):
        raise M8S2ProtocolViolation("S2 live runtime-policy binding differs")
    return bound


def verify_runtime_policy_document(
    policy: Mapping[str, object], *, execution_commit: str, candidate: Mapping[str, object]
) -> None:
    """Check all frozen and required policy fields before science import."""

    upstream = candidate.get("upstream")
    checkpoint = candidate.get("checkpoint")
    if not isinstance(upstream, Mapping) or not isinstance(checkpoint, Mapping):
        raise M8S2ProtocolViolation("S2 candidate identity is malformed")
    identity = {
        "architecture": candidate.get("architecture"),
        "candidate_manifest_sha256": CANDIDATE_MANIFEST_SHA256,
        "upstream_commit": upstream.get("commit"),
        "config_sha256": upstream.get("config_sha256"),
        "checkpoint_sha256": checkpoint.get("sha256"),
    }
    exact = {
        "schema_version": RUNTIME_POLICY_SCHEMA,
        "repository_execution_commit": execution_commit,
        "candidate_identity": identity,
        "s2_protocol_sha256": PROTOCOL_SHA256,
        "s2_partitions_sha256": PARTITIONS_SHA256,
        "s2_input_freeze_sha256": INPUT_FREEZE_SHA256,
        "s2_full_ledger_sha256": FULL_LEDGER_SHA256,
        "s2_compact_manifest_sha256": COMPACT_MANIFEST_SHA256,
    }
    if any(policy.get(key) != value for key, value in exact.items()):
        raise M8S2ProtocolViolation("S2 runtime policy frozen identity differs")
    for key in (
        "gpu_name",
        "gpu_uuid",
        "nvidia_driver",
        "cuda_runtime",
        "python_exact_version",
        "pytorch_exact_version",
        "spconv",
        "torch_scatter",
        "numpy",
    ):
        if not isinstance(policy.get(key), str) or not str(policy[key]).strip():
            raise M8S2ProtocolViolation(f"S2 runtime policy {key} is absent")
    vram = policy.get("gpu_vram_bytes")
    if isinstance(vram, bool) or not isinstance(vram, int) or vram <= 0:
        raise M8S2ProtocolViolation("S2 runtime policy VRAM is absent")


def verify_structural_preflight(structural: Mapping[str, object]) -> None:
    """Validate seven ordered, same-runtime candidate coordinate comparisons."""

    exact = {
        "schema_version": "laserperception.m8.s2.structural-preflight.v1",
        "status": "ACCEPTED",
        "sentinel_XYZIT_exact": 28,
        "B2_A2_cuda_coordinate_identity": 7,
        "D2_C2_cuda_coordinate_identity": 7,
        "ground_truth_loaded": False,
    }
    if any(structural.get(key) != value for key, value in exact.items()):
        raise M8S2ProtocolViolation("S2 structural preflight gate differs")
    sentinels = structural.get("sentinels")
    if not isinstance(sentinels, list) or len(sentinels) != 7:
        raise M8S2ProtocolViolation("S2 structural sentinel evidence differs")
    for frame_id, entry in zip(SENTINELS, sentinels, strict=True):
        if not isinstance(entry, Mapping) or entry.get("frame_id") != frame_id:
            raise M8S2ProtocolViolation("S2 structural sentinel order differs")
        counts = entry.get("candidate_pillar_counts")
        hashes = entry.get("candidate_coordinate_sha256")
        if (
            not isinstance(counts, Mapping)
            or set(counts) != {"A2", *ARMS}
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in counts.values()
            )
        ):
            raise M8S2ProtocolViolation("S2 structural candidate counts differ")
        if not isinstance(hashes, Mapping) or set(hashes) != {"A2", *ARMS}:
            raise M8S2ProtocolViolation("S2 structural coordinate hashes differ")
        for name in ("A2", *ARMS):
            _sha(hashes[name], "S2 structural coordinate")
        if (
            counts["A2"] != counts["B2"]
            or counts["C2"] != counts["D2"]
            or hashes["A2"] != hashes["B2"]
            or hashes["C2"] != hashes["D2"]
            or entry.get("A2_B2_coordinate_identity") is not True
            or entry.get("C2_D2_coordinate_identity") is not True
        ):
            raise M8S2ProtocolViolation("S2 structural pair identity differs")


def verify_qualification_receipt(
    path: Path, *, execution_commit: str, runtime_policy_sha256: str, input_gate_receipt_sha256: str
) -> str:
    """Require accepted, GT-blind, same-runtime structural qualification."""

    record = _mapping(path)
    expected = {
        "schema_version": QUALIFICATION_RECEIPT_SCHEMA,
        "status": "ACCEPTED",
        "execution_commit": execution_commit,
        "runtime_policy_binding_sha256": runtime_policy_sha256,
        "input_gate_receipt_sha256": input_gate_receipt_sha256,
        "full_ledger_sha256": FULL_LEDGER_SHA256,
        "sentinel_XYZIT_exact": 28,
        "B2_A2_cuda_coordinate_identity": 7,
        "D2_C2_cuda_coordinate_identity": 7,
        "ground_truth_loaded": False,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise M8S2ProtocolViolation("S2 runtime qualification receipt differs")
    structural = record.get("structural_preflight")
    if not isinstance(structural, Mapping):
        raise M8S2ProtocolViolation("S2 structural preflight receipt is absent")
    verify_structural_preflight(structural)
    return sha256_file(path)


def verify_repeatability_review(
    path: Path,
    *,
    execution_commit: str,
    runtime_policy_sha256: str,
    input_gate_receipt_sha256: str,
) -> str:
    """Require ten accepted processes and an explicit owner review for full-pass scope."""

    record = _mapping(path)
    expected = {
        "schema_version": "laserperception.m8.s2.repeatability-review.v1",
        "status": "ACCEPTED",
        "processes": 10,
        "calls_per_process": 28,
        "accepted_calls": 280,
        "execution_commit": execution_commit,
        "runtime_policy_sha256": runtime_policy_sha256,
        "input_gate_receipt_sha256": input_gate_receipt_sha256,
        "owner_reviewed": True,
        "full_corpus_authorized": False,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise M8S2ProtocolViolation("S2 accepted owner-reviewed repeatability receipt differs")
    process_uuids = record.get("process_uuids")
    if (
        not isinstance(process_uuids, list)
        or len(process_uuids) != 10
        or len(set(process_uuids)) != 10
    ):
        raise M8S2ProtocolViolation("S2 repeatability process identities differ")
    return sha256_file(path)


@dataclass(frozen=True, slots=True)
class AttemptIdentity:
    """One non-spliceable future process identity and its frozen bindings."""

    mode: str
    logical_pass_id: str
    attempt_id: str
    process_uuid: str
    process_id: int
    execution_commit: str
    runtime_policy_sha256: str
    input_gate_receipt_sha256: str

    def __post_init__(self) -> None:
        if self.logical_pass_id not in logical_pass_ids(self.mode):
            raise M8S2ProtocolViolation("S2 logical pass ID differs")
        if not self.attempt_id or not self.process_uuid or self.process_id <= 0:
            raise M8S2ProtocolViolation("S2 attempt identity is incomplete")
        uuid.UUID(self.process_uuid)
        for label, value in (
            ("execution commit", self.execution_commit),
            ("runtime policy", self.runtime_policy_sha256),
            ("input receipt", self.input_gate_receipt_sha256),
        ):
            _sha(value, label, length=40 if label == "execution commit" else 64)

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "logical_pass_id": self.logical_pass_id,
            "attempt_id": self.attempt_id,
            "process_uuid": self.process_uuid,
            "process_id": self.process_id,
            "execution_commit": self.execution_commit,
            "runtime_policy_sha256": self.runtime_policy_sha256,
            "input_gate_receipt_sha256": self.input_gate_receipt_sha256,
            "full_ledger_sha256": FULL_LEDGER_SHA256,
            "protocol_sha256": PROTOCOL_SHA256,
        }


class AtomicAttempt:
    """Append-only ordered attempt; incomplete output has zero accepted calls."""

    def __init__(
        self, root: Path, identity: AttemptIdentity, *, condition_order: Sequence[str] | None = None
    ) -> None:
        self.root = root
        self.identity = identity
        self.condition_ids = tuple(
            condition_ids(identity.mode) if condition_order is None else condition_order
        )
        if condition_order is not None and self.condition_ids != condition_ids(identity.mode):
            raise M8S2ProtocolViolation("S2 attempt condition order differs")
        self.completed: list[str] = []
        self.failed_calls = 0
        self.status = "IN_PROGRESS"
        self.started_at = datetime.now(timezone.utc).isoformat()
        root.mkdir(parents=True, exist_ok=False)
        self._write("IN_PROGRESS", None)

    def _payload(self, status: str, reason: str | None) -> dict[str, object]:
        return {
            "schema_version": ATTEMPT_SCHEMA,
            "status": status,
            **self.identity.to_dict(),
            "started_at_utc": self.started_at,
            "ended_at_utc": datetime.now(timezone.utc).isoformat()
            if status != "IN_PROGRESS"
            else None,
            "expected_calls": len(self.condition_ids),
            "attempted_calls": len(self.completed) + self.failed_calls,
            "completed_calls": len(self.completed),
            "accepted_canonical_calls": len(self.completed) if status == "COMPLETE" else 0,
            "failed_calls": self.failed_calls,
            "failure_reason": reason,
            "completed_condition_ids": list(self.completed),
            "next_condition_id": self.condition_ids[len(self.completed)]
            if len(self.completed) < len(self.condition_ids)
            else None,
        }

    def _write(self, status: str, reason: str | None) -> None:
        atomic_write_json(self.root / "attempt_manifest.json", self._payload(status, reason))

    def record(self, condition_id: str, payload: Mapping[str, object]) -> None:
        if (
            self.status != "IN_PROGRESS"
            or len(self.completed) >= len(self.condition_ids)
            or condition_id != self.condition_ids[len(self.completed)]
        ):
            raise M8S2ProtocolViolation("S2 condition order or call count differs")
        record: dict[str, object] = {
            "schema_version": CONDITION_SCHEMA,
            "status": "COMPLETE",
            "identity": self.identity.to_dict(),
            "condition_id": condition_id,
            "payload": dict(payload),
        }
        record["record_sha256"] = canonical_json_sha256(record)
        atomic_write_json(self.root / "conditions" / f"{len(self.completed):04d}.json", record)
        self.completed.append(condition_id)
        self._write("IN_PROGRESS", None)

    def fail(self, reason: str, *, count_failed_call: bool = True) -> dict[str, object]:
        if self.status != "IN_PROGRESS":
            raise M8S2ProtocolViolation("S2 attempt is already terminal")
        if count_failed_call:
            self.failed_calls += 1
        self.status = "INCOMPLETE"
        self._write("INCOMPLETE", reason)
        return self._payload("INCOMPLETE", reason)

    def finalize(self) -> dict[str, object]:
        if (
            self.status != "IN_PROGRESS"
            or tuple(self.completed) != self.condition_ids
            or self.failed_calls
        ):
            raise M8S2ProtocolViolation("S2 attempt is incomplete and cannot be canonical")
        files = sorted((self.root / "conditions").glob("*.json"))
        if len(files) != len(self.condition_ids):
            raise M8S2ProtocolViolation("S2 condition files are incomplete")
        record = self._payload("COMPLETE", None)
        record["condition_file_sha256"] = [sha256_file(path) for path in files]
        record["result_sha256"] = canonical_json_sha256(record)
        atomic_write_json(self.root / "final_pass_manifest.json", record)
        self.status = "COMPLETE"
        self._write("COMPLETE", None)
        return record
