"""GT-blind sizing, separate from scientific scoring; safe to import on CPU.

Only the authorized worker entry imports Torch/backend. Synthetic tests supply
runtime and backend doubles. Prediction objects never enter evidence builders.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import socket
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from laserperception.detection.m8_s1_runtime import (
    CANDIDATE_MANIFEST_PATH,
    atomic_write_json,
    sha256_file,
)
from laserperception.detection.m8_s2_input_gate import verify_consumed_input
from laserperception.detection.m8_s2_planning import estimate_future_cost
from laserperception.detection.m8_s2_runtime import (
    COMPACT_MANIFEST_PATH,
    COMPACT_MANIFEST_SHA256,
    FULL_LEDGER_SHA256,
    INPUT_FREEZE_SHA256,
    PARTITIONS_SHA256,
    PROTOCOL_SHA256,
    SIZING_PLAN_PATH,
    SIZING_PLAN_SHA256,
    SIZING_PROCESS_IDS,
    M8S2ProtocolViolation,
    require_authorization,
    verify_authorization,
)

ATTEMPT_SCHEMA = "laserperception.m8.s2.sizing-attempt.v1"
CALL_KEYS = frozenset(
    {
        "condition_id",
        "arm",
        "input_sha256",
        "M7_XYZT_sha256",
        "selected_global_row_sha256",
        "point_count",
        "cpu_analytic_candidate_pillar_count",
        "sequence_index",
        "warmup",
        "elapsed_seconds",
        "memory",
        "process_uuid",
    }
)
MEMORY_KEYS = frozenset(
    {"torch_allocated_before_bytes", "torch_peak_allocated_bytes", "torch_reserved_after_bytes"}
)
TEMPORARY_NAMES = (
    "calls.json",
    "manifest.json",
    "sizing-claims.json",
    "authorization.json",
    "sizing-retained-copy",
    "sizing-failed-manifest",
)


def sizing_plan(root: Path) -> dict[str, Any]:
    """Read the unchanged preregistration, never regenerate a selection."""
    path = root / SIZING_PLAN_PATH
    if sha256_file(path) != SIZING_PLAN_SHA256:
        raise M8S2ProtocolViolation("S2 sizing plan bytes differ")
    plan: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return plan


def planned_ids(plan: Mapping[str, Any]) -> tuple[str, ...]:
    """Two fixed warmups followed by sixteen fixed measured conditions."""
    return tuple(plan["warmup_condition_ids_per_process"]) + tuple(
        row["condition_id"] for row in plan["measured_conditions_per_process"]
    )


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise M8S2ProtocolViolation("sizing numeric evidence differs")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise M8S2ProtocolViolation("sizing numeric evidence must be finite and nonnegative")
    return result


def _json(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise M8S2ProtocolViolation("sizing evidence is not an object")
    return record


def verify_call(record: Mapping[str, Any], index: int, ids: Sequence[str], process: str) -> None:
    """Reject semantic payloads, reordered calls and invalid telemetry."""
    if set(record) != CALL_KEYS or record["condition_id"] != ids[index]:
        raise M8S2ProtocolViolation("sizing call schema or frozen order differs")
    if record["sequence_index"] != index or record["warmup"] is not (index < 2):
        raise M8S2ProtocolViolation("sizing warmup/measurement boundary differs")
    if record["process_uuid"] != process or record["arm"] != ids[index].rsplit("/", 1)[1]:
        raise M8S2ProtocolViolation("sizing process or arm differs")
    for name in ("input_sha256", "M7_XYZT_sha256", "selected_global_row_sha256"):
        value = record[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise M8S2ProtocolViolation("sizing input hash differs")
    for name in ("point_count", "cpu_analytic_candidate_pillar_count"):
        if isinstance(record[name], bool) or not isinstance(record[name], int) or record[name] <= 0:
            raise M8S2ProtocolViolation("sizing input characterization differs")
    _number(record["elapsed_seconds"])
    memory = record["memory"]
    if not isinstance(memory, dict) or set(memory) != MEMORY_KEYS:
        raise M8S2ProtocolViolation("sizing telemetry schema differs")
    for value in memory.values():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise M8S2ProtocolViolation("sizing Torch allocator metric differs")


def verified_condition_loader(root: Path, date_root: Path, m6_ledger: Path) -> Callable[..., Any]:
    """Reconstruct once and return the exact, just-verified detector array."""
    from laserperception.detection.m8_s2_reconstruction import frame_inputs, sources

    m7, m8, compact = sources(repository_root=root, date_root=date_root, m6_ledger=m6_ledger)

    def load(condition_id: str) -> tuple[np.ndarray, dict[str, object]]:
        frame_id, arm = condition_id.rsplit("/", 1)
        _, items = frame_inputs(frame_id, m7, m8)
        item = next(item for item in items if item.arm == arm)
        bound = compact[condition_id]
        evidence = verify_consumed_input(condition_id, item.points, item.selected_row_sha256, bound)
        return item.points, {
            "condition_id": condition_id,
            "arm": arm,
            "input_sha256": evidence["input_sha256"],
            "M7_XYZT_sha256": evidence["M7_XYZT_sha256"],
            "selected_global_row_sha256": evidence["selected_global_row_sha256"],
            "point_count": bound["point_count"],
            "cpu_analytic_candidate_pillar_count": bound["cpu_analytic_candidate_pillar_count"],
        }

    return load


def measure_calls(
    *,
    plan: Mapping[str, Any],
    process_uuid: str,
    backend_factory: Callable[[], Any],
    runtime: Any,
    load: Callable[..., Any],
    record: Callable[[dict[str, Any]], None],
    clock: Callable[[], float] = time.perf_counter,
    on_initialized: Callable[[float], None] | None = None,
    on_call_start: Callable[[], None] | None = None,
) -> float:
    """Synchronize around one inference; discard its result without inspection."""
    runtime.synchronize()
    start = clock()
    backend = backend_factory()
    runtime.synchronize()
    initialization = clock() - start
    _number(initialization)
    if on_initialized is not None:
        on_initialized(initialization)
    for index, condition in enumerate(planned_ids(plan)):
        points, consumed = load(condition)  # reconstruction/hash checks outside timing
        runtime.synchronize()
        runtime.reset_peak_memory_stats()
        before = int(runtime.memory_allocated())
        if on_call_start is not None:
            on_call_start()
        start = clock()
        backend.run_gt_blind_timing_call(points)  # existing boundary discards GPU semantics
        runtime.synchronize()
        duration = clock() - start
        payload = {
            **consumed,
            "sequence_index": index,
            "warmup": index < 2,
            "elapsed_seconds": duration,
            "process_uuid": process_uuid,
            "memory": {
                "torch_allocated_before_bytes": before,
                "torch_peak_allocated_bytes": int(runtime.max_memory_allocated()),
                "torch_reserved_after_bytes": int(runtime.memory_reserved()),
            },
        }
        verify_call(payload, index, planned_ids(plan), process_uuid)
        record(payload)
    return initialization


def _bindings(commit: str, policy: str, input_sha: str, qualification: str) -> dict[str, object]:
    return {
        "execution_commit": commit,
        "runtime_policy_binding_sha256": policy,
        "input_gate_receipt_sha256": input_sha,
        "qualification_receipt_sha256": qualification,
        "protocol_sha256": PROTOCOL_SHA256,
        "partitions_sha256": PARTITIONS_SHA256,
        "input_freeze_sha256": INPUT_FREEZE_SHA256,
        "full_ledger_sha256": FULL_LEDGER_SHA256,
        "compact_manifest_sha256": COMPACT_MANIFEST_SHA256,
        "sizing_plan_sha256": SIZING_PLAN_SHA256,
    }


def _atomic_write_bytes(target: Path, data: bytes, *, retained_copy: bool = False) -> None:
    """Keep canonical bytes intact until a flushed sibling copy is complete."""
    name = "sizing-retained-copy" if retained_copy else target.name
    temporary = target.with_name(f".{name}.{os.getpid()}.tmp")
    if temporary.is_symlink():
        raise M8S2ProtocolViolation("sizing byte-copy temporary is a symlink")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def run_sizing_attempt(
    *,
    repository_root: Path,
    date_root: Path,
    m6_ledger: Path,
    campaign_root: Path,
    attempt_root: Path,
    logical_process_id: str,
    execution_commit: str,
    runtime_policy_sha256: str,
    input_gate_receipt_sha256: str,
    qualification_receipt_sha256: str,
    authorization_path: Path,
    candidate_readiness_receipt_sha256: str,
    attempt_id: str | None = None,
    worker: Callable[..., float] | None = None,
) -> dict[str, Any]:
    """Claim once atomically; seal partial attempts instead of accepting timings."""
    plan = sizing_plan(repository_root)
    bindings = _bindings(
        execution_commit,
        runtime_policy_sha256,
        input_gate_receipt_sha256,
        qualification_receipt_sha256,
    )
    grant = require_authorization(
        authorization_path,
        scope="sizing-only",
        execution_commit=execution_commit,
        logical_pass_id=logical_process_id,
        runtime_policy_sha256=runtime_policy_sha256,
        input_gate_receipt_sha256=input_gate_receipt_sha256,
        qualification_receipt_sha256=qualification_receipt_sha256,
        candidate_readiness_receipt_sha256=candidate_readiness_receipt_sha256,
        campaign_root=campaign_root,
        sizing_plan_sha256=SIZING_PLAN_SHA256,
    )
    if campaign_root.is_symlink() or attempt_root.is_symlink():
        raise M8S2ProtocolViolation("sizing campaign or attempt cannot be a symlink")
    campaign_root = campaign_root.resolve()
    attempt_root = attempt_root.resolve()
    attempt_id = attempt_id or attempt_root.name
    if not attempt_id.strip():
        raise M8S2ProtocolViolation("sizing attempt identifier is absent")
    if attempt_root.parent != campaign_root or attempt_root.exists() or campaign_root.is_symlink():
        raise M8S2ProtocolViolation("sizing attempt must be new and directly under campaign")
    campaign_root.mkdir(parents=True, exist_ok=True)
    process_uuid = str(uuid.uuid4())
    authorization_bytes = authorization_path.read_bytes()
    manifest: dict[str, Any] = {
        "schema_version": ATTEMPT_SCHEMA,
        **bindings,
        "status": "RUNNING",
        "logical_process_id": logical_process_id,
        "attempt_id": attempt_id,
        "process_uuid": process_uuid,
        "process_id": os.getpid(),
        "recovery_process_ids": [],
        "worker_hostname": socket.gethostname(),
        "authorization_id": grant["authorization_id"],
        "authorization_sha256": hashlib.sha256(authorization_bytes).hexdigest(),
        "initialization_seconds": None,
        "engineering_calls_completed": 0,
        "engineering_calls_started": 0,
        "accepted_engineering_calls": 0,
        "scientific_calls": 0,
        "ground_truth_loaded": False,
        "semantic_predictions_retained": False,
        "files": {},
        "failure_type": None,
    }
    lock = campaign_root / ".sizing.lock"
    lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(
        lock_fd,
        json.dumps(
            {
                "process_id": os.getpid(),
                "hostname": socket.gethostname(),
                "attempt": attempt_root.name,
                "bootstrap": {
                    "manifest": manifest,
                    "authorization_text": authorization_bytes.decode("utf-8"),
                },
            }
        ).encode("utf-8"),
    )
    os.close(lock_fd)
    ledger_path = campaign_root / "sizing-claims.json"
    ledger: dict[str, Any] = {"bindings": bindings, "attempts": []}
    sealed = False
    claimed = False
    prepared = False
    calls: list[dict[str, Any]] = []
    try:
        if ledger_path.exists():
            ledger = _json(ledger_path)
        if ledger["bindings"] != bindings:
            raise M8S2ProtocolViolation("sizing campaign bindings differ")
        for entry in ledger["attempts"]:
            if entry["attempt_id"] == attempt_id:
                raise M8S2ProtocolViolation("sizing attempt identifier cannot be reused")
            if entry["status"] not in {"COMPLETE", "INCOMPLETE"}:
                raise M8S2ProtocolViolation("sizing campaign contains an unsealed claim")
            old = _attempt_path(campaign_root, entry["attempt"])
            if sha256_file(old / "manifest.json") != entry["manifest_sha256"]:
                raise M8S2ProtocolViolation("sizing prior attempt seal differs")
            _verify_files(old, _json(old / "manifest.json"))
            if entry["status"] == "COMPLETE" and entry["logical_process_id"] == logical_process_id:
                raise M8S2ProtocolViolation("complete sizing logical process cannot be rerun")
            if entry["process_id"] == os.getpid():
                raise M8S2ProtocolViolation("sizing requires a fresh OS process")
        attempt_root.mkdir()
        prepared = True
        _atomic_write_bytes(attempt_root / "authorization.json", authorization_bytes)
        entry = {
            "attempt": attempt_root.name,
            "attempt_id": attempt_id,
            "logical_process_id": logical_process_id,
            "process_uuid": process_uuid,
            "process_id": os.getpid(),
            "status": "RUNNING",
            "manifest_sha256": None,
        }
        ledger["attempts"].append(entry)
        atomic_write_json(ledger_path, ledger)
        claimed = True
        atomic_write_json(attempt_root / "manifest.json", manifest)
        atomic_write_json(attempt_root / "calls.json", {"calls": []})

        def record(call: dict[str, Any]) -> None:
            verify_call(call, len(calls), planned_ids(plan), process_uuid)
            atomic_write_json(attempt_root / "calls.json", {"calls": [*calls, call]})
            calls.append(call)

        def initialized(seconds: float) -> None:
            manifest["initialization_seconds"] = _number(seconds)
            atomic_write_json(attempt_root / "manifest.json", manifest)

        def call_started() -> None:
            manifest["engineering_calls_started"] += 1
            atomic_write_json(attempt_root / "manifest.json", manifest)

        initialization = (worker or _external_worker)(
            plan=plan,
            process_uuid=process_uuid,
            record=record,
            repository_root=repository_root,
            date_root=date_root,
            m6_ledger=m6_ledger,
            on_initialized=initialized,
            on_call_start=call_started,
        )
        _number(initialization)
        if len(calls) != 18:
            raise M8S2ProtocolViolation("sizing process requires all 18 engineering calls")
        manifest.update(
            status="COMPLETE", initialization_seconds=initialization, accepted_engineering_calls=18
        )
    except BaseException as error:
        if claimed:
            manifest.update(status="INCOMPLETE", failure_type=type(error).__name__)
        raise
    finally:
        if claimed:
            # Replacement may complete immediately before an interrupt. Count
            # only the canonical document, never an uncommitted in-memory row.
            calls = _json(attempt_root / "calls.json")["calls"]
            manifest["engineering_calls_completed"] = len(calls)
            manifest["engineering_calls_started"] = max(
                manifest["engineering_calls_started"], len(calls)
            )
            manifest["files"] = {
                n: sha256_file(attempt_root / n) for n in ("calls.json", "authorization.json")
            }
            own_manifest_temp = attempt_root / f".manifest.json.{os.getpid()}.tmp"
            if own_manifest_temp.exists():
                if own_manifest_temp.is_symlink():
                    raise M8S2ProtocolViolation("sizing failed manifest temporary is a symlink")
                # The final canonical-manifest write uses this same PID's temp
                # name, so preserve its prior bytes under a distinct exact name.
                _atomic_write_bytes(
                    attempt_root / f".sizing-failed-manifest.{os.getpid()}.tmp",
                    own_manifest_temp.read_bytes(),
                    retained_copy=True,
                )
            for name in TEMPORARY_NAMES:
                temporary = attempt_root / f".{name}.{os.getpid()}.tmp"
                if name != "manifest.json" and temporary.exists():
                    if temporary.is_symlink():
                        raise M8S2ProtocolViolation("sizing failed writer temporary is a symlink")
                    manifest["files"][temporary.name] = sha256_file(temporary)
            atomic_write_json(attempt_root / "manifest.json", manifest)
            entry.update(
                status=manifest["status"],
                manifest_sha256=sha256_file(attempt_root / "manifest.json"),
            )
            atomic_write_json(ledger_path, ledger)
            sealed = True
        if (not claimed and not prepared) or sealed:
            lock.unlink()
    return manifest


def _external_worker(**kwargs: Any) -> float:
    """Only reached after runner authorization, live policy checks and claim."""
    torch = importlib.import_module("torch")
    backend = importlib.import_module("laserperception.detection.m8_backend")
    root = kwargs.pop("repository_root")
    load = verified_condition_loader(root, kwargs.pop("date_root"), kwargs.pop("m6_ledger"))
    return measure_calls(
        **kwargs,
        runtime=torch.cuda,
        load=load,
        backend_factory=lambda: backend.DsvtBackend.from_environment(
            manifest_path=root / CANDIDATE_MANIFEST_PATH
        ),
    )


def _verify_files(root: Path, manifest: Mapping[str, Any]) -> None:
    allowed = {
        "schema_version",
        "status",
        "logical_process_id",
        "attempt_id",
        "process_uuid",
        "process_id",
        "recovery_process_ids",
        "worker_hostname",
        "authorization_id",
        "authorization_sha256",
        "initialization_seconds",
        "engineering_calls_completed",
        "engineering_calls_started",
        "accepted_engineering_calls",
        "scientific_calls",
        "ground_truth_loaded",
        "semantic_predictions_retained",
        "files",
        "failure_type",
        *_bindings("", "", "", ""),
    }
    if set(manifest) != allowed or (root / "manifest.json").is_symlink():
        raise M8S2ProtocolViolation("sizing manifest schema differs")
    required_files = {"calls.json", "authorization.json"}
    writers = [manifest["process_id"], *manifest["recovery_process_ids"]]
    if any(type(pid) is not int or pid <= 0 for pid in writers) or len(set(writers)) != len(
        writers
    ):
        raise M8S2ProtocolViolation("sizing writer identities differ")
    owned_temporaries = {f".{name}.{pid}.tmp" for pid in writers for name in TEMPORARY_NAMES}
    recorded_files = set(manifest["files"])
    if (
        not required_files <= recorded_files
        or not recorded_files <= required_files | owned_temporaries
    ):
        raise M8S2ProtocolViolation("sizing sealed file inventory differs")
    if {p.name for p in root.iterdir()} != recorded_files | {"manifest.json"}:
        raise M8S2ProtocolViolation("sizing attempt has unexpected retained files")
    for name, expected in manifest["files"].items():
        if (root / name).is_symlink() or sha256_file(root / name) != expected:
            raise M8S2ProtocolViolation("sizing sealed file hash differs")


def _attempt_path(campaign: Path, name: str) -> Path:
    if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}:
        raise M8S2ProtocolViolation("sizing claim path differs")
    path = campaign / name
    if path.is_symlink() or path.resolve().parent != campaign.resolve():
        raise M8S2ProtocolViolation("sizing claim must resolve directly under campaign")
    return path


def _recover_initial_claim(
    campaign: Path, attempt: Path, lock: Mapping[str, Any], execution_commit: str
) -> None:
    """Restore only zero-call bootstrap records, never uncommitted measurements."""
    bootstrap = lock.get("bootstrap")
    if bootstrap is None:
        return  # Older locks require their existing canonical files.
    initial = bootstrap["manifest"]
    authorization = bootstrap["authorization_text"].encode("utf-8")
    bindings = _bindings(
        execution_commit,
        initial["runtime_policy_binding_sha256"],
        initial["input_gate_receipt_sha256"],
        initial["qualification_receipt_sha256"],
    )
    if (
        any(initial.get(k) != v for k, v in bindings.items())
        or initial["process_id"] != lock["process_id"]
        or initial["worker_hostname"] != lock["hostname"]
        or initial["status"] != "RUNNING"
        or initial["initialization_seconds"] is not None
        or initial["engineering_calls_started"] != 0
        or initial["engineering_calls_completed"] != 0
        or hashlib.sha256(authorization).hexdigest() != initial["authorization_sha256"]
    ):
        raise M8S2ProtocolViolation("sizing initial claim snapshot differs")
    verify_authorization(
        json.loads(authorization),
        scope="sizing-only",
        execution_commit=execution_commit,
        logical_pass_id=initial["logical_process_id"],
        campaign_root=campaign,
        runtime_policy_sha256=initial["runtime_policy_binding_sha256"],
        input_gate_receipt_sha256=initial["input_gate_receipt_sha256"],
        qualification_receipt_sha256=initial["qualification_receipt_sha256"],
        candidate_readiness_receipt_sha256=json.loads(authorization).get(
            "candidate_readiness_receipt_sha256"
        ),
        sizing_plan_sha256=SIZING_PLAN_SHA256,
    )
    ledger_path = campaign / "sizing-claims.json"
    ledger: dict[str, Any] = (
        _json(ledger_path) if ledger_path.exists() else {"bindings": bindings, "attempts": []}
    )
    entries = [e for e in ledger["attempts"] if e["attempt"] == attempt.name]
    missing = any(
        not (attempt / n).exists() for n in ("manifest.json", "calls.json", "authorization.json")
    )
    if not missing and entries:
        return
    if ledger["bindings"] != bindings or len(entries) > 1:
        raise M8S2ProtocolViolation("sizing bootstrap campaign differs")
    if not ledger_path.exists() and any(p.is_dir() and p != attempt for p in campaign.iterdir()):
        raise M8S2ProtocolViolation("sizing missing ledger has other attempts")
    current = _json(attempt / "manifest.json") if (attempt / "manifest.json").exists() else initial
    calls = _json(attempt / "calls.json")["calls"] if (attempt / "calls.json").exists() else []
    if (
        {**current, "recovery_process_ids": []} != initial
        or calls != []
        or any(
            e["attempt_id"] == initial["attempt_id"]
            or e["process_id"] == initial["process_id"]
            or e["status"] not in {"COMPLETE", "INCOMPLETE"}
            or (
                e["status"] == "COMPLETE"
                and e["logical_process_id"] == initial["logical_process_id"]
            )
            for e in ledger["attempts"]
            if e not in entries
        )
    ):
        raise M8S2ProtocolViolation(
            "sizing bootstrap cannot reconstruct executed or conflicting claim"
        )
    attempt.mkdir(exist_ok=True)
    auth_path = attempt / "authorization.json"
    if auth_path.exists() and (auth_path.is_symlink() or auth_path.read_bytes() != authorization):
        raise M8S2ProtocolViolation("sizing bootstrap authorization copy differs")
    if not auth_path.exists():
        _atomic_write_bytes(auth_path, authorization)
    if not (attempt / "manifest.json").exists():
        atomic_write_json(attempt / "manifest.json", initial)
    if not (attempt / "calls.json").exists():
        atomic_write_json(attempt / "calls.json", {"calls": []})
    if not entries:
        ledger["attempts"].append(
            {
                "attempt": attempt.name,
                "attempt_id": initial["attempt_id"],
                "logical_process_id": initial["logical_process_id"],
                "process_uuid": initial["process_uuid"],
                "process_id": initial["process_id"],
                "status": "RUNNING",
                "manifest_sha256": None,
            }
        )
        atomic_write_json(ledger_path, ledger)


@contextmanager
def _recovery_guard(campaign: Path) -> Iterator[None]:
    """Kernel-owned exclusive recovery lock; process death releases ownership."""
    path = campaign / ".sizing-recovery-guard"
    if path.is_symlink():
        raise M8S2ProtocolViolation("sizing recovery guard cannot be a symlink")
    # Keep the same inode/path permanently: unlinking an advisory-lock file
    # could let another process lock a new inode while this one remains held.
    with path.open("a+b") as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            windows_lock: Any = importlib.import_module("msvcrt")

            try:
                windows_lock.locking(stream.fileno(), windows_lock.LK_NBLCK, 1)
            except OSError as error:
                raise M8S2ProtocolViolation("another sizing recovery owns the guard") from error
            try:
                yield
            finally:
                stream.seek(0)
                windows_lock.locking(stream.fileno(), windows_lock.LK_UNLCK, 1)
        else:
            unix_lock: Any = importlib.import_module("fcntl")

            try:
                unix_lock.flock(stream.fileno(), unix_lock.LOCK_EX | unix_lock.LOCK_NB)
            except OSError as error:
                raise M8S2ProtocolViolation("another sizing recovery owns the guard") from error
            try:
                yield
            finally:
                unix_lock.flock(stream.fileno(), unix_lock.LOCK_UN)


def seal_interrupted_sizing(
    campaign_root: Path, attempt_root: Path, *, execution_commit: str
) -> dict[str, Any]:
    """Serialize original-worker recovery without a read/replace claim race."""
    with _recovery_guard(campaign_root.resolve()):
        return _seal_interrupted_sizing_owned(
            campaign_root, attempt_root, execution_commit=execution_commit
        )


def _seal_interrupted_sizing_owned(
    campaign_root: Path, attempt_root: Path, *, execution_commit: str
) -> dict[str, Any]:
    """Seal a dead process on its original worker; never resume partial calls."""
    campaign = campaign_root.resolve()
    attempt = _attempt_path(campaign, attempt_root.name)
    if attempt.resolve() != attempt_root.resolve():
        raise M8S2ProtocolViolation("interrupted sizing attempt path differs")
    lock_path = campaign / ".sizing.lock"
    lock = _json(lock_path)
    if lock["hostname"] != socket.gethostname() or lock["attempt"] != attempt.name:
        raise M8S2ProtocolViolation("sizing recovery belongs to another worker/attempt")
    try:
        os.kill(lock["process_id"], 0)
    except ProcessLookupError:
        pass
    else:
        raise M8S2ProtocolViolation("sizing process remains live or cannot be proven dead")
    recovery_pids = lock.get("recovery_process_ids", [])
    if not isinstance(recovery_pids, list) or any(
        type(pid) is not int or pid <= 0 for pid in recovery_pids
    ):
        raise M8S2ProtocolViolation("sizing recovery writer inventory differs")
    for pid in recovery_pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise M8S2ProtocolViolation("a sizing recovery writer remains live")
    # Register the recovery writer before it can create attempt/ledger temporaries.
    # If this registration write itself is killed, canonical evidence is untouched.
    dead_writers = [lock["process_id"], *recovery_pids]
    recovery_pids.append(os.getpid())
    lock["recovery_process_ids"] = recovery_pids
    atomic_write_json(lock_path, lock)
    _recover_initial_claim(campaign, attempt, lock, execution_commit)
    ledger = _json(campaign / "sizing-claims.json")
    entries = [e for e in ledger["attempts"] if e["attempt"] == attempt.name]
    if len(entries) != 1:
        raise M8S2ProtocolViolation("interrupted sizing claim differs")
    entry = entries[0]
    manifest = _json(attempt / "manifest.json")
    if manifest["execution_commit"] != execution_commit:
        raise M8S2ProtocolViolation("sizing recovery execution commit differs")
    if (
        manifest["process_id"] != lock["process_id"]
        or manifest["process_uuid"] != entry["process_uuid"]
    ):
        raise M8S2ProtocolViolation("interrupted sizing process identity differs")
    # Preserve the verified dead writer's uncommitted bytes, never promote them
    # to canonical calls/manifests. An atomic campaign-ledger temp is copied into
    # the sealed attempt too, without deleting its original valuable state.
    owned: dict[str, str] = {}
    for pid in dead_writers:
        for name in TEMPORARY_NAMES:
            temporary_name = f".{name}.{pid}.tmp"
            source = (campaign if name == "sizing-claims.json" else attempt) / temporary_name
            if not source.exists() and not source.is_symlink():
                continue
            if source.is_symlink() or not source.is_file():
                raise M8S2ProtocolViolation("sizing owned atomic temporary is not a regular file")
            target = attempt / temporary_name
            if source != target:
                if target.is_symlink():
                    raise M8S2ProtocolViolation("sizing recovery temporary copy is a symlink")
                if target.exists() and sha256_file(target) != sha256_file(source):
                    raise M8S2ProtocolViolation("sizing recovery temporary copy differs")
                _atomic_write_bytes(target, source.read_bytes(), retained_copy=True)
            owned[temporary_name] = sha256_file(target)
    manifest["recovery_process_ids"] = recovery_pids
    if manifest["status"] == "RUNNING":
        calls = _json(attempt / "calls.json")["calls"]
        manifest.update(
            status="INCOMPLETE",
            accepted_engineering_calls=0,
            engineering_calls_completed=len(calls),
            failure_type="Interrupted",
            files={
                **{n: sha256_file(attempt / n) for n in ("calls.json", "authorization.json")},
                **owned,
            },
        )
        atomic_write_json(attempt / "manifest.json", manifest)
    elif manifest["status"] not in {"COMPLETE", "INCOMPLETE"}:
        raise M8S2ProtocolViolation("sizing recovery status differs")
    else:
        # A committed COMPLETE manifest keeps its status; only preservation
        # inventory is extended with the dead writer's exact temporary bytes.
        manifest["files"].update(owned)
        atomic_write_json(attempt / "manifest.json", manifest)
    _verify_files(attempt, manifest)
    entry.update(status=manifest["status"], manifest_sha256=sha256_file(attempt / "manifest.json"))
    atomic_write_json(campaign / "sizing-claims.json", ledger)
    lock_path.unlink()
    return manifest


def write_sizing_result(output: Path, roots: Sequence[Path], result: Mapping[str, Any]) -> None:
    """Write a new result without overwriting any retained campaign evidence."""
    destination = output.resolve()
    campaigns = {root.resolve().parent for root in roots}
    if (
        output.is_symlink()
        or destination.exists()
        or any(
            destination.is_relative_to(campaign)
            and (
                destination.parent != campaign
                or destination.name.startswith(".")
                or destination.name == "sizing-claims.json"
            )
            for campaign in campaigns
        )
    ):
        raise M8S2ProtocolViolation("sizing output overlaps retained evidence or already exists")
    atomic_write_json(destination, result)


def aggregate_sizing(
    roots: Sequence[Path],
    *,
    repository_root: Path,
    execution_commit: str,
    usd_per_hour: float,
) -> dict[str, Any]:
    """Validate exactly two claimed processes; reuse the existing cost estimator."""
    if len(roots) != 2 or len({p.resolve().parent for p in roots}) != 1:
        raise M8S2ProtocolViolation("sizing aggregation requires two attempts in one campaign")
    campaign = roots[0].resolve().parent
    if (campaign / ".sizing.lock").exists():
        raise M8S2ProtocolViolation("sizing campaign has a live or interrupted claim")
    ledger = _json(campaign / "sizing-claims.json")
    plan = sizing_plan(repository_root)
    ids = planned_ids(plan)
    compact = _json(repository_root / COMPACT_MANIFEST_PATH)
    if sha256_file(repository_root / COMPACT_MANIFEST_PATH) != COMPACT_MANIFEST_SHA256:
        raise M8S2ProtocolViolation("sizing compact input manifest bytes differ")
    by_id = {r["condition_id"]: r for r in compact["conditions"]}
    complete = [e for e in ledger["attempts"] if e["status"] == "COMPLETE"]
    if len(complete) != 2 or {e["attempt"] for e in complete} != {p.name for p in roots}:
        raise M8S2ProtocolViolation("sizing complete claims differ or were cherry-picked")
    records: list[dict[str, Any]] = []
    all_calls: list[dict[str, Any]] = []
    accepted_timed_seconds = 0.0
    seen_process_ids: set[int] = set()
    for entry in ledger["attempts"]:
        root = _attempt_path(campaign, entry["attempt"])
        manifest = _json(root / "manifest.json")
        _verify_files(root, manifest)
        if manifest["process_id"] in seen_process_ids:
            raise M8S2ProtocolViolation("sizing attempt history reused an OS process")
        seen_process_ids.add(manifest["process_id"])
        if (
            sha256_file(root / "manifest.json") != entry["manifest_sha256"]
            or manifest["status"] != entry["status"]
            or manifest["process_uuid"] != entry["process_uuid"]
            or manifest["process_id"] != entry["process_id"]
            or manifest["logical_process_id"] != entry["logical_process_id"]
            or manifest["attempt_id"] != entry["attempt_id"]
        ):
            raise M8S2ProtocolViolation("sizing manifest/claim identity differs")
        bindings = ledger["bindings"]
        if bindings != _bindings(
            execution_commit,
            bindings["runtime_policy_binding_sha256"],
            bindings["input_gate_receipt_sha256"],
            bindings["qualification_receipt_sha256"],
        ):
            raise M8S2ProtocolViolation("sizing frozen aggregation bindings differ")
        if any(manifest.get(k) != v for k, v in bindings.items()):
            raise M8S2ProtocolViolation("sizing process binding differs")
        grant = require_authorization(
            root / "authorization.json",
            scope="sizing-only",
            execution_commit=execution_commit,
            logical_pass_id=manifest["logical_process_id"],
            campaign_root=campaign,
            runtime_policy_sha256=bindings["runtime_policy_binding_sha256"],
            input_gate_receipt_sha256=bindings["input_gate_receipt_sha256"],
            qualification_receipt_sha256=bindings["qualification_receipt_sha256"],
            candidate_readiness_receipt_sha256=_json(root / "authorization.json").get(
                "candidate_readiness_receipt_sha256"
            ),
            sizing_plan_sha256=SIZING_PLAN_SHA256,
        )
        if manifest["authorization_id"] != grant["authorization_id"] or manifest[
            "authorization_sha256"
        ] != sha256_file(root / "authorization.json"):
            raise M8S2ProtocolViolation("sizing authorization identity differs")
        call_document = _json(root / "calls.json")
        if set(call_document) != {"calls"}:
            raise M8S2ProtocolViolation("sizing call document has unexpected retained fields")
        calls = call_document["calls"]
        if not isinstance(calls, list) or len(calls) > 18:
            raise M8S2ProtocolViolation("sizing call count differs")
        for index, call in enumerate(calls):
            verify_call(call, index, ids, manifest["process_uuid"])
            bound = by_id[call["condition_id"]]
            for actual, frozen in {
                "input_sha256": "full_XYZIT_sha256",
                "M7_XYZT_sha256": "M7_expected_XYZT_sha256",
                "selected_global_row_sha256": "selected_global_row_sha256",
                "point_count": "point_count",
                "cpu_analytic_candidate_pillar_count": "cpu_analytic_candidate_pillar_count",
            }.items():
                if call[actual] != bound[frozen]:
                    raise M8S2ProtocolViolation("sizing recorded frozen input identity differs")
        if (
            manifest["schema_version"] != ATTEMPT_SCHEMA
            or manifest["scientific_calls"] != 0
            or manifest["ground_truth_loaded"] is not False
            or manifest["semantic_predictions_retained"] is not False
            or manifest["engineering_calls_completed"] != len(calls)
            or not len(calls) <= manifest["engineering_calls_started"] <= 18
        ):
            raise M8S2ProtocolViolation("sizing attempt schema or accounting differs")
        if entry["status"] == "INCOMPLETE":
            if manifest["accepted_engineering_calls"] != 0:
                raise M8S2ProtocolViolation("incomplete sizing cannot supply accepted timings")
            continue
        if (
            entry["status"] != "COMPLETE"
            or len(calls) != 18
            or manifest["accepted_engineering_calls"] != 18
        ):
            raise M8S2ProtocolViolation("sizing process is not complete")
        _number(manifest["initialization_seconds"])
        records.append(manifest)
        all_calls.extend(calls[2:])
        accepted_timed_seconds += sum(_number(c["elapsed_seconds"]) for c in calls)
    if (
        {r["logical_process_id"] for r in records} != set(SIZING_PROCESS_IDS)
        or len({r["process_uuid"] for r in records}) != 2
        or len({r["process_id"] for r in records}) != 2
        or len({r["worker_hostname"] for r in records}) != 1
    ):
        raise M8S2ProtocolViolation("sizing process independence differs")
    # Stable logical-process order, regardless of --pass-input ordering.
    ordered = sorted(records, key=lambda r: r["logical_process_id"])
    calls = [c for r in ordered for c in all_calls if c["process_uuid"] == r["process_uuid"]]
    durations = [_number(c["elapsed_seconds"]) for c in calls]
    starts = [_number(r["initialization_seconds"]) for r in ordered]
    estimate = estimate_future_cost(
        measured_call_seconds=durations, initialization_seconds=starts, usd_per_hour=usd_per_hour
    )
    estimates = estimate["estimates"]
    return {
        "schema_version": "laserperception.m8.s2.sizing-result.v1",
        "status": "ACCEPTED",
        **ledger["bindings"],
        "process_uuids": [r["process_uuid"] for r in ordered],
        "initialization_seconds": starts,
        "measured_call_count": 32,
        "warmup_call_count": 4,
        "engineering_calls": 36,
        "scientific_calls": 0,
        "measured_call_seconds": durations,
        "measured_calls": calls,
        "observed_minimum_seconds": min(durations),
        "median_seconds": median(durations),
        "observed_maximum_seconds": max(durations),
        "memory_summary": {
            key: {
                "minimum": min(c["memory"][key] for c in calls),
                "median": median(c["memory"][key] for c in calls),
                "maximum": max(c["memory"][key] for c in calls),
            }
            for key in sorted(MEMORY_KEYS)
        },
        "memory_method": "Torch allocator only; no device/system measurement",
        "cost_estimate": estimate,
        "observed_sizing_timed_seconds_including_warmups_and_initialization": accepted_timed_seconds
        + sum(starts),
        "observed_sizing_timed_gpu_cost_at_input_rate": (accepted_timed_seconds + sum(starts))
        / 3600
        * _number(usd_per_hour),
        "budget_decision": {
            workload: {
                scenario: estimates[workload][scenario]["usd_at_input_rate"]  # type: ignore[index]
                for scenario in ("median", "observed_maximum")
            }
            for workload in ("repeatability", "three_full_passes", "total_accepted_science")
        },
        "ground_truth_loaded": False,
        "semantic_predictions_retained": False,
        "repeatability_authorized": False,
        "full_pass_authorized": False,
    }
