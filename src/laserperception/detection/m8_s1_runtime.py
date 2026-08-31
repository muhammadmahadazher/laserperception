"""Fail-closed, CPU-testable contracts for the frozen M8 P1-S1 runtime.

This module deliberately has no ground-truth, evaluator, Torch, CUDA, or DSVT
import.  Scientific entry points must pass its authorization barrier before a
caller is allowed to import or construct the candidate backend.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import TypeVar

import numpy as np

PROTOCOL_FREEZE_COMMIT = "5061d5d2c6a6057fed1f3f537c5857d2d84f6b3f"
PHASE1E_IMPLEMENTATION_COMMIT = "77369c02e3486650cd06624cb796cf1efbc6e3d4"
PHASE1E_MERGE_COMMIT = "8fcf71f527104e439a59bb8cc2376ec332fa5841"
PROTOCOL_MARKDOWN_PATH = Path("docs/m8/M8_S1_PROTOCOL.md")
PROTOCOL_MARKDOWN_BYTES = 23_802
PROTOCOL_MARKDOWN_SHA256 = "1ad58ebbdd04897558ef9802fee6288b806c5e633d393f1bed957ecc6d6f6b10"
PROTOCOL_JSON_PATH = Path("benchmarks/m8/preregistration/m8_s1_protocol.json")
PROTOCOL_JSON_BYTES = 15_956
PROTOCOL_JSON_SHA256 = "c132f60257c6a39debb548461c79bd59c98325484d233db6095b441c638d8e88"
CANDIDATE_MANIFEST_PATH = Path("configs/m8/dsvt_nuscenes_pillar.json")
CANDIDATE_MANIFEST_SHA256 = "aa456e0386e46e9d089a957b1f1a8a4f74ceae70435c7ad8e6ca5e67bb90f4e7"
INPUT_LEDGER_PATH = Path("benchmarks/m8/diagnostics/m8_input_projection_ledger.json")
INPUT_LEDGER_SHA256 = "474e87e34c64d669750d4b6f7a64ac46fc9c5c462693fad79ff7c9547a7f1f7c"
INPUT_REVALIDATION_PATH = Path("benchmarks/m8/diagnostics/m8_input_projection_revalidation.json")
INPUT_REVALIDATION_SHA256 = "71ac9418c29da5efd64f9eaeb03e859f85d6b1c56dc2fe47cef6563a9f960341"
ORDERED_FRAME_SHA256 = "76bd5f7adac3d892ad2fb00cb9cf5f4f73dd475682ee011b7ea9524060c46c95"
UPSTREAM_COMMIT = "8cfc2a6f23eed0b10aabcdc4768c60b184357061"
CONFIG_SHA256 = "b0832e03ad11d4e0b61f0fb07d977e687763caae472a4f87ed750bdc2d13be0f"
CHECKPOINT_SHA256 = "a675149d095eef8ddc0c137ae46eeac075ccc504c7608162c71e7adf318793fb"
EVALUATOR_IDENTITY = "m6b-r2-score-0.25-oriented-bev-iou-0.30-0.50-0.70"

STAGE_R_FRAMES = (
    "2011_09_26_drive_0001/0000000010",
    "2011_09_26_drive_0001/0000000011",
    "2011_09_26_drive_0001/0000000015",
    "2011_09_26_drive_0001/0000000083",
    "2011_09_26_drive_0091/0000000010",
    "2011_09_26_drive_0091/0000000011",
    "2011_09_26_drive_0091/0000000012",
)
SCIENTIFIC_MODES = frozenset({"stage-r", "primary-pass", "zero-intensity-pass"})
PASS_MODES = frozenset({"primary-pass", "zero-intensity-pass"})
TBackend = TypeVar("TBackend")


class M8S1ProtocolViolation(ValueError):
    """Raised before work can cross a frozen M8 P1-S1 boundary."""


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading a potentially large external artifact at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_frame_ids() -> tuple[str, ...]:
    """Return and verify the frozen 428-frame M6/M8 order."""

    frames = tuple(
        [f"2011_09_26_drive_0001/{index:010d}" for index in range(10, 108)]
        + [f"2011_09_26_drive_0091/{index:010d}" for index in range(10, 340)]
    )
    identity = hashlib.sha256(("\n".join(frames) + "\n").encode()).hexdigest()
    if len(frames) != 428 or identity != ORDERED_FRAME_SHA256:
        raise AssertionError("frozen M8 frame order is internally inconsistent")
    return frames


def canonical_condition_ids() -> tuple[str, ...]:
    """Return the 856-condition frame-major, H10-then-H5 order."""

    return tuple(
        f"{frame_id}/{history}" for frame_id in canonical_frame_ids() for history in ("H10", "H5")
    )


def stage_r_condition_ids() -> tuple[str, ...]:
    """Return the frozen 14-call Stage R order."""

    return tuple(
        f"{frame_id}/{history}" for frame_id in STAGE_R_FRAMES for history in ("H10", "H5")
    )


def zero_intensity_copy(points: np.ndarray) -> np.ndarray:
    """Return a row-preserving copy with candidate intensity set to float32 +0."""

    primary = np.asarray(points)
    if primary.ndim != 2 or primary.shape[1] != 5 or primary.dtype != np.float32:
        raise M8S1ProtocolViolation("M8 S1 input must be float32 with five columns")
    result = np.ascontiguousarray(primary.copy())
    before_non_intensity = np.ascontiguousarray(primary[:, [0, 1, 2, 4]]).tobytes()
    result[:, 3] = np.float32(0.0)
    after_non_intensity = np.ascontiguousarray(result[:, [0, 1, 2, 4]]).tobytes()
    if before_non_intensity != after_non_intensity:
        raise M8S1ProtocolViolation("zero-intensity intervention changed another feature")
    intensity_bits = result[:, 3].view(np.uint32)
    if np.any(intensity_bits != np.uint32(0)):
        raise M8S1ProtocolViolation("zero-intensity intervention is not float32 positive zero")
    return result


@dataclass(frozen=True, slots=True)
class StaticBindingReport:
    """Verified, non-authorizing static identities."""

    repository_head: str
    protocol_json: Mapping[str, object]
    candidate_manifest: Mapping[str, object]


def _load_mapping(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise M8S1ProtocolViolation(f"{path.name} must be a JSON object")
    return payload


def _require_identity(path: Path, expected_bytes: int | None, expected_sha256: str) -> None:
    if not path.is_file():
        raise M8S1ProtocolViolation(f"required M8 S1 artifact is missing: {path}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise M8S1ProtocolViolation(f"M8 S1 artifact byte count changed: {path}")
    if sha256_file(path) != expected_sha256:
        raise M8S1ProtocolViolation(f"M8 S1 artifact SHA256 changed: {path}")


def _repository_head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_git_objects(root: Path) -> None:
    for commit in (PROTOCOL_FREEZE_COMMIT, PHASE1E_IMPLEMENTATION_COMMIT, PHASE1E_MERGE_COMMIT):
        result = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=root,
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            raise M8S1ProtocolViolation(f"required frozen Git commit is unavailable: {commit}")


def verify_static_bindings(
    repository_root: str | Path,
    *,
    upstream_root: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> StaticBindingReport:
    """Verify every frozen local identity and optional external runtime artifact."""

    root = Path(repository_root).resolve()
    _require_git_objects(root)
    for relative, byte_count, identity in (
        (PROTOCOL_MARKDOWN_PATH, PROTOCOL_MARKDOWN_BYTES, PROTOCOL_MARKDOWN_SHA256),
        (PROTOCOL_JSON_PATH, PROTOCOL_JSON_BYTES, PROTOCOL_JSON_SHA256),
        (CANDIDATE_MANIFEST_PATH, None, CANDIDATE_MANIFEST_SHA256),
        (INPUT_LEDGER_PATH, None, INPUT_LEDGER_SHA256),
        (INPUT_REVALIDATION_PATH, None, INPUT_REVALIDATION_SHA256),
    ):
        _require_identity(root / relative, byte_count, identity)
    protocol = _load_mapping(root / PROTOCOL_JSON_PATH)
    authorization = protocol.get("authorization")
    if not isinstance(authorization, Mapping) or authorization.get("protocol_frozen") is not True:
        raise M8S1ProtocolViolation("M8 S1 protocol is not frozen")
    if any(
        authorization.get(key) is not False
        for key in (
            "stage_r_authorized",
            "ground_truth_relative_v2_measurement_authorized",
            "explicit_inference_authorization_committed",
        )
    ):
        raise M8S1ProtocolViolation("frozen M8 S1 protocol authorization state changed")
    manifest = _load_mapping(root / CANDIDATE_MANIFEST_PATH)
    if manifest.get("architecture") != "DSVT-Pillar with TransFusion head":
        raise M8S1ProtocolViolation("M8 candidate architecture changed")
    upstream = manifest.get("upstream")
    checkpoint = manifest.get("checkpoint")
    if not isinstance(upstream, Mapping) or not isinstance(checkpoint, Mapping):
        raise M8S1ProtocolViolation("M8 candidate identity is malformed")
    if upstream.get("commit") != UPSTREAM_COMMIT or upstream.get("config_sha256") != CONFIG_SHA256:
        raise M8S1ProtocolViolation("M8 upstream/config identity changed")
    if checkpoint.get("sha256") != CHECKPOINT_SHA256:
        raise M8S1ProtocolViolation("M8 checkpoint identity changed")

    if (upstream_root is None) != (checkpoint_path is None):
        raise M8S1ProtocolViolation("upstream and checkpoint paths must be verified together")
    if upstream_root is not None and checkpoint_path is not None:
        external_root = Path(upstream_root).resolve()
        actual_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=external_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if actual_commit != UPSTREAM_COMMIT:
            raise M8S1ProtocolViolation("external DSVT commit changed")
        config_relative = upstream.get("config_relative_path")
        if not isinstance(config_relative, str):
            raise M8S1ProtocolViolation("candidate config path is malformed")
        _require_identity(external_root / config_relative, None, CONFIG_SHA256)
        expected_bytes = checkpoint.get("bytes")
        if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
            raise M8S1ProtocolViolation("candidate checkpoint byte count is malformed")
        _require_identity(Path(checkpoint_path).resolve(), expected_bytes, CHECKPOINT_SHA256)
    return StaticBindingReport(_repository_head(root), protocol, manifest)


@dataclass(frozen=True, slots=True)
class AuthorizationIdentity:
    """Expected future owner authorization bindings."""

    measurement_runtime_commit: str
    runtime_binding_identity: str

    def to_dict(self) -> dict[str, object]:
        """Return exact fields bound by a future separate authorization artifact."""

        return {
            "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
            "protocol_json_sha256": PROTOCOL_JSON_SHA256,
            "measurement_runtime_reviewed_commit": self.measurement_runtime_commit,
            "runtime_binding_identity": self.runtime_binding_identity,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "config_sha256": CONFIG_SHA256,
            "candidate_manifest_sha256": CANDIDATE_MANIFEST_SHA256,
            "input_ledger_sha256": INPUT_LEDGER_SHA256,
            "evaluator_identity": EVALUATOR_IDENTITY,
        }


def verify_scientific_authorization(
    authorization: Mapping[str, object], expected: AuthorizationIdentity
) -> None:
    """Reject anything except an exact, separately issued future owner authorization."""

    fixed = expected.to_dict()
    required = {
        "schema_version",
        "scientific_inference_authorized",
        "authorization_role",
        "owner_approval",
        "authorization_timestamp_utc",
        "authorization_provenance",
        *fixed,
    }
    if set(authorization) != required:
        raise M8S1ProtocolViolation("M8 S1 authorization schema fields differ")
    if authorization.get("schema_version") != "laserperception.m8.s1.authorization.v1":
        raise M8S1ProtocolViolation("M8 S1 authorization schema is invalid")
    if authorization.get("scientific_inference_authorized") is not True:
        raise M8S1ProtocolViolation("M8 S1 scientific inference is not authorized")
    if authorization.get("authorization_role") != "owner_scientific_inference_authorization":
        raise M8S1ProtocolViolation("M8 S1 authorization role is invalid")
    if authorization.get("owner_approval") is not True:
        raise M8S1ProtocolViolation("M8 S1 owner approval is absent")
    for name in ("authorization_timestamp_utc", "authorization_provenance"):
        value = authorization.get(name)
        if not isinstance(value, str) or not value.strip():
            raise M8S1ProtocolViolation(f"M8 S1 authorization {name} is absent")
    for name, value in fixed.items():
        if authorization.get(name) != value:
            raise M8S1ProtocolViolation(f"M8 S1 authorization binding mismatch: {name}")


def require_scientific_authorization(
    mode: str,
    path: str | Path | None,
    expected: AuthorizationIdentity,
) -> dict[str, object]:
    """Normal fail-closed barrier used before any DSVT scientific construction."""

    if mode not in SCIENTIFIC_MODES:
        raise M8S1ProtocolViolation(f"mode is not a scientific inference mode: {mode}")
    if path is None or not Path(path).is_file():
        raise M8S1ProtocolViolation(
            "M8 S1 scientific inference is normally disabled: no separate owner "
            "authorization artifact exists"
        )
    payload = _load_mapping(Path(path))
    verify_scientific_authorization(payload, expected)
    return payload


def authorize_then_construct(
    mode: str,
    path: str | Path | None,
    expected: AuthorizationIdentity,
    factory: Callable[[], TBackend],
) -> TBackend:
    """Testable ordering guarantee: authorization always precedes construction."""

    require_scientific_authorization(mode, path, expected)
    return factory()


def atomic_write_json(path: str | Path, record: Mapping[str, object]) -> None:
    """Durably write JSON through a sibling temporary file and atomic replace."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    encoded = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def canonical_json_sha256(record: Mapping[str, object]) -> str:
    """Return a deterministic compact JSON identity."""

    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class AttemptIdentity:
    """Identity of one non-spliceable fresh-process scientific attempt."""

    mode: str
    logical_pass_id: str
    attempt_id: str
    process_uuid: str
    process_id: int
    runtime_commit: str

    def __post_init__(self) -> None:
        if self.mode not in SCIENTIFIC_MODES:
            raise M8S1ProtocolViolation("attempt mode is not scientific")
        if not all((self.logical_pass_id, self.attempt_id, self.process_uuid)):
            raise M8S1ProtocolViolation("attempt identity fields must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Serialize the process-bound attempt identity."""

        return {
            "mode": self.mode,
            "logical_pass_id": self.logical_pass_id,
            "attempt_id": self.attempt_id,
            "process_uuid": self.process_uuid,
            "process_id": self.process_id,
            "runtime_commit": self.runtime_commit,
        }


class AtomicAttempt:
    """Ordered attempt ledger that cannot resume or splice a failed process."""

    def __init__(
        self,
        root: str | Path,
        identity: AttemptIdentity,
        *,
        condition_ids: Sequence[str] | None = None,
        allow_test_fixture: bool = False,
    ) -> None:
        self.root = Path(root)
        self.identity = identity
        canonical = (
            stage_r_condition_ids() if identity.mode == "stage-r" else canonical_condition_ids()
        )
        self.condition_ids = tuple(canonical if condition_ids is None else condition_ids)
        if not allow_test_fixture and self.condition_ids != canonical:
            raise M8S1ProtocolViolation("scientific attempt order differs from the frozen order")
        if not self.condition_ids or len(set(self.condition_ids)) != len(self.condition_ids):
            raise M8S1ProtocolViolation("attempt condition IDs must be non-empty and unique")
        if self.root.exists():
            raise M8S1ProtocolViolation("attempt directory already exists; attempts never resume")
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.completed: list[str] = []
        self.failed_calls = 0
        self._write_progress("IN_PROGRESS", None)

    def _progress(self, status: str, failure_reason: str | None) -> dict[str, object]:
        return {
            "schema_version": "laserperception.m8.s1.attempt.v1",
            "status": status,
            "identity": self.identity.to_dict(),
            "start_timestamp_utc": self.started_at,
            "end_timestamp_utc": (
                datetime.now(timezone.utc).isoformat()
                if status in {"COMPLETE", "INCOMPLETE"}
                else None
            ),
            "expected_calls": len(self.condition_ids),
            "attempted_calls": len(self.completed) + self.failed_calls,
            "completed_calls": len(self.completed),
            "accepted_canonical_calls": len(self.completed) if status == "COMPLETE" else 0,
            "failed_calls": self.failed_calls,
            "failure_reason": failure_reason,
            "next_condition_id": (
                self.condition_ids[len(self.completed)]
                if len(self.completed) < len(self.condition_ids)
                else None
            ),
            "completed_condition_ids": list(self.completed),
        }

    def _write_progress(self, status: str, failure_reason: str | None) -> None:
        atomic_write_json(
            self.root / "attempt_manifest.json", self._progress(status, failure_reason)
        )

    def record(self, condition_id: str, payload: Mapping[str, object]) -> None:
        """Write exactly the next condition; a duplicate or skip invalidates the attempt."""

        if len(self.completed) >= len(self.condition_ids):
            raise M8S1ProtocolViolation("attempt already contains every expected condition")
        expected = self.condition_ids[len(self.completed)]
        if condition_id != expected:
            raise M8S1ProtocolViolation(
                f"attempt condition order changed: expected {expected}, received {condition_id}"
            )
        record = {
            "schema_version": "laserperception.m8.s1.condition.v1",
            "status": "COMPLETE",
            "identity": self.identity.to_dict(),
            "condition_id": condition_id,
            "payload": dict(payload),
        }
        record["record_sha256"] = canonical_json_sha256(record)
        atomic_write_json(self.root / "conditions" / f"{len(self.completed):04d}.json", record)
        self.completed.append(condition_id)
        self._write_progress("IN_PROGRESS", None)

    def fail(self, reason: str) -> dict[str, object]:
        """Freeze this process attempt as noncanonical and nonresumable."""

        self.failed_calls += 1
        record = self._progress("INCOMPLETE", reason)
        self._write_progress("INCOMPLETE", reason)
        return record

    def finalize(self) -> dict[str, object]:
        """Atomically expose a canonical final manifest only at the exact call count."""

        if tuple(self.completed) != self.condition_ids or self.failed_calls:
            raise M8S1ProtocolViolation(
                f"cannot finalize {len(self.completed)}/{len(self.condition_ids)} conditions"
            )
        condition_hashes = [
            sha256_file(path) for path in sorted((self.root / "conditions").glob("*.json"))
        ]
        final = self._progress("COMPLETE", None)
        final["condition_file_sha256"] = condition_hashes
        final["result_sha256"] = canonical_json_sha256(final)
        atomic_write_json(self.root / "final_pass_manifest.json", final)
        self._write_progress("COMPLETE", None)
        return final


def verify_three_process_realizations(records: Sequence[Mapping[str, object]]) -> None:
    """Require exactly three distinct complete fresh-process pass identities."""

    if len(records) != 3:
        raise M8S1ProtocolViolation("S1 aggregation requires exactly three canonical passes")
    modes = {record.get("mode") for record in records}
    identities = {record.get("process_uuid") for record in records}
    if len(modes) != 1 or modes - PASS_MODES or len(identities) != 3:
        raise M8S1ProtocolViolation("three S1 passes must use one mode and three processes")


def verify_cross_mode_process_separation(
    primary: Sequence[Mapping[str, object]], zero_intensity: Sequence[Mapping[str, object]]
) -> None:
    """Reject any process identity reused across primary and zero-intensity evidence."""

    primary_ids = {record.get("process_uuid") for record in primary}
    zero_ids = {record.get("process_uuid") for record in zero_intensity}
    if None in primary_ids or None in zero_ids or primary_ids.intersection(zero_ids):
        raise M8S1ProtocolViolation(
            "primary and zero-intensity evidence cannot share a process identity"
        )


def summarize_three(values: Sequence[float]) -> dict[str, object]:
    """Summarize three process realizations without inferential statistics."""

    if len(values) != 3 or not np.isfinite(np.asarray(values, dtype=np.float64)).all():
        raise M8S1ProtocolViolation("S1 spread summary requires three finite values")
    return {
        "pass_values": [float(value) for value in values],
        "minimum": float(min(values)),
        "median": float(median(values)),
        "maximum": float(max(values)),
    }


def paired_history_delta(
    h10_recall: Sequence[float], h5_recall: Sequence[float]
) -> dict[str, object]:
    """Compute H5-H10 per pass before the frozen three-value summary."""

    if len(h10_recall) != 3 or len(h5_recall) != 3:
        raise M8S1ProtocolViolation("paired history contrast requires three aligned passes")
    deltas = [float(h5) - float(h10) for h10, h5 in zip(h10_recall, h5_recall, strict=True)]
    return {
        "formula": "history_delta_i = recall(E2_i) - recall(A2_i)",
        "prohibited_formula_used": False,
        **summarize_three(deltas),
    }


def validate_scientific_condition_payload(payload: Mapping[str, object]) -> None:
    """Validate the future raw evidence envelope without evaluating predictions."""

    required = {
        "frame_id",
        "history",
        "input_sha256",
        "detection_frame_sha256",
        "predictions",
        "classes",
        "evaluator_provenance",
    }
    if not required.issubset(payload):
        raise M8S1ProtocolViolation("scientific condition payload is incomplete")
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise M8S1ProtocolViolation("scientific predictions must be an ordered list")
    prediction_fields = {
        "stable_prediction_index",
        "class_name",
        "score",
        "box_lidar",
        "box_sha256",
        "inside_annotation_fov",
        "primary_disposition",
        "matched_gt_identity",
        "matched_iou",
    }
    if any(
        not isinstance(item, Mapping) or not prediction_fields.issubset(item)
        for item in predictions
    ):
        raise M8S1ProtocolViolation("scientific prediction evidence is incomplete")


def external_artifact_descriptor(
    path: str | Path, *, logical_name: str, role: str, schema_version: str, producer: str
) -> dict[str, object]:
    """Return the compact identity for a reviewed external evidence artifact."""

    source = Path(path)
    return {
        "logical_name": logical_name,
        "role": role,
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
        "schema_version": schema_version,
        "producer_identity": producer,
    }
