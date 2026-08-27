"""Authorization, repeatability, resume, and frozen metric helpers for future M7 execution."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np

from benchmarks.m7.evidence import PairedSets, PoseKey
from benchmarks.m7.protocol import (
    CHECKPOINT_SHA256,
    ENGINE_SHA256,
    EVALUATOR_IDENTITY,
    ONNX_SHA256,
    PROTOCOL_FREEZE_COMMIT,
    REPEATABILITY_REPETITIONS,
    ProtocolViolation,
    canonical_condition_ids,
)
from benchmarks.m7.provenance import atomic_write_json, canonical_json_sha256, model_ready_sha256
from laserperception.detection.types import Detection3D
from laserperception.evaluation.kitti_m6b import M6bGroundTruthBox, MatchSummary, match_detections

TDetector = TypeVar("TDetector")
TResult = TypeVar("TResult")
RAW_OUTPUT_NAMES = ("cls_score", "bbox_pred", "dir_cls_pred")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ExecutionIdentity:
    """All identities that must match before detector construction is reachable."""

    implementation_commit: str
    input_ledger_sha256: str
    engine_sha256: str = ENGINE_SHA256
    checkpoint_sha256: str = CHECKPOINT_SHA256
    onnx_sha256: str = ONNX_SHA256
    evaluator_identity: str = EVALUATOR_IDENTITY
    protocol_commit: str = PROTOCOL_FREEZE_COMMIT

    def to_dict(self) -> dict[str, str]:
        """Return the frozen authorization identity mapping."""

        return {
            "m7_protocol_commit": self.protocol_commit,
            "m7_implementation_commit": self.implementation_commit,
            "m7_input_ledger_sha256": self.input_ledger_sha256,
            "engine_sha256": self.engine_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "onnx_sha256": self.onnx_sha256,
            "evaluator_identity": self.evaluator_identity,
        }


@dataclass(frozen=True, slots=True)
class RuntimeArtifacts:
    """External artifact paths verified before canonical detector construction is reachable."""

    input_ledger: Path
    engine: Path
    checkpoint: Path
    onnx: Path
    evaluator_identity: str

    def verify_input_ledger(self, expected: ExecutionIdentity) -> None:
        """Verify the authorized ledger file before parsing or detector construction."""

        if not self.input_ledger.is_file():
            raise FileNotFoundError(f"M7 input ledger is missing: {self.input_ledger}")
        actual = _sha256_file(self.input_ledger)
        if actual != expected.input_ledger_sha256:
            raise ProtocolViolation(
                "M7 input ledger SHA256 mismatch: "
                f"expected {expected.input_ledger_sha256}, found {actual}"
            )

    def verify_runtime(self, expected: ExecutionIdentity) -> None:
        """Hash frozen detector artifacts after strict ledger validation."""

        for name, path, expected_sha256 in (
            ("engine", self.engine, expected.engine_sha256),
            ("checkpoint", self.checkpoint, expected.checkpoint_sha256),
            ("ONNX", self.onnx, expected.onnx_sha256),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"M7 {name} is missing: {path}")
            actual = _sha256_file(path)
            if actual != expected_sha256:
                raise ProtocolViolation(
                    f"M7 {name} SHA256 mismatch: expected {expected_sha256}, found {actual}"
                )
        if self.evaluator_identity != expected.evaluator_identity:
            raise ProtocolViolation("M7 evaluator identity mismatch")

    def verify(self, expected: ExecutionIdentity) -> None:
        """Compatibility helper verifying ledger then remaining runtime artifacts."""

        self.verify_input_ledger(expected)
        self.verify_runtime(expected)


def verify_inference_authorization(
    authorization: Mapping[str, object],
    expected: ExecutionIdentity,
) -> None:
    """Hard-stop on missing or mismatched owner authorization before detector initialization."""

    if authorization.get("schema_version") != "laserperception.m7.inference-authorization.v1":
        raise ProtocolViolation("M7 inference authorization schema is missing or invalid")
    if authorization.get("authorized_for_inference") is not True:
        raise ProtocolViolation("M7 detector inference is not explicitly authorized")
    for name, expected_value in expected.to_dict().items():
        if authorization.get(name) != expected_value:
            raise ProtocolViolation(f"M7 inference authorization identity mismatch: {name}")


def load_inference_authorization(
    path: str | Path, expected: ExecutionIdentity
) -> dict[str, object]:
    """Load and verify a future owner-created authorization artifact."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ProtocolViolation("M7 inference authorization must be a JSON object")
    record = dict(value)
    verify_inference_authorization(record, expected)
    return record


def _run_authorized_for_test(
    authorization: Mapping[str, object],
    expected: ExecutionIdentity,
    detector_factory: Callable[[], TDetector],
    execute: Callable[[TDetector], TResult],
) -> TResult:
    """Private unit-test helper; the canonical runner has no arbitrary execute callback."""

    verify_inference_authorization(authorization, expected)
    detector = detector_factory()
    return execute(detector)


@dataclass(frozen=True, slots=True)
class DetectorObservation:
    """Fixed detector-return boundary consumed by the canonical corpus runner."""

    raw_outputs: Mapping[str, np.ndarray]
    detection_frame: Mapping[str, object]
    payload: Mapping[str, object]

    def hashes(self) -> ObservationHashes:
        """Return frozen raw and DetectionFrame identities."""

        return observation_hashes(self.raw_outputs, self.detection_frame)


def require_actual_input(points: np.ndarray, expected_sha256: str) -> str:
    """Hash the exact immutable array immediately before its detector invocation."""

    if points.flags.writeable:
        raise ProtocolViolation("M7 detector input must be read-only after regeneration")
    actual = model_ready_sha256(points)
    if actual != expected_sha256:
        raise ProtocolViolation(
            f"M7 actual detector input SHA256 mismatch: expected {expected_sha256}, found {actual}"
        )
    return actual


@dataclass(frozen=True, slots=True)
class ObservationHashes:
    """Raw-network and DetectionFrame identities for one detector invocation."""

    cls_score: str
    bbox_pred: str
    dir_cls_pred: str
    detection_frame: str

    def to_dict(self) -> dict[str, str]:
        """Return stable repeatability/checkpoint fields."""

        return {
            "cls_score": self.cls_score,
            "bbox_pred": self.bbox_pred,
            "dir_cls_pred": self.dir_cls_pred,
            "detection_frame": self.detection_frame,
        }


def _tensor_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    header = f"{value.dtype.str}|{value.shape}".encode("ascii")
    return hashlib.sha256(header + b"\0" + value.tobytes(order="C")).hexdigest()


def observation_hashes(
    raw_outputs: Mapping[str, np.ndarray],
    detection_frame: Mapping[str, object],
) -> ObservationHashes:
    """Hash the three frozen raw tensors and canonical DetectionFrame JSON."""

    if set(raw_outputs) != set(RAW_OUTPUT_NAMES):
        raise ProtocolViolation("M7 raw detector outputs do not match the frozen tensor names")
    return ObservationHashes(
        cls_score=_tensor_sha256(raw_outputs["cls_score"]),
        bbox_pred=_tensor_sha256(raw_outputs["bbox_pred"]),
        dir_cls_pred=_tensor_sha256(raw_outputs["dir_cls_pred"]),
        detection_frame=canonical_json_sha256(dict(detection_frame)),
    )


def repeatability_condition(
    execute: Callable[[], ObservationHashes],
    *,
    repetitions: int = REPEATABILITY_REPETITIONS,
) -> ObservationHashes:
    """Execute exactly ten repeats and return repeat 1 only after exact identity PASS."""

    if repetitions != REPEATABILITY_REPETITIONS:
        raise ProtocolViolation("canonical M7 repeatability requires exactly ten repetitions")
    observed = tuple(execute() for _ in range(repetitions))
    canonical = observed[0]
    for index, value in enumerate(observed[1:], start=2):
        if value != canonical:
            differing = [
                name
                for name in canonical.__dataclass_fields__
                if getattr(canonical, name) != getattr(value, name)
            ]
            raise ProtocolViolation(
                f"M7 repeatability differs at repetition {index}: {', '.join(differing)}"
            )
    return canonical


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    """Protocol/runtime identity embedded in every atomic condition checkpoint."""

    implementation_commit: str
    input_ledger_sha256: str
    engine_sha256: str = ENGINE_SHA256
    checkpoint_sha256: str = CHECKPOINT_SHA256
    onnx_sha256: str = ONNX_SHA256
    evaluator_identity: str = EVALUATOR_IDENTITY
    protocol_commit: str = PROTOCOL_FREEZE_COMMIT

    def to_dict(self) -> dict[str, str]:
        """Return exact resume-bound identities."""

        return {
            "protocol_commit": self.protocol_commit,
            "implementation_commit": self.implementation_commit,
            "input_ledger_sha256": self.input_ledger_sha256,
            "engine_sha256": self.engine_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "onnx_sha256": self.onnx_sha256,
            "evaluator_identity": self.evaluator_identity,
        }


class M7CheckpointStore:
    """Atomic, identity-checked progress that never alters the frozen corpus order."""

    def __init__(
        self,
        root: str | Path,
        identity: CheckpointIdentity,
        condition_ids: Sequence[str] | None = None,
        *,
        _allow_synthetic_fixture: bool = False,
    ) -> None:
        self.root = Path(root)
        self.identity = identity
        self.condition_ids = tuple(condition_ids or canonical_condition_ids())
        if len(self.condition_ids) != len(set(self.condition_ids)) or not self.condition_ids:
            raise ProtocolViolation(
                "M7 checkpoint condition identities must be nonempty and unique"
            )
        if not _allow_synthetic_fixture and self.condition_ids != canonical_condition_ids():
            raise ProtocolViolation(
                "canonical M7 resume cannot shorten or reorder the 1,712 conditions"
            )
        self._load_or_initialize_progress()

    def _checkpoint_path(self, condition: str) -> Path:
        frame_id, arm = condition.split("|", 1)
        drive_id, frame_index = frame_id.split("/", 1)
        return self.root / "conditions" / drive_id / f"{frame_index}_{arm}.json"

    def _progress_path(self) -> Path:
        return self.root / "progress.json"

    def _base_progress(self) -> dict[str, object]:
        return {
            "schema_version": "laserperception.m7.progress.v1",
            "identity": self.identity.to_dict(),
            "condition_ids": list(self.condition_ids),
            "conditions": {condition: {"status": "PENDING"} for condition in self.condition_ids},
        }

    def _load_or_initialize_progress(self) -> None:
        path = self._progress_path()
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("identity") != self.identity.to_dict():
                raise ProtocolViolation("M7 progress identity differs from the frozen execution")
            if value.get("condition_ids") != list(self.condition_ids):
                raise ProtocolViolation("M7 progress corpus is shortened, reordered, or changed")
            conditions = value.get("conditions")
            if not isinstance(conditions, Mapping) or set(conditions) != set(self.condition_ids):
                raise ProtocolViolation("M7 progress condition set is malformed")
            self.progress = dict(value)
            return
        self.progress = self._base_progress()
        atomic_write_json(path, self.progress)

    def load_complete(
        self,
        condition: str,
        *,
        expected_input_sha256: str,
    ) -> dict[str, object] | None:
        """Load a completed checkpoint only after every frozen identity and payload hash agrees."""

        if condition not in self.condition_ids:
            raise ProtocolViolation(f"condition is outside the frozen M7 corpus: {condition}")
        conditions = self.progress["conditions"]
        assert isinstance(conditions, Mapping)
        status_record = conditions[condition]
        assert isinstance(status_record, Mapping)
        path = self._checkpoint_path(condition)
        if not path.exists():
            if status_record.get("status") == "COMPLETE":
                raise ProtocolViolation(f"completed M7 condition lacks its checkpoint: {condition}")
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "COMPLETE" or record.get("identity") != self.identity.to_dict():
            raise ProtocolViolation(f"M7 checkpoint identity is invalid: {condition}")
        if record.get("condition_id") != condition:
            raise ProtocolViolation(f"M7 checkpoint targets another condition: {condition}")
        for field in ("input_sha256", "raw_output_hashes", "detection_frame_sha256"):
            if field not in record:
                raise ProtocolViolation(f"M7 checkpoint is missing {field}: {condition}")
        if record.get("input_sha256") != expected_input_sha256:
            raise ProtocolViolation(f"M7 checkpoint input identity differs: {condition}")
        if record.get("checkpoint_payload_sha256") != canonical_json_sha256(
            {key: value for key, value in record.items() if key != "checkpoint_payload_sha256"}
        ):
            raise ProtocolViolation(f"M7 checkpoint payload hash differs: {condition}")
        return dict(record)

    def save_complete(
        self,
        condition: str,
        *,
        input_sha256: str,
        hashes: ObservationHashes,
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        """Atomically save one completed result and refuse any duplicate inference result."""

        if self.load_complete(condition, expected_input_sha256=input_sha256) is not None:
            raise ProtocolViolation(
                f"M7 completed condition cannot be rerun or overwritten: {condition}"
            )
        record: dict[str, object] = {
            "schema_version": "laserperception.m7.checkpoint.v1",
            "status": "COMPLETE",
            "identity": self.identity.to_dict(),
            "condition_id": condition,
            "input_sha256": input_sha256,
            "raw_output_hashes": {
                "cls_score": hashes.cls_score,
                "bbox_pred": hashes.bbox_pred,
                "dir_cls_pred": hashes.dir_cls_pred,
            },
            "detection_frame_sha256": hashes.detection_frame,
            "payload": dict(payload),
        }
        record["checkpoint_payload_sha256"] = canonical_json_sha256(record)
        atomic_write_json(self._checkpoint_path(condition), record)
        conditions = self.progress["conditions"]
        assert isinstance(conditions, dict)
        conditions[condition] = {
            "status": "COMPLETE",
            "checkpoint_payload_sha256": record["checkpoint_payload_sha256"],
        }
        atomic_write_json(self._progress_path(), self.progress)
        return record

    def pending_condition_ids(self) -> tuple[str, ...]:
        """Return incomplete conditions in unchanged canonical order."""

        conditions = self.progress["conditions"]
        assert isinstance(conditions, Mapping)
        pending = []
        for condition in self.condition_ids:
            record = conditions[condition]
            assert isinstance(record, Mapping)
            if record.get("status") != "COMPLETE":
                pending.append(condition)
        return tuple(pending)


def gap_recovery(tp_x: int, *, baseline_tp: int, gap: int) -> float:
    """Return the unclamped frozen continuous recovery value."""

    return (tp_x - baseline_tp) / gap


def paired_recovery(detected: set[PoseKey], sets: PairedSets) -> dict[str, float]:
    """Return exact E-only/shared/neither rates for one deterministic arm."""

    def rate(values: tuple[PoseKey, ...]) -> float:
        return len(detected.intersection(values)) / len(values) if values else 0.0

    return {
        "r_gain": rate(sets.e_only),
        "r_shared": rate(sets.shared),
        "r_novel": rate(sets.neither),
    }


def car_interpretation(g_car: float, r_gain: float, r_shared: float) -> bool:
    """Apply only the owner-approved conservative three-part Car gate."""

    return g_car >= 0.50 and r_gain >= 0.50 and r_shared >= 15 / 16


def factorial_contrasts(*, a: float, b: float, c: float, d: float) -> dict[str, float]:
    """Return frozen descriptive L/P/I contrasts; F is intentionally absent."""

    return {
        "L": ((b - a) + (d - c)) / 2,
        "P": ((c - a) + (d - b)) / 2,
        "I": d - b - c + a,
    }


def frozen_primary_match(
    predictions: Sequence[Detection3D],
    targets: Sequence[M6bGroundTruthBox],
    neighbour_ignores: Sequence[M6bGroundTruthBox],
    *,
    class_name: str,
) -> MatchSummary:
    """Delegate the M7 primary match to the unchanged M6b matcher and frozen thresholds."""

    return match_detections(
        predictions,
        targets,
        neighbour_ignores,
        class_name=class_name,
        iou_threshold=0.50,
        score_threshold=0.25,
    )
