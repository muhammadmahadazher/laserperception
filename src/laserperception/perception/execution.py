"""Explicit execution metadata and fail-closed authorization verification."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from .inputs import InputDescription
from .manifests import ModelManifest

POINTPILLARS_CONFIG_SHA256 = "1ffe085179a48b1bf47e15c12674fe0d58d518cb117ce7a2b7fa10dbbdbd4db1"
POINTPILLARS_CHECKPOINT_SHA256 = "f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0"
POINTPILLARS_UPSTREAM_COMMIT = "fe25f7a51d36e3702f961e198894580d83c4387b"
POINTPILLARS_CONFIG_RELATIVE = (
    "configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py"
)


class RuntimeTarget(str, Enum):
    EXTERNAL_CUDA = "external_cuda"
    CPU_TEST = "cpu_test"


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"


@dataclass(frozen=True)
class ExecutionAuthorization:
    """A hashed owner artifact; a reference never manufactures authorization."""

    kind: Literal["generic_prediction", "m8_s1"]
    path: Path
    sha256: str
    mode: str | None = None
    logical_pass_id: str | None = None
    runtime_policy_path: Path | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"generic_prediction", "m8_s1"}:
            raise ValueError("unknown execution authorization kind")
        if not isinstance(self.path, Path):
            raise TypeError("authorization path must be a pathlib.Path")
        if not isinstance(self.sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("authorization reference requires lowercase SHA256")
        if self.runtime_policy_path is not None and not isinstance(self.runtime_policy_path, Path):
            raise TypeError("runtime_policy_path must be a pathlib.Path")
        if self.kind == "m8_s1" and not all(
            isinstance(value, str) and value.strip() for value in (self.mode, self.logical_pass_id)
        ):
            raise ValueError("M8 authorization requires mode and logical pass")
        if self.kind == "m8_s1" and self.runtime_policy_path is None:
            raise ValueError("M8 authorization requires a runtime policy path")
        if self.kind == "generic_prediction" and any(
            (self.mode, self.logical_pass_id, self.runtime_policy_path)
        ):
            raise ValueError("generic prediction authorization cannot carry M8 pass fields")


@dataclass(frozen=True)
class ModelResources:
    """Explicit external model paths; never searched or downloaded implicitly."""

    config_path: Path | None = None
    checkpoint_path: Path | None = None
    candidate_manifest_path: Path | None = None
    upstream_root: Path | None = None

    def __post_init__(self) -> None:
        for name in (
            "config_path",
            "checkpoint_path",
            "candidate_manifest_path",
            "upstream_root",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                raise TypeError(f"{name} must be a pathlib.Path")


@dataclass(frozen=True)
class ExecutionContext:
    """An explicit target description; construction performs no hardware discovery."""

    runtime_target: RuntimeTarget
    precision: Precision
    device_target: str
    runtime_id: str
    task_id: str
    execution_commit: str
    deterministic: bool
    authorization: ExecutionAuthorization | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_target, RuntimeTarget):
            raise TypeError("runtime_target must be RuntimeTarget")
        if not isinstance(self.precision, Precision):
            raise TypeError("precision must be Precision")
        if type(self.deterministic) is not bool:
            raise TypeError("deterministic must be a boolean")
        if self.authorization is not None and not isinstance(
            self.authorization, ExecutionAuthorization
        ):
            raise TypeError("authorization must be ExecutionAuthorization")
        for name in ("device_target", "runtime_id", "task_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be explicit and non-empty")
        if not isinstance(self.execution_commit, str) or not re.fullmatch(
            r"[0-9a-f]{40}", self.execution_commit
        ):
            raise ValueError("execution_commit must be a full Git SHA")
        if self.runtime_target is RuntimeTarget.EXTERNAL_CUDA and not self.device_target.startswith(
            "cuda:"
        ):
            raise ValueError("external_cuda requires an explicit CUDA device target")
        if self.runtime_target is RuntimeTarget.CPU_TEST and self.device_target != "cpu":
            raise ValueError("cpu_test requires device_target='cpu'")


def file_sha256(path: Path) -> str:
    """Hash a file incrementally without importing a model runtime."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execution_context_errors(
    manifest: ModelManifest,
    context: ExecutionContext,
    *,
    supported_precisions: tuple[Precision, ...],
    required_device: str,
) -> tuple[str, ...]:
    """Validate static execution bindings without inspecting the current machine."""

    errors: list[str] = []
    if context.runtime_target.value not in manifest.capabilities.runtime_targets:
        errors.append(f"runtime target {context.runtime_target.value!r} is unsupported")
    if context.runtime_target is not RuntimeTarget.EXTERNAL_CUDA:
        errors.append("built-in detector execution requires external_cuda")
    if context.device_target != required_device:
        errors.append(f"backend requires explicit device {required_device!r}")
    if context.precision not in supported_precisions:
        allowed = ", ".join(precision.value for precision in supported_precisions)
        errors.append(f"backend precision must be one of: {allowed}")
    if not context.deterministic:
        errors.append("backend execution requires deterministic=True")
    return tuple(errors)


def _artifact_sha256(manifest: ModelManifest, role: str) -> str:
    matches = tuple(item for item in manifest.artifacts if item.role == role)
    if len(matches) != 1:
        raise ValueError(f"model manifest lacks one exact {role} artifact")
    if matches[0].sha256 is not None:
        return matches[0].sha256
    if manifest.model_id == "pointpillars-nuscenes-v0.3" and role == "config":
        # Reuse the accepted M7 identity without changing the historical manifest.
        return POINTPILLARS_CONFIG_SHA256
    raise ValueError(f"model manifest lacks one exact {role} SHA256")


def require_canonical_execution_manifest(manifest: ModelManifest) -> None:
    """Prevent a custom registry record from changing a built-in adapter identity."""

    from .registry import builtin_registry

    if builtin_registry().get(manifest.model_id) != manifest:
        raise ValueError("built-in detector execution requires its canonical packaged manifest")


def _verify_pointpillars_checkout(resources: ModelResources) -> None:
    root = resources.upstream_root
    config = resources.config_path
    if root is None or config is None:
        raise ValueError("PointPillars upstream root and config path are required")
    resolved_root = root.expanduser().resolve()
    expected_config = (resolved_root / POINTPILLARS_CONFIG_RELATIVE).resolve()
    if config.expanduser().resolve() != expected_config:
        raise ValueError("PointPillars config path differs from the pinned upstream location")
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=resolved_root,
            check=False,
            capture_output=True,
            text=True,
        )
        flags = subprocess.run(
            ["git", "ls-files", "-v"],
            cwd=resolved_root,
            check=False,
            capture_output=True,
            text=True,
        )
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=resolved_root,
            check=False,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=resolved_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise ValueError("PointPillars upstream Git checkout cannot be verified") from error
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != resolved_root:
        raise ValueError("PointPillars upstream root is not its Git repository top level")
    if flags.returncode != 0 or any(
        not line.startswith("H ") for line in flags.stdout.splitlines()
    ):
        raise ValueError("PointPillars upstream checkout has hidden or unusual index flags")
    if head.returncode != 0 or head.stdout.strip() != POINTPILLARS_UPSTREAM_COMMIT:
        raise ValueError("PointPillars upstream checkout commit mismatch")
    if status.returncode != 0 or status.stdout.strip():
        raise ValueError("PointPillars upstream checkout has tracked modifications")


def _strict_json_object(path: Path) -> dict[str, object]:
    if path.stat().st_size > 64 * 1024:
        raise ValueError("authorization artifact exceeds the 64 KiB limit")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate authorization JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value: {value}")

    payload: object = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise ValueError("authorization artifact must be a JSON object")
    return payload


def generic_prediction_authorization_fields(
    context: ExecutionContext,
    manifest: ModelManifest,
    description: InputDescription,
    *,
    input_sha256: str,
) -> dict[str, object]:
    """Return exact bindings an owner authorization artifact must contain."""

    if not re.fullmatch(r"[0-9a-f]{64}", input_sha256):
        raise ValueError("input_sha256 must be lowercase SHA256")
    return {
        "schema_version": "laserperception.perception.execution-authorization.v1",
        "authorization_role": "owner_detector_execution_authorization",
        "owner_approved": True,
        "action": "detection_predict",
        "model_id": manifest.model_id,
        "runtime_target": context.runtime_target.value,
        "precision": context.precision.value,
        "device_target": context.device_target,
        "runtime_id": context.runtime_id,
        "task_id": context.task_id,
        "execution_commit": context.execution_commit,
        "deterministic": context.deterministic,
        "sample_id": description.sample_id,
        "input_description_sha256": hashlib.sha256(description.to_json().encode()).hexdigest(),
        "input_payload_sha256": input_sha256,
        "config_sha256": _artifact_sha256(manifest, "config"),
        "checkpoint_sha256": _artifact_sha256(manifest, "checkpoint"),
        "upstream_commit": POINTPILLARS_UPSTREAM_COMMIT,
    }


def verify_generic_prediction_authorization(
    context: ExecutionContext,
    manifest: ModelManifest,
    resources: ModelResources,
    description: InputDescription,
    *,
    input_sha256: str,
) -> Mapping[str, object]:
    """Verify owner, input, and resource bindings before a detector import."""

    reference = context.authorization
    if reference is None or reference.kind != "generic_prediction":
        raise ValueError("generic prediction requires a generic_prediction authorization")
    if not reference.path.is_file() or file_sha256(reference.path) != reference.sha256:
        raise ValueError("generic prediction authorization file identity mismatch")
    payload = _strict_json_object(reference.path)
    expected = generic_prediction_authorization_fields(
        context,
        manifest,
        description,
        input_sha256=input_sha256,
    )
    if (
        payload != expected
        or type(payload.get("owner_approved")) is not bool
        or type(payload.get("deterministic")) is not bool
    ):
        raise ValueError("generic prediction authorization scope or schema mismatch")

    _verify_pointpillars_checkout(resources)

    for name, path, expected_sha256 in (
        ("config", resources.config_path, _artifact_sha256(manifest, "config")),
        ("checkpoint", resources.checkpoint_path, _artifact_sha256(manifest, "checkpoint")),
    ):
        if path is None or not path.is_file():
            raise ValueError(f"authorized {name} file is missing")
        if file_sha256(path) != expected_sha256:
            raise ValueError(f"authorized {name} SHA256 mismatch")
    return payload
