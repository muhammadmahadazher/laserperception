"""Task records separate persistence, qualification and scientific authorization."""

import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Literal

from laserperception.perception.serialization import JsonRecord

from .artifacts import VerifiedArtifact
from .paths import validate_relative_path


class TaskType(str, Enum):
    ENGINEERING = "engineering_only"
    QUALIFICATION = "qualification"
    STAGE_R = "stage_r"
    PRIMARY = "primary"
    TRAINING = "training"


@dataclass(frozen=True)
class RuntimeDescriptor(JsonRecord):
    runtime_id: str
    provider: str | None
    external: bool
    description: str


@dataclass(frozen=True)
class EnvironmentEntry(JsonRecord):
    name: Literal["os", "python", "architecture", "framework", "container_digest"]
    value: str


@dataclass(frozen=True)
class AuthorizationReference(JsonRecord):
    role: TaskType
    relative_path: str
    sha256: str
    runtime_id: str
    execution_commit: str

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_relative_path(self.relative_path)
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("authorization reference requires SHA256")
        if not re.fullmatch(r"[0-9a-f]{40}", self.execution_commit):
            raise ValueError("authorization reference requires full execution commit")


@dataclass(frozen=True)
class TaskManifest(JsonRecord):
    schema_version: Literal["1.0"]
    task_id: str
    created_utc: str
    repository: str
    execution_commit: str
    branch: str | None
    task_type: TaskType
    runtime: RuntimeDescriptor
    environment: tuple[EnvironmentEntry, ...]
    inputs: tuple[VerifiedArtifact, ...]
    outputs: tuple[VerifiedArtifact, ...]
    steps: tuple[str, ...]
    authorizations: tuple[AuthorizationReference, ...]
    attempted_calls: int
    accepted_calls: int
    failed_calls: int
    checkpoint_state: Literal["pending", "persisted", "verified"]
    completion_state: Literal["planned", "running", "failed", "complete"]
    unique_worker_state_remaining: bool

    def __post_init__(self) -> None:
        super().__post_init__()
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", self.task_id):
            raise ValueError("task_id must be one portable identifier")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.repository):
            raise ValueError("repository must be owner/name, never a credential-bearing URL")
        if not re.fullmatch(r"[0-9a-f]{40}", self.execution_commit):
            raise ValueError("execution_commit must be a full Git SHA")
        stamp = datetime.fromisoformat(self.created_utc.replace("Z", "+00:00"))
        offset = stamp.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise ValueError("creation time must be UTC")
        counts = (self.attempted_calls, self.accepted_calls, self.failed_calls)
        if min(counts) < 0 or self.accepted_calls + self.failed_calls > self.attempted_calls:
            raise ValueError("call accounting is inconsistent")
        if self.task_type in {TaskType.ENGINEERING, TaskType.QUALIFICATION} and any(counts):
            raise ValueError("engineering/qualification tasks cannot record detector calls")
        if len({e.name for e in self.environment}) != len(self.environment):
            raise ValueError("duplicate environment key")
        for artifacts in (self.inputs, self.outputs):
            if len({a.relative_path for a in artifacts}) != len(artifacts):
                raise ValueError("duplicate artifact path")
        if self.completion_state == "complete" and (
            self.unique_worker_state_remaining or self.checkpoint_state != "verified"
        ):
            raise ValueError(
                "complete tasks require verified persistence and no unique worker state"
            )
        for auth in self.authorizations:
            if (
                auth.runtime_id != self.runtime.runtime_id
                or auth.execution_commit != self.execution_commit
            ):
                raise ValueError("authorization reference runtime/commit mismatch")
        if self.task_type != TaskType.ENGINEERING and not any(
            a.role == self.task_type for a in self.authorizations
        ):
            raise ValueError("task type requires a matching scoped authorization reference")
        if self.task_type != TaskType.ENGINEERING and not self.runtime.external:
            raise ValueError("qualification/scientific tasks require an external runtime")

    @property
    def capsule_path(self) -> str:
        return f"_CLOUD_WORK/{self.task_id}"
