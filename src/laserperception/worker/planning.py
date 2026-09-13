"""Worker plans are descriptions, never provider requests or execution permission."""

from dataclasses import dataclass
from typing import Literal

from laserperception.perception import registry
from laserperception.perception.manifests import ArtifactReference
from laserperception.perception.serialization import JsonRecord

from .manifests import TaskManifest, TaskType


@dataclass(frozen=True)
class WorkerPlan(JsonRecord):
    schema_version: Literal["1.0"]
    task: TaskManifest
    model_id: str
    runtime_requirements: tuple[str, ...]
    model_artifacts: tuple[ArtifactReference, ...]
    capsule_target: str
    steps: tuple[str, ...]
    remote_commands: tuple[tuple[str, ...], ...]
    authorization_state: str
    executes: Literal[False] = False


def plan_worker(task: TaskManifest, model_id: str) -> WorkerPlan:
    manifest = registry.get(model_id)
    if task.task_type == TaskType.TRAINING:
        raise ValueError("training is outside the authorized engineering workflow")
    commands: tuple[tuple[str, ...], ...] = (
        ("git", "rev-parse", "HEAD"),
        ("laserperception", "worker", "manifest", "validate", "task.json"),
        (
            "laserperception",
            "worker",
            "bootstrap",
            "task.json",
            "--external-worker",
            "--expected-commit",
            task.execution_commit,
            "--runtime-id",
            task.runtime.runtime_id,
            "--qualification",
            "--root",
            ".",
            "--output",
            "qualification.json",
        ),
    )
    if task.task_type != TaskType.QUALIFICATION:
        commands = commands[:2]
    return WorkerPlan(
        "1.0",
        task,
        model_id,
        manifest.runtime_requirements,
        manifest.artifacts,
        f"lpdrive:{task.capsule_path}",
        (
            "Owner selects external runtime and qualification scope.",
            "Verify repository commit and each required artifact size/SHA256.",
            "Run GT-blind qualification only within separately authorized external context.",
            "Review and bind machine-specific policy; obtain new Stage-R-only authorization.",
            "Repeat Stage R; persist/review raw evidence before any new primary authorization.",
            "Verify GitHub/Drive persistence before worker retirement.",
        ),
        commands,
        "References recorded; not verified authorization or permission to execute.",
    )
