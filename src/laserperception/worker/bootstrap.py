"""Explicit external-only qualification entry point. Never imports GPU code at module load."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from laserperception.perception.serialization import JsonRecord

from .artifacts import VerifiedArtifact, file_identity, verify_artifact
from .manifests import TaskManifest, TaskType
from .paths import resolve_artifact_path


@dataclass(frozen=True)
class GPURecord(JsonRecord):
    model: str
    vram_mib: int
    free_mib: int
    driver: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 <= self.free_mib <= self.vram_mib or self.vram_mib <= 0:
            raise ValueError("invalid GPU memory capacity record")


@dataclass(frozen=True)
class FrameworkVersion(JsonRecord):
    name: str
    version: str


@dataclass(frozen=True)
class QualificationRecord(JsonRecord):
    schema_version: Literal["1.0"]
    task_id: str
    runtime_id: str
    execution_commit: str
    gpus: tuple[GPURecord, ...]
    cuda: str
    pytorch: str
    frameworks: tuple[FrameworkVersion, ...]
    verified_inputs: tuple[VerifiedArtifact, ...]
    status: Literal["recorded_for_owner_review_not_scientific_authorization"]


def _git(command: Sequence[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    return result.stdout.strip()


def _collect_external_runtime(task: TaskManifest) -> QualificationRecord:
    """Private implementation detail called only after bootstrap's full guard.

    No scientific inference, memory benchmark, training or provider provisioning occurs.
    The public supported entry point is bootstrap(), never this collector directly.
    """
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    records: list[GPURecord] = []
    for line in result.stdout.splitlines():
        name, total, free, driver = (part.strip() for part in line.split(","))
        records.append(GPURecord(name, int(total), int(free), driver))
    if not records:
        raise RuntimeError("external qualification returned no GPU records")
    torch = importlib.import_module("torch")
    frameworks: list[FrameworkVersion] = []
    for name in ("spconv-cu118", "torch-scatter", "pcdet", "numpy"):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "not_installed"
        frameworks.append(FrameworkVersion(name, version))
    return QualificationRecord(
        "1.0",
        task.task_id,
        task.runtime.runtime_id,
        task.execution_commit,
        tuple(records),
        str(torch.version.cuda),
        str(torch.__version__),
        tuple(frameworks),
        task.inputs,
        "recorded_for_owner_review_not_scientific_authorization",
    )


def bootstrap(
    task: TaskManifest,
    *,
    root: Path,
    output: Path,
    external_worker: bool,
    expected_commit: str,
    runtime_id: str,
    qualification: bool,
    git_reader: Callable[[Sequence[str]], str] = _git,
    collector: Callable[[TaskManifest], QualificationRecord] | None = None,
) -> QualificationRecord:
    """Validate context, owner scope, repository and inputs before hardware collection.

    Failed attempts leave a sanitized event journal; no raw subprocess output is persisted.
    An explicit external-worker flag alone is insufficient.
    """
    if not external_worker or not qualification or not task.runtime.external:
        raise ValueError("bootstrap requires explicit external-worker qualification context")
    if task.task_type != TaskType.QUALIFICATION:
        raise ValueError("bootstrap accepts qualification tasks only; no scientific execution")
    if expected_commit != task.execution_commit or runtime_id != task.runtime.runtime_id:
        raise ValueError("bootstrap runtime/execution identity mismatch")
    if output.exists() or output.is_symlink():
        raise FileExistsError("qualification output must be fresh")
    expected_origin = f"https://github.com/{task.repository}.git"
    if git_reader(["git", "-C", str(root), "rev-parse", "HEAD"]) != expected_commit:
        raise ValueError("worker repository HEAD differs from expected commit")
    origin = git_reader(["git", "-C", str(root), "config", "--get", "remote.origin.url"])
    if origin not in (
        expected_origin,
        expected_origin.removesuffix(".git"),
        f"git@github.com:{task.repository}.git",
    ):
        raise ValueError("worker repository origin differs from manifest")
    if git_reader(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"]):
        raise ValueError("worker checkout has tracked modifications")
    auth = next(a for a in task.authorizations if a.role == TaskType.QUALIFICATION)
    auth_path = resolve_artifact_path(root, auth.relative_path)
    if file_identity(auth_path)[1] != auth.sha256:
        raise ValueError("qualification authorization SHA256 mismatch")
    payload = json.loads(auth_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "laserperception.worker.qualification-authorization.v1",
        "owner_approved": True,
        "task_id": task.task_id,
        "runtime_id": runtime_id,
        "execution_commit": expected_commit,
        "mode": "qualification",
    }
    if payload != expected or type(payload.get("owner_approved")) is not bool:
        raise ValueError("qualification authorization does not match owner-scoped task")
    for artifact in task.inputs:
        verify_artifact(resolve_artifact_path(root, artifact.relative_path), artifact)
    output.parent.mkdir(parents=True, exist_ok=True)
    journal = output.with_name(output.name + ".events.jsonl")
    with journal.open("x", encoding="utf-8") as events:
        events.write('{"state":"collecting","scientific_calls":0}\n')
        events.flush()
        try:
            record = (collector or _collect_external_runtime)(task)
            if (
                record.task_id != task.task_id
                or record.runtime_id != runtime_id
                or record.execution_commit != expected_commit
                or record.verified_inputs != task.inputs
            ):
                raise ValueError("qualification collector returned a different task identity")
            with output.open("x", encoding="utf-8") as stream:
                stream.write(record.to_json())
        except Exception:
            events.write('{"state":"failed","scientific_calls":0}\n')
            raise
        events.write('{"state":"recorded_for_owner_review","scientific_calls":0}\n')
    return record
