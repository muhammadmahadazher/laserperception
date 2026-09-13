"""Non-destructive individual-object rclone transfers with full byte verification."""

import json
import re
import shutil
import subprocess
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from laserperception.perception.serialization import JsonRecord

from .artifacts import VerifiedArtifact, verify_artifact
from .paths import resolve_artifact_path, validate_relative_path

CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, check=False, timeout=3600)
    except FileNotFoundError:
        raise RuntimeError("rclone is required; install/configure it separately") from None


@dataclass(frozen=True)
class RemoteObject(JsonRecord):
    remote: str
    relative_path: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]*", self.remote):
            raise ValueError(
                "remote must be a configured rclone name, without credentials or colon"
            )
        validate_relative_path(self.relative_path)

    @property
    def address(self) -> str:
        return f"{self.remote}:{self.relative_path}"


class ArtifactTransfer:
    """All attempts retain staged bytes and a sanitized journal in the selected log root.

    Remote paths should be unique per capsule/artifact. Rclone ignore-existing prevents
    intentional overwrite; round-trip verification detects a racing differing object.
    This is not a distributed transaction or a permission to delete failed attempts.
    """

    def __init__(self, attempt_root: Path, *, runner: CommandRunner = run_command) -> None:
        self.attempt_root = attempt_root
        self.runner = runner

    def _command(self, args: list[str], *, missing_allowed: bool = False) -> bool:
        result = self.runner(["rclone", *args])
        if missing_allowed and result.returncode in (3, 4):
            return False
        if result.returncode != 0:
            # Raw stderr may expose external credential/config details; retain safe status only.
            raise RuntimeError(f"rclone {args[0]} failed with exit code {result.returncode}")
        return True

    def _attempt(self, operation: str, artifact: VerifiedArtifact, remote: RemoteObject) -> Path:
        path = self.attempt_root / uuid.uuid4().hex
        path.mkdir(parents=True, exist_ok=False)
        self._journal(path, operation, artifact, remote, "running")
        return path

    @staticmethod
    def _journal(
        path: Path, operation: str, artifact: VerifiedArtifact, remote: RemoteObject, state: str
    ) -> None:
        # Append events instead of overwriting earlier failed-attempt state.
        with (path / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "operation": operation,
                        "artifact": artifact.to_dict(),
                        "remote": remote.to_dict(),
                        "state": state,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    def pull(
        self, artifact: VerifiedArtifact, remote: RemoteObject, destination_root: Path
    ) -> Path:
        destination = resolve_artifact_path(destination_root, artifact.relative_path)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("pull requires a fresh destination")
        attempt = self._attempt("pull", artifact, remote)
        try:
            staged = attempt / "download"
            self._command(["copyto", remote.address, str(staged), "--ignore-existing"])
            verify_artifact(staged, artifact)
            destination = resolve_artifact_path(destination_root, artifact.relative_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as output, staged.open("rb") as source:
                shutil.copyfileobj(source, output, 1024 * 1024)
            verify_artifact(destination, artifact)
        except Exception:
            self._journal(attempt, "pull", artifact, remote, "failed")
            raise
        self._journal(attempt, "pull", artifact, remote, "verified")
        return destination

    def push(self, artifact: VerifiedArtifact, source_root: Path, remote: RemoteObject) -> Path:
        source = resolve_artifact_path(source_root, artifact.relative_path)
        verify_artifact(source, artifact)
        attempt = self._attempt("push", artifact, remote)
        try:
            exists = self._command(["lsjson", remote.address, "--stat"], missing_allowed=True)
            if exists:
                raise FileExistsError(
                    "push requires a fresh remote object; existing objects are preserved"
                )
            self._command(
                ["copyto", str(source), remote.address, "--ignore-existing", "--immutable"]
            )
            roundtrip = attempt / "roundtrip"
            self._command(["copyto", remote.address, str(roundtrip), "--ignore-existing"])
            verify_artifact(roundtrip, artifact)
        except Exception:
            self._journal(attempt, "push", artifact, remote, "failed")
            raise
        self._journal(attempt, "push", artifact, remote, "verified")
        return attempt
