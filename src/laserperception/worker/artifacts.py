"""Streaming artifact identities; no downloads or optional runtime discovery."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from laserperception.perception.serialization import JsonRecord

from .paths import validate_relative_path


@dataclass(frozen=True)
class VerifiedArtifact(JsonRecord):
    name: str
    relative_path: str
    byte_size: int
    sha256: str
    role: str
    provenance: str

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_relative_path(self.relative_path)
        if self.byte_size < 0 or not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("artifact requires a non-negative byte size and lowercase SHA256")


def file_identity(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def verify_artifact(path: Path, expected: VerifiedArtifact) -> None:
    size, digest = file_identity(path)
    if size != expected.byte_size:
        raise ValueError(f"artifact byte size mismatch: expected {expected.byte_size}, got {size}")
    if digest != expected.sha256:
        raise ValueError("artifact SHA256 mismatch")
