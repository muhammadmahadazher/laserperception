"""Framework-independent metadata for external deployment artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from laserperception.detection.mmdet3d_backend import sha256_file


@dataclass(frozen=True, slots=True)
class ExternalArtifactMetadata:
    """Sanitized identity for an artifact that must remain outside Git."""

    logical_name: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        logical_path = PurePosixPath(self.logical_name)
        windows_path = PureWindowsPath(self.logical_name)
        digest = self.sha256.lower()
        if (
            logical_path.is_absolute()
            or windows_path.is_absolute()
            or bool(windows_path.drive)
            or bool(windows_path.root)
            or ".." in logical_path.parts
            or ".." in windows_path.parts
        ):
            raise ValueError("logical_name must be a safe relative path")
        if (
            not self.logical_name.strip()
            or "\\" in self.logical_name
            or self.logical_name != logical_path.as_posix()
        ):
            raise ValueError("logical_name must be a non-empty POSIX-style relative path")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("sha256 must contain exactly 64 lowercase hexadecimal characters")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise TypeError("size_bytes must be an integer")
        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        object.__setattr__(self, "sha256", digest)

    @classmethod
    def from_file(cls, path: str | Path, *, logical_name: str) -> ExternalArtifactMetadata:
        """Measure one existing external artifact without retaining its host path."""

        artifact = Path(path)
        if not artifact.is_file():
            raise FileNotFoundError("external deployment artifact was not found")
        return cls(
            logical_name=logical_name,
            sha256=sha256_file(artifact),
            size_bytes=artifact.stat().st_size,
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible metadata containing no absolute path."""

        return {
            "logical_name": self.logical_name,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "committed": False,
        }
