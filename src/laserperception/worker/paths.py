"""Portable artifact paths; reject ambiguity before transfers or file creation."""

import re
import unicodedata
from pathlib import Path


def validate_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("artifact path must be a non-empty portable relative POSIX path")
    if any(unicodedata.category(c).startswith("C") for c in value):
        raise ValueError("artifact path contains control characters")
    for part in value.split("/"):
        if part in {"", ".", ".."} or part.endswith((" ", ".")):
            raise ValueError("artifact path contains an empty, dot or ambiguous component")
        if re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", part):
            raise ValueError("artifact path contains a reserved Windows device name")
        if any(c in part for c in '<>"|?*'):
            raise ValueError("artifact path contains a non-portable character")
    return value


def resolve_artifact_path(root: Path, relative: str) -> Path:
    """Resolve only the selected path, including existing symlinks; never walk the root."""
    validate_relative_path(relative)
    base = root.resolve()
    target = base.joinpath(*relative.split("/"))
    if not target.resolve().is_relative_to(base):
        raise ValueError("artifact path escapes selected root through a symlink")
    return target
