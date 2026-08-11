"""Small sanitized runtime metadata helpers shared by measured commands."""

from __future__ import annotations

import subprocess
from pathlib import Path


def repository_git_sha(repository_root: str | Path) -> str:
    """Resolve the current 40-character Git commit in Windows or WSL checkouts."""

    root = Path(repository_root).resolve()
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    sha = process.stdout.strip().lower()
    if _is_sha(sha):
        return sha

    pointer = root / ".git"
    if pointer.is_file():
        value = pointer.read_text(encoding="utf-8").strip()
        if value.startswith("gitdir: "):
            raw_path = value.removeprefix("gitdir: ")
            if len(raw_path) >= 3 and raw_path[1:3] == ":/":
                git_directory = Path("/mnt") / raw_path[0].lower() / raw_path[3:]
            else:
                git_directory = Path(raw_path)
                if not git_directory.is_absolute():
                    git_directory = root / git_directory
            process = subprocess.run(
                [
                    "git",
                    f"--git-dir={git_directory}",
                    f"--work-tree={root}",
                    "rev-parse",
                    "HEAD",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            sha = process.stdout.strip().lower()
            if _is_sha(sha):
                return sha
    raise RuntimeError("cannot resolve the measurement commit SHA from this checkout")


def nvidia_smi_value(field: str) -> str:
    """Return the first GPU's sanitized nvidia-smi query value."""

    if not field or any(character not in "abcdefghijklmnopqrstuvwxyz_" for character in field):
        raise ValueError("nvidia-smi field must contain only lowercase letters and underscores")
    completed = subprocess.run(
        ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip().splitlines()[0]


def _is_sha(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)
