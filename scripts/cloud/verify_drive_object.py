#!/usr/bin/env python3
"""Verify a single canonical-Drive object's SHA256 without downloading it."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess

SHA256 = re.compile(r"^[0-9a-f]{64}$")


def verify(relative_path: str, expected: str) -> None:
    """Fail unless rclone reports the expected SHA256 for one lpdrive object."""
    path = relative_path.lstrip("/")
    if not path or ".." in path.split("/"):
        raise ValueError("Drive path must be relative, non-empty, and contain no '..' component")
    expected = expected.lower()
    if not SHA256.fullmatch(expected):
        raise ValueError("expected SHA256 must contain exactly 64 lowercase hexadecimal digits")
    process = subprocess.Popen(
        ["rclone", "cat", f"lpdrive:{path}"],
        stdout=subprocess.PIPE,
    )
    assert process.stdout is not None
    digest = hashlib.sha256()
    for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
        digest.update(chunk)
    if process.wait() != 0:
        raise RuntimeError("rclone failed while reading Drive object")
    actual = digest.hexdigest()
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch: expected {expected}, got {actual}")
    print(f"verified {actual}  lpdrive:{path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("relative_path")
    parser.add_argument("expected_sha256")
    args = parser.parse_args()
    verify(args.relative_path, args.expected_sha256)


if __name__ == "__main__":
    main()
