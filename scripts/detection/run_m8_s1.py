#!/usr/bin/env python3
"""Narrow entry point for the frozen M8 P1-S1 measurement runtime."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from laserperception.detection.m8_s1_runtime import (
    CANDIDATE_MANIFEST_PATH,
    AuthorizationIdentity,
    M8S1ProtocolViolation,
    atomic_write_json,
    require_scientific_authorization,
    verify_static_bindings,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("preflight", "stage-r", "primary-pass", "zero-intensity-pass", "aggregate"),
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--runtime-binding-identity")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--full-ledger", type=Path)
    parser.add_argument("--date-root", type=Path)
    parser.add_argument("--census", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempt-root", type=Path)
    parser.add_argument("--logical-pass-id")
    parser.add_argument("--attempt-id")
    parser.add_argument("--pass-input", action="append", type=Path, default=[])
    return parser


def _require_path(path: Path | None, name: str) -> Path:
    if path is None:
        raise M8S1ProtocolViolation(f"{name} is required for this mode")
    return path


def _require_text(value: str | None, name: str) -> str:
    if value is None or not value.strip():
        raise M8S1ProtocolViolation(f"{name} is required for this mode")
    return value


def _external_runtime_paths(repository_root: Path) -> tuple[str, str]:
    manifest = json.loads((repository_root / CANDIDATE_MANIFEST_PATH).read_text(encoding="utf-8"))
    environment = manifest["environment"]
    upstream_name = environment["upstream_root_variable"]
    checkpoint_name = environment["checkpoint_variable"]
    upstream = os.environ.get(upstream_name)
    checkpoint = os.environ.get(checkpoint_name)
    if not upstream or not checkpoint:
        raise M8S1ProtocolViolation(f"set {upstream_name} and {checkpoint_name}")
    return upstream, checkpoint


def main() -> int:
    args = _parser().parse_args()
    root = args.repository_root.resolve()
    if args.mode == "preflight":
        script = Path(__file__).with_name("run_m8_s1_preflight.py")
        command = [
            sys.executable,
            str(script),
            "--repository-root",
            str(root),
            "--full-ledger",
            str(_require_path(args.full_ledger, "--full-ledger")),
            "--date-root",
            str(_require_path(args.date_root, "--date-root")),
            "--census",
            str(_require_path(args.census, "--census")),
            "--runtime-commit",
            args.runtime_commit,
            "--output",
            str(args.output),
        ]
        return subprocess.run(command, cwd=root, check=False).returncode

    if args.mode == "aggregate":
        if len(args.pass_input) != 3:
            raise M8S1ProtocolViolation("aggregate requires exactly three --pass-input files")
        from laserperception.evaluation.m8_s1_aggregation import aggregate_three_passes

        records = [json.loads(path.read_text(encoding="utf-8")) for path in args.pass_input]
        atomic_write_json(args.output, aggregate_three_passes(records))
        return 0

    binding = verify_static_bindings(root)
    if binding.repository_head != args.runtime_commit:
        raise M8S1ProtocolViolation("scientific runner HEAD differs from --runtime-commit")
    expected = AuthorizationIdentity(
        args.runtime_commit,
        _require_text(args.runtime_binding_identity, "--runtime-binding-identity"),
    )
    require_scientific_authorization(args.mode, args.authorization, expected)
    upstream, checkpoint = _external_runtime_paths(root)
    verify_static_bindings(
        root,
        upstream_root=upstream,
        checkpoint_path=checkpoint,
    )

    # The import below is deliberately unreachable until the separate future
    # authorization passes. It is the only CLI path that loads GT/evaluation.
    from laserperception.evaluation.m8_s1_science import run_scientific_attempt

    run_scientific_attempt(
        mode=args.mode,
        repository_root=root,
        full_ledger=_require_path(args.full_ledger, "--full-ledger").resolve(),
        date_root=_require_path(args.date_root, "--date-root").resolve(),
        runtime_commit=args.runtime_commit,
        attempt_root=_require_path(args.attempt_root, "--attempt-root").resolve(),
        logical_pass_id=args.logical_pass_id or "",
        attempt_id=args.attempt_id or "",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
