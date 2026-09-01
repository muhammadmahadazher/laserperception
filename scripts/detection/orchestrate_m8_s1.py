#!/usr/bin/env python3
"""Spawn canonical M8 S1 logical passes as separate fresh Python processes."""

from __future__ import annotations

import argparse
import subprocess
import sys
import uuid
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stage-r", "primary-pass", "zero-intensity-pass"))
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--runtime-policy-binding", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--full-ledger", type=Path, required=True)
    parser.add_argument("--date-root", type=Path, required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    repetitions = 10 if args.mode == "stage-r" else 3
    worker = Path(__file__).with_name("run_m8_s1.py")
    session = str(uuid.uuid4())
    for logical_index in range(1, repetitions + 1):
        logical_id = f"{args.mode}-{logical_index}"
        attempt_id = f"{session}-attempt-1"
        root = args.attempt_root / logical_id / attempt_id
        command = [
            sys.executable,
            str(worker),
            args.mode,
            "--repository-root",
            str(args.repository_root),
            "--runtime-commit",
            args.runtime_commit,
            "--runtime-policy-binding",
            str(args.runtime_policy_binding),
            "--authorization",
            str(args.authorization),
            "--full-ledger",
            str(args.full_ledger),
            "--date-root",
            str(args.date_root),
            "--attempt-root",
            str(root),
            "--logical-pass-id",
            logical_id,
            "--attempt-id",
            attempt_id,
            "--output",
            str(root / "raw_pass.json"),
        ]
        status = subprocess.run(command, cwd=args.repository_root, check=False)
        if status.returncode != 0:
            raise RuntimeError(
                f"{logical_id} attempt is INCOMPLETE; preserve it and restart the entire "
                "logical pass from condition 1 in a new process"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
