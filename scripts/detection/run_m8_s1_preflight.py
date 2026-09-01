#!/usr/bin/env python3
"""Run the separately authorized GT-blind M8 S1 runtime-sizing preflight."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from laserperception.detection.m8_s1_preflight import (
    combine_sizing_workers,
    run_sizing_worker,
)
from laserperception.detection.m8_s1_runtime import atomic_write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--full-ledger", type=Path, required=True)
    parser.add_argument("--date-root", type=Path, required=True)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--worker-output", type=Path)
    return parser


def _worker(args: argparse.Namespace) -> int:
    if args.worker_output is None or args.worker_index not in (1, 2):
        raise ValueError("worker mode requires --worker-index 1/2 and --worker-output")
    run_sizing_worker(
        repository_root=args.repository_root.resolve(),
        full_ledger=args.full_ledger.resolve(),
        date_root=args.date_root.resolve(),
        census_path=args.census.resolve(),
        runtime_commit=args.runtime_commit,
        worker_index=args.worker_index,
        output=args.worker_output.resolve(),
    )
    return 0


def main() -> int:
    args = _parser().parse_args()
    if args.worker_index is not None:
        return _worker(args)
    script = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="laserperception-m8-s1-preflight-") as directory:
        temporary = Path(directory)
        worker_paths = []
        for worker_index in (1, 2):
            worker_output = temporary / f"worker_{worker_index}.json"
            command = [
                sys.executable,
                str(script),
                "--repository-root",
                str(args.repository_root),
                "--full-ledger",
                str(args.full_ledger),
                "--date-root",
                str(args.date_root),
                "--census",
                str(args.census),
                "--runtime-commit",
                args.runtime_commit,
                "--output",
                str(args.output),
                "--worker-index",
                str(worker_index),
                "--worker-output",
                str(worker_output),
            ]
            subprocess.run(command, check=True, cwd=args.repository_root)
            worker_paths.append(worker_output)
        workers = [json.loads(path.read_text(encoding="utf-8")) for path in worker_paths]
        result = combine_sizing_workers(
            workers,
            runtime_commit=args.runtime_commit,
            census_path=args.census.resolve(),
        )
        atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
