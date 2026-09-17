#!/usr/bin/env python3
"""Run the owner-authorized GT-blind M8 S1 maximum-pillar capacity review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from laserperception.detection.m8_s1_preflight import run_max_pillar_capacity_review
from laserperception.worker.guards import require_external_worker


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-worker", action="store_true")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--full-ledger", type=Path, required=True)
    parser.add_argument("--date-root", type=Path, required=True)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    require_external_worker(args.external_worker)
    result = run_max_pillar_capacity_review(
        repository_root=args.repository_root.resolve(),
        full_ledger=args.full_ledger.resolve(),
        date_root=args.date_root.resolve(),
        census_path=args.census.resolve(),
        runtime_commit=args.runtime_commit,
        output=args.output.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
