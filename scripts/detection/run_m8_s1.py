#!/usr/bin/env python3
"""Narrow entry point for the frozen M8 P1-S1 measurement runtime."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from laserperception.detection.m8_s1_input_gate import (
    DEFAULT_PRIMARY_INPUT_REVALIDATION_WORKERS,
    validate_primary_input_revalidation_workers,
    verify_input_gate_receipt,
)
from laserperception.detection.m8_s1_runtime import (
    CANDIDATE_MANIFEST_PATH,
    AuthorizationIdentity,
    M8S1ProtocolViolation,
    atomic_write_json,
    require_scientific_authorization,
    sha256_file,
    verify_runtime_policy_binding,
    verify_static_bindings,
)
from laserperception.detection.m8_s1_runtime_policy import capture_runtime_policy
from laserperception.worker.guards import require_external_worker


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-worker", action="store_true")
    parser.add_argument(
        "mode",
        choices=(
            "preflight",
            "runtime-binding",
            "stage-r",
            "primary-pass",
            "zero-intensity-pass",
            "aggregate",
        ),
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--runtime-policy-binding", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--input-gate-receipt", type=Path)
    parser.add_argument("--input-revalidation-workers", type=int)
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
    if args.mode != "aggregate":
        require_external_worker(args.external_worker)
    root = args.repository_root.resolve()
    if args.mode == "preflight":
        script = Path(__file__).with_name("run_m8_s1_preflight.py")
        command = [
            sys.executable,
            str(script),
            "--external-worker",
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
        raise M8S1ProtocolViolation("M8 S1 runner HEAD differs from --runtime-commit")
    if args.mode == "runtime-binding":
        atomic_write_json(
            args.output,
            capture_runtime_policy(args.runtime_commit, binding.candidate_manifest),
        )
        return 0

    logical_pass_id = _require_text(args.logical_pass_id, "--logical-pass-id")
    runtime_policy_path = _require_path(
        args.runtime_policy_binding,
        "--runtime-policy-binding",
    )
    receipt_modes = {"stage-r", "primary-pass"}
    input_gate_receipt = (
        _require_path(args.input_gate_receipt, "--input-gate-receipt").resolve()
        if args.mode in receipt_modes
        else None
    )
    if args.mode not in receipt_modes and args.input_gate_receipt is not None:
        raise M8S1ProtocolViolation(
            "--input-gate-receipt is valid only for stage-r and primary-pass"
        )
    if args.mode == "primary-pass":
        input_revalidation_workers = validate_primary_input_revalidation_workers(
            args.input_revalidation_workers
            if args.input_revalidation_workers is not None
            else DEFAULT_PRIMARY_INPUT_REVALIDATION_WORKERS
        )
    else:
        if args.input_revalidation_workers is not None:
            raise M8S1ProtocolViolation(
                "--input-revalidation-workers is valid only for primary-pass"
            )
        input_revalidation_workers = 1
    expected = AuthorizationIdentity(
        args.runtime_commit,
        input_gate_receipt_sha256=(
            sha256_file(input_gate_receipt) if input_gate_receipt is not None else None
        ),
    )
    authorization = require_scientific_authorization(
        args.mode,
        logical_pass_id,
        args.authorization,
        expected,
    )
    runtime_policy_sha256 = authorization["runtime_policy_binding_sha256"]
    if not isinstance(runtime_policy_sha256, str):
        raise AssertionError("verified runtime-policy SHA256 changed type")
    live_policy = capture_runtime_policy(args.runtime_commit, binding.candidate_manifest)
    verify_runtime_policy_binding(
        runtime_policy_path,
        runtime_policy_sha256,
        live_policy,
    )
    if args.mode in receipt_modes:
        assert input_gate_receipt is not None
        verify_input_gate_receipt(
            input_gate_receipt,
            repository_root=root,
            full_ledger=_require_path(args.full_ledger, "--full-ledger").resolve(),
            execution_commit=args.runtime_commit,
        )
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
        logical_pass_id=logical_pass_id,
        attempt_id=args.attempt_id or "",
        input_gate_receipt=input_gate_receipt,
        input_revalidation_workers=input_revalidation_workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
