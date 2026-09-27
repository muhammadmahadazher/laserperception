#!/usr/bin/env python3
"""Future M8 S2 runner with a fail-closed authorization-before-import boundary."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from laserperception.detection.m8_s1_runtime import (
    CANDIDATE_MANIFEST_PATH,
    atomic_write_json,
    sha256_file,
)
from laserperception.detection.m8_s1_runtime import (
    verify_static_bindings as verify_s1_candidate,
)
from laserperception.detection.m8_s2_input_gate import (
    make_input_gate_receipt,
    verify_input_gate_receipt,
    write_input_gate_receipt,
)
from laserperception.detection.m8_s2_planning import qualification_plan
from laserperception.detection.m8_s2_runtime import (
    M8S2ProtocolViolation,
    require_authorization,
    verify_qualification_receipt,
    verify_qualification_worker,
    verify_repeatability_review,
    verify_runtime_policy,
    verify_runtime_policy_document,
    verify_static_bindings,
)
from laserperception.worker.guards import require_external_worker


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "input-gate",
            "runtime-binding",
            "qualification-plan",
            "qualification",
            "repeatability",
            "full-pass",
            "aggregate",
        ),
    )
    parser.add_argument("--external-worker", action="store_true")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--execution-commit")
    parser.add_argument("--full-ledger", type=Path)
    parser.add_argument("--input-gate-receipt", type=Path)
    parser.add_argument("--runtime-policy-binding", type=Path)
    parser.add_argument("--qualification-receipt", type=Path)
    parser.add_argument("--repeatability-review", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--m6-ledger", type=Path)
    parser.add_argument("--date-root", type=Path)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--attempt-root", type=Path)
    parser.add_argument("--logical-pass-id")
    parser.add_argument("--attempt-id")
    parser.add_argument("--pass-input", action="append", type=Path, default=[])
    parser.add_argument("--aggregate-mode", choices=("repeatability", "full-pass"))
    parser.add_argument("--output", type=Path)
    return parser


def _path(value: Path | None, name: str) -> Path:
    if value is None:
        raise M8S2ProtocolViolation(f"{name} is required")
    return value.resolve()


def _text(value: str | None, name: str) -> str:
    if value is None or not value.strip():
        raise M8S2ProtocolViolation(f"{name} is required")
    return value


def _external_candidate(root: Path, upstream: Path, checkpoint: Path) -> None:
    # S1's accepted static candidate verifier performs only Git/file checks.
    verify_s1_candidate(root, upstream_root=upstream, checkpoint_path=checkpoint)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mode in {
        "input-gate",
        "runtime-binding",
        "qualification",
        "repeatability",
        "full-pass",
    }:
        require_external_worker(args.external_worker)
    root = args.repository_root.resolve()

    if args.mode == "aggregate":
        from laserperception.evaluation.m8_s2_aggregation import (
            aggregate_three_full_passes,
            load_completed_attempt,
            review_repeatability,
        )

        selected = _text(args.aggregate_mode, "--aggregate-mode")
        expected_count = 10 if selected == "repeatability" else 3
        if len(args.pass_input) != expected_count:
            raise M8S2ProtocolViolation(f"S2 aggregate requires {expected_count} --pass-input dirs")
        records = [
            load_completed_attempt(path.resolve(), mode=selected) for path in args.pass_input
        ]
        result = (
            review_repeatability(records)
            if selected == "repeatability"
            else aggregate_three_full_passes(records, repository_root=root)
        )
        atomic_write_json(_path(args.output, "--output"), result)
        return 0

    commit = _text(args.execution_commit, "--execution-commit")
    verify_static_bindings(root, commit)
    if args.mode == "qualification-plan":
        atomic_write_json(_path(args.output, "--output"), qualification_plan(root))
        return 0

    if args.mode == "input-gate":
        receipt = make_input_gate_receipt(root, _path(args.full_ledger, "--full-ledger"), commit)
        write_input_gate_receipt(_path(args.output, "--output"), receipt)
        return 0

    if args.mode == "runtime-binding":
        authorization = require_authorization(
            args.authorization,
            scope="qualification-only",
            execution_commit=commit,
            logical_pass_id=None,
        )
        verify_qualification_worker(authorization)
        verify_input_gate_receipt(
            _path(args.input_gate_receipt, "--input-gate-receipt"),
            root=root,
            ledger=_path(args.full_ledger, "--full-ledger"),
            execution_commit=commit,
        )
        _external_candidate(
            root,
            _path(args.upstream_root, "--upstream-root"),
            _path(args.checkpoint, "--checkpoint"),
        )
        candidate = json.loads((root / CANDIDATE_MANIFEST_PATH).read_text(encoding="utf-8"))
        policy_module = importlib.import_module("laserperception.detection.m8_s2_runtime_policy")
        policy = policy_module.capture_runtime_policy(commit, candidate)
        atomic_write_json(_path(args.output, "--output"), policy)
        return 0

    if args.mode == "qualification":
        authorization = require_authorization(
            args.authorization,
            scope="qualification-only",
            execution_commit=commit,
            logical_pass_id=None,
        )
        verify_qualification_worker(authorization)
        input_sha = verify_input_gate_receipt(
            _path(args.input_gate_receipt, "--input-gate-receipt"),
            root=root,
            ledger=_path(args.full_ledger, "--full-ledger"),
            execution_commit=commit,
        )
        policy_path = _path(args.runtime_policy_binding, "--runtime-policy-binding")
        policy_sha = sha256_file(policy_path)
        bound_policy = json.loads(policy_path.read_text(encoding="utf-8"))
        candidate = json.loads((root / CANDIDATE_MANIFEST_PATH).read_text(encoding="utf-8"))
        if not isinstance(bound_policy, dict):
            raise M8S2ProtocolViolation("S2 qualification runtime policy is malformed")
        verify_runtime_policy_document(bound_policy, execution_commit=commit, candidate=candidate)
        _external_candidate(
            root,
            _path(args.upstream_root, "--upstream-root"),
            _path(args.checkpoint, "--checkpoint"),
        )
        policy_module = importlib.import_module("laserperception.detection.m8_s2_runtime_policy")
        live_policy = policy_module.capture_runtime_policy(commit, candidate)
        verify_runtime_policy(policy_path, policy_sha, live_policy)
        qualification = importlib.import_module("laserperception.detection.m8_s2_qualification")
        receipt = qualification.run_future_qualification(
            repository_root=root,
            date_root=_path(args.date_root, "--date-root"),
            m6_ledger=_path(args.m6_ledger, "--m6-ledger"),
            execution_commit=commit,
            runtime_policy_sha256=policy_sha,
            input_gate_receipt_sha256=input_sha,
        )
        atomic_write_json(_path(args.output, "--output"), receipt)
        return 0

    # From here, every failure must occur before importing the science module.
    mode = args.mode
    logical_pass_id = _text(args.logical_pass_id, "--logical-pass-id")
    full_ledger = _path(args.full_ledger, "--full-ledger")
    input_receipt_sha = verify_input_gate_receipt(
        _path(args.input_gate_receipt, "--input-gate-receipt"),
        root=root,
        ledger=full_ledger,
        execution_commit=commit,
    )
    policy_path = _path(args.runtime_policy_binding, "--runtime-policy-binding")
    policy_sha = sha256_file(policy_path)
    bound_policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if not isinstance(bound_policy, dict):
        raise M8S2ProtocolViolation("S2 runtime policy is malformed or bound to another commit")
    candidate = json.loads((root / CANDIDATE_MANIFEST_PATH).read_text(encoding="utf-8"))
    verify_runtime_policy_document(bound_policy, execution_commit=commit, candidate=candidate)
    qualification_sha = verify_qualification_receipt(
        _path(args.qualification_receipt, "--qualification-receipt"),
        execution_commit=commit,
        runtime_policy_sha256=policy_sha,
        input_gate_receipt_sha256=input_receipt_sha,
    )
    review_sha = (
        verify_repeatability_review(
            _path(args.repeatability_review, "--repeatability-review"),
            execution_commit=commit,
            runtime_policy_sha256=policy_sha,
            input_gate_receipt_sha256=input_receipt_sha,
        )
        if mode == "full-pass"
        else None
    )
    require_authorization(
        args.authorization,
        scope="repeatability-only" if mode == "repeatability" else "full-pass-only",
        execution_commit=commit,
        logical_pass_id=logical_pass_id,
        runtime_policy_sha256=policy_sha,
        input_gate_receipt_sha256=input_receipt_sha,
        qualification_receipt_sha256=qualification_sha,
        repeatability_review_sha256=review_sha,
    )
    _external_candidate(
        root,
        _path(args.upstream_root, "--upstream-root"),
        _path(args.checkpoint, "--checkpoint"),
    )
    # First accelerator import occurs after exact owner scope and bindings.
    policy_module = importlib.import_module("laserperception.detection.m8_s2_runtime_policy")
    live_policy = policy_module.capture_runtime_policy(commit, candidate)
    verify_runtime_policy(policy_path, policy_sha, live_policy)
    science = importlib.import_module("laserperception.evaluation.m8_s2_science")
    science.run_scientific_attempt(
        mode=mode,
        repository_root=root,
        date_root=_path(args.date_root, "--date-root"),
        m6_ledger=_path(args.m6_ledger, "--m6-ledger"),
        execution_commit=commit,
        runtime_policy_sha256=policy_sha,
        input_gate_receipt_sha256=input_receipt_sha,
        attempt_root=_path(args.attempt_root, "--attempt-root"),
        logical_pass_id=logical_pass_id,
        attempt_id=_text(args.attempt_id, "--attempt-id"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
