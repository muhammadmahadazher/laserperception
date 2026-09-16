"""CPU-safe prediction planning command."""

from __future__ import annotations

import argparse
from pathlib import Path

from .api import load_model
from .execution import (
    ExecutionAuthorization,
    ExecutionContext,
    ModelResources,
    Precision,
    RuntimeTarget,
)
from .inputs import InputDescription


def configure_predict(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, type=Path, dest="input_description")
    parser.add_argument(
        "--runtime-target",
        required=True,
        choices=tuple(target.value for target in RuntimeTarget),
    )
    parser.add_argument(
        "--precision",
        required=True,
        choices=tuple(precision.value for precision in Precision),
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--runtime-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--deterministic", required=True, action="store_true")
    parser.add_argument(
        "--authorization-kind",
        choices=("generic_prediction", "m8_s1"),
    )
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--authorization-sha256")
    parser.add_argument("--mode")
    parser.add_argument("--logical-pass-id")
    parser.add_argument("--runtime-policy-binding", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--dry-run", required=True, action="store_true")


def _authorization(args: argparse.Namespace) -> ExecutionAuthorization | None:
    supplied = (
        args.authorization_kind,
        args.authorization,
        args.authorization_sha256,
        args.mode,
        args.logical_pass_id,
        args.runtime_policy_binding,
    )
    if not any(value is not None for value in supplied):
        return None
    if args.authorization_kind is None or args.authorization is None:
        raise ValueError("authorization kind and path must be supplied together")
    if args.authorization_sha256 is None:
        raise ValueError("authorization SHA256 is required when an authorization is supplied")
    return ExecutionAuthorization(
        args.authorization_kind,
        args.authorization,
        args.authorization_sha256,
        args.mode,
        args.logical_pass_id,
        args.runtime_policy_binding,
    )


def run_predict(args: argparse.Namespace) -> int:
    """Print one deterministic plan without loading a backend."""

    path: Path = args.input_description
    description = InputDescription.from_json(path.read_text(encoding="utf-8"))
    context = ExecutionContext(
        RuntimeTarget(args.runtime_target),
        Precision(args.precision),
        args.device,
        args.runtime_id,
        args.task_id,
        args.execution_commit,
        args.deterministic,
        _authorization(args),
    )
    resources = ModelResources(
        args.config, args.checkpoint, args.candidate_manifest, args.upstream_root
    )
    plan = load_model(args.model).plan(description, context, resources)
    print(plan.to_json(), end="")
    return 0
