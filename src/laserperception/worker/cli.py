"""Worker CLI handlers; planning and artifact verification never inspect hardware."""

import argparse
from pathlib import Path

from .artifacts import VerifiedArtifact, verify_artifact
from .manifests import TaskManifest
from .paths import resolve_artifact_path
from .planning import plan_worker


def configure_worker(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="worker_action", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("manifest", type=Path)
    plan.add_argument("--model", required=True)
    validate = commands.add_parser("manifest").add_subparsers(dest="manifest_action", required=True)
    validate.add_parser("validate").add_argument("manifest", type=Path)
    verify = commands.add_parser("verify-artifact")
    verify.add_argument("artifact", type=Path, help="VerifiedArtifact JSON identity")
    verify.add_argument("--root", required=True, type=Path)
    bootstrap_parser = commands.add_parser("bootstrap")
    bootstrap_parser.add_argument("manifest", type=Path)
    bootstrap_parser.add_argument("--external-worker", action="store_true")
    bootstrap_parser.add_argument("--qualification", action="store_true")
    bootstrap_parser.add_argument("--expected-commit", required=True)
    bootstrap_parser.add_argument("--runtime-id", required=True)
    bootstrap_parser.add_argument("--root", required=True, type=Path)
    bootstrap_parser.add_argument("--output", required=True, type=Path)


def run_worker(args: argparse.Namespace) -> int:
    if args.worker_action == "verify-artifact":
        artifact = VerifiedArtifact.from_json(args.artifact.read_text(encoding="utf-8"))
        verify_artifact(resolve_artifact_path(args.root, artifact.relative_path), artifact)
        print("Artifact bytes and SHA256 verified; no execution authorized.")
        return 0
    task = TaskManifest.from_json(args.manifest.read_text(encoding="utf-8"))
    if args.worker_action == "manifest":
        print(task.to_json(), end="")
    elif args.worker_action == "plan":
        print(plan_worker(task, args.model).to_json(), end="")
    else:
        from .bootstrap import bootstrap

        result = bootstrap(
            task,
            root=args.root,
            output=args.output,
            external_worker=args.external_worker,
            expected_commit=args.expected_commit,
            runtime_id=args.runtime_id,
            qualification=args.qualification,
        )
        print(result.to_json(), end="")
    return 0
