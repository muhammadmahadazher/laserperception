"""CPU-safe developer CLI; historical scripts retain their existing entry points."""

import argparse
import json
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    from laserperception.perception import registry

    parser = argparse.ArgumentParser(prog="laserperception")
    commands = parser.add_subparsers(dest="command", required=True)
    from laserperception.perception.cli import configure_predict, run_predict
    from laserperception.semantic.cli import configure_semantic, run_semantic
    from laserperception.tracking.cli import configure_track, run_track
    from laserperception.worker.cli import configure_worker, run_worker

    configure_semantic(commands.add_parser("semantic"))
    configure_track(commands.add_parser("track"))
    configure_worker(commands.add_parser("worker"))
    configure_predict(commands.add_parser("predict"))
    models = commands.add_parser("models").add_subparsers(dest="action", required=True)
    listing = models.add_parser("list")
    listing.add_argument("--json", action="store_true")
    inspect = models.add_parser("inspect")
    inspect.add_argument("model_id")
    inspect.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "semantic":
            return run_semantic(args)
        if args.command == "track":
            return run_track(args)
        if args.command == "worker":
            return run_worker(args)
        if args.command == "predict":
            return run_predict(args)
        if args.action == "list":
            manifests = registry.list_models()
            if args.json:
                print(json.dumps([m.to_dict() for m in manifests], sort_keys=True, allow_nan=False))
            else:
                for manifest in manifests:
                    print(
                        f"{manifest.model_id}\t{','.join(t.value for t in manifest.tasks)}"
                        f"\t{manifest.status}"
                        f"\t{','.join(manifest.capabilities.runtime_targets)}"
                        f"\t{manifest.temporal.mode}"
                    )
        else:
            print(registry.get(args.model_id).to_json(), end="")
    except (TypeError, ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    return 0
