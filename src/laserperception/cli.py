"""CPU-safe developer CLI; historical scripts retain their existing entry points."""

import argparse
import json
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    from laserperception.perception import registry

    parser = argparse.ArgumentParser(prog="laserperception")
    commands = parser.add_subparsers(dest="command", required=True)
    from laserperception.worker.cli import configure_worker, run_worker

    configure_worker(commands.add_parser("worker"))
    models = commands.add_parser("models").add_subparsers(dest="action", required=True)
    listing = models.add_parser("list")
    listing.add_argument("--json", action="store_true")
    inspect = models.add_parser("inspect")
    inspect.add_argument("model_id")
    inspect.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "worker":
            return run_worker(args)
        if args.action == "list":
            manifests = registry.list_models()
            if args.json:
                print(json.dumps([m.to_dict() for m in manifests], sort_keys=True, allow_nan=False))
            else:
                for m in manifests:
                    print(
                        f"{m.model_id}\t{','.join(t.value for t in m.tasks)}\t{m.status}"
                        f"\t{','.join(m.capabilities.runtime_targets)}\t{m.temporal.mode}"
                    )
        else:
            print(registry.get(args.model_id).to_json(), end="")
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    return 0
