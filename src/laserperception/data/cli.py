"""Discover adapters and inspect local inputs on CPU."""

import argparse
import json
from pathlib import Path

from .contracts import InputOptions
from .inspection import inspect_input
from .registry import registry


def configure_data(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="data_action", required=True)
    adapters = commands.add_parser("adapters").add_subparsers(dest="adapter_action", required=True)
    listing = adapters.add_parser("list")
    listing.add_argument("--json", action="store_true")
    manifest = adapters.add_parser("inspect")
    manifest.add_argument("adapter_id")
    manifest.add_argument("--json", action="store_true")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("path", type=Path)
    inspect.add_argument("--adapter")
    inspect.add_argument("--label-path", type=Path)
    inspect.add_argument("--timestamp-microseconds", type=int)
    inspect.add_argument("--sample-id")
    inspect.add_argument("--split")
    inspect.add_argument("--sequence", action="append")
    inspect.add_argument("--index", type=int, default=0)
    inspect.add_argument("--date-root", type=Path)
    inspect.add_argument("--model")
    inspect.add_argument("--json", action="store_true")


def run_data(args: argparse.Namespace) -> int:
    if args.data_action == "adapters":
        if args.adapter_action == "inspect":
            print(registry.get(args.adapter_id).to_json(), end="")
        elif args.json:
            print(
                json.dumps(
                    [m.to_dict() for m in registry.list_adapters()], sort_keys=True, allow_nan=False
                )
            )
        else:
            for manifest in registry.list_adapters():
                print(f"{manifest.adapter_id}\t{manifest.category}\t{','.join(manifest.formats)}")
    else:
        options = InputOptions(
            args.label_path,
            args.timestamp_microseconds,
            args.sample_id,
            args.split,
            tuple(args.sequence) if args.sequence else None,
            args.index,
            args.date_root,
        )
        result = inspect_input(
            args.path, adapter_id=args.adapter, options=options, model_id=args.model
        )
        if args.json:
            print(result.to_json(), end="")
        else:
            print(
                f"{result.adapter.adapter_id}: {result.point_count} points; "
                f"labels={result.labels_available}; timestamp_ns={result.timestamp_ns}"
            )
            if result.compatibility:
                print(
                    f"model={result.model_id} compatible={result.compatibility.valid}; "
                    f"verified={result.compatibility_verified}; "
                    f"errors={result.compatibility.errors}"
                )
    return 0
