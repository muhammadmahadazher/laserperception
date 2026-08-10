"""Find a bounded nuScenes-mini sample with a qualifying M1 detection."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from laserperception.detection.mmdet3d_backend import (
    DetectionEnvironmentError,
    Mmdet3dBackend,
)


def _manifest() -> dict[str, Any]:
    path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "detection"
        / "m1_pointpillars_nuscenes.yaml"
    )
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def _data_root(value: str | None) -> Path:
    raw = value or os.environ.get("LASERPERCEPTION_NUSCENES_ROOT")
    if not raw:
        raise ValueError("set LASERPERCEPTION_NUSCENES_ROOT or pass --data-root")
    return Path(raw).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="nuScenes root; defaults to environment variable")
    parser.add_argument("--split", choices=("mini_train", "mini_val"), default="mini_val")
    parser.add_argument("--class-name", help="exact official class name")
    parser.add_argument("--min-score", type=float, help="fixed search threshold")
    parser.add_argument("--max-samples", type=int, help="bounded number of samples to scan")
    parser.add_argument("--json", type=Path, help="optional sanitized match summary")
    parser.add_argument("--config", type=Path, help="override pinned upstream config")
    parser.add_argument("--checkpoint", type=Path, help="override verified checkpoint cache")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = _manifest()
    model_info = manifest["model"]
    checkpoint_info = model_info["checkpoint"]
    checkout = Path(str(model_info["upstream_checkout"])).expanduser()
    config = args.config or checkout / str(model_info["upstream_config"])
    checkpoint = args.checkpoint or Path(
        str(checkpoint_info["cache_directory"])
    ).expanduser() / str(checkpoint_info["filename"])
    search = manifest["visualization"]["pedestrian_search"]
    class_name = args.class_name or str(search["class_name"])
    threshold = float(args.min_score) if args.min_score is not None else float(search["min_score"])
    max_samples = (
        int(args.max_samples) if args.max_samples is not None else int(search["max_samples"])
    )
    if not class_name:
        raise SystemExit("error: class name must not be empty")
    if not 0.0 <= threshold <= 1.0:
        raise SystemExit("error: min-score must be between 0 and 1")
    if max_samples <= 0:
        raise SystemExit("error: max-samples must be positive")

    try:
        backend = Mmdet3dBackend(
            config,
            checkpoint,
            checkpoint_sha256=str(checkpoint_info["sha256"]),
        )
        data_root = _data_root(args.data_root)
        scan_count = min(max_samples, backend.dataset_size(data_root, args.split))
        match: dict[str, object] | None = None
        for index in range(scan_count):
            prepared = backend.prepare_sample(data_root, split=args.split, index=index)
            raw_frame = backend.run_prepared(prepared)
            qualifying = tuple(
                detection
                for detection in raw_frame.detections
                if detection.class_name == class_name and detection.score >= threshold
            )
            if qualifying:
                match = {
                    "sample_id": prepared.sample_id,
                    "sample_index": index,
                    "split": args.split,
                    "class_name": class_name,
                    "min_score": threshold,
                    "qualifying_count": len(qualifying),
                    "best_score": max(detection.score for detection in qualifying),
                    "scanned_samples": index + 1,
                }
                break
    except (
        DetectionEnvironmentError,
        FileNotFoundError,
        IndexError,
        RuntimeError,
        ValueError,
    ) as error:
        raise SystemExit(f"error: {error}") from error

    if match is None:
        print(
            f"No {class_name!r} detection at score >= {threshold:.2f} "
            f"in the bounded scan of {scan_count} {args.split} samples."
        )
        return 1

    print(json.dumps(match, indent=2, sort_keys=True))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(match, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("Sanitized match summary written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
