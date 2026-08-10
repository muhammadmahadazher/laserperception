"""Run official pretrained M1 PointPillars inference on one nuScenes-mini sample."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from laserperception.detection.m1_assets import resolve_m1_asset_paths
from laserperception.detection.mmdet3d_backend import (
    DetectionEnvironmentError,
    Mmdet3dBackend,
)


def _manifest() -> dict[str, Any]:
    path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "detection"
        / ("m1_pointpillars_nuscenes.yaml")
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
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--min-score", type=float, help="export filter; does not alter model output"
    )
    parser.add_argument("--json", type=Path, help="optional sanitized JSON output")
    parser.add_argument("--config", type=Path, help="override pinned upstream config")
    parser.add_argument("--checkpoint", type=Path, help="override verified checkpoint cache")
    return parser


def _print_frame(raw_count: int, threshold: float, frame: object) -> None:
    detections = frame.detections
    print(f"Sample: {frame.sample_id}")
    print(f"Split/index: {frame.metadata['split']}/{frame.metadata['sample_index']}")
    print(f"Detections: raw={raw_count}, exported={len(detections)}, min_score={threshold:.2f}")
    print("class                  score   center_xyz (m)            size_lwh (m)       yaw (rad)")
    print("-" * 92)
    for detection in detections:
        center = "(" + ", ".join(f"{value:6.2f}" for value in detection.center_xyz) + ")"
        size = "(" + ", ".join(f"{value:5.2f}" for value in detection.size_lwh) + ")"
        print(
            f"{detection.class_name:<22} {detection.score:5.3f}   "
            f"{center:<27} {size:<20} {detection.yaw_rad:8.3f}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = _manifest()
    assets = resolve_m1_asset_paths(manifest)
    config = args.config or assets.mmdet3d_root / str(manifest["model"]["upstream_config"])
    checkpoint_info = manifest["model"]["checkpoint"]
    checkpoint = args.checkpoint or assets.checkpoint_path
    threshold = (
        float(args.min_score)
        if args.min_score is not None
        else float(manifest["visualization"]["score_threshold"])
    )
    try:
        backend = Mmdet3dBackend(
            config,
            checkpoint,
            checkpoint_sha256=str(checkpoint_info["sha256"]),
        )
        prepared = backend.prepare_sample(
            _data_root(args.data_root), split=args.split, index=args.index
        )
        raw_frame = backend.run_prepared(prepared)
        frame = raw_frame.filtered(threshold)
    except (DetectionEnvironmentError, FileNotFoundError, IndexError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error

    _print_frame(len(raw_frame.detections), threshold, frame)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(frame.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print("Sanitized JSON written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
