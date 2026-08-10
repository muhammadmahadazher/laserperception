"""Render an original headless BEV image for one M1 nuScenes-mini sample."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from laserperception.detection.mmdet3d_backend import (
    DetectionEnvironmentError,
    Mmdet3dBackend,
)
from laserperception.detection.visualization import render_bev


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
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--min-score", type=float, help="visualization-only score threshold")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/m1/pointpillars_bev.png"),
        help="ignored PNG or SVG artifact path",
    )
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
    viz_info = manifest["visualization"]
    threshold = (
        float(args.min_score) if args.min_score is not None else float(viz_info["score_threshold"])
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
        destination = render_bev(
            prepared.points_xyz,
            raw_frame,
            args.output,
            min_score=threshold,
            x_limits=tuple(float(value) for value in viz_info["x_limits_m"]),
            y_limits=tuple(float(value) for value in viz_info["y_limits_m"]),
            max_points=int(viz_info["max_points"]),
        )
    except (
        DetectionEnvironmentError,
        FileNotFoundError,
        IndexError,
        RuntimeError,
        ValueError,
    ) as error:
        raise SystemExit(f"error: {error}") from error

    count = len(raw_frame.filtered(threshold).detections)
    print(
        f"Rendered sample {prepared.sample_id} with {count} detections at score >= {threshold:.2f}."
    )
    print(f"Artifact: {destination.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
