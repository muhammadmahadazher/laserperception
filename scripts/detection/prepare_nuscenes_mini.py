"""Validate and prepare official nuScenes v1.0-mini metadata with MMDetection3D."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

MMDET3D_COMMIT = "fe25f7a51d36e3702f961e198894580d83c4387b"


def _configured_root(value: str | None) -> Path:
    raw = value or os.environ.get("LASERPERCEPTION_NUSCENES_ROOT")
    if not raw:
        raise ValueError(
            "set LASERPERCEPTION_NUSCENES_ROOT or pass --data-root after obtaining "
            "nuScenes v1.0-mini from the official nuScenes download page"
        )
    return Path(raw).expanduser().resolve()


def _require_outside_repository(path: Path, repository_root: Path, name: str) -> None:
    if path == repository_root or path.is_relative_to(repository_root):
        raise ValueError(f"{name} must be outside the LaserPerception repository")


def _validate_raw_mini(root: Path) -> None:
    required = (
        root / "v1.0-mini",
        root / "samples" / "LIDAR_TOP",
        root / "sweeps" / "LIDAR_TOP",
    )
    missing = [path.relative_to(root).as_posix() for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError("nuScenes v1.0-mini is incomplete; missing: " + ", ".join(missing))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="nuScenes root; defaults to environment variable")
    parser.add_argument("--out-dir", help="prepared output directory; defaults to data root")
    parser.add_argument(
        "--mmdet3d-root",
        default="~/.cache/laserperception/mmdetection3d-v1.4.0",
        help="pinned official MMDetection3D checkout",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="validate and print no private paths"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[2]
    try:
        data_root = _configured_root(args.data_root)
        out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else data_root
        mmdet3d_root = Path(args.mmdet3d_root).expanduser().resolve()
        _require_outside_repository(data_root, repository_root, "data root")
        _require_outside_repository(out_dir, repository_root, "output directory")
        _validate_raw_mini(data_root)
        tool = mmdet3d_root / "tools" / "create_data.py"
        if not tool.is_file():
            raise FileNotFoundError("pinned MMDetection3D data preparation tool was not found")
        commit = subprocess.check_output(
            ["git", "-C", str(mmdet3d_root), "rev-parse", "HEAD"], text=True
        ).strip()
        if commit != MMDET3D_COMMIT:
            raise RuntimeError(f"MMDetection3D commit mismatch: expected {MMDET3D_COMMIT}")
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error

    command = [
        sys.executable,
        str(tool),
        "nuscenes",
        "--root-path",
        str(data_root),
        "--out-dir",
        str(out_dir),
        "--extra-tag",
        "nuscenes",
        "--version",
        "v1.0-mini",
        "--max-sweeps",
        "10",
    ]
    if args.dry_run:
        print("Validated official nuScenes v1.0-mini structure and pinned MMDetection3D checkout.")
        print("Preparation command is ready; paths are intentionally redacted.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=mmdet3d_root, check=True)

    from mmengine import load

    counts: dict[str, int] = {}
    for split in ("train", "val"):
        info_path = out_dir / f"nuscenes_infos_{split}.pkl"
        if not info_path.is_file():
            raise SystemExit(f"error: preparation did not create the {split} info file")
        document = load(info_path)
        counts[split] = len(document["data_list"])
    print("Prepared nuScenes v1.0-mini with the official MMDetection3D converter.")
    print(f"Observed prepared sample counts: train={counts['train']}, val={counts['val']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
