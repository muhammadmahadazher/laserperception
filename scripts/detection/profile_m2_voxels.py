"""Profile official M2 voxel tensor shapes across the complete nuScenes mini_val split."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any

import yaml

from laserperception.detection.m1_assets import resolve_m1_asset_paths
from laserperception.detection.m2_assets import resolve_m2_asset_paths
from laserperception.detection.m2_backend import M2Backend


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _manifest(name: str) -> dict[str, Any]:
    path = _repository_root() / "configs" / "detection" / name
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def _data_root(value: str | None) -> Path:
    raw = value or os.environ.get("LASERPERCEPTION_NUSCENES_ROOT")
    if not raw:
        raise ValueError("set LASERPERCEPTION_NUSCENES_ROOT or pass --data-root")
    return Path(raw).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="nuScenes root; defaults to environment variable")
    parser.add_argument("--output", type=Path, help="override external sanitized profile JSON")
    return parser


def _nearest_rank(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    index = max(0, ceil((percentile / 100.0) * len(ordered)) - 1)
    return ordered[index]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    m1_manifest = _manifest("m1_pointpillars_nuscenes.yaml")
    m2_manifest = _manifest("m2_pointpillars_tensorrt.yaml")
    m1_assets = resolve_m1_asset_paths(m1_manifest)
    m2_assets = resolve_m2_asset_paths(m2_manifest)
    output = args.output or m2_assets.artifact_directory / "voxel_profile.json"
    model_info = m1_manifest["model"]
    checkpoint_info = model_info["checkpoint"]
    deploy_relative = str(m2_manifest["deployment"]["official_deployment_config"])
    backend = M2Backend(
        m1_assets.mmdet3d_root / str(model_info["upstream_config"]),
        m1_assets.checkpoint_path,
        m2_assets.mmdeploy_root / deploy_relative,
        checkpoint_sha256=str(checkpoint_info["sha256"]),
    )
    data_root = _data_root(args.data_root)
    split_size = backend.dataset_size(data_root, "mini_val")
    expected_size = int(m2_manifest["profile"]["preferred_samples_scanned"])
    if split_size != expected_size:
        raise SystemExit(
            f"error: expected mini_val size {expected_size}, found {split_size}; refusing profile"
        )

    samples: list[dict[str, object]] = []
    voxel_counts: list[int] = []
    for index in range(split_size):
        prepared = backend.prepare_sample(data_root, split="mini_val", index=index)
        voxelized = backend.voxelize(prepared)
        voxel_counts.append(voxelized.voxel_count)
        samples.append(
            {
                "index": index,
                "sample_id": prepared.sample_id,
                "shapes": voxelized.shapes,
            }
        )
        print(f"profiled mini_val index {index}: {voxelized.voxel_count} voxels")

    minimum = min(voxel_counts)
    maximum = max(voxel_counts)
    p50 = _nearest_rank(voxel_counts, 50.0)
    p90 = _nearest_rank(voxel_counts, 90.0)
    p95 = _nearest_rank(voxel_counts, 95.0)
    optimum = p50
    upstream_limit = int(m2_manifest["voxel_contract"]["upstream_max_voxels"]["validation"])
    if maximum > upstream_limit:
        raise SystemExit(
            f"error: observed voxel count {maximum} exceeds upstream limit {upstream_limit}"
        )
    official_profile_max = int(
        backend.deploy_config["backend_config"]["model_inputs"][0]["input_shapes"]["voxels"][
            "max_shape"
        ][0]
    )
    headroom_target = ceil(maximum * 1.25)
    selected_maximum = min(upstream_limit, max(official_profile_max, headroom_target))
    result = {
        "schema_version": "1.0",
        "status": "measured",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "nuScenes",
            "version": "v1.0-mini",
            "split": "mini_val",
            "samples_scanned": split_size,
        },
        "official_voxelization": "MMDetection3D Det3DDataPreprocessor",
        "observed_voxel_counts": {
            "minimum": minimum,
            "p50_nearest_rank": p50,
            "p90_nearest_rank": p90,
            "p95_nearest_rank": p95,
            "maximum": maximum,
            "official_mmdeploy_profile_maximum": official_profile_max,
            "upstream_validation_limit": upstream_limit,
        },
        "selected_profile": {
            "voxels": {
                "min_shape": [minimum, 64, 4],
                "opt_shape": [optimum, 64, 4],
                "max_shape": [selected_maximum, 64, 4],
            },
            "num_points": {
                "min_shape": [minimum],
                "opt_shape": [optimum],
                "max_shape": [selected_maximum],
            },
            "coors": {
                "min_shape": [minimum, 4],
                "opt_shape": [optimum, 4],
                "max_shape": [selected_maximum, 4],
            },
        },
        "selection_rationale": (
            "exact observed minimum and nearest-rank p50 over all mini_val samples; "
            "the official MMDeploy maximum of 30000 is retained because it exceeds "
            "the 25 percent measured-headroom target and remains below the upstream "
            "validation hard limit of 40000"
        ),
        "samples": samples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["observed_voxel_counts"], indent=2, sort_keys=True))
    print("Profile written outside the repository.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
