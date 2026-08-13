#!/usr/bin/env python3
"""Validate the external v0.1 detector, dataset, and ROS demo assets."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from laserperception.detection.m1_assets import resolve_m1_asset_paths
from laserperception.detection.m2_assets import resolve_m2_asset_paths
from laserperception.detection.mmdet3d_backend import sha256_file

EXPECTED_CHECKPOINT_SHA256 = "f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0"
EXPECTED_ONNX_SHA256 = "61ce22a8ca31498675c32576bfb94f0093d31dc95d2762f7254bf915a59ecc16"
EXPECTED_ENGINE_SHA256 = "a005f75852097cd9b193750560b214cc3d5237ae9b6c106c7fca3d4fc348714b"


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a mapping")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required configuration is missing: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(_mapping(value, str(path)))


def _verify_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA256 mismatch: expected {expected}, found {actual}")


def _verify_ros_config(path: Path) -> None:
    config = _load_yaml(path)
    detector = _mapping(config.get("laserperception_detector"), "detector config")
    detector_params = _mapping(detector.get("ros__parameters"), "detector parameters")
    replay = _mapping(config.get("laserperception_replay"), "replay config")
    replay_params = _mapping(replay.get("ros__parameters"), "replay parameters")
    expected = {
        "voxelization_mode": "exact_fast",
        "provenance_mode": "live",
        "start_index": 42,
        "sample_count": 1,
        "loop": True,
    }
    actual = {
        "voxelization_mode": detector_params.get("voxelization_mode"),
        "provenance_mode": detector_params.get("provenance_mode"),
        "start_index": replay_params.get("start_index"),
        "sample_count": replay_params.get("sample_count"),
        "loop": replay_params.get("loop"),
    }
    if actual != expected:
        raise RuntimeError(f"release ROS config mismatch: expected {expected}, found {actual}")


def validate_release_assets(repo_root: Path, data_root: Path, ros_share: Path) -> None:
    """Validate frozen assets and the installed production demo configuration."""

    m1_manifest = _load_yaml(repo_root / "configs/detection/m1_pointpillars_nuscenes.yaml")
    m2_manifest = _load_yaml(repo_root / "configs/detection/m2_pointpillars_tensorrt.yaml")
    m1_assets = resolve_m1_asset_paths(m1_manifest)
    m2_assets = resolve_m2_asset_paths(m2_manifest)

    model = _mapping(m1_manifest.get("model"), "M1 model")
    checkpoint = _mapping(model.get("checkpoint"), "M1 checkpoint")
    if checkpoint.get("sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("M1 manifest no longer identifies the frozen v0.1 checkpoint")

    artifacts = _mapping(m2_manifest.get("artifacts"), "M2 artifacts")
    onnx = _mapping(artifacts.get("onnx"), "M2 ONNX artifact")
    engine = _mapping(artifacts.get("engine"), "M2 engine artifact")
    if onnx.get("sha256") != EXPECTED_ONNX_SHA256:
        raise RuntimeError("M2 manifest no longer identifies the frozen v0.1 ONNX artifact")
    if engine.get("sha256") != EXPECTED_ENGINE_SHA256:
        raise RuntimeError("M2 manifest no longer identifies the frozen v0.1 TensorRT engine")

    config_path = m1_assets.mmdet3d_root / str(model["upstream_config"])
    if not config_path.is_file():
        raise FileNotFoundError(
            f"pinned MMDetection3D config is missing: {config_path}; run setup_detection_m2.sh"
        )
    deployment = _mapping(m2_manifest.get("deployment"), "M2 deployment")
    deploy_config = m2_assets.mmdeploy_root / str(deployment["official_deployment_config"])
    if not deploy_config.is_file():
        raise FileNotFoundError(
            f"pinned MMDeploy config is missing: {deploy_config}; run setup_detection_m2.sh"
        )

    _verify_hash(m1_assets.checkpoint_path, EXPECTED_CHECKPOINT_SHA256, "checkpoint")
    _verify_hash(m2_assets.artifact_directory / "pointpillars.onnx", EXPECTED_ONNX_SHA256, "ONNX")
    _verify_hash(
        m2_assets.engine_directory / "pointpillars_fp16.engine",
        EXPECTED_ENGINE_SHA256,
        "TensorRT engine",
    )

    required_dataset_paths = (
        data_root / "v1.0-mini/sample.json",
        data_root / "samples/LIDAR_TOP",
        data_root / "nuscenes_infos_train.pkl",
        data_root / "nuscenes_infos_val.pkl",
    )
    missing = [path for path in required_dataset_paths if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "nuScenes v1.0-mini is missing or unprepared; obtain it under its own terms, then run "
            f"prepare_nuscenes_mini.py. Missing:\n{formatted}"
        )

    _verify_ros_config(repo_root / "ros2/laserperception_ros/config/m3_ros2.yaml")
    _verify_ros_config(ros_share / "config/m3_ros2.yaml")

    print("LaserPerception v0.1 external assets verified")
    print(f"  checkpoint sha256: {EXPECTED_CHECKPOINT_SHA256}")
    print(f"  ONNX sha256:       {EXPECTED_ONNX_SHA256}")
    print(f"  engine sha256:     {EXPECTED_ENGINE_SHA256}")
    print("  dataset:           nuScenes v1.0-mini prepared metadata present")
    print("  ROS policy:        exact_fast / live, W1 mini_val index 42")


def main() -> None:
    """Parse arguments and validate the release demo prerequisites."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--ros-share", type=Path, required=True)
    args = parser.parse_args()
    validate_release_assets(
        args.repo_root.expanduser().resolve(),
        args.data_root.expanduser().resolve(),
        args.ros_share.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
