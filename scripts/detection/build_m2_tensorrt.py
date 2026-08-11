"""Build the profiled PointPillars FP16 engine with MMDeploy's official converter."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Sequence
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import onnx
import yaml
from mmengine import Config

from laserperception.detection.artifacts import ExternalArtifactMetadata
from laserperception.detection.m2_assets import resolve_m2_asset_paths
from laserperception.detection.tensorrt_backend import inspect_engine, load_tensorrt


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _manifest() -> dict[str, Any]:
    path = _repository_root() / "configs" / "detection" / "m2_pointpillars_tensorrt.yaml"
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def _profile(manifest: dict[str, Any]) -> dict[str, dict[str, list[int]]]:
    profile = manifest["profile"]
    result: dict[str, dict[str, list[int]]] = {}
    for name in ("voxels", "num_points", "coors"):
        result[name] = {
            "min_shape": list(profile["selected_min_shapes"][name]),
            "opt_shape": list(profile["selected_opt_shapes"][name]),
            "max_shape": list(profile["selected_max_shapes"][name]),
        }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, help="override external ONNX path")
    parser.add_argument("--output", type=Path, help="override external engine path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = _manifest()
    assets = resolve_m2_asset_paths(manifest)
    onnx_path = args.onnx or assets.artifact_directory / "pointpillars.onnx"
    engine_path = args.output or assets.engine_directory / "pointpillars_fp16.engine"
    if engine_path.suffix != ".engine":
        raise SystemExit("error: engine output must use the .engine suffix")
    onnx_model = onnx.load(str(onnx_path), load_external_data=True)
    onnx.checker.check_model(onnx_model, full_check=True)
    trt = load_tensorrt()
    if not bool(trt.Builder(trt.Logger(trt.Logger.WARNING)).platform_has_fast_fp16):
        raise SystemExit("error: TensorRT builder reports no fast FP16 support")

    deploy_relative = str(manifest["deployment"]["official_deployment_config"])
    official_path = assets.mmdeploy_root / deploy_relative
    deploy_config = Config.fromfile(str(official_path))
    effective_profile = _profile(manifest)
    deploy_config["backend_config"]["common_config"]["fp16_mode"] = True
    deploy_config["backend_config"]["model_inputs"][0]["input_shapes"] = deepcopy(effective_profile)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    deploy_config.dump(str(assets.artifact_directory / "effective_deploy_config.py"))

    from mmdeploy.apis.tensorrt import onnx2tensorrt

    started = time.perf_counter()
    onnx2tensorrt(
        str(engine_path.parent),
        engine_path.name,
        0,
        deploy_config,
        str(onnx_path),
        device="cuda:0",
    )
    build_time_seconds = time.perf_counter() - started
    expected_bindings = [
        "voxels",
        "num_points",
        "coors",
        "cls_score0",
        "bbox_pred0",
        "dir_cls_pred0",
    ]
    inspection = inspect_engine(
        engine_path,
        expected_bindings=expected_bindings,
        expected_profile=effective_profile,
    )
    artifact = ExternalArtifactMetadata.from_file(
        engine_path, logical_name=str(manifest["artifacts"]["engine"]["logical_name"])
    )

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise SystemExit("error: built TensorRT engine could not be deserialized")
    inspector = engine.create_engine_inspector()
    layer_information = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    inspector_path = assets.artifact_directory / "engine_inspector.json"
    inspector_path.write_text(layer_information + "\n", encoding="utf-8")

    result = {
        "schema_version": "1.0",
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder": {
            "name": "MMDeploy official onnx2tensorrt",
            "mmdeploy_version": str(manifest["deployment"]["exporter_version"]),
            "mmdeploy_commit": str(manifest["deployment"]["exporter_commit"]),
            "official_config": deploy_relative,
            "tensorrt_version": str(trt.__version__),
            "fp16_mode": True,
            "int8_mode": False,
            "workspace_size_bytes": int(
                deploy_config["backend_config"]["common_config"]["max_workspace_size"]
            ),
            "build_time_seconds": build_time_seconds,
        },
        "profile": effective_profile,
        "engine": {
            **artifact.to_dict(),
            **inspection,
            "inspector_sha256": hashlib.sha256(layer_information.encode("utf-8")).hexdigest(),
            "inspector_size_bytes": len(layer_information.encode("utf-8")),
            "inspector_fp16_token_count": layer_information.lower().count("half")
            + layer_information.lower().count("fp16"),
        },
    }
    metadata_path = assets.artifact_directory / "engine_metadata.json"
    metadata_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("TensorRT FP16 engine built and verified outside the repository.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
