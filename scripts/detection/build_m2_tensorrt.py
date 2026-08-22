"""Build the profiled PointPillars FP16 engine with MMDeploy's official converter."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import subprocess
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
from laserperception.detection.m6b_engine_remediation import (
    load_engine_manifest,
    profile_shapes,
    resolve_build_manifest_path,
    validate_candidate_manifest,
)
from laserperception.detection.runtime_metadata import nvidia_smi_value
from laserperception.detection.tensorrt_backend import inspect_engine, load_tensorrt


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _manifest(path: Path) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def _profile(manifest: dict[str, Any]) -> dict[str, dict[str, list[int]]]:
    return profile_shapes(manifest)


def _git_commit(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="deployment manifest; defaults to the historical M2 30k manifest",
    )
    parser.add_argument("--onnx", type=Path, help="override external ONNX path")
    parser.add_argument("--output", type=Path, help="override external engine path")
    parser.add_argument("--metadata-output", type=Path, help="override external build metadata")
    parser.add_argument("--inspector-output", type=Path, help="override external inspector JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository_root = _repository_root()
    manifest_path = resolve_build_manifest_path(repository_root, args.manifest)
    manifest = _manifest(manifest_path)
    historical_path = resolve_build_manifest_path(repository_root, None)
    if manifest_path != historical_path:
        validate_candidate_manifest(manifest, load_engine_manifest(historical_path))
    assets = resolve_m2_asset_paths(manifest)
    onnx_path = args.onnx or assets.artifact_directory / "pointpillars.onnx"
    engine_filename = Path(str(manifest["artifacts"]["engine"]["logical_name"])).name
    engine_path = args.output or assets.engine_directory / engine_filename
    if engine_path.suffix != ".engine":
        raise SystemExit("error: engine output must use the .engine suffix")
    onnx_artifact = ExternalArtifactMetadata.from_file(
        onnx_path, logical_name=str(manifest["artifacts"]["onnx"]["logical_name"])
    )
    if onnx_artifact.sha256 != str(manifest["artifacts"]["onnx"]["sha256"]):
        raise SystemExit("error: source ONNX SHA256 differs from the selected manifest")
    onnx_model = onnx.load(str(onnx_path), load_external_data=True)
    onnx.checker.check_model(onnx_model, full_check=True)
    trt = load_tensorrt()
    torch = importlib.import_module("torch")
    mmdeploy = importlib.import_module("mmdeploy")
    expected_mmdeploy = str(manifest["deployment"]["exporter_version"])
    if str(mmdeploy.__version__) != expected_mmdeploy:
        raise SystemExit(
            f"error: MMDeploy version differs: expected {expected_mmdeploy}, "
            f"found {mmdeploy.__version__}"
        )
    expected_commit = str(manifest["deployment"]["exporter_commit"])
    actual_commit = _git_commit(assets.mmdeploy_root)
    if actual_commit != expected_commit:
        raise SystemExit(
            f"error: MMDeploy commit differs: expected {expected_commit}, found {actual_commit}"
        )
    if str(torch.version.cuda) != "11.8":
        raise SystemExit(
            f"error: PyTorch CUDA runtime differs: expected 11.8, found {torch.version.cuda}"
        )
    if str(onnx.__version__) != "1.14.1":
        raise SystemExit(f"error: ONNX version differs: expected 1.14.1, found {onnx.__version__}")
    if not bool(trt.Builder(trt.Logger(trt.Logger.WARNING)).platform_has_fast_fp16):
        raise SystemExit("error: TensorRT builder reports no fast FP16 support")

    deploy_relative = str(manifest["deployment"]["official_deployment_config"])
    official_path = assets.mmdeploy_root / deploy_relative
    deploy_config = Config.fromfile(str(official_path))
    effective_profile = _profile(manifest)
    deploy_config["backend_config"]["common_config"]["fp16_mode"] = True
    deploy_config["backend_config"]["model_inputs"][0]["input_shapes"] = deepcopy(effective_profile)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_default = assets.artifact_directory / "engine_metadata.json"
    effective_config_path = (args.metadata_output or metadata_default).with_name(
        "effective_deploy_config.py"
    )
    effective_config_path.parent.mkdir(parents=True, exist_ok=True)
    deploy_config.dump(str(effective_config_path))

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
    inspector_path = args.inspector_output or assets.artifact_directory / "engine_inspector.json"
    inspector_path.parent.mkdir(parents=True, exist_ok=True)
    inspector_path.write_text(layer_information + "\n", encoding="utf-8")

    result = {
        "schema_version": "1.0",
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "measurement_commit": subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "manifest": {
            "path": str(manifest_path.relative_to(repository_root)),
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "gpu_name": str(torch.cuda.get_device_name(0)),
            "nvidia_driver": nvidia_smi_value("driver_version"),
            "torch_cuda_runtime": str(torch.version.cuda),
            "onnx_version": str(onnx.__version__),
        },
        "source_onnx": onnx_artifact.to_dict(),
        "builder": {
            "name": "MMDeploy official onnx2tensorrt",
            "mmdeploy_version": str(manifest["deployment"]["exporter_version"]),
            "mmdeploy_commit": str(manifest["deployment"]["exporter_commit"]),
            "official_config": deploy_relative,
            "tensorrt_version": str(trt.__version__),
            "requested_builder_flags": ["FP16"],
            "fp16_mode": True,
            "int8_mode": False,
            "tf32_policy": "TensorRT 8.6.1 builder default; not overridden",
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
    historical_engine = manifest["artifacts"].get("historical_engine")
    if isinstance(historical_engine, dict):
        historical_memory = int(historical_engine["device_memory_size_bytes"])
        candidate_memory = int(inspection["engine_device_memory_size_bytes"])
        result["engine_memory_comparison"] = {
            "api": "ICudaEngine.getDeviceMemorySize / engine.device_memory_size",
            "historical_30k_bytes": historical_memory,
            "candidate_40k_bytes": candidate_memory,
            "difference_bytes": candidate_memory - historical_memory,
            "required": True,
        }
    metadata_path = args.metadata_output or assets.artifact_directory / "engine_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("TensorRT FP16 engine built and verified outside the repository.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
