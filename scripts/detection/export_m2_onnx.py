"""Export the pinned PointPillars network through MMDeploy's official ONNX API."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import onnx
import yaml

from laserperception.detection.artifacts import ExternalArtifactMetadata
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


def _value_info(value: Any) -> dict[str, object]:
    tensor_type = value.type.tensor_type
    dimensions: list[int | str] = []
    for dimension in tensor_type.shape.dim:
        dimensions.append(
            str(dimension.dim_param) if dimension.dim_param else int(dimension.dim_value)
        )
    return {
        "name": str(value.name),
        "element_type": onnx.TensorProto.DataType.Name(tensor_type.elem_type),
        "shape": dimensions,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="nuScenes root; defaults to environment variable")
    parser.add_argument("--index", type=int, default=0, help="real trace sample from mini_val")
    parser.add_argument("--output", type=Path, help="override external ONNX artifact path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    m1_manifest = _manifest("m1_pointpillars_nuscenes.yaml")
    m2_manifest = _manifest("m2_pointpillars_tensorrt.yaml")
    m1_assets = resolve_m1_asset_paths(m1_manifest)
    m2_assets = resolve_m2_asset_paths(m2_manifest)
    output = args.output or m2_assets.artifact_directory / "pointpillars.onnx"
    if output.suffix != ".onnx":
        raise SystemExit("error: ONNX output must use the .onnx suffix")
    model_info = m1_manifest["model"]
    checkpoint_info = model_info["checkpoint"]
    deploy_relative = str(m2_manifest["deployment"]["official_deployment_config"])
    backend = M2Backend(
        m1_assets.mmdet3d_root / str(model_info["upstream_config"]),
        m1_assets.checkpoint_path,
        m2_assets.mmdeploy_root / deploy_relative,
        checkpoint_sha256=str(checkpoint_info["sha256"]),
    )
    prepared = backend.prepare_sample(
        _data_root(args.data_root), split="mini_val", index=args.index
    )
    voxelized = backend.voxelize(prepared)
    deploy_config = deepcopy(backend.deploy_config)
    onnx_config = deploy_config["onnx_config"]
    output.parent.mkdir(parents=True, exist_ok=True)

    from mmdeploy.apis.onnx import export

    export(
        backend._model,
        (voxelized.voxels, voxelized.num_points, voxelized.coors),
        output_path_prefix=str(output.with_suffix("")),
        backend="tensorrt",
        input_metas={
            "data_samples": deepcopy(list(voxelized.data_samples)),
            "mode": "predict",
        },
        context_info={"deploy_cfg": deploy_config},
        input_names=list(onnx_config["input_names"]),
        output_names=list(onnx_config["output_names"]),
        opset_version=int(onnx_config["opset_version"]),
        dynamic_axes=dict(onnx_config["dynamic_axes"]),
        verbose=bool(onnx_config.get("verbose", False)),
        keep_initializers_as_inputs=bool(onnx_config["keep_initializers_as_inputs"]),
        optimize=bool(onnx_config["optimize"]),
    )
    model = onnx.load(str(output), load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    input_records = [_value_info(value) for value in model.graph.input]
    output_records = [_value_info(value) for value in model.graph.output]
    expected_inputs = list(onnx_config["input_names"])
    expected_outputs = list(onnx_config["output_names"])
    if [record["name"] for record in input_records] != expected_inputs:
        raise SystemExit("error: exported ONNX inputs differ from the official contract")
    if [record["name"] for record in output_records] != expected_outputs:
        raise SystemExit("error: exported ONNX outputs differ from the official contract")
    for record in input_records:
        if record["shape"][0] != "voxels_num":
            raise SystemExit("error: exported ONNX voxel dimension is not dynamic")

    artifact = ExternalArtifactMetadata.from_file(
        output, logical_name=str(m2_manifest["artifacts"]["onnx"]["logical_name"])
    )
    result = {
        "schema_version": "1.0",
        "status": "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "exporter": {
            "name": "MMDeploy",
            "version": str(m2_manifest["deployment"]["exporter_version"]),
            "commit": str(m2_manifest["deployment"]["exporter_commit"]),
            "official_config": deploy_relative,
        },
        "source_model": {
            "mmdet3d_commit": str(m2_manifest["source_model"]["framework_commit"]),
            "checkpoint_sha256": str(checkpoint_info["sha256"]),
            "trace_split": "mini_val",
            "trace_index": args.index,
            "trace_sample_id": prepared.sample_id,
            "trace_voxel_shapes": voxelized.shapes,
        },
        "onnx": {
            **artifact.to_dict(),
            "opset": int(onnx_config["opset_version"]),
            "checker": "pass",
            "inputs": input_records,
            "outputs": output_records,
        },
    }
    metadata_path = m2_assets.artifact_directory / "onnx_metadata.json"
    metadata_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["onnx"], indent=2, sort_keys=True))
    print("ONNX export and checker passed; artifact remains outside the repository.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
