#!/usr/bin/env python3
"""Smoke the official partial DSVT TensorRT boundary on the selected config.

The upstream boundary starts after the dynamic VFE and DSVT InputLayer and
ends after the four DSVT transformer blocks. It is not detector-network or
end-to-end latency evidence. Generated ONNX/engine files remain external.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from laserperception.detection.m8_backend import DsvtBackend
from laserperception.detection.m8_input import M8PointCloud


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--nuscenes-root", type=Path, required=True)
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    backend = DsvtBackend(
        manifest_path=args.manifest,
        upstream_root=args.upstream_root,
        checkpoint_path=args.checkpoint,
    )
    torch = backend._torch
    nn = torch.nn
    datasets = __import__("pcdet.datasets", fromlist=["build_dataloader"])
    cfg = backend._cfg
    cfg.DATA_CONFIG.DATA_PATH = str(args.nuscenes_root)
    cfg.DATA_CONFIG.VERSION = "v1.0-mini"
    cfg.DATA_CONFIG.INFO_PATH["test"] = ["nuscenes_infos_10sweeps_val.pkl"]
    cfg.DATA_CONFIG.BALANCED_RESAMPLING = False
    cfg.DATA_CONFIG.DATA_PROCESSOR[1].SHUFFLE_ENABLED["test"] = False
    dataset, _, _ = datasets.build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=0,
        logger=logging.getLogger("laserperception.m8.deployment"),
        training=False,
    )
    source = M8PointCloud(np.ascontiguousarray(dataset[0]["points"], dtype=np.float32))
    points = torch.from_numpy(
        np.concatenate(
            (np.zeros((source.points.shape[0], 1), dtype=np.float32), source.points), axis=1
        )
    ).cuda(0)
    batch: dict[str, Any] = {"batch_size": 1, "points": points}
    model = backend._model
    with torch.inference_mode():
        batch = model.vfe(batch)
        voxel_info = model.backbone_3d.input_layer(batch)

    class SelectedDsvtBlocks(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = model.backbone_3d.stage_0
            self.norms = model.backbone_3d.residual_norm_stage_0

        def forward(
            self,
            src: Any,
            indices_shift_0: Any,
            indices_shift_1: Any,
            masks_shift_0: Any,
            masks_shift_1: Any,
            position_embeddings: Any,
        ) -> Any:
            output = src
            for block_id in range(4):
                residual = output
                indices = indices_shift_0 if block_id % 2 == 0 else indices_shift_1
                masks = masks_shift_0 if block_id % 2 == 0 else masks_shift_1
                for set_id in range(2):
                    output = self.blocks[block_id].encoder_list[set_id](
                        output,
                        indices[set_id],
                        masks[set_id],
                        position_embeddings[block_id, set_id],
                        True,
                    )
                output = self.norms[block_id](residual + output)
            return output

    inputs = (
        batch["voxel_features"],
        voxel_info["set_voxel_inds_stage0_shift0"],
        voxel_info["set_voxel_inds_stage0_shift1"],
        voxel_info["set_voxel_mask_stage0_shift0"],
        voxel_info["set_voxel_mask_stage0_shift1"],
        torch.stack(
            [
                torch.stack(
                    [
                        voxel_info[f"pos_embed_stage0_block{block_id}_shift{shift_id}"]
                        for shift_id in range(2)
                    ],
                    dim=0,
                )
                for block_id in range(4)
            ],
            dim=0,
        ),
    )
    names = (
        "src",
        "set_voxel_inds_tensor_shift_0",
        "set_voxel_inds_tensor_shift_1",
        "set_voxel_masks_tensor_shift_0",
        "set_voxel_masks_tensor_shift_1",
        "pos_embed_tensor",
    )
    dynamic_axes = {
        "src": {0: "voxel_number"},
        names[1]: {1: "set_number_shift_0"},
        names[2]: {1: "set_number_shift_1"},
        names[3]: {1: "set_number_shift_0"},
        names[4]: {1: "set_number_shift_1"},
        names[5]: {2: "voxel_number"},
        "output": {0: "voxel_number"},
    }
    args.artifact_directory.mkdir(parents=True, exist_ok=True)
    onnx_path = args.artifact_directory / "dsvt_nuscenes_selected_boundary.onnx"
    engine_path = args.artifact_directory / "dsvt_nuscenes_selected_boundary_fp16.engine"
    wrapper = SelectedDsvtBlocks().eval().cuda(0)
    with torch.inference_mode():
        reference = wrapper(*inputs)
        torch.onnx.export(
            wrapper,
            inputs,
            onnx_path,
            input_names=list(names),
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=14,
        )

    onnx = __import__("onnx")
    model_proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(model_proto)
    trt = __import__("tensorrt")
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed: " + " | ".join(errors))
    build_config = builder.create_builder_config()
    build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    build_config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    input_shapes = {
        name: tuple(int(value) for value in tensor.shape)
        for name, tensor in zip(names, inputs, strict=True)
    }
    for name, shape in input_shapes.items():
        profile.set_shape(name, shape, shape, shape)
    build_config.add_optimization_profile(profile)
    serialized_engine = builder.build_serialized_network(network, build_config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT failed to build the selected DSVT boundary")
    engine_path.write_bytes(serialized_engine)
    runtime = trt.Runtime(trt_logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    if engine is None:
        raise RuntimeError("TensorRT failed to deserialize the selected DSVT boundary")

    record = {
        "schema_version": "1.0",
        "status": "m8_phase1_selected_config_partial_deployment_smoke_pass",
        "scientific_performance_measurement": False,
        "boundary": {
            "start": "after DynPillarVFE and DSVT InputLayer",
            "end": "after four DSVT transformer blocks",
            "excluded": [
                "point input",
                "DynPillarVFE",
                "DSVT InputLayer",
                "BEV scatter",
                "2D backbone",
                "TransFusion head",
                "postprocess",
            ],
        },
        "selected_config": "dsvt_plain_1f_onestage_nusences.yaml",
        "d_model": int(reference.shape[1]),
        "input_shapes": {key: list(value) for key, value in input_shapes.items()},
        "output_shape": list(reference.shape),
        "onnx": {
            "logical_name": onnx_path.name,
            "bytes": onnx_path.stat().st_size,
            "sha256": _sha256(onnx_path),
            "committed": False,
        },
        "engine": {
            "logical_name": engine_path.name,
            "bytes": engine_path.stat().st_size,
            "sha256": _sha256(engine_path),
            "committed": False,
            "tensorrt": trt.__version__,
            "fp16": True,
            "deserialized": True,
        },
        "builder_warnings_retained": [
            "ONNX INT64 weights were cast to INT32 and out-of-range values were clamped",
            "FP16 LayerNorm after self-attention may overflow",
            "8 FP32 infinity weights converted to FP16 infinity",
            "93 weights produced subnormal FP16 values",
        ],
        "upstream_scope": (
            "faithful selected-config adaptation of official tools/deploy.py partial backbone3D "
            "boundary; not the separate DSVT-AI-TRT end-to-end implementation"
        ),
        "reported_upstream_context_only": {
            "partial_boundary_ms_rtx3090": 13.8,
            "partial_boundary_excludes_input_layer": True,
            "full_deployed_dsvt_pillar_ms": 37.0,
            "full_deployed_dsvt_pillar_hz": 27.0,
            "laserperception_measurement": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
