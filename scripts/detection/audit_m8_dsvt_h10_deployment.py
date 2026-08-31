#!/usr/bin/env python3
"""Audit and execute the selected partial DSVT boundary at H10 capacity.

This engineering-only route begins after DynPillarVFE/DSVT InputLayer and
ends after the four DSVT transformer blocks. It is not a detector parity,
accuracy, or latency measurement. ONNX and TensorRT artifacts stay external.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from run_m8_h10_capacity_smoke import _load_mapping, _reconstruct_selected

from laserperception.detection.m8_backend import DsvtBackend
from laserperception.detection.m8_input import M8PointCloud

NAMES = (
    "src",
    "set_voxel_inds_tensor_shift_0",
    "set_voxel_inds_tensor_shift_1",
    "set_voxel_masks_tensor_shift_0",
    "set_voxel_masks_tensor_shift_1",
    "pos_embed_tensor",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--nuscenes-root", type=Path, required=True)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--full-ledger", type=Path, required=True)
    parser.add_argument("--date-root", type=Path, required=True)
    parser.add_argument("--existing-engine", type=Path, required=True)
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _source_points(backend: DsvtBackend, root: Path) -> np.ndarray:
    datasets = __import__("pcdet.datasets", fromlist=["build_dataloader"])
    cfg = backend._cfg
    cfg.DATA_CONFIG.DATA_PATH = str(root)
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
        logger=logging.getLogger("laserperception.m8.h10_deployment"),
        training=False,
    )
    return M8PointCloud(np.ascontiguousarray(dataset[0]["points"], dtype=np.float32)).points


def _boundary_inputs(backend: DsvtBackend, points: np.ndarray) -> tuple[Any, ...]:
    torch = backend._torch
    batch, _ = backend._prepare_batch(points)
    with torch.inference_mode():
        batch = backend._model.vfe(batch)
        voxel_info = backend._model.backbone_3d.input_layer(batch)
        torch.cuda.synchronize(0)
    return (
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


def _shapes(inputs: tuple[Any, ...]) -> dict[str, tuple[int, ...]]:
    return {
        name: tuple(int(value) for value in tensor.shape)
        for name, tensor in zip(NAMES, inputs, strict=True)
    }


def _profile_shapes(
    source: dict[str, tuple[int, ...]], h10: dict[str, tuple[int, ...]]
) -> dict[str, dict[str, tuple[int, ...]]]:
    result = {}
    for name in NAMES:
        if len(source[name]) != len(h10[name]):
            raise RuntimeError(f"boundary rank changed for {name}")
        result[name] = {
            "min": tuple(min(a, b) for a, b in zip(source[name], h10[name], strict=True)),
            "opt": h10[name],
            "max": tuple(max(a, b) for a, b in zip(source[name], h10[name], strict=True)),
        }
    return result


def _torch_dtype(torch: Any, trt: Any, value: Any) -> Any:
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    if value not in mapping:
        raise RuntimeError(f"unsupported TensorRT I/O dtype: {value}")
    return mapping[value]


def _execute_h10(engine: Any, trt: Any, torch: Any, inputs: tuple[Any, ...]) -> dict[str, object]:
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("TensorRT failed to create an execution context")
    retained: list[Any] = []
    for name, tensor in zip(NAMES, inputs, strict=True):
        expected_dtype = _torch_dtype(torch, trt, engine.get_tensor_dtype(name))
        value = tensor.to(device="cuda:0", dtype=expected_dtype).contiguous()
        if not context.set_input_shape(name, tuple(int(item) for item in value.shape)):
            raise RuntimeError(f"TensorRT rejected H10 shape for {name}")
        context.set_tensor_address(name, int(value.data_ptr()))
        retained.append(value)
    unresolved = context.infer_shapes()
    if unresolved:
        raise RuntimeError(f"TensorRT shape inference left unresolved tensors: {unresolved}")
    output_shape = tuple(int(value) for value in context.get_tensor_shape("output"))
    output_dtype = _torch_dtype(torch, trt, engine.get_tensor_dtype("output"))
    output = torch.empty(output_shape, device="cuda:0", dtype=output_dtype)
    context.set_tensor_address("output", int(output.data_ptr()))
    stream = torch.cuda.current_stream(0)
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("TensorRT H10 boundary execution failed")
    stream.synchronize()
    finite = bool(torch.isfinite(output).all().item())
    if not finite:
        raise RuntimeError("TensorRT H10 boundary emitted non-finite features")
    del retained, output
    return {"attempted": True, "passed": True, "output_shape": list(output_shape), "finite": True}


def main() -> None:
    args = _parse_args()
    census = _load_mapping(args.census)
    summary = census.get("summary")
    records = census.get("records")
    if not isinstance(summary, dict) or not isinstance(records, list):
        raise ValueError("capacity census has an unexpected schema")
    condition_id = summary.get("max_condition_id")
    selected = next(
        (
            record
            for record in records
            if isinstance(record, dict) and record.get("condition_id") == condition_id
        ),
        None,
    )
    if not isinstance(condition_id, str) or not isinstance(selected, dict):
        raise ValueError("capacity census maximum identity is missing")
    h10_points = _reconstruct_selected(
        full_ledger=args.full_ledger,
        date_root=args.date_root,
        condition_id=condition_id,
    )
    if hashlib.sha256(h10_points.tobytes(order="C")).hexdigest() != selected.get(
        "candidate_feature_sha256"
    ):
        raise RuntimeError("H10 deployment input identity changed")

    backend = DsvtBackend(
        manifest_path=args.manifest,
        upstream_root=args.upstream_root,
        checkpoint_path=args.checkpoint,
    )
    torch = backend._torch
    source_inputs = _boundary_inputs(backend, _source_points(backend, args.nuscenes_root))
    h10_inputs = _boundary_inputs(backend, h10_points)
    source_shapes = _shapes(source_inputs)
    h10_shapes = _shapes(h10_inputs)
    profile_shapes = _profile_shapes(source_shapes, h10_shapes)

    model = backend._model
    nn = torch.nn

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

    dynamic_axes = {
        "src": {0: "voxel_number"},
        NAMES[1]: {1: "set_number_shift_0"},
        NAMES[2]: {1: "set_number_shift_1"},
        NAMES[3]: {1: "set_number_shift_0"},
        NAMES[4]: {1: "set_number_shift_1"},
        NAMES[5]: {2: "voxel_number"},
        "output": {0: "voxel_number"},
    }
    args.artifact_directory.mkdir(parents=True, exist_ok=True)
    onnx_path = args.artifact_directory / "dsvt_h10_selected_boundary.onnx"
    engine_path = args.artifact_directory / "dsvt_h10_selected_boundary_fp16.engine"
    wrapper = SelectedDsvtBlocks().eval().cuda(0)
    with torch.inference_mode():
        reference = wrapper(*h10_inputs)
        torch.onnx.export(
            wrapper,
            h10_inputs,
            onnx_path,
            input_names=list(NAMES),
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=14,
        )

    onnx = __import__("onnx")
    model_proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(model_proto)
    trt = __import__("tensorrt")
    trt_logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(trt_logger)
    old_serialized = args.existing_engine.read_bytes()
    old_engine = runtime.deserialize_cuda_engine(old_serialized)
    if old_engine is None:
        raise RuntimeError("TensorRT failed to deserialize the source-shape engine")
    old_profile = {
        name: {
            key: list(shape)
            for key, shape in zip(
                ("min", "opt", "max"), old_engine.get_tensor_profile_shape(name, 0), strict=True
            )
        }
        for name in NAMES
    }
    old_accepts_h10 = all(
        all(
            minimum <= observed <= maximum
            for minimum, observed, maximum in zip(
                old_profile[name]["min"], h10_shapes[name], old_profile[name]["max"], strict=True
            )
        )
        for name in NAMES
    )

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
    for name in NAMES:
        shape = profile_shapes[name]
        profile.set_shape(name, shape["min"], shape["opt"], shape["max"])
        recorded = profile.get_shape(name)
        expected = (shape["min"], shape["opt"], shape["max"])
        if tuple(tuple(int(item) for item in values) for values in recorded) != expected:
            raise RuntimeError(f"TensorRT rejected the H10 profile for {name}")
    build_config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, build_config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the H10 DSVT boundary")
    engine_path.write_bytes(serialized)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("TensorRT failed to deserialize the H10 DSVT boundary")
    execution = _execute_h10(engine, trt, torch, h10_inputs)

    record = {
        "schema_version": "1.0",
        "status": "m8_phase1e_owner_review_h10_partial_deployment_smoke_pass",
        "scientific_measurement": False,
        "latency_claim": False,
        "detector_parity_claim": False,
        "ground_truth_loaded": False,
        "condition_id": condition_id,
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
        "onnx_dynamic_axes": dynamic_axes,
        "source_domain_input_shapes": {name: list(shape) for name, shape in source_shapes.items()},
        "h10_input_shapes": {name: list(shape) for name, shape in h10_shapes.items()},
        "h10_reference_output_shape": list(reference.shape),
        "existing_source_shape_engine": {
            "logical_name": args.existing_engine.name,
            "sha256": _sha256(args.existing_engine),
            "profile": old_profile,
            "accepts_h10_shapes": old_accepts_h10,
        },
        "new_external_profile": {
            name: {key: list(value) for key, value in shapes.items()}
            for name, shapes in profile_shapes.items()
        },
        "onnx": {
            "logical_name": onnx_path.name,
            "bytes": onnx_path.stat().st_size,
            "sha256": _sha256(onnx_path),
            "checked": True,
            "committed": False,
        },
        "engine": {
            "logical_name": engine_path.name,
            "bytes": engine_path.stat().st_size,
            "sha256": _sha256(engine_path),
            "tensorrt": trt.__version__,
            "fp16": True,
            "built": True,
            "deserialized": True,
            "device_memory_bytes": int(engine.device_memory_size),
            "committed": False,
        },
        "h10_execution": execution,
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
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
