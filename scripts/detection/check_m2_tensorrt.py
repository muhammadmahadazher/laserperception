"""Run the mandatory TensorRT Gate 0 build and execution smoke test."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from laserperception.detection.m2_assets import resolve_m2_asset_paths

EXPECTED_TENSORRT_VERSION = "8.6.1"


def _manifest() -> dict[str, Any]:
    path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "detection"
        / "m2_pointpillars_tensorrt.yaml"
    )
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def _nvidia_smi(field: str) -> str:
    completed = subprocess.run(
        ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip().splitlines()[0]


def _write_json(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="override external sanitized result path")
    parser.add_argument("--engine", type=Path, help="override external smoke-engine path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = _manifest()
    assets = resolve_m2_asset_paths(manifest)
    engine_path = args.engine or assets.engine_directory / "gate0_fp16.engine"
    json_path = args.json or assets.artifact_directory / "gate0.json"

    try:
        import tensorrt as trt
        import torch
    except ImportError as error:
        raise SystemExit(f"TensorRT Gate 0 import failed: {error}") from error

    if trt.__version__ != EXPECTED_TENSORRT_VERSION:
        raise SystemExit(
            f"TensorRT Gate 0 requires {EXPECTED_TENSORRT_VERSION}, found {trt.__version__}"
        )
    if torch.version.cuda != "11.8":
        raise SystemExit(f"TensorRT Gate 0 requires PyTorch CUDA 11.8, found {torch.version.cuda}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise SystemExit("TensorRT Gate 0 requires an available CUDA device")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    if not builder.platform_has_fast_fp16:
        raise SystemExit("TensorRT Gate 0 requires a device with fast FP16 support")

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    input_tensor = network.add_input("input", trt.float32, (2, 4))
    if input_tensor is None:
        raise SystemExit("TensorRT Gate 0 could not create the network input")
    identity = network.add_identity(input_tensor)
    if identity is None:
        raise SystemExit("TensorRT Gate 0 could not create the identity layer")
    identity.precision = trt.float16
    identity.set_output_type(0, trt.float16)
    output_tensor = identity.get_output(0)
    output_tensor.name = "output"
    output_tensor.dtype = trt.float16
    network.mark_output(output_tensor)

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("TensorRT Gate 0 failed to build the serialized FP16 network")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise SystemExit("TensorRT Gate 0 failed to deserialize the FP16 engine")
    context = engine.create_execution_context()
    if context is None:
        raise SystemExit("TensorRT Gate 0 failed to create an execution context")

    device_input = torch.tensor(
        [[-2.0, -0.5, 0.0, 1.5], [2.0, 3.5, 7.0, 11.0]],
        dtype=torch.float32,
        device="cuda",
    )
    device_output = torch.empty((2, 4), dtype=torch.float16, device="cuda")
    bindings = [0] * engine.num_bindings
    bindings[engine.get_binding_index("input")] = device_input.data_ptr()
    bindings[engine.get_binding_index("output")] = device_output.data_ptr()
    stream = torch.cuda.current_stream()
    executed = context.execute_async_v2(bindings, stream.cuda_stream)
    stream.synchronize()
    if not executed:
        raise SystemExit("TensorRT Gate 0 execution returned false")

    expected = device_input.to(dtype=torch.float16)
    comparison_passed = bool(torch.equal(device_output, expected))
    max_absolute_error = float(torch.max(torch.abs(device_output.float() - device_input)).item())
    if not comparison_passed:
        raise SystemExit(
            f"TensorRT Gate 0 output comparison failed (max abs error {max_absolute_error})"
        )

    result: dict[str, object] = {
        "schema_version": "1.0",
        "gate": "M2 TensorRT Gate 0",
        "status": "pass",
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "pytorch_cuda_version": torch.version.cuda,
        "tensorrt_version": trt.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "driver_version": _nvidia_smi("driver_version"),
        "compute_capability": ".".join(str(value) for value in torch.cuda.get_device_capability(0)),
        "platform_has_fast_fp16": bool(builder.platform_has_fast_fp16),
        "fp16_builder_flag": bool(config.get_flag(trt.BuilderFlag.FP16)),
        "serialized_engine_size_bytes": engine_path.stat().st_size,
        "deserialization_passed": True,
        "execution_passed": bool(executed),
        "comparison_passed": comparison_passed,
        "maximum_absolute_error": max_absolute_error,
    }
    _write_json(json_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print("TensorRT Gate 0 passed; artifacts were written outside the repository.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
