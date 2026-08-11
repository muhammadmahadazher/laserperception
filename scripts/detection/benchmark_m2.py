"""M2 benchmark of native PyTorch FP32 and TensorRT FP16 in isolated blocks."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from laserperception.detection.artifacts import ExternalArtifactMetadata
from laserperception.detection.benchmark import bytes_to_gib, latency_statistics_ms
from laserperception.detection.m1_assets import resolve_m1_asset_paths
from laserperception.detection.m2_assets import resolve_m2_asset_paths
from laserperception.detection.m2_backend import M2Backend, VoxelizedM2Sample
from laserperception.detection.m2_benchmark import (
    build_fidelity_diagnostic_record,
    build_parity_v2_benchmark_record,
)
from laserperception.detection.m2_diagnostics import benchmark_review_flags
from laserperception.detection.runtime_metadata import (
    nvidia_smi_value,
    repository_git_sha,
)
from laserperception.detection.tensorrt_backend import inspect_engine
from laserperception.detection.types import DetectionFrame

CANONICAL_GPU = "NVIDIA GeForce RTX 4060 Laptop GPU"
TRACKED_RESULT_NAME = "rtx4060_pytorch_fp32_vs_tensorrt_fp16.json"


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


def _profile(manifest: dict[str, Any]) -> dict[str, dict[str, list[int]]]:
    profile = manifest["profile"]
    return {
        name: {
            "min_shape": list(profile["selected_min_shapes"][name]),
            "opt_shape": list(profile["selected_opt_shapes"][name]),
            "max_shape": list(profile["selected_max_shapes"][name]),
        }
        for name in ("voxels", "num_points", "coors")
    }


def _cuda_measure(torch: Any, operation: Callable[[], object]) -> tuple[float, object]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = operation()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)), result


def _torch_memory(torch: Any, *, method: str) -> dict[str, object]:
    allocated = int(torch.cuda.max_memory_allocated(0))
    reserved = int(torch.cuda.max_memory_reserved(0))
    return {
        "peak_allocated_bytes": allocated,
        "peak_allocated_gib": bytes_to_gib(allocated),
        "peak_reserved_bytes": reserved,
        "peak_reserved_gib": bytes_to_gib(reserved),
        "method": method,
    }


def _measure_networks(
    backend: M2Backend,
    voxelized: VoxelizedM2Sample,
    engine_path: Path,
    *,
    warmup: int,
    iterations: int,
    torch: Any,
) -> tuple[list[float], list[float]]:
    for _ in range(warmup):
        backend.run_native_pytorch_raw(voxelized)
    torch.cuda.synchronize(0)
    native_latencies: list[float] = []
    for _ in range(iterations):
        latency, _ = _cuda_measure(torch, lambda: backend.run_native_pytorch_raw(voxelized))
        native_latencies.append(latency)

    for _ in range(warmup):
        backend.run_tensorrt_raw(voxelized, engine_path)
    torch.cuda.synchronize(0)
    tensorrt_latencies: list[float] = []
    for _ in range(iterations):
        latency, _ = _cuda_measure(torch, lambda: backend.run_tensorrt_raw(voxelized, engine_path))
        tensorrt_latencies.append(latency)
    return native_latencies, tensorrt_latencies


def _end_to_end_operation(
    backend: M2Backend,
    data_root: Path,
    engine_path: Path,
    *,
    index: int,
    runtime: str,
    torch: Any,
) -> tuple[float, DetectionFrame]:
    started = time.perf_counter()
    prepared = backend.prepare_sample(data_root, split="mini_val", index=index)
    voxelized = backend.voxelize(prepared)
    if runtime == "native_pytorch":
        raw = backend.run_native_pytorch_raw(voxelized)
        frame = backend.postprocess_raw(
            raw,
            voxelized,
            backend_name="native_mmdetection3d_pytorch",
            precision="fp32",
        )
    else:
        raw = backend.run_tensorrt_raw(voxelized, engine_path)
        frame = backend.postprocess_raw(
            raw,
            voxelized,
            backend_name="tensorrt",
            precision="fp16",
        )
    torch.cuda.synchronize(0)
    return (time.perf_counter() - started) * 1000.0, frame


def _measure_end_to_end(
    backend: M2Backend,
    data_root: Path,
    engine_path: Path,
    *,
    index: int,
    warmup: int,
    iterations: int,
    torch: Any,
) -> tuple[list[float], list[float], DetectionFrame, DetectionFrame]:
    for _ in range(warmup):
        _end_to_end_operation(
            backend,
            data_root,
            engine_path,
            index=index,
            runtime="native_pytorch",
            torch=torch,
        )
    native_latencies: list[float] = []
    native_frame: DetectionFrame | None = None
    for _ in range(iterations):
        latency, native_frame = _end_to_end_operation(
            backend,
            data_root,
            engine_path,
            index=index,
            runtime="native_pytorch",
            torch=torch,
        )
        native_latencies.append(float(latency))

    for _ in range(warmup):
        _end_to_end_operation(
            backend,
            data_root,
            engine_path,
            index=index,
            runtime="tensorrt",
            torch=torch,
        )
    tensorrt_latencies: list[float] = []
    tensorrt_frame: DetectionFrame | None = None
    for _ in range(iterations):
        latency, tensorrt_frame = _end_to_end_operation(
            backend,
            data_root,
            engine_path,
            index=index,
            runtime="tensorrt",
            torch=torch,
        )
        tensorrt_latencies.append(float(latency))

    if native_frame is None or tensorrt_frame is None:
        raise RuntimeError("isolated-block benchmark produced no final DetectionFrames")
    return native_latencies, tensorrt_latencies, native_frame, tensorrt_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="nuScenes root; defaults to environment variable")
    parser.add_argument("--engine", type=Path, help="override external TensorRT engine")
    parser.add_argument("--parity", type=Path, help="override external full parity JSON")
    parser.add_argument(
        "--fidelity", type=Path, help="override external native-vs-rewritten diagnosis JSON"
    )
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/m2/raw/rtx4060_pytorch_fp32_vs_tensorrt_fp16.json"),
        help="sanitized result; use benchmarks/m2/results only for a canonical passing run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository_root = _repository_root()
    m1_manifest = _manifest("m1_pointpillars_nuscenes.yaml")
    m2_manifest = _manifest("m2_pointpillars_tensorrt.yaml")
    parity_manifest = _manifest("m2_parity_v2.yaml")
    m1_assets = resolve_m1_asset_paths(m1_manifest)
    m2_assets = resolve_m2_asset_paths(m2_manifest)
    benchmark_config = m2_manifest["benchmark"]
    canonical_warmup = int(benchmark_config["warmup_iterations_per_runtime_and_boundary"])
    canonical_iterations = int(benchmark_config["measured_iterations_per_runtime_and_boundary"])
    warmup = int(args.warmup) if args.warmup is not None else canonical_warmup
    iterations = int(args.iterations) if args.iterations is not None else canonical_iterations
    if warmup < 0 or iterations <= 0:
        raise SystemExit("error: warmup must be non-negative and iterations positive")
    tracked_result_path = (
        repository_root / "benchmarks" / "m2" / "results" / TRACKED_RESULT_NAME
    ).resolve()
    tracked_promotion = args.output.resolve() == tracked_result_path
    if tracked_promotion and (warmup != canonical_warmup or iterations != canonical_iterations):
        raise SystemExit("error: tracked benchmark promotion requires the frozen 10/100 counts")
    index = int(benchmark_config["sample_index"])
    engine_path = args.engine or m2_assets.engine_directory / "pointpillars_fp16.engine"
    onnx_path = m2_assets.artifact_directory / "pointpillars.onnx"
    parity_path = args.parity or m2_assets.artifact_directory / "parity_v2.json"
    parity_bytes = parity_path.read_bytes()
    parity = json.loads(parity_bytes)
    fidelity_path = args.fidelity or (
        m2_assets.artifact_directory / "diagnostics" / "m2_diagnosis.json"
    )
    fidelity_bytes = fidelity_path.read_bytes()
    fidelity = json.loads(fidelity_bytes)
    frozen_indices = [int(value) for value in parity_manifest["dataset"]["sample_indices"]]
    current_commit = repository_git_sha(repository_root)

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
    prepared = backend.prepare_sample(data_root, split="mini_val", index=index)
    voxelized = backend.voxelize(prepared)
    profile = _profile(m2_manifest)
    if (
        not profile["voxels"]["min_shape"][0]
        <= voxelized.voxel_count
        <= profile["voxels"]["max_shape"][0]
    ):
        raise SystemExit("error: benchmark sample is outside the frozen TensorRT profile")
    torch = importlib.import_module("torch")
    gpu_name = str(torch.cuda.get_device_name(0))
    if tracked_promotion and gpu_name != CANONICAL_GPU:
        message = (
            f"error: tracked {TRACKED_RESULT_NAME} promotion requires "
            f"{CANONICAL_GPU}, found {gpu_name}"
        )
        raise SystemExit(message)

    onnx_artifact = ExternalArtifactMetadata.from_file(
        onnx_path, logical_name=str(m2_manifest["artifacts"]["onnx"]["logical_name"])
    )
    engine_artifact = ExternalArtifactMetadata.from_file(
        engine_path, logical_name=str(m2_manifest["artifacts"]["engine"]["logical_name"])
    )
    try:
        parity_record = build_parity_v2_benchmark_record(
            parity,
            parity_bytes,
            current_commit=current_commit,
            frozen_indices=frozen_indices,
            onnx_sha256=onnx_artifact.sha256,
            engine_sha256=engine_artifact.sha256,
        )
        fidelity_record = build_fidelity_diagnostic_record(
            fidelity,
            fidelity_bytes,
            current_commit=current_commit,
            frozen_indices=frozen_indices,
            checkpoint_sha256=str(checkpoint_info["sha256"]),
            onnx_sha256=onnx_artifact.sha256,
            engine_sha256=engine_artifact.sha256,
        )
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error

    engine_inspection = inspect_engine(
        engine_path,
        expected_bindings=[
            "voxels",
            "num_points",
            "coors",
            "cls_score0",
            "bbox_pred0",
            "dir_cls_pred0",
        ],
        expected_profile=profile,
    )

    native_network, tensorrt_network = _measure_networks(
        backend,
        voxelized,
        engine_path,
        warmup=warmup,
        iterations=iterations,
        torch=torch,
    )
    native_e2e, tensorrt_e2e, native_frame, tensorrt_frame = _measure_end_to_end(
        backend,
        data_root,
        engine_path,
        index=index,
        warmup=warmup,
        iterations=iterations,
        torch=torch,
    )

    torch.cuda.reset_peak_memory_stats(0)
    backend.run_native_pytorch_raw(voxelized)
    torch.cuda.synchronize(0)
    pytorch_network_memory = _torch_memory(
        torch,
        method=(
            "torch.cuda peak allocator counters after reset for one native PyTorch "
            "network call in the already initialized process"
        ),
    )
    torch.cuda.reset_peak_memory_stats(0)
    _end_to_end_operation(
        backend,
        data_root,
        engine_path,
        index=index,
        runtime="native_pytorch",
        torch=torch,
    )
    pytorch_e2e_memory = _torch_memory(
        torch,
        method=(
            "torch.cuda peak allocator counters after reset for one native PyTorch "
            "end-to-end call in the already initialized process"
        ),
    )

    native_network_stats = latency_statistics_ms(native_network)
    tensorrt_network_stats = latency_statistics_ms(tensorrt_network)
    native_e2e_stats = latency_statistics_ms(native_e2e)
    tensorrt_e2e_stats = latency_statistics_ms(tensorrt_e2e)
    network_speedup = float(native_network_stats["median_ms"]) / float(
        tensorrt_network_stats["median_ms"]
    )
    e2e_speedup = float(native_e2e_stats["median_ms"]) / float(tensorrt_e2e_stats["median_ms"])
    review_flags = benchmark_review_flags(
        native_e2e_median_ms=float(native_e2e_stats["median_ms"]),
        native_network_median_ms=float(native_network_stats["median_ms"]),
        tensorrt_e2e_median_ms=float(tensorrt_e2e_stats["median_ms"]),
        tensorrt_network_median_ms=float(tensorrt_network_stats["median_ms"]),
        native_e2e_p95_ms=float(native_e2e_stats["p95_ms"]),
        tensorrt_e2e_p95_ms=float(tensorrt_e2e_stats["p95_ms"]),
    )
    if tracked_promotion and review_flags:
        raise SystemExit(
            "error: tracked benchmark promotion requires reviewer inspection: "
            + ", ".join(review_flags)
        )
    result = {
        "schema_version": "1.0",
        "status": "measured",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit_sha": current_commit,
        "milestone": "M2",
        "same_session": True,
        "measurement_order": "isolated native-PyTorch block followed by isolated TensorRT block",
        "baseline_roles": {
            "parity_reference": "mmdeploy_rewritten_pytorch_fp32",
            "performance_baseline": "native_mmdetection3d_pytorch_fp32",
        },
        "warmup_iterations_per_runtime_and_boundary": warmup,
        "measured_iterations_per_runtime_and_boundary": iterations,
        "batch_size": 1,
        "dataset": {
            "name": "nuScenes",
            "version": "v1.0-mini",
            "split": "mini_val",
            "sample_index": index,
            "sample_id": prepared.sample_id,
            "cache_state": "warm-cache repeated-single-sample microbenchmark",
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            **dict(backend.versions),
            "mmdeploy": str(importlib.import_module("mmdeploy").__version__),
            "mmdeploy_commit": str(m2_manifest["deployment"]["exporter_commit"]),
            "onnx": str(importlib.import_module("onnx").__version__),
            "tensorrt": str(importlib.import_module("tensorrt").__version__),
            "torch_cuda_runtime": str(torch.version.cuda),
            "nvidia_driver": nvidia_smi_value("driver_version"),
            "gpu_name": gpu_name,
            "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        },
        "artifacts": {
            "checkpoint_sha256": str(checkpoint_info["sha256"]),
            "onnx": onnx_artifact.to_dict(),
            "engine": engine_artifact.to_dict(),
        },
        "parity": parity_record,
        "native_vs_rewritten_fidelity": fidelity_record,
        "benchmark_sanity": {
            "review_required": bool(review_flags),
            "flags": review_flags,
            "promotion_refused_when_flagged": True,
        },
        "measurements": {
            "network": {
                "boundary": str(benchmark_config["network_boundary"]),
                "clock": "torch.cuda.Event with per-iteration end-event synchronization",
                "native_pytorch_fp32": native_network_stats,
                "tensorrt_fp16": tensorrt_network_stats,
                "pytorch_over_tensorrt_median_speedup": network_speedup,
            },
            "end_to_end": {
                "boundary": str(benchmark_config["end_to_end_boundary"]),
                "clock": "time.perf_counter with torch.cuda.synchronize before stop",
                "native_pytorch_fp32": native_e2e_stats,
                "tensorrt_fp16": tensorrt_e2e_stats,
                "pytorch_over_tensorrt_median_speedup": e2e_speedup,
                "headline": True,
            },
        },
        "memory": {
            "native_pytorch_network": pytorch_network_memory,
            "native_pytorch_end_to_end": pytorch_e2e_memory,
            "tensorrt": {
                "serialized_engine_size_bytes": engine_artifact.size_bytes,
                "engine_device_memory_size_bytes": engine_inspection[
                    "engine_device_memory_size_bytes"
                ],
                "method": "TensorRT serialized file size and ICudaEngine.device_memory_size",
            },
            "comparable_process_level_gpu_memory": "Pending measurement",
        },
        "final_detection_counts": {
            "native_pytorch": len(native_frame.detections),
            "tensorrt": len(tensorrt_frame.detections),
        },
        "interpretation_limits": [
            "not cold-storage I/O latency",
            "not whole-dataset sequential throughput",
            "FPS derived from median latency is not a sensor-throughput guarantee",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result["measurements"], indent=2, sort_keys=True))
    print(f"Sanitized same-session benchmark written to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
