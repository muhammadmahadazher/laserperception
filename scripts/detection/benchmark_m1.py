"""Benchmark the pinned FP32 M1 detector on actual nuScenes-mini data."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from laserperception.detection.benchmark import bytes_to_gib, latency_statistics_ms
from laserperception.detection.mmdet3d_backend import (
    DetectionEnvironmentError,
    Mmdet3dBackend,
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _manifest() -> dict[str, Any]:
    path = _repository_root() / "configs" / "detection" / "m1_pointpillars_nuscenes.yaml"
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def _data_root(value: str | None) -> Path:
    raw = value or os.environ.get("LASERPERCEPTION_NUSCENES_ROOT")
    if not raw:
        raise ValueError("set LASERPERCEPTION_NUSCENES_ROOT or pass --data-root")
    return Path(raw).expanduser()


def _git_sha() -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_repository_root(),
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip()


def _driver_version() -> str:
    process = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.splitlines()[0].strip()


def _linux_metadata() -> dict[str, str | int]:
    release: dict[str, str] = {}
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            release[key] = value.strip().strip('"')
    cpu_name = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
            if line.startswith("model name")
        ),
        "unknown",
    )
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    page_count = int(os.sysconf("SC_PHYS_PAGES"))
    return {
        "ubuntu": release.get("PRETTY_NAME", "unknown"),
        "wsl_kernel": platform.release(),
        "cpu": cpu_name,
        "system_ram_bytes_visible_to_wsl": page_size * page_count,
    }


def _memory_record(torch: Any) -> dict[str, float | int]:
    allocated = int(torch.cuda.max_memory_allocated(0))
    reserved = int(torch.cuda.max_memory_reserved(0))
    return {
        "peak_allocated_bytes": allocated,
        "peak_allocated_gib": bytes_to_gib(allocated),
        "peak_reserved_bytes": reserved,
        "peak_reserved_gib": bytes_to_gib(reserved),
        "method": "torch.cuda peak memory counters after reset on cuda:0",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", help="nuScenes root; defaults to environment variable")
    parser.add_argument("--split", choices=("mini_train", "mini_val"), default="mini_val")
    parser.add_argument("--index", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/m1/raw/pointpillars_fp32.json"),
        help="ignored sanitized raw result; written only after a complete run",
    )
    parser.add_argument("--config", type=Path, help="override pinned upstream config")
    parser.add_argument("--checkpoint", type=Path, help="override verified checkpoint cache")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = _manifest()
    benchmark_info = manifest["benchmark"]
    warmup = (
        int(args.warmup) if args.warmup is not None else int(benchmark_info["warmup_iterations"])
    )
    iterations = (
        int(args.iterations)
        if args.iterations is not None
        else int(benchmark_info["measured_iterations"])
    )
    index = int(args.index) if args.index is not None else int(benchmark_info["sample_index"])
    if warmup < 0:
        raise SystemExit("error: warmup must be non-negative")
    if iterations <= 0:
        raise SystemExit("error: iterations must be positive")

    model_info = manifest["model"]
    checkpoint_info = model_info["checkpoint"]
    checkout = Path(str(model_info["upstream_checkout"])).expanduser()
    config = args.config or checkout / str(model_info["upstream_config"])
    checkpoint = args.checkpoint or Path(
        str(checkpoint_info["cache_directory"])
    ).expanduser() / str(checkpoint_info["filename"])

    try:
        torch = importlib.import_module("torch")
        backend = Mmdet3dBackend(
            config,
            checkpoint,
            checkpoint_sha256=str(checkpoint_info["sha256"]),
        )
        data_root = _data_root(args.data_root)
        split_size = backend.dataset_size(data_root, args.split)
        prepared = backend.prepare_sample(data_root, split=args.split, index=index)

        for _ in range(warmup):
            backend.run_model(prepared)
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
        model_latencies: list[float] = []
        raw_prediction: object | None = None
        for _ in range(iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            raw_prediction = backend.run_model(prepared)
            end_event.record()
            end_event.synchronize()
            model_latencies.append(float(start_event.elapsed_time(end_event)))
        model_memory = _memory_record(torch)
        if raw_prediction is None:
            raise RuntimeError("model benchmark produced no prediction")

        for _ in range(warmup):
            warm_sample = backend.prepare_sample(data_root, split=args.split, index=index)
            backend.run_prepared(warm_sample)
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
        end_to_end_latencies: list[float] = []
        final_frame = None
        for _ in range(iterations):
            started = time.perf_counter()
            measured_sample = backend.prepare_sample(data_root, split=args.split, index=index)
            final_frame = backend.run_prepared(measured_sample)
            torch.cuda.synchronize(0)
            end_to_end_latencies.append((time.perf_counter() - started) * 1000.0)
        end_to_end_memory = _memory_record(torch)
        if final_frame is None:
            raise RuntimeError("end-to-end benchmark produced no result")

        properties = torch.cuda.get_device_properties(0)
        result = {
            "schema_version": "1.0",
            "status": "measured",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "commit_sha": _git_sha(),
            "milestone": "M1",
            "task": "pretrained 3D LiDAR object detection inference",
            "precision": "fp32",
            "batch_size": 1,
            "device": "cuda:0",
            "warmup_iterations_per_boundary": warmup,
            "measured_iterations_per_boundary": iterations,
            "dataset": {
                "name": "nuScenes",
                "version": "v1.0-mini",
                "split": args.split,
                "observed_split_size": split_size,
                "sample_index": index,
                "sample_id": prepared.sample_id,
            },
            "model": {
                "architecture": str(model_info["architecture"]),
                "upstream_config": str(model_info["upstream_config"]),
                "checkpoint_filename": str(checkpoint_info["filename"]),
                "checkpoint_sha256": str(checkpoint_info["sha256"]),
                "raw_detection_count": len(final_frame.detections),
                "trained_by_laserperception": False,
            },
            "environment": {
                "platform": platform.platform(),
                "wsl_distribution": os.environ.get("WSL_DISTRO_NAME", "unknown"),
                **_linux_metadata(),
                "python": platform.python_version(),
                **dict(backend.versions),
                "torch_cuda_runtime": str(torch.version.cuda),
                "nvidia_driver": _driver_version(),
                "gpu_name": str(properties.name),
                "gpu_total_memory_bytes": int(properties.total_memory),
                "gpu_total_memory_gib": bytes_to_gib(int(properties.total_memory)),
            },
            "measurements": {
                "model_gpu": {
                    "clock": "torch.cuda.Event",
                    "boundary": str(benchmark_info["model_gpu_boundary"]),
                    "statistics": latency_statistics_ms(model_latencies),
                    "memory": model_memory,
                },
                "end_to_end": {
                    "clock": "time.perf_counter with torch.cuda.synchronize",
                    "boundary": str(benchmark_info["end_to_end_boundary"]),
                    "statistics": latency_statistics_ms(end_to_end_latencies),
                    "memory": end_to_end_memory,
                },
            },
        }
    except (
        DetectionEnvironmentError,
        FileNotFoundError,
        ImportError,
        IndexError,
        RuntimeError,
        subprocess.CalledProcessError,
        ValueError,
    ) as error:
        raise SystemExit(f"error: {error}") from error

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result["measurements"], indent=2, sort_keys=True))
    print(f"Sanitized benchmark written to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
