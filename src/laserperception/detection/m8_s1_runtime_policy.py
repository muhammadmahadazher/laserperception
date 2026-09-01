"""GT-blind, pre-DSVT runtime-policy capture for M8 P1-S1.

This module may query the pinned Torch/CUDA stack, but it does not import a
ground-truth loader, evaluator, DSVT/OpenPCDet module, or detector backend.
It records stable policy identity only; process RNG states and seeds are
deliberately outside the cross-process equality contract.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from typing import Any

from laserperception.detection.m8_s1_runtime import (
    CANDIDATE_MANIFEST_SHA256,
    OPERATIONAL_CONSTRAINTS,
    POINT_ORDER_POLICY,
    RANDOM_POLICY,
    RUNTIME_POLICY_SCHEMA,
    M8S1ProtocolViolation,
)


def _mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise M8S1ProtocolViolation(f"M8 candidate manifest {key} is malformed")
    return value


def _text(parent: Mapping[str, object], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise M8S1ProtocolViolation(f"M8 candidate manifest {key} is malformed")
    return value


def _nvidia_identity(
    command_runner: Callable[..., subprocess.CompletedProcess[str]],
) -> tuple[str, str | None]:
    try:
        result = command_runner(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=driver_version,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise M8S1ProtocolViolation("nvidia-smi runtime identity query failed") from error
    if result.returncode != 0 or not result.stdout.strip():
        raise M8S1ProtocolViolation("nvidia-smi runtime identity query failed")
    values = [value.strip() for value in result.stdout.strip().splitlines()[0].split(",", 1)]
    if len(values) != 2 or not values[0]:
        raise M8S1ProtocolViolation("nvidia-smi runtime identity is malformed")
    uuid = values[1] if values[1] and values[1].lower() not in {"n/a", "not supported"} else None
    return values[0], uuid


def capture_runtime_policy(
    repository_execution_commit: str,
    candidate_manifest: Mapping[str, object],
    *,
    module_loader: Callable[[str], Any] = importlib.import_module,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
    """Capture stable runtime policy without constructing DSVT or loading science data."""

    if len(repository_execution_commit) != 40:
        raise M8S1ProtocolViolation("runtime-policy execution commit is malformed")
    torch = module_loader("torch")
    spconv = module_loader("spconv")
    torch_scatter = module_loader("torch_scatter")
    numpy = module_loader("numpy")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise M8S1ProtocolViolation("M8 S1 runtime-policy capture requires CUDA device 0")
    driver, gpu_uuid = _nvidia_identity(command_runner)

    upstream = _mapping(candidate_manifest, "upstream")
    checkpoint = _mapping(candidate_manifest, "checkpoint")
    feature_contract = _mapping(candidate_manifest, "feature_contract")
    point_order_policy = _text(feature_contract, "point_order_policy")
    if point_order_policy != POINT_ORDER_POLICY:
        raise M8S1ProtocolViolation("M8 candidate point-order policy changed")

    policy: dict[str, object] = {
        "schema_version": RUNTIME_POLICY_SCHEMA,
        "repository_execution_commit": repository_execution_commit,
        "python_exact_version": sys.version,
        "pytorch_exact_version": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "nvidia_driver": driver,
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "gpu_uuid": gpu_uuid,
        "spconv": str(spconv.__version__),
        "torch_scatter": str(torch_scatter.__version__),
        "numpy": str(numpy.__version__),
        "tf32": {
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        },
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "CUDA_MODULE_LOADING": os.environ.get("CUDA_MODULE_LOADING"),
        "point_order_policy": point_order_policy,
        "candidate_identity": {
            "architecture": _text(candidate_manifest, "architecture"),
            "candidate_manifest_sha256": CANDIDATE_MANIFEST_SHA256,
            "upstream_commit": _text(upstream, "commit"),
            "config_sha256": _text(upstream, "config_sha256"),
            "checkpoint_sha256": _text(checkpoint, "sha256"),
        },
        "random_policy": {
            "python": RANDOM_POLICY,
            "numpy": RANDOM_POLICY,
            "torch": RANDOM_POLICY,
            "process_rng_state_or_seed_bound": False,
        },
        "operational_constraints": dict(OPERATIONAL_CONSTRAINTS),
    }
    if policy["PYTORCH_CUDA_ALLOC_CONF"] is not None:
        raise M8S1ProtocolViolation("PYTORCH_CUDA_ALLOC_CONF must remain unset")
    if policy["cudnn_deterministic"] is not False:
        raise M8S1ProtocolViolation("cuDNN deterministic policy changed")
    if policy["torch_deterministic_algorithms"] is not False:
        raise M8S1ProtocolViolation("Torch deterministic-algorithm policy changed")
    return policy
