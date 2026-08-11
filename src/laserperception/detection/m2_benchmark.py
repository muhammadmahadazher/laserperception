"""CPU-testable validation for promoted M2 benchmark evidence."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any


def build_parity_v2_benchmark_record(
    parity: Mapping[str, Any],
    parity_bytes: bytes,
    *,
    current_commit: str,
    frozen_indices: Sequence[int],
    onnx_sha256: str,
    engine_sha256: str,
) -> dict[str, object]:
    """Validate full parity-v2 evidence and build its benchmark provenance record."""
    if parity.get("schema_version") != "2.0" or parity.get("protocol_version") != 2:
        raise ValueError("benchmarking requires parity evidence with protocol_version 2")
    if parity.get("status") != "pass" or parity.get("diagnostic_only") is not False:
        raise ValueError("benchmarking requires a passing, non-diagnostic parity-v2 run")
    if parity.get("overall_pass") is not True:
        raise ValueError("benchmarking requires overall_pass=true in parity-v2 evidence")
    if parity.get("commit_sha") != current_commit:
        raise ValueError("parity evidence must come from the current implementation commit")

    expected_indices = [int(value) for value in frozen_indices]
    dataset = parity.get("dataset")
    if (
        not isinstance(dataset, Mapping)
        or len(expected_indices) != 20
        or dataset.get("sample_indices") != expected_indices
    ):
        raise ValueError("benchmarking requires the full frozen 20-sample parity-v2 run")

    stage_1 = parity.get("stage_1")
    if not isinstance(stage_1, Mapping) or stage_1.get("overall_pass") is not True:
        raise ValueError("benchmarking requires a passing parity-v2 Stage 1 summary")

    artifacts = parity.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("parity-v2 evidence is missing artifact provenance")
    onnx = artifacts.get("onnx")
    engine = artifacts.get("engine")
    if (
        not isinstance(onnx, Mapping)
        or onnx.get("sha256") != onnx_sha256
        or not isinstance(engine, Mapping)
        or engine.get("sha256") != engine_sha256
    ):
        raise ValueError("parity-v2 evidence does not identify the current ONNX and engine")

    return {
        "protocol_version": 2,
        "status": "pass",
        "commit_sha": current_commit,
        "sample_count": len(expected_indices),
        "result_sha256": hashlib.sha256(parity_bytes).hexdigest(),
        "overall_pass": True,
        "stage_1": dict(stage_1),
    }
