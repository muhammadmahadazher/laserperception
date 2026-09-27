"""Future external-only S2 live policy capture; importing is CPU safe."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from laserperception.detection.m8_s1_runtime_policy import (
    capture_runtime_policy as capture_s1_policy,
)
from laserperception.detection.m8_s2_runtime import (
    COMPACT_MANIFEST_SHA256,
    FULL_LEDGER_SHA256,
    INPUT_FREEZE_SHA256,
    PARTITIONS_SHA256,
    PROTOCOL_SHA256,
    RUNTIME_POLICY_SCHEMA,
    M8S2ProtocolViolation,
    single_visible_gpu_uuid,
)


def capture_runtime_policy(
    execution_commit: str,
    candidate_manifest: Mapping[str, object],
) -> dict[str, object]:
    """Capture a fresh, exact runtime only after the external-worker barrier."""

    visible_uuid = single_visible_gpu_uuid()
    base = capture_s1_policy(execution_commit, candidate_manifest)
    torch: Any = importlib.import_module("torch")
    if torch.cuda.device_count() != 1 or base.get("gpu_uuid") != visible_uuid:
        raise M8S2ProtocolViolation("S2 CUDA and NVIDIA-SMI GPU identities are ambiguous")
    vram = int(torch.cuda.get_device_properties(0).total_memory)
    if vram <= 0 or not base.get("gpu_uuid"):
        raise M8S2ProtocolViolation("S2 live GPU identity or VRAM is unavailable")
    return {
        **base,
        "schema_version": RUNTIME_POLICY_SCHEMA,
        "gpu_vram_bytes": vram,
        "s2_protocol_sha256": PROTOCOL_SHA256,
        "s2_partitions_sha256": PARTITIONS_SHA256,
        "s2_input_freeze_sha256": INPUT_FREEZE_SHA256,
        "s2_full_ledger_sha256": FULL_LEDGER_SHA256,
        "s2_compact_manifest_sha256": COMPACT_MANIFEST_SHA256,
    }
