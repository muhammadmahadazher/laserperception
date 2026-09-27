"""Future authorized, GT-blind same-runtime S2 structural qualification."""

from __future__ import annotations

import importlib
from pathlib import Path

from laserperception.detection.m8_capacity import candidate_dynamic_pillar_coordinates_cuda
from laserperception.detection.m8_s2_preflight import (
    check_sentinel_coordinates,
    make_qualification_receipt,
)
from laserperception.detection.m8_s2_reconstruction import frame_inputs, sources
from laserperception.detection.m8_s2_runtime import SENTINELS


def run_future_qualification(
    *,
    repository_root: Path,
    date_root: Path,
    m6_ledger: Path,
    execution_commit: str,
    runtime_policy_sha256: str,
    input_gate_receipt_sha256: str,
) -> dict[str, object]:
    """Invoke only after external-worker, policy, and qualification authorization checks."""

    m7_source, m8_source, compact_by_id = sources(
        repository_root=repository_root, date_root=date_root, m6_ledger=m6_ledger
    )
    frames: list[dict[str, object]] = []
    for frame_id in SENTINELS:
        a2, inputs = frame_inputs(frame_id, m7_source, m8_source)
        frames.append(
            {
                "frame_id": frame_id,
                "A2": a2,
                **{item.arm: (item.points, item.selected_row_sha256) for item in inputs},
            }
        )
    # Importing Torch and invoking CUDA is confined to this future external call.
    torch = importlib.import_module("torch")
    structural = check_sentinel_coordinates(
        frames,
        compact_by_id=compact_by_id,
        coordinate_fn=lambda points: candidate_dynamic_pillar_coordinates_cuda(
            points, torch_module=torch
        ),
    )
    return make_qualification_receipt(
        structural,
        execution_commit=execution_commit,
        runtime_policy_sha256=runtime_policy_sha256,
        input_gate_receipt_sha256=input_gate_receipt_sha256,
    )
