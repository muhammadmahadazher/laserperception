"""Future same-runtime S2 structural gate, with injectable coordinate arithmetic.

This module has no accelerator dependency. A separately authorized external
worker must supply the bound candidate CUDA coordinate function; CPU mocks in
unit tests only exercise the comparison and receipt contract.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from laserperception.detection.m8_s2_input_gate import verify_consumed_input
from laserperception.detection.m8_s2_runtime import ARMS, SENTINELS, M8S2ProtocolViolation


def _coordinates(
    points: np.ndarray, coordinate_fn: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    coordinates = np.asarray(coordinate_fn(points))
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise M8S2ProtocolViolation("S2 candidate coordinates must have two columns")
    if not np.issubdtype(coordinates.dtype, np.integer):
        raise M8S2ProtocolViolation("S2 candidate coordinates must be integer")
    ordered = np.asarray(sorted(map(tuple, coordinates.tolist())), dtype=np.int64).reshape(-1, 2)
    if len(ordered) != len(np.unique(ordered, axis=0)):
        raise M8S2ProtocolViolation("S2 candidate coordinates contain duplicates")
    return ordered


def check_sentinel_coordinates(
    frames: Sequence[Mapping[str, object]],
    *,
    compact_by_id: Mapping[str, Mapping[str, object]],
    coordinate_fn: Callable[[np.ndarray], np.ndarray],
) -> dict[str, object]:
    """Require 28 frozen inputs and seven exact A2/B2 and C2/D2 CUDA pairs."""

    if len(frames) != 7:
        raise M8S2ProtocolViolation("S2 structural preflight requires seven sentinels")
    results: list[dict[str, object]] = []
    for frame_id, frame in zip(SENTINELS, frames, strict=True):
        if frame.get("frame_id") != frame_id:
            raise M8S2ProtocolViolation("S2 structural sentinel order differs")
        a2 = frame.get("A2")
        if not isinstance(a2, np.ndarray):
            raise M8S2ProtocolViolation("S2 structural A2 source is absent")
        coords: dict[str, np.ndarray] = {"A2": _coordinates(a2, coordinate_fn)}
        for arm in ARMS:
            condition_id = f"{frame_id}/{arm}"
            entry = frame.get(arm)
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise M8S2ProtocolViolation("S2 structural arm input is absent")
            points, selected_sha = entry
            if not isinstance(points, np.ndarray) or not isinstance(selected_sha, str):
                raise M8S2ProtocolViolation("S2 structural arm input is malformed")
            expected = compact_by_id.get(condition_id)
            if expected is None:
                raise M8S2ProtocolViolation("S2 structural frozen condition is absent")
            verify_consumed_input(condition_id, points, selected_sha, expected)
            coords[arm] = _coordinates(points, coordinate_fn)
        if not np.array_equal(coords["A2"], coords["B2"]):
            raise M8S2ProtocolViolation("S2 same-runtime A2/B2 CUDA coordinates differ")
        if not np.array_equal(coords["C2"], coords["D2"]):
            raise M8S2ProtocolViolation("S2 same-runtime C2/D2 CUDA coordinates differ")
        results.append(
            {
                "frame_id": frame_id,
                "candidate_pillar_counts": {name: len(value) for name, value in coords.items()},
                "candidate_coordinate_sha256": {
                    name: hashlib.sha256(value.tobytes(order="C")).hexdigest()
                    for name, value in coords.items()
                },
                "A2_B2_coordinate_identity": True,
                "C2_D2_coordinate_identity": True,
            }
        )
    return {
        "schema_version": "laserperception.m8.s2.structural-preflight.v1",
        "status": "ACCEPTED",
        "sentinel_XYZIT_exact": 28,
        "B2_A2_cuda_coordinate_identity": 7,
        "D2_C2_cuda_coordinate_identity": 7,
        "ground_truth_loaded": False,
        "sentinels": results,
    }


def make_qualification_receipt(
    structural: Mapping[str, object],
    *,
    execution_commit: str,
    runtime_policy_sha256: str,
    input_gate_receipt_sha256: str,
    qualification_authorization_id: str,
    qualification_authorization_sha256: str,
) -> dict[str, object]:
    """Build a future GT-blind receipt from same-runtime structural evidence."""

    from laserperception.detection.m8_s2_runtime import (
        FULL_LEDGER_SHA256,
        QUALIFICATION_RECEIPT_SCHEMA,
        verify_structural_preflight,
    )

    verify_structural_preflight(structural)
    if not qualification_authorization_id.strip() or len(qualification_authorization_sha256) != 64:
        raise M8S2ProtocolViolation("S2 qualification owner grant binding is absent")
    return {
        "schema_version": QUALIFICATION_RECEIPT_SCHEMA,
        "status": "ACCEPTED",
        "execution_commit": execution_commit,
        "runtime_policy_binding_sha256": runtime_policy_sha256,
        "input_gate_receipt_sha256": input_gate_receipt_sha256,
        "qualification_authorization_id": qualification_authorization_id,
        "qualification_authorization_sha256": qualification_authorization_sha256,
        "full_ledger_sha256": FULL_LEDGER_SHA256,
        "sentinel_XYZIT_exact": 28,
        "B2_A2_cuda_coordinate_identity": 7,
        "D2_C2_cuda_coordinate_identity": 7,
        "ground_truth_loaded": False,
        "structural_preflight": dict(structural),
    }
