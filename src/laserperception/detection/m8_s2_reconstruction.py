"""Reconstruct frozen S2 inputs once for future preflight or science consumption.

The module is CPU-safe to import; no detector, GT, Torch, or CUDA is imported.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from laserperception.detection.m8_s1_frozen_input import FrozenInputSource
from laserperception.detection.m8_s1_runtime import canonical_frame_ids
from laserperception.detection.m8_s2_input import (
    S2Input,
    array_sha256,
    canonical_xyzt,
    lift_b2,
    lift_c2,
    lift_d2,
    lift_f2,
)
from laserperception.detection.m8_s2_input_gate import verify_consumed_input
from laserperception.detection.m8_s2_runtime import (
    ARMS,
    COMPACT_MANIFEST_PATH,
    SENTINELS,
    M8S2ProtocolViolation,
)


def frame_inputs(
    frame_id: str, m7_source: Any, m8_source: FrozenInputSource
) -> tuple[np.ndarray, tuple[S2Input, ...]]:
    """Lift unchanged M7 B/C/D/F outputs into a verified M8 source pair."""

    intervention = importlib.import_module("benchmarks.m7.interventions")
    provenance = importlib.import_module("benchmarks.m7.provenance")
    m7 = m7_source.frame_sources(frame_id)
    (a2, a_record), (e2, e_record) = m8_source.pair(frame_id)
    if array_sha256(a2) != a_record["input_sha256"] or array_sha256(e2) != e_record["input_sha256"]:
        raise M8S2ProtocolViolation("S2 live A2/E2 source input differs")
    if (
        array_sha256(canonical_xyzt(a2)) != m7.expected_a_sha256
        or array_sha256(canonical_xyzt(e2)) != m7.expected_e_sha256
    ):
        raise M8S2ProtocolViolation("S2 live M7 source projection differs")
    drive_id, frame_text = frame_id.split("/", 1)
    b, scale = intervention.construct_b(m7.a_points, m7.e_points, m7.a_provenance, m7.e_provenance)
    c = intervention.construct_c(
        m7.a_points,
        m7.e_points,
        m7.a_provenance,
        m7.e_provenance,
        drive_id=drive_id,
        frame_index=int(frame_text),
    )
    d = intervention.construct_d(c, scale)
    f = intervention.construct_f(m7.a_points, m7.a_provenance)
    c2 = lift_c2(a2, c.intervention, e2_count=len(e2))
    result = (lift_b2(a2, b), c2, lift_d2(c2, d), lift_f2(a2, f))
    if tuple(item.arm for item in result) != ARMS:
        raise M8S2ProtocolViolation("S2 live arm order differs")
    for item in result:
        if provenance.selected_rows_sha256(item.selected_global_rows) != item.selected_row_sha256:
            raise M8S2ProtocolViolation("S2 live selected-row SHA differs")
    return a2, result


def sources(
    *, repository_root: Path, date_root: Path, m6_ledger: Path
) -> tuple[Any, FrozenInputSource, dict[str, Mapping[str, object]]]:
    """Open the frozen sources and compact identities without loading GT."""

    manifest = json.loads((repository_root / COMPACT_MANIFEST_PATH).read_text(encoding="utf-8"))
    by_id = {row["condition_id"]: row for row in manifest["conditions"]}
    m7_module = importlib.import_module("benchmarks.m7.prepare_inputs")
    m7_source = m7_module.CanonicalM7SourceAdapter(date_root, m6_ledger)
    m8_source = FrozenInputSource.load(
        date_root=date_root,
        full_ledger=m6_ledger,
        accepted_ledger=repository_root
        / "benchmarks/m8/diagnostics/m8_input_projection_ledger.json",
    )
    return m7_source, m8_source, by_id


def verified_inputs(
    *, mode: str, repository_root: Path, date_root: Path, m6_ledger: Path
) -> Iterator[tuple[str, np.ndarray, dict[str, object]]]:
    """Yield each just-verified array in detector-consumption order."""

    m7_source, m8_source, by_id = sources(
        repository_root=repository_root, date_root=date_root, m6_ledger=m6_ledger
    )
    frames = SENTINELS if mode == "repeatability" else canonical_frame_ids()
    for frame_id in frames:
        _, items = frame_inputs(frame_id, m7_source, m8_source)
        for item in items:
            condition_id = f"{frame_id}/{item.arm}"
            bound = by_id.get(condition_id)
            if not isinstance(bound, Mapping):
                raise M8S2ProtocolViolation("S2 frozen compact condition is absent")
            evidence = verify_consumed_input(
                condition_id, item.points, item.selected_row_sha256, bound
            )
            yield condition_id, item.points, evidence
