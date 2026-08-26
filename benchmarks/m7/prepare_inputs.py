"""Input-only M7 freeze entrypoint; deliberately independent of TensorRT and CUDA."""

from __future__ import annotations

import json
import platform
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmarks.m7.evidence import InputLedgerIdentity, build_input_ledger, write_input_ledger
from benchmarks.m7.interventions import construct_b, construct_c, construct_d, construct_f
from benchmarks.m7.protocol import (
    H5_ORDERED_HASHES_SHA256,
    H10_ORDERED_HASHES_SHA256,
    M6B_INPUT_LEDGER_FULL_BYTES,
    M6B_INPUT_LEDGER_FULL_SHA256,
    M6B_RESULT_FULL_BYTES,
    M6B_RESULT_FULL_SHA256,
    ORDERED_CORPUS_SHA256,
    PROTOCOL_FREEZE_COMMIT,
    Arm,
    ProtocolViolation,
    canonical_frame_ids,
)
from benchmarks.m7.provenance import (
    SweepProvenance,
    model_ready_sha256,
    verify_external_asset,
    xyz_sha256,
)
from benchmarks.m7.structural_validation import (
    PillarStructure,
    validate_b_against_a,
    validate_c_against_a_e,
    validate_d_against_c,
    validate_f_against_a,
)


@dataclass(frozen=True, slots=True)
class InputFreezePrerequisites:
    """External assets and identities required before any real intervention construction."""

    implementation_commit: str
    m6b_input_asset: Path
    m6b_result_asset: Path
    protocol_commit: str = PROTOCOL_FREEZE_COMMIT
    ordered_corpus_sha256: str = ORDERED_CORPUS_SHA256
    h10_ordered_hashes_sha256: str = H10_ORDERED_HASHES_SHA256
    h5_ordered_hashes_sha256: str = H5_ORDERED_HASHES_SHA256

    def verify(self) -> InputLedgerIdentity:
        """Fail closed on every frozen protocol/source identity."""

        if self.protocol_commit != PROTOCOL_FREEZE_COMMIT:
            raise ProtocolViolation("M7 protocol freeze identity mismatch")
        if self.ordered_corpus_sha256 != ORDERED_CORPUS_SHA256:
            raise ProtocolViolation("M7 ordered corpus identity mismatch")
        if self.h10_ordered_hashes_sha256 != H10_ORDERED_HASHES_SHA256:
            raise ProtocolViolation("M7 H10 ordered input commitment mismatch")
        if self.h5_ordered_hashes_sha256 != H5_ORDERED_HASHES_SHA256:
            raise ProtocolViolation("M7 H5 ordered input commitment mismatch")
        if len(self.implementation_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.implementation_commit
        ):
            raise ProtocolViolation("M7 implementation identity must be a lowercase 40-hex commit")
        input_sha = verify_external_asset(
            self.m6b_input_asset,
            expected_bytes=M6B_INPUT_LEDGER_FULL_BYTES,
            expected_sha256=M6B_INPUT_LEDGER_FULL_SHA256,
        )
        result_sha = verify_external_asset(
            self.m6b_result_asset,
            expected_bytes=M6B_RESULT_FULL_BYTES,
            expected_sha256=M6B_RESULT_FULL_SHA256,
        )
        return InputLedgerIdentity(
            implementation_commit=self.implementation_commit,
            m6b_input_asset_sha256=input_sha,
            m6b_result_asset_sha256=result_sha,
        )


@dataclass(frozen=True, slots=True)
class FrameSources:
    """Verified frozen A/E arrays plus explicit per-row provenance for one frame."""

    frame_id: str
    a_points: np.ndarray
    e_points: np.ndarray
    a_provenance: SweepProvenance
    e_provenance: SweepProvenance
    expected_a_sha256: str
    expected_e_sha256: str


def _source_commitments(path: Path) -> tuple[tuple[str, str, str], ...]:
    value = json.loads(path.read_text(encoding="utf-8"))
    frames = value.get("frames")
    if not isinstance(frames, list):
        raise ProtocolViolation("full M6b input asset lacks its frozen frame ledger")
    commitments: list[tuple[str, str, str]] = []
    for record in frames:
        if not isinstance(record, dict):
            raise ProtocolViolation("full M6b input frame record is malformed")
        h10 = record.get("h10")
        h5 = record.get("h5")
        if not isinstance(h10, dict) or not isinstance(h5, dict):
            raise ProtocolViolation("full M6b A/E source commitment is malformed")
        frame_id = record.get("frame_id")
        a_sha = h10.get("model_ready_sha256")
        e_sha = h5.get("model_ready_sha256")
        if not all(isinstance(value, str) for value in (frame_id, a_sha, e_sha)):
            raise ProtocolViolation("full M6b A/E source commitment identity is malformed")
        commitments.append((frame_id, a_sha, e_sha))
    if tuple(value[0] for value in commitments) != canonical_frame_ids():
        raise ProtocolViolation("full M6b source asset does not match the frozen corpus order")
    return tuple(commitments)


def _lag_record(points: np.ndarray) -> tuple[list[str], int, float]:
    lags = np.asarray(points, dtype=np.float32)[:, 3]
    unique = np.unique(lags)
    bits = unique.view(np.uint32)
    return (
        [f"0x{int(value):08x}" for value in bits],
        len(unique),
        float(unique.max() - unique.min()),
    )


def _sweep_record(provenance: SweepProvenance) -> tuple[list[str], dict[str, int]]:
    result_ids: list[str] = []
    counts: dict[str, int] = {}
    for rank in sorted(int(value) for value in np.unique(provenance.history_rank)):
        positions = np.flatnonzero(provenance.history_rank == rank)
        sweep_id = provenance.source_sweep_id[int(positions[0])]
        result_ids.append(sweep_id)
        counts[str(rank)] = len(positions)
    return result_ids, counts


def _condition_record(
    *,
    frame_id: str,
    arm: Arm,
    generation_commit: str,
    source_a_sha256: str,
    source_e_sha256: str,
    result_points: np.ndarray,
    provenance: SweepProvenance,
    selected_row_sha256: str,
    structure: PillarStructure,
    lag_scale: object,
    quota: object,
    seeds: object,
) -> dict[str, object]:
    drive_id, frame_text = frame_id.split("/", 1)
    lag_bits, support_count, lag_span = _lag_record(result_points)
    sweep_ids, sweep_counts = _sweep_record(provenance)
    return {
        "condition_id": f"{frame_id}|{arm.value}",
        "drive_id": drive_id,
        "frame_index": int(frame_text),
        "arm": arm.value,
        "generation_commit": generation_commit,
        "source_a_sha256": source_a_sha256,
        "source_e_sha256": source_e_sha256,
        "point_count": len(result_points),
        "xyz_sha256": xyz_sha256(result_points),
        "model_ready_sha256": model_ready_sha256(result_points),
        "selected_row_sha256": selected_row_sha256,
        "lag_bit_patterns": lag_bits,
        "lag_support_count": support_count,
        "lag_span_seconds": lag_span,
        "sweep_ids": sweep_ids,
        "per_sweep_point_counts": sweep_counts,
        "pillar_structure": structure.to_dict(),
        "lag_scale_provenance": lag_scale,
        "quota_provenance": quota,
        "seed_provenance": seeds,
        "f_history_ranks": [2, 4, 6, 8, 10] if arm is Arm.F else None,
        "runtime_versions": {"python": platform.python_version(), "numpy": np.__version__},
    }


def prepare_frame_conditions(
    source: FrameSources,
    *,
    implementation_commit: str,
) -> tuple[dict[str, object], ...]:
    """Construct and structurally validate one B/C/D/F frame without a detector dependency."""

    if source.frame_id not in canonical_frame_ids():
        raise ProtocolViolation(f"frame is outside the frozen M7 corpus: {source.frame_id}")
    actual_a = model_ready_sha256(source.a_points)
    actual_e = model_ready_sha256(source.e_points)
    if actual_a != source.expected_a_sha256 or actual_e != source.expected_e_sha256:
        raise ProtocolViolation(f"frozen A/E input commitment mismatch at {source.frame_id}")
    drive_id, frame_text = source.frame_id.split("/", 1)
    b, scale = construct_b(
        source.a_points,
        source.e_points,
        source.a_provenance,
        source.e_provenance,
    )
    c = construct_c(
        source.a_points,
        source.e_points,
        source.a_provenance,
        source.e_provenance,
        drive_id=drive_id,
        frame_index=int(frame_text),
    )
    d = construct_d(c, scale)
    f = construct_f(source.a_points, source.a_provenance)
    a_structure = PillarStructure.from_points(source.a_points)
    b_structure = PillarStructure.from_points(b.points)
    c_structure = PillarStructure.from_points(c.intervention.points)
    d_structure = PillarStructure.from_points(d.points)
    f_structure = PillarStructure.from_points(f.points)
    validate_b_against_a(source.a_points, b, a_structure, b_structure)
    validate_c_against_a_e(source.a_points, source.e_points, c)
    validate_d_against_c(c, d, c_structure, d_structure)
    validate_f_against_a(source.a_points, source.a_provenance, f)
    return (
        _condition_record(
            frame_id=source.frame_id,
            arm=Arm.B,
            generation_commit=implementation_commit,
            source_a_sha256=actual_a,
            source_e_sha256=actual_e,
            result_points=b.points,
            provenance=b.provenance,
            selected_row_sha256=b.selected_row_sha256,
            structure=b_structure,
            lag_scale=scale.to_dict(),
            quota=None,
            seeds=None,
        ),
        _condition_record(
            frame_id=source.frame_id,
            arm=Arm.C,
            generation_commit=implementation_commit,
            source_a_sha256=actual_a,
            source_e_sha256=actual_e,
            result_points=c.intervention.points,
            provenance=c.intervention.provenance,
            selected_row_sha256=c.intervention.selected_row_sha256,
            structure=c_structure,
            lag_scale=None,
            quota=c.quota.to_dict(),
            seeds=list(c.seed_identities),
        ),
        _condition_record(
            frame_id=source.frame_id,
            arm=Arm.D,
            generation_commit=implementation_commit,
            source_a_sha256=actual_a,
            source_e_sha256=actual_e,
            result_points=d.points,
            provenance=d.provenance,
            selected_row_sha256=d.selected_row_sha256,
            structure=d_structure,
            lag_scale=scale.to_dict(),
            quota=c.quota.to_dict(),
            seeds=list(c.seed_identities),
        ),
        _condition_record(
            frame_id=source.frame_id,
            arm=Arm.F,
            generation_commit=implementation_commit,
            source_a_sha256=actual_a,
            source_e_sha256=actual_e,
            result_points=f.points,
            provenance=f.provenance,
            selected_row_sha256=f.selected_row_sha256,
            structure=f_structure,
            lag_scale=None,
            quota=None,
            seeds=None,
        ),
    )


def prepare_input_freeze(
    prerequisites: InputFreezePrerequisites,
    sources: Iterable[FrameSources],
    output_path: str | Path,
) -> dict[str, object]:
    """Future canonical input-only entrypoint; it never imports or constructs a detector."""

    identity = prerequisites.verify()
    commitments = _source_commitments(prerequisites.m6b_input_asset)
    source_iterator = iter(sources)
    conditions: list[dict[str, object]] = []
    for expected_frame, expected_a, expected_e in commitments:
        try:
            source = next(source_iterator)
        except StopIteration as error:
            raise ProtocolViolation(
                "M7 input sources ended before all 428 frozen frames"
            ) from error
        if (
            source.frame_id != expected_frame
            or source.expected_a_sha256 != expected_a
            or source.expected_e_sha256 != expected_e
        ):
            raise ProtocolViolation(f"M7 source commitment mismatch at {expected_frame}")
        conditions.extend(
            prepare_frame_conditions(
                source,
                implementation_commit=prerequisites.implementation_commit,
            )
        )
    try:
        next(source_iterator)
    except StopIteration:
        pass
    else:
        raise ProtocolViolation(
            "M7 input sources contain frames beyond the frozen 428-frame corpus"
        )
    ledger = build_input_ledger(identity, conditions, require_full_corpus=True)
    write_input_ledger(output_path, ledger)
    return ledger
