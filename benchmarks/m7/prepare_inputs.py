"""Input-only M7 freeze entrypoint; deliberately independent of TensorRT and CUDA."""

from __future__ import annotations

import json
import platform
from collections.abc import Mapping
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
    FLOAT32_LE,
    UINT32_LE,
    UINT64_LE,
    RankSourceIdentity,
    SweepProvenance,
    canonical_model_ready,
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
from laserperception.datasets.kitti_raw import KittiRawSequence, KittiReconstructionResult
from laserperception.detection.multisweep import MultiSweepBuilder, MultiSweepBuilderConfig
from laserperception.evaluation.m6b_input_oracle import reconstruct_from_frozen_transforms

PROVENANCE_SCHEMA = "laserperception.m7.sweep-provenance.v2"


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
    """Internal/synthetic A/E value object; not a public scientific input boundary."""

    frame_id: str
    a_points: np.ndarray
    e_points: np.ndarray
    a_provenance: SweepProvenance
    e_provenance: SweepProvenance
    expected_a_sha256: str
    expected_e_sha256: str


@dataclass(frozen=True, slots=True)
class GeneratedCondition:
    """One immutable regenerated condition and its complete compact-ledger record."""

    arm: Arm
    points: np.ndarray
    record: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class GeneratedFrame:
    """Canonical B/C/D/F conditions regenerated from one authoritative M6 source frame."""

    frame_id: str
    conditions: tuple[GeneratedCondition, ...]

    def condition(self, arm: Arm) -> GeneratedCondition:
        """Return one arm without accepting alternate condition order or identity."""

        for value in self.conditions:
            if value.arm is arm:
                return value
        raise ProtocolViolation(f"canonical generated frame lacks arm {arm.value}")


def _validated_m6_frame_records(path: Path) -> tuple[Mapping[str, object], ...]:
    value = json.loads(path.read_text(encoding="utf-8"))
    frames = value.get("frames")
    if not isinstance(frames, list):
        raise ProtocolViolation("full M6b input asset lacks its frozen frame ledger")
    records: list[Mapping[str, object]] = []
    for record in frames:
        if not isinstance(record, Mapping):
            raise ProtocolViolation("full M6b input frame record is malformed")
        records.append(record)
    if tuple(record.get("frame_id") for record in records) != canonical_frame_ids():
        raise ProtocolViolation("full M6b source asset does not match the frozen corpus order")
    return tuple(records)


def _source_commitments(path: Path) -> tuple[tuple[str, str, str], ...]:
    commitments: list[tuple[str, str, str]] = []
    for record in _validated_m6_frame_records(path):
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
    return tuple(commitments)


def _required_mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolViolation(f"full M6b source record lacks mapping: {key}")
    return value


def _required_string(parent: Mapping[str, object], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise ProtocolViolation(f"full M6b source record lacks string: {key}")
    return value


class CanonicalM7SourceAdapter:
    """Reconstruct frozen A/E and row provenance only through the authoritative M6 path."""

    def __init__(self, dataset_root: str | Path, m6b_input_asset: str | Path) -> None:
        self.dataset_root = Path(dataset_root)
        self.m6b_input_asset = Path(m6b_input_asset)
        verify_external_asset(
            self.m6b_input_asset,
            expected_bytes=M6B_INPUT_LEDGER_FULL_BYTES,
            expected_sha256=M6B_INPUT_LEDGER_FULL_SHA256,
        )
        records = _validated_m6_frame_records(self.m6b_input_asset)
        self._records = {_required_string(record, "frame_id"): record for record in records}
        self._sequences: dict[str, KittiRawSequence] = {}
        self._verify_dataset_contract()

    def _verify_dataset_contract(self) -> None:
        required = (
            self.dataset_root / "calib_imu_to_velo.txt",
            self.dataset_root / "calib_velo_to_cam.txt",
            self.dataset_root / "calib_cam_to_cam.txt",
        )
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(
                    f"canonical KITTI date-root prerequisite is missing: {path}"
                )
        drive_ids = {frame_id.split("/", 1)[0] for frame_id in self._records}
        for drive_id in sorted(drive_ids):
            drive_root = self.dataset_root / f"{drive_id}_sync"
            for path in (
                drive_root / "velodyne_points/data",
                drive_root / "velodyne_points/timestamps.txt",
                drive_root / "oxts/data",
            ):
                if not path.exists():
                    raise FileNotFoundError(
                        f"canonical KITTI drive prerequisite is missing: {path}"
                    )

    def _sequence(self, drive_id: str) -> KittiRawSequence:
        sequence = self._sequences.get(drive_id)
        if sequence is None:
            sequence = KittiRawSequence(
                self.dataset_root,
                self.dataset_root / f"{drive_id}_sync",
            )
            self._sequences[drive_id] = sequence
        return sequence

    @staticmethod
    def _transform_records(record: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
        raw = record.get("frozen_sweep_transforms")
        if not isinstance(raw, list) or len(raw) != 10:
            raise ProtocolViolation("M6b frozen transform ledger must contain ten transforms")
        result: list[Mapping[str, object]] = []
        for value in raw:
            if not isinstance(value, Mapping):
                raise ProtocolViolation("M6b frozen transform record is malformed")
            result.append(value)
        return tuple(result)

    @staticmethod
    def _derive_provenance(
        sequence: KittiRawSequence,
        reconstruction: KittiReconstructionResult,
        *,
        history_depth: int,
    ) -> SweepProvenance:
        points = canonical_model_ready(reconstruction.point_cloud.points_xyzt)
        selected_indices = reconstruction.selected_indices[: history_depth + 1]
        if selected_indices != tuple(
            reconstruction.current_index - rank for rank in range(history_depth + 1)
        ):
            raise ProtocolViolation("M6 source indices are not in frozen nearest-to-oldest order")
        current_timestamp = sequence.timestamps[reconstruction.current_index]
        lag_bits = points[:, 3].view(UINT32_LE)
        ranks = np.full(len(points), -1, dtype=np.int16)
        source_ids = [""] * len(points)
        ordinals = np.zeros(len(points), dtype=UINT64_LE)
        rank_sources: list[RankSourceIdentity] = []
        for rank, source_index in enumerate(selected_indices):
            frame = sequence.frame(source_index)
            timestamp = sequence.timestamps[source_index]
            lag = (
                np.float32(0.0)
                if rank == 0
                else np.float32(
                    current_timestamp.microseconds / 1_000_000 - timestamp.microseconds / 1_000_000
                )
            )
            bits = int(np.asarray([lag], dtype=FLOAT32_LE).view(UINT32_LE)[0])
            positions = np.flatnonzero(lag_bits == np.uint32(bits))
            if len(positions) == 0:
                raise ProtocolViolation(f"M6 reconstruction has no rows for history rank {rank}")
            if np.any(ranks[positions] != -1):
                raise ProtocolViolation("M6 rank lag bit patterns overlap")
            ranks[positions] = rank
            for position in positions:
                source_ids[int(position)] = frame.source_id
            ordinals[positions] = np.arange(len(positions), dtype=UINT64_LE)
            rank_sources.append(
                RankSourceIdentity(
                    history_rank=rank,
                    source_sweep_id=frame.source_id,
                    source_index=source_index,
                    timestamp_text=timestamp.original_text,
                    timestamp_nanoseconds=timestamp.nanoseconds,
                    timestamp_microseconds=timestamp.microseconds,
                    lag_float32_bits=bits,
                )
            )
        if np.any(ranks == -1) or any(not value for value in source_ids):
            raise ProtocolViolation("M6 reconstruction contains an unbound time-lag support")
        provenance = SweepProvenance(
            history_rank=ranks,
            source_sweep_id=tuple(source_ids),
            within_sweep_ordinal=ordinals,
            global_a_row_index=np.arange(len(points), dtype=UINT64_LE),
            rank_sources=tuple(rank_sources),
        )
        provenance.validate_chronology(points, expected_ranks=tuple(range(history_depth + 1)))
        return provenance

    @staticmethod
    def _verify_reconstruction(
        reconstruction: KittiReconstructionResult,
        expected: Mapping[str, object],
        *,
        frame_id: str,
    ) -> np.ndarray:
        points = canonical_model_ready(reconstruction.point_cloud.points_xyzt)
        expected_indices = expected.get("selected_indices")
        expected_lags = expected.get("time_lag_values")
        if expected_indices != list(reconstruction.selected_indices):
            raise ProtocolViolation(f"frozen source index mismatch at {frame_id}")
        if expected.get("point_count") != len(points):
            raise ProtocolViolation(f"frozen point-count mismatch at {frame_id}")
        if expected.get("model_ready_sha256") != model_ready_sha256(points):
            raise ProtocolViolation(f"frozen model-ready SHA256 mismatch at {frame_id}")
        if expected_lags != [float(value) for value in np.unique(points[:, 3])]:
            raise ProtocolViolation(f"frozen time-lag support mismatch at {frame_id}")
        points.setflags(write=False)
        return points

    def frame_sources(self, frame_id: str) -> FrameSources:
        """Derive one real A/E frame from frozen identity; no caller provenance is accepted."""

        if frame_id not in canonical_frame_ids():
            raise ProtocolViolation(f"frame is outside the frozen M7 corpus: {frame_id}")
        record = self._records[frame_id]
        drive_id, frame_text = frame_id.split("/", 1)
        frame_index = int(frame_text)
        if record.get("frame_index") != frame_index:
            raise ProtocolViolation(f"M6 frame index identity mismatch at {frame_id}")
        sequence = self._sequence(drive_id)
        transforms = self._transform_records(record)
        a_reconstruction = reconstruct_from_frozen_transforms(
            sequence,
            frame_index,
            transforms,
            builder=MultiSweepBuilder(MultiSweepBuilderConfig(max_historical_sweeps=10)),
        )
        e_reconstruction = reconstruct_from_frozen_transforms(
            sequence,
            frame_index,
            transforms,
            builder=MultiSweepBuilder(MultiSweepBuilderConfig(max_historical_sweeps=5)),
        )
        expected_a = _required_mapping(record, "h10")
        expected_e = _required_mapping(record, "h5")
        a_points = self._verify_reconstruction(a_reconstruction, expected_a, frame_id=frame_id)
        e_points = self._verify_reconstruction(e_reconstruction, expected_e, frame_id=frame_id)
        return FrameSources(
            frame_id=frame_id,
            a_points=a_points,
            e_points=e_points,
            a_provenance=self._derive_provenance(sequence, a_reconstruction, history_depth=10),
            e_provenance=self._derive_provenance(sequence, e_reconstruction, history_depth=5),
            expected_a_sha256=_required_string(expected_a, "model_ready_sha256"),
            expected_e_sha256=_required_string(expected_e, "model_ready_sha256"),
        )


def _lag_record(points: np.ndarray) -> tuple[list[str], int, float]:
    lags = np.asarray(points, dtype=np.float32)[:, 3]
    unique = np.unique(lags)
    bits = unique.view(np.uint32)
    return (
        [f"0x{int(value):08x}" for value in bits],
        len(unique),
        float(unique.max() - unique.min()),
    )


def _sweep_record(
    provenance: SweepProvenance,
) -> tuple[list[str], dict[str, int], list[dict[str, object]]]:
    result_ids: list[str] = []
    counts: dict[str, int] = {}
    for rank in sorted(int(value) for value in np.unique(provenance.history_rank)):
        positions = np.flatnonzero(provenance.history_rank == rank)
        sweep_id = provenance.source_sweep_id[int(positions[0])]
        result_ids.append(sweep_id)
        counts[str(rank)] = len(positions)
    return result_ids, counts, [source.to_dict() for source in provenance.rank_sources]


def _rank_lag_record(points: np.ndarray, provenance: SweepProvenance) -> dict[str, str]:
    lag_bits = canonical_model_ready(points)[:, 3].view(UINT32_LE)
    result: dict[str, str] = {}
    for rank in sorted(int(value) for value in np.unique(provenance.history_rank)):
        bits = np.unique(lag_bits[provenance.history_rank == rank])
        if len(bits) != 1:
            raise ProtocolViolation(f"condition history rank {rank} has multiple lag patterns")
        result[str(rank)] = f"0x{int(bits[0]):08x}"
    return result


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
    sweep_ids, sweep_counts, rank_sources = _sweep_record(provenance)
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
        "provenance_schema": PROVENANCE_SCHEMA,
        "rank_source_identities": rank_sources,
        "rank_to_lag_bit_pattern": _rank_lag_record(result_points, provenance),
        "pillar_structure": structure.to_dict(),
        "lag_scale_provenance": lag_scale,
        "quota_provenance": quota,
        "seed_provenance": seeds,
        "f_history_ranks": [2, 4, 6, 8, 10] if arm is Arm.F else None,
        "runtime_versions": {"python": platform.python_version(), "numpy": np.__version__},
    }


def _prepare_frame_conditions_for_test(
    source: FrameSources,
    *,
    implementation_commit: str,
) -> tuple[dict[str, object], ...]:
    """Synthetic test-only helper; real input freeze uses the canonical source adapter."""

    generated = _generate_frame_conditions(source, implementation_commit=implementation_commit)
    return tuple(dict(value.record) for value in generated.conditions)


def _generate_frame_conditions(
    source: FrameSources,
    *,
    implementation_commit: str,
) -> GeneratedFrame:
    """Construct immutable B/C/D/F arrays and records from one already-bound source value."""

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
    records = (
        GeneratedCondition(
            arm=Arm.B,
            points=b.points,
            record=_condition_record(
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
        ),
        GeneratedCondition(
            arm=Arm.C,
            points=c.intervention.points,
            record=_condition_record(
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
        ),
        GeneratedCondition(
            arm=Arm.D,
            points=d.points,
            record=_condition_record(
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
        ),
        GeneratedCondition(
            arm=Arm.F,
            points=f.points,
            record=_condition_record(
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
        ),
    )
    return GeneratedFrame(source.frame_id, records)


def generate_canonical_frame(
    adapter: CanonicalM7SourceAdapter,
    frame_id: str,
    *,
    implementation_commit: str,
) -> GeneratedFrame:
    """Canonical real source-to-intervention path with no caller-controlled arrays/provenance."""

    return _generate_frame_conditions(
        adapter.frame_sources(frame_id),
        implementation_commit=implementation_commit,
    )


def prepare_input_freeze(
    prerequisites: InputFreezePrerequisites,
    dataset_root: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Canonical input-only entrypoint deriving every source from frozen M6/KITTI identity."""

    identity = prerequisites.verify()
    commitments = _source_commitments(prerequisites.m6b_input_asset)
    adapter = CanonicalM7SourceAdapter(dataset_root, prerequisites.m6b_input_asset)
    conditions: list[dict[str, object]] = []
    for expected_frame, expected_a, expected_e in commitments:
        source = adapter.frame_sources(expected_frame)
        if (
            source.frame_id != expected_frame
            or source.expected_a_sha256 != expected_a
            or source.expected_e_sha256 != expected_e
        ):
            raise ProtocolViolation(f"M7 source commitment mismatch at {expected_frame}")
        generated = _generate_frame_conditions(
            source,
            implementation_commit=prerequisites.implementation_commit,
        )
        conditions.extend(dict(value.record) for value in generated.conditions)
    ledger = build_input_ledger(identity, conditions, require_full_corpus=True)
    write_input_ledger(output_path, ledger)
    return ledger
