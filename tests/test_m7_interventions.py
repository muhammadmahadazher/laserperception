from __future__ import annotations

import numpy as np
import pytest

from benchmarks.m7.interventions import (
    allocate_quotas,
    construct_b,
    construct_c,
    construct_d,
    construct_f,
    seed_identity,
    seed_text,
    select_lowest_ordinals,
    splitmix64_key,
)
from benchmarks.m7.prepare_inputs import FrameSources, prepare_frame_conditions
from benchmarks.m7.protocol import F_HISTORY_RANKS, ProtocolViolation
from benchmarks.m7.provenance import SweepProvenance, model_ready_sha256, selected_rows_bytes
from benchmarks.m7.structural_validation import (
    PillarStructure,
    validate_b_against_a,
    validate_c_against_a_e,
    validate_d_against_c,
    validate_f_against_a,
)


def _fixture(
    rows_per_rank: int = 3,
) -> tuple[np.ndarray, np.ndarray, SweepProvenance, SweepProvenance]:
    a_rows: list[list[float]] = []
    a_ranks: list[int] = []
    a_ids: list[str] = []
    a_ordinals: list[int] = []
    for rank in range(11):
        for ordinal in range(rows_per_rank):
            row = len(a_rows)
            lag = 0.0 if rank == 0 else -0.1 * rank
            a_rows.append([-45.0 + 0.5 * row, -20.0 + rank, 0.1 * ordinal, lag])
            a_ranks.append(rank)
            a_ids.append(f"sweep-{rank}")
            a_ordinals.append(ordinal)
    e_rows: list[list[float]] = []
    e_ranks: list[int] = []
    e_ids: list[str] = []
    e_ordinals: list[int] = []
    for rank in range(6):
        for ordinal in range(rows_per_rank):
            row = len(e_rows)
            lag = 0.0 if rank == 0 else -0.1 * rank
            e_rows.append([-44.0 + 0.5 * row, 10.0 + rank, 0.2 * ordinal, lag])
            e_ranks.append(rank)
            e_ids.append(f"e-sweep-{rank}")
            e_ordinals.append(ordinal)
    a = np.asarray(a_rows, dtype=np.float32)
    e = np.asarray(e_rows, dtype=np.float32)
    a_provenance = SweepProvenance(
        np.asarray(a_ranks), tuple(a_ids), np.asarray(a_ordinals), np.arange(len(a))
    )
    e_provenance = SweepProvenance(
        np.asarray(e_ranks), tuple(e_ids), np.asarray(e_ordinals), np.arange(len(e))
    )
    return a, e, a_provenance, e_provenance


def test_canonical_model_hash_ignores_view_contiguity_and_native_endianness() -> None:
    source = np.arange(48, dtype=np.float32).reshape(6, 8)[:, ::2]
    assert source.shape == (6, 4) and not source.flags.c_contiguous
    contiguous = np.ascontiguousarray(source, dtype="<f4")
    big_endian = contiguous.astype(">f4")

    assert model_ready_sha256(source) == model_ready_sha256(contiguous)
    assert model_ready_sha256(big_endian) == model_ready_sha256(contiguous)
    assert selected_rows_bytes(np.asarray([1, 256], dtype=np.int64)) == bytes.fromhex(
        "01000000000000000001000000000000"
    )


def test_arm_b_known_float32_bits_and_exact_binary64_scale() -> None:
    a, e, a_provenance, e_provenance = _fixture(rows_per_rank=1)
    b, scale = construct_b(a, e, a_provenance, e_provenance)

    assert scale.scale_hex == "0x1.0000000000000p-1"
    assert scale.scale_bits == 0x3FE0000000000000
    assert b.points[:, 3].view(np.uint32).tolist() == [
        0x00000000,
        0xBD4CCCCD,
        0xBDCCCCCD,
        0xBE19999A,
        0xBE4CCCCD,
        0xBE800000,
        0xBE99999A,
        0xBEB33333,
        0xBECCCCCD,
        0xBEE66666,
        0xBF000000,
    ]
    assert np.array_equal(b.points[:, :3].view(np.uint32), a[:, :3].view(np.uint32))
    assert not np.signbit(b.points[0, 3])
    assert np.all(np.signbit(b.points[1:, 3]))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_arm_b_rejects_nonfinite_lags(bad_value: float) -> None:
    a, e, a_provenance, e_provenance = _fixture(rows_per_rank=1)
    a[1, 3] = bad_value
    with pytest.raises(ProtocolViolation, match="finite"):
        construct_b(a, e, a_provenance, e_provenance)


def test_arm_b_rejects_zero_t10_t5_unexpected_support_and_cast_collapse() -> None:
    a, e, a_provenance, e_provenance = _fixture(rows_per_rank=1)
    zero_a = a.copy()
    zero_a[1:, 3] = np.float32(-0.0)
    with pytest.raises(ProtocolViolation, match="historical rows"):
        construct_b(zero_a, e, a_provenance, e_provenance)

    zero_e = e.copy()
    zero_e[1:, 3] = np.float32(0.0)
    with pytest.raises(ProtocolViolation, match="historical rows"):
        construct_b(a, zero_e, a_provenance, e_provenance)

    short_provenance = SweepProvenance(
        a_provenance.history_rank[:-1],
        a_provenance.source_sweep_id[:-1],
        a_provenance.within_sweep_ordinal[:-1],
        a_provenance.global_a_row_index[:-1],
    )
    with pytest.raises(ProtocolViolation, match="original row positions"):
        construct_b(a, e, short_provenance, e_provenance)

    tiny_e = e.copy()
    tiny_e[1:, 3] = -np.arange(1, 6, dtype=np.float32) * np.nextafter(np.float32(0), np.float32(1))
    with pytest.raises(ProtocolViolation, match="collapsed"):
        construct_b(a, tiny_e, a_provenance, e_provenance)


def test_integer_largest_remainder_edges_and_rank_tie_break() -> None:
    tied = allocate_quotas((1,) * 10, 5)
    assert tied.initial_quotas == (0,) * 10
    assert tied.incremented_ranks == (1, 2, 3, 4, 5)
    assert tied.final_quotas == (1, 1, 1, 1, 1, 0, 0, 0, 0, 0)
    assert allocate_quotas((1,) * 10, 0).final_quotas == (0,) * 10
    assert allocate_quotas((1,) * 10, 10).final_quotas == (1,) * 10
    assert allocate_quotas((1, 99, 0, 0, 0, 0, 0, 0, 0, 0), 1).final_quotas == (
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    with pytest.raises(ProtocolViolation):
        allocate_quotas((0,) * 10, 0)


def test_splitmix64_known_answers_selection_and_ties() -> None:
    expected_text = b"laserperception-m7-c-v1|2011_09_26_drive_0001|0000000010|1"
    text, digest, seed = seed_identity("2011_09_26_drive_0001", 10, 1)
    assert seed_text("2011_09_26_drive_0001", 10, 1) == expected_text
    assert text == expected_text
    assert digest == "66b9030d4f6a151ec1bed274fbd8ec3f85ec46a62a5e7d4353d8c5a063215ed7"
    assert seed == 0x66B9030D4F6A151E
    assert [splitmix64_key(seed, ordinal) for ordinal in (0, 1, 2, 9)] == [
        0x7CA05F4081CE1706,
        0x9A0BC31C44BD2C7B,
        0xE930618898348E1A,
        0x2F8939F2BEEE0DC0,
    ]
    keys = tuple(splitmix64_key(seed, ordinal) for ordinal in range(10))
    assert select_lowest_ordinals(keys, 3) == (5, 9, 7)
    assert select_lowest_ordinals((5, 5, 4), 2) == (2, 0)


def test_c_restores_global_order_and_d_reuses_exact_selection() -> None:
    a, e, a_provenance, e_provenance = _fixture()
    b, scale = construct_b(a, e, a_provenance, e_provenance)
    c = construct_c(
        a,
        e,
        a_provenance,
        e_provenance,
        drive_id="2011_09_26_drive_0001",
        frame_index=10,
    )
    d = construct_d(c, scale)

    assert len(c.intervention.points) == len(e)
    assert np.all(np.diff(c.intervention.selected_global_rows.astype(np.int64)) > 0)
    assert np.array_equal(
        c.intervention.points.view(np.uint32),
        a[c.intervention.selected_global_rows.astype(np.int64)].view(np.uint32),
    )
    assert d.selected_row_sha256 == c.intervention.selected_row_sha256
    assert np.array_equal(d.selected_global_rows, c.intervention.selected_global_rows)
    assert np.array_equal(
        d.points[:, :3].view(np.uint32), c.intervention.points[:, :3].view(np.uint32)
    )
    validate_c_against_a_e(a, e, c)
    validate_b_against_a(
        a,
        b,
        PillarStructure.from_points(a),
        PillarStructure.from_points(b.points),
    )
    validate_d_against_c(
        c,
        d,
        PillarStructure.from_points(c.intervention.points),
        PillarStructure.from_points(d.points),
    )


def test_provenance_rejects_duplicate_and_out_of_range_global_rows() -> None:
    a, e, a_provenance, e_provenance = _fixture()
    with pytest.raises(ProtocolViolation, match="unique"):
        SweepProvenance(
            a_provenance.history_rank,
            a_provenance.source_sweep_id,
            a_provenance.within_sweep_ordinal,
            np.zeros(len(a), dtype=np.uint64),
        )
    bad = SweepProvenance(
        a_provenance.history_rank,
        a_provenance.source_sweep_id,
        a_provenance.within_sweep_ordinal,
        np.arange(1, len(a) + 1),
    )
    with pytest.raises(ProtocolViolation, match="original row positions"):
        construct_c(
            a,
            e,
            bad,
            e_provenance,
            drive_id="2011_09_26_drive_0001",
            frame_index=10,
        )


def test_f_selects_only_complete_current_and_even_history_ranks() -> None:
    a, _, a_provenance, _ = _fixture()
    f = construct_f(a, a_provenance)

    assert tuple(np.unique(f.provenance.history_rank)) == (0, *F_HISTORY_RANKS)
    assert 1 not in f.provenance.history_rank
    assert 2 in f.provenance.history_rank
    assert 10 in f.provenance.history_rank
    for rank in (0, *F_HISTORY_RANKS):
        assert np.sum(f.provenance.history_rank == rank) == np.sum(
            a_provenance.history_rank == rank
        )
    validate_f_against_a(a, a_provenance, f)


def test_structural_validators_fail_closed_on_identity_difference() -> None:
    a, e, a_provenance, e_provenance = _fixture()
    b, _ = construct_b(a, e, a_provenance, e_provenance)
    structure = PillarStructure.from_points(a)
    changed = PillarStructure(
        coordinate_order_sha256="0" * 64,
        candidate_key_order_sha256=structure.candidate_key_order_sha256,
        retained_selection_sha256=structure.retained_selection_sha256,
        discarded_selection_sha256=structure.discarded_selection_sha256,
        candidate_count=structure.candidate_count,
        retained_count=structure.retained_count,
        discarded_count=structure.discarded_count,
    )
    with pytest.raises(ProtocolViolation, match="coordinate_order_sha256"):
        validate_b_against_a(a, b, structure, changed)


def test_input_only_frame_orchestration_emits_b_c_d_f_records_without_detector() -> None:
    a, e, a_provenance, e_provenance = _fixture()
    source = FrameSources(
        frame_id="2011_09_26_drive_0001/0000000010",
        a_points=a,
        e_points=e,
        a_provenance=a_provenance,
        e_provenance=e_provenance,
        expected_a_sha256=model_ready_sha256(a),
        expected_e_sha256=model_ready_sha256(e),
    )

    records = prepare_frame_conditions(source, implementation_commit="1" * 40)

    assert [record["arm"] for record in records] == [
        "H10_LAG_COMPRESSED",
        "H10_POINT_COUNT_MATCHED",
        "H10_LAG_COMPRESSED_POINT_COUNT_MATCHED",
        "H10_ALTERNATE_FULL_SPAN",
    ]
    assert records[1]["point_count"] == len(e)
    assert records[2]["selected_row_sha256"] == records[1]["selected_row_sha256"]
    assert records[3]["f_history_ranks"] == [2, 4, 6, 8, 10]
