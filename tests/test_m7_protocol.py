from __future__ import annotations

import sys

from benchmarks.m7.protocol import (
    ENGINE_PROFILE,
    ENGINE_SHA256,
    F_HISTORY_RANKS,
    NEW_ARMS,
    Arm,
    canonical_condition_ids,
    canonical_frame_ids,
)


def test_frozen_protocol_constants_and_corpus_order() -> None:
    frames = canonical_frame_ids()
    conditions = canonical_condition_ids()

    assert ENGINE_SHA256 == "2e790b1cdbdc1b88c2aafdc81b5921ebee152edd8408158f88437ae4dd1f3e7f"
    assert ENGINE_PROFILE == (4352, 18207, 40000)
    assert F_HISTORY_RANKS == (2, 4, 6, 8, 10)
    assert NEW_ARMS == (Arm.B, Arm.C, Arm.D, Arm.F)
    assert len(frames) == 428
    assert frames[0] == "2011_09_26_drive_0001/0000000010"
    assert frames[-1] == "2011_09_26_drive_0091/0000000339"
    assert len(conditions) == 1712
    assert conditions[:4] == tuple(f"{frames[0]}|{arm.value}" for arm in NEW_ARMS)


def test_input_generation_import_has_no_tensorrt_or_cuda_side_effect() -> None:
    before = set(sys.modules)
    __import__("benchmarks.m7.prepare_inputs")
    added = set(sys.modules).difference(before)

    assert "tensorrt" not in added
    assert not any(name.startswith("mmdeploy") for name in added)
