"""Immutable identities and ordering for the frozen M7 protocol."""

from __future__ import annotations

from enum import Enum


class ProtocolViolation(ValueError):
    """Raised when data would violate a frozen M7 scientific rule."""


SOURCE_DRAFT_COMMIT = "7700216c234c0c4bf908dba6ab5a7106e730a627"
PROTOCOL_FREEZE_COMMIT = "fd4a143621ffc0692206c100279a9edfd5572d35"
PROTOCOL_PATH = "docs/m7/M7_PROTOCOL.md"

CHECKPOINT_SHA256 = "f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0"
ONNX_SHA256 = "61ce22a8ca31498675c32576bfb94f0093d31dc95d2762f7254bf915a59ecc16"
ENGINE_SHA256 = "2e790b1cdbdc1b88c2aafdc81b5921ebee152edd8408158f88437ae4dd1f3e7f"
ENGINE_PROFILE = (4_352, 18_207, 40_000)
SCORE_THRESHOLD = 0.25
PRIMARY_IOU_THRESHOLD = 0.50
EVALUATOR_IDENTITY = "m6b-r2-score-0.25-oriented-bev-iou-0.50"

M6B_INPUT_LEDGER_FULL_NAME = "pre_inference_input_ledger_full.json"
M6B_INPUT_LEDGER_FULL_BYTES = 5_837_452
M6B_INPUT_LEDGER_FULL_SHA256 = "e25b3d62113cc7e8c1fcf736caa68b1ab698f965f007c758ff91d3e498ca6caa"
M6B_RESULT_FULL_NAME = "kitti_raw_cross_domain_characterization_full.json"
M6B_RESULT_FULL_BYTES = 41_987_113
M6B_RESULT_FULL_SHA256 = "87870b2aa0cc2a91d39331afc8154fdad0c8c796f1cabfb4f8530a3eb106de27"

ORDERED_CORPUS_SHA256 = "76bd5f7adac3d892ad2fb00cb9cf5f4f73dd475682ee011b7ea9524060c46c95"
H10_ORDERED_HASHES_SHA256 = "63f4bd20d33a62948dc9a2593b57509380848cb48980827d0b0352c47fa37469"
H5_ORDERED_HASHES_SHA256 = "e5f43d6511d96f6db232c880f94b5464ab5d217f5e5bfdf34bd1626ab8ac7f89"


class Arm(str, Enum):
    """Frozen M7 arm identities."""

    A = "H10_NATIVE"
    B = "H10_LAG_COMPRESSED"
    C = "H10_POINT_COUNT_MATCHED"
    D = "H10_LAG_COMPRESSED_POINT_COUNT_MATCHED"
    E = "H5_NATIVE"
    F = "H10_ALTERNATE_FULL_SPAN"


NEW_ARMS = (Arm.B, Arm.C, Arm.D, Arm.F)
F_HISTORY_RANKS = (2, 4, 6, 8, 10)
CORPUS_RANGES = (
    ("2011_09_26_drive_0001", 10, 107),
    ("2011_09_26_drive_0091", 10, 339),
)
SENTINEL_FRAMES = (
    "2011_09_26_drive_0001/0000000010",
    "2011_09_26_drive_0001/0000000083",
    "2011_09_26_drive_0001/0000000011",
    "2011_09_26_drive_0001/0000000015",
    "2011_09_26_drive_0091/0000000010",
)
REPEATABILITY_REPETITIONS = 10


def canonical_frame_ids() -> tuple[str, ...]:
    """Return the frozen 428-frame M6b order without reading point data."""

    frames = tuple(
        f"{drive}/{frame_index:010d}"
        for drive, first, last in CORPUS_RANGES
        for frame_index in range(first, last + 1)
    )
    if len(frames) != 428 or len(set(frames)) != 428:
        raise AssertionError("frozen M7 corpus identity is internally inconsistent")
    return frames


def condition_id(frame_id: str, arm: Arm) -> str:
    """Return one canonical frame/arm condition identity."""

    if frame_id not in _CANONICAL_FRAME_SET:
        raise ProtocolViolation(f"frame is outside the frozen M7 corpus: {frame_id}")
    if arm not in NEW_ARMS:
        raise ProtocolViolation(f"arm is not a new canonical M7 condition: {arm}")
    return f"{frame_id}|{arm.value}"


def canonical_condition_ids() -> tuple[str, ...]:
    """Return all 1,712 new conditions in frozen frame then B/C/D/F order."""

    conditions = tuple(
        condition_id(frame_id, arm) for frame_id in canonical_frame_ids() for arm in NEW_ARMS
    )
    if len(conditions) != 1_712 or len(set(conditions)) != 1_712:
        raise AssertionError("frozen M7 condition identity is internally inconsistent")
    return conditions


_CANONICAL_FRAME_SET = frozenset(canonical_frame_ids())
