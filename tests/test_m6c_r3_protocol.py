from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/m6/M6C_PROTOCOL_R3.md"

IMPLEMENTATION_HASHES = {
    "scripts/ros2/validate_m6c_r3_inputs.py": (
        "97068272cac11d26d83da0b3d81839ba74ef1c6a2510773bb512a0985529988d"
    ),
    "scripts/ros2/validate_m6c_r3_detector.py": (
        "e3cd25c5932a9d4baa0a2ff43d10c0bc61617115282d45b58c34bfc9fd5c0fcf"
    ),
    "scripts/ros2/validate_m6c_kitti_detector.py": (
        "44ca8737c57478da56da6df5469674d978297e011f84f915a34bb8379122b20d"
    ),
    "src/laserperception/detection/m6c_contract.py": (
        "5ceac7ab5c69a091bc7e6f56f99d45a02c7684e74179bc14b5fa8646f3343d96"
    ),
    "src/laserperception/evaluation/m6c_projected_reference.py": (
        "4930e34ace88f6a4c6d8c45a45a930efa797c3cf7ceb3f671126040e778659b4"
    ),
    "src/laserperception/datasets/kitti_ros_replay.py": (
        "d128ea170c00a9c8459f336145e15ec8293576dd4ccfe0ecbbcb86140949bee3"
    ),
    "ros2/laserperception_ros/laserperception_ros/kitti_raw_replay_node.py": (
        "32122a000f9f650ae061893e425bdc09acae61a2777dec4f9ded133c2b0138e9"
    ),
    "ros2/laserperception_ros/laserperception_ros/multisweep_node.py": (
        "715debf2a7ea000575a04cd1d63f6c13e54124f67149a810adbe8bc9d872a22e"
    ),
}


def test_final_r3_protocol_is_frozen_and_preserves_final_cycle() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "FROZEN — FINAL M6c EXECUTION CYCLE" in text
    assert "No canonical live R3 output was observed before this protocol freeze." in text
    assert "There is no automatic R4" in text
    assert "24/24" in text
    assert "856/856" in text
    assert "860/860" in text
    assert "10/10" in text


def test_protocol_sha256_tokens_are_lowercase_and_exactly_64_hexadecimal_characters() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    hexadecimal_tokens = re.findall(r"`([0-9A-Fa-f]{60,68})`", text)
    assert hexadecimal_tokens
    assert all(re.fullmatch(r"[0-9a-f]{64}", token) for token in hexadecimal_tokens)
    assert "6f7f63b7db7de179db11bad0a4793ab79208e6db009ff76362b2160351eaa1d2" not in text


def test_protocol_binds_exact_measurement_implementation_files() -> None:
    for relative, expected in IMPLEMENTATION_HASHES.items():
        observed = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert observed == expected


def test_protocol_preserves_shared_and_independent_claim_boundary() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    for required in (
        "PointCloud2 transport",
        "time-aware",
        "live history selection",
        "does not independently revalidate official KITTI pose derivation",
        "not statistically derived for quaternion-projection input noise",
        "does not claim that ROS reproduces the original M6a",
    ):
        assert required in text
