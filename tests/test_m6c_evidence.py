from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SENTINEL_PATH = ROOT / "benchmarks/m6c/preregistration/detector_sentinels.json"
SENTINEL_SHA256 = "e80e803fe8923a52ced1bdecd17841a4cf915352a56ea5427607746409cf3be3"


def test_m6c_detector_sentinels_are_frozen_before_inference() -> None:
    payload = SENTINEL_PATH.read_bytes()
    record = json.loads(payload)
    assert hashlib.sha256(payload).hexdigest() == SENTINEL_SHA256
    assert len(payload) == 1_492_097
    assert len(payload) < 5_000_000
    assert record["status"] == "FROZEN_BEFORE_M6C_DETECTOR_EXECUTION"
    assert record["selection"]["condition_count"] == 10
    assert record["selection"]["selected_without_M6c_detector_results"] is True
    assert record["ros_output_contract"]["velocity_exposed"] is False
    assert record["oracle_independence"]["builder_node_calls_KittiRawSequence"] is False
    assert len(record["sentinels"]) == 10


def test_m6c_preregistration_contains_no_private_or_binary_payloads() -> None:
    text = SENTINEL_PATH.read_text(encoding="utf-8")
    lowered = text.lower()
    assert "j:\\" not in lowered
    assert "c:\\users" not in lowered
    assert "/home/" not in lowered
    assert "/root/" not in lowered
    assert '"points_xyz"' not in text
    assert '"raw_tensor_values"' not in text
    assert '.engine"' not in lowered


def test_m6c_protocol_freezes_implementation_and_exact_gates() -> None:
    protocol = (ROOT / "docs/m6/M6C_PROTOCOL.md").read_text(encoding="utf-8")
    assert "0b74d048423e78ad349c35a55cdc8a9cc082eb8b" in protocol
    assert "24/24" in protocol
    assert "856/856" in protocol
    assert "10/10" in protocol
    assert "corpus must never be reduced" in protocol
    assert "does not expose `velocity_xy`" in protocol
