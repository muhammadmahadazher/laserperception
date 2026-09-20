from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

import laserperception.detection.m8_s1_input_gate as gate
from laserperception.detection.m8_s1_runtime import M8S1ProtocolViolation, sha256_file


def _gate_result(commit: str = "a" * 40) -> dict[str, object]:
    return {
        "result": "PASS",
        "detector_inference_performed": False,
        "ground_truth_loaded": False,
        "conditions_checked": 856,
        "H10_exact_count": 428,
        "H5_exact_count": 428,
        "full_XYZIT_exact_count": 856,
        "XYZT_projection_exact_count": 856,
        "intensity_exact_count": 856,
        "range_drop_exact_count": 856,
        "mismatch_count": 0,
        "mismatch_ids": [],
        "final_implementation_commit": commit,
        "existing_ledger": {"sha256": gate.INPUT_LEDGER_SHA256},
        "source_m6b_full_ledger": {"sha256": "b" * 64},
    }


def _receipt(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setattr(gate, "_git_head", lambda root: "a" * 40)
    monkeypatch.setattr(
        gate,
        "_full_ledger_bindings",
        lambda path: ("b" * 64, "c" * 64, gate.ORDERED_FRAME_SHA256),
    )
    return gate.create_input_gate_receipt(
        repository_root=Path("."),
        full_ledger=Path("full.json"),
        execution_commit="a" * 40,
        gate_result=_gate_result(),
    )


def test_complete_856_receipt_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    receipt = _receipt(monkeypatch)
    assert receipt["complete"] is True
    assert (receipt["conditions_exact"], receipt["H10_exact"], receipt["H5_exact"]) == (
        856,
        428,
        428,
    )
    gate.verify_input_gate_receipt_payload(receipt)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"conditions_checked": 855}, "conditions_checked"),
        ({"mismatch_count": 1, "mismatch_ids": ["condition"]}, "mismatch_count"),
        ({"final_implementation_commit": "d" * 40}, "final_implementation_commit"),
    ],
)
def test_partial_mismatch_or_old_commit_cannot_create_complete_receipt(
    monkeypatch: pytest.MonkeyPatch, mutation: dict[str, object], message: str
) -> None:
    monkeypatch.setattr(gate, "_git_head", lambda root: "a" * 40)
    result = {**_gate_result(), **mutation}
    with pytest.raises(M8S1ProtocolViolation, match=message):
        gate.create_input_gate_receipt(
            repository_root=Path("."),
            full_ledger=Path("full.json"),
            execution_commit="a" * 40,
            gate_result=result,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("accepted_input_ledger_sha256", "0" * 64),
        ("conditions_exact", 855),
    ],
)
def test_receipt_binding_mutations_fail_closed(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    receipt = deepcopy(_receipt(monkeypatch))
    receipt[field] = value
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256")
    receipt["receipt_sha256"] = gate.canonical_json_sha256(unsigned)
    with pytest.raises(M8S1ProtocolViolation, match="binding mismatch|live mismatch"):
        gate.verify_input_gate_receipt_payload(receipt)


def test_unknown_field_and_bad_self_hash_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    receipt = _receipt(monkeypatch)
    receipt["skip_input_gate"] = True
    with pytest.raises(M8S1ProtocolViolation, match="fields differ"):
        gate.verify_input_gate_receipt_payload(receipt)
    receipt.pop("skip_input_gate")
    receipt["receipt_sha256"] = "0" * 64
    with pytest.raises(M8S1ProtocolViolation, match="SHA256 mismatch"):
        gate.verify_input_gate_receipt_payload(receipt)


@pytest.mark.parametrize(
    ("live_bindings", "message"),
    [
        (("e" * 64, "c" * 64, gate.ORDERED_FRAME_SHA256), "source_manifest_identity"),
        (("b" * 64, "e" * 64, gate.ORDERED_FRAME_SHA256), "full_transform_ledger_sha256"),
    ],
)
def test_live_receipt_rejects_old_head_and_wrong_source_or_transform(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    live_bindings: tuple[str, str, str],
    message: str,
) -> None:
    receipt = _receipt(monkeypatch)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(gate, "_git_head", lambda root: "d" * 40)
    with pytest.raises(M8S1ProtocolViolation, match="execution commit changed"):
        gate.verify_input_gate_receipt(
            path,
            repository_root=tmp_path,
            full_ledger=tmp_path / "full.json",
            execution_commit="a" * 40,
        )
    monkeypatch.setattr(gate, "_git_head", lambda root: "a" * 40)
    monkeypatch.setattr(
        gate,
        "_full_ledger_bindings",
        lambda path: live_bindings,
    )
    monkeypatch.setattr(gate, "sha256_file", lambda path: gate.INPUT_LEDGER_SHA256)
    with pytest.raises(M8S1ProtocolViolation, match=message):
        gate.verify_input_gate_receipt(
            path,
            repository_root=tmp_path,
            full_ledger=tmp_path / "full.json",
            execution_commit="a" * 40,
        )


def test_receipt_serialization_is_deterministic_and_atomic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receipt = _receipt(monkeypatch)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    gate.write_input_gate_receipt(first, receipt)
    gate.write_input_gate_receipt(second, receipt)
    assert first.read_bytes() == second.read_bytes()
    assert sha256_file(first) == sha256_file(second)


def test_input_gate_module_imports_no_torch_or_cuda() -> None:
    code = (
        "import sys; import laserperception.detection.m8_s1_input_gate; "
        "assert 'torch' not in sys.modules; "
        "assert not any(n.startswith('cuda') for n in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
