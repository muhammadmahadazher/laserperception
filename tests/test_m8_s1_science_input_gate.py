from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import laserperception.evaluation.m8_s1_science as science
from laserperception.detection.m8_s1_runtime import canonical_condition_ids


class _Source:
    def __init__(self, frames: tuple[str, ...]) -> None:
        self.frames = {frame: {} for frame in frames}
        self.pair_calls: list[str] = []

    def pair(self, frame_id: str) -> tuple[tuple[np.ndarray, dict[str, object]], ...]:
        self.pair_calls.append(frame_id)
        return tuple(
            (
                np.zeros((1, 5), dtype=np.float32),
                {
                    "condition_id": f"{frame_id}/{history}",
                    "history": history,
                    "input_point_count": 1,
                    "input_sha256": history.lower().ljust(64, "0"),
                },
            )
            for history in ("H10", "H5")
        )


class _Sampler:
    samples: tuple[()] = ()

    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def begin_block(self, name: str) -> None:
        pass

    def end_block(self, name: str) -> None:
        pass


class _Backend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def runtime_state(self) -> dict[str, object]:
        return {"mock": True}

    def infer(self, points: np.ndarray, *, sample_id: str) -> SimpleNamespace:
        self.calls.append(sample_id)
        return SimpleNamespace(detections=())


class _Attempt:
    def __init__(self, root: Path, identity: object, **kwargs: object) -> None:
        self.evidence_bindings = dict(kwargs.get("evidence_bindings", {}))
        self.completed: list[str] = []

    def record(self, condition_id: str, payload: object) -> None:
        self.completed.append(condition_id)

    def finalize(self) -> dict[str, object]:
        return {"evidence_bindings": self.evidence_bindings, "completed": len(self.completed)}

    def fail(self, reason: str) -> dict[str, object]:
        return {"status": "INCOMPLETE", "failure_reason": reason}


def test_stage_r_revalidates_exactly_14_consumed_inputs() -> None:
    source = _Source(science.STAGE_R_FRAMES)
    consumed, evidence = science._revalidate_stage_r(source)
    assert tuple(consumed) == science.stage_r_condition_ids()
    assert len(source.pair_calls) == 7
    assert evidence["conditions_exact"] == 14


def test_full_revalidation_remains_856_for_canonical_passes() -> None:
    frames = tuple(
        dict.fromkeys(identifier.rsplit("/", 1)[0] for identifier in canonical_condition_ids())
    )
    source = _Source(frames)
    assert science._revalidate_all(source) == {
        "H10_exact": 428,
        "H5_exact": 428,
        "conditions_exact": 856,
    }
    assert len(source.pair_calls) == 428


def test_missing_or_invalid_stage_r_receipt_precedes_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    constructed = False

    def construct(**kwargs: object) -> object:
        nonlocal constructed
        constructed = True
        return object()

    monkeypatch.setattr(science.DsvtBackend, "from_environment", construct)
    common = {
        "mode": "stage-r",
        "repository_root": tmp_path,
        "full_ledger": tmp_path / "ledger.json",
        "date_root": tmp_path,
        "runtime_commit": "a" * 40,
        "attempt_root": tmp_path / "attempt",
        "logical_pass_id": "stage-r-1",
        "attempt_id": "attempt-1",
    }
    with pytest.raises(ValueError, match="requires a complete"):
        science.run_scientific_attempt(**common)
    monkeypatch.setattr(
        science,
        "verify_input_gate_receipt",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("invalid receipt")),
    )
    with pytest.raises(ValueError, match="invalid receipt"):
        science.run_scientific_attempt(
            **common,
            input_gate_receipt=tmp_path / "receipt.json",
        )
    assert constructed is False


def _run_mock_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> tuple[_Source, _Backend, dict[str, object]]:
    frame_ids = (
        science.STAGE_R_FRAMES
        if mode == "stage-r"
        else tuple(
            dict.fromkeys(identifier.rsplit("/", 1)[0] for identifier in canonical_condition_ids())
        )
    )
    source = _Source(frame_ids)
    backend = _Backend()
    monkeypatch.setattr(science, "verify_input_gate_receipt", lambda *args, **kwargs: "f" * 64)
    monkeypatch.setattr(science.FrozenInputSource, "load", lambda **kwargs: source)
    monkeypatch.setattr(science, "_load_gt", lambda path: (object(), {}))
    monkeypatch.setattr(science.DsvtBackend, "from_environment", lambda **kwargs: backend)
    monkeypatch.setattr(science, "AtomicAttempt", _Attempt)
    monkeypatch.setattr(science, "NvidiaSmiSampler", _Sampler)
    monkeypatch.setattr(science, "summarize_gpu_telemetry", lambda samples: {})
    monkeypatch.setattr(science, "_condition_evidence", lambda *args, **kwargs: {"mock": True})
    result = science.run_scientific_attempt(
        mode=mode,
        repository_root=tmp_path,
        full_ledger=tmp_path / "ledger.json",
        date_root=tmp_path,
        runtime_commit="a" * 40,
        attempt_root=tmp_path / mode,
        logical_pass_id=f"{mode}-1",
        attempt_id="attempt-1",
        input_gate_receipt=(tmp_path / "receipt.json" if mode == "stage-r" else None),
    )
    return source, backend, result


def test_valid_receipt_reaches_mock_backend_without_full_revalidation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        science,
        "_revalidate_all",
        lambda source: (_ for _ in ()).throw(AssertionError("full replay called")),
    )
    source, backend, result = _run_mock_attempt(monkeypatch, tmp_path, "stage-r")
    assert len(source.pair_calls) == 7
    assert backend.calls == list(science.stage_r_condition_ids())
    bindings = result["final_manifest"]["evidence_bindings"]
    assert bindings["input_gate_receipt_sha256"] == "f" * 64
    assert bindings["stage_r_freshly_revalidated_conditions"] == 14


@pytest.mark.parametrize("mode", ["primary-pass", "zero-intensity-pass"])
def test_corpus_modes_still_run_full_856_revalidation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    full_revalidations = 0

    def full(source: _Source) -> dict[str, object]:
        nonlocal full_revalidations
        full_revalidations += 1
        return {"H10_exact": 428, "H5_exact": 428, "conditions_exact": 856}

    monkeypatch.setattr(science, "_revalidate_all", full)
    _, backend, _ = _run_mock_attempt(monkeypatch, tmp_path, mode)
    assert full_revalidations == 1
    assert backend.calls == list(canonical_condition_ids())
