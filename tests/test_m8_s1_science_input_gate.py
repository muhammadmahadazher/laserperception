from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import laserperception.evaluation.m8_s1_science as science
from laserperception.detection.m8_s1_input_gate import revalidate_primary_inputs
from laserperception.detection.m8_s1_runtime import canonical_condition_ids


def _canonical_frames() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(identifier.rsplit("/", 1)[0] for identifier in canonical_condition_ids())
    )


class _Source:
    def __init__(
        self,
        frames: tuple[str, ...],
        *,
        shared: dict[str, Any] | None = None,
        corrupt_frame: str | None = None,
        delays: dict[str, float] | None = None,
    ) -> None:
        self.frames = {frame: {} for frame in frames}
        self.pair_calls: list[str] = []
        self.corrupt_frame = corrupt_frame
        self.delays = delays or {}
        self.shared = shared or {
            "lock": threading.Lock(),
            "all_calls": [],
            "active_sources": set(),
            "source_max_active": {},
            "clones": [],
        }

    def __getstate__(
        self,
    ) -> tuple[tuple[str, ...], str | None, dict[str, float]]:
        return tuple(self.frames), self.corrupt_frame, self.delays

    def __setstate__(self, state: tuple[tuple[str, ...], str | None, dict[str, float]]) -> None:
        frames, corrupt_frame, delays = state
        self.__init__(frames, corrupt_frame=corrupt_frame, delays=delays)

    def isolated(self) -> _Source:
        clone = _Source(
            tuple(self.frames),
            shared=self.shared,
            corrupt_frame=self.corrupt_frame,
            delays=self.delays,
        )
        with self.shared["lock"]:
            self.shared["clones"].append(clone)
        return clone

    def pair(self, frame_id: str) -> tuple[tuple[np.ndarray, dict[str, object]], ...]:
        source_id = id(self)
        with self.shared["lock"]:
            if source_id in self.shared["active_sources"]:
                raise AssertionError("one mutable source was used concurrently")
            self.shared["active_sources"].add(source_id)
            self.shared["source_max_active"][source_id] = 1
            self.shared["all_calls"].append(frame_id)
            self.pair_calls.append(frame_id)
        try:
            delay = self.delays.get(frame_id, 0.0)
            if delay:
                time.sleep(delay)
            if frame_id == self.corrupt_frame:
                raise ValueError(f"preflight input identity changed: {frame_id}/H10")
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
        finally:
            with self.shared["lock"]:
                self.shared["active_sources"].remove(source_id)


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


def _install_mock_runtime(
    monkeypatch: pytest.MonkeyPatch,
    source: _Source,
) -> tuple[_Backend, dict[str, int]]:
    backend = _Backend()
    calls = {
        "backend_constructions": 0,
        "gt_loads": 0,
        "gate_completions": 0,
        "pair_calls_before_gt": -1,
        "pair_calls_before_backend": -1,
    }

    real_revalidation = science.revalidate_primary_inputs

    def revalidate(*args: object, **kwargs: object) -> dict[str, object]:
        result = real_revalidation(*args, **kwargs)  # type: ignore[arg-type]
        calls["gate_completions"] += 1
        return result

    def load_gt(path: Path) -> tuple[object, dict[str, object]]:
        calls["gt_loads"] += 1
        calls["pair_calls_before_gt"] = len(source.shared["all_calls"])
        return object(), {}

    def construct(**kwargs: object) -> _Backend:
        calls["backend_constructions"] += 1
        calls["pair_calls_before_backend"] = len(source.shared["all_calls"])
        return backend

    monkeypatch.setattr(science, "verify_input_gate_receipt", lambda *args, **kwargs: "f" * 64)
    monkeypatch.setattr(science, "revalidate_primary_inputs", revalidate)
    monkeypatch.setattr(science.FrozenInputSource, "load", lambda **kwargs: source)
    monkeypatch.setattr(science, "_load_gt", load_gt)
    monkeypatch.setattr(science.DsvtBackend, "from_environment", construct)
    monkeypatch.setattr(science, "AtomicAttempt", _Attempt)
    monkeypatch.setattr(science, "NvidiaSmiSampler", _Sampler)
    monkeypatch.setattr(science, "summarize_gpu_telemetry", lambda samples: {})
    monkeypatch.setattr(science, "_condition_evidence", lambda *args, **kwargs: {"mock": True})
    return backend, calls


def _run_mock_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    *,
    source: _Source | None = None,
    worker_count: int = 1,
) -> tuple[_Source, _Backend, dict[str, object], dict[str, int]]:
    frame_ids = science.STAGE_R_FRAMES if mode == "stage-r" else _canonical_frames()
    source = source or _Source(frame_ids)
    backend, calls = _install_mock_runtime(monkeypatch, source)
    result = science.run_scientific_attempt(
        mode=mode,
        repository_root=tmp_path,
        full_ledger=tmp_path / "ledger.json",
        date_root=tmp_path,
        runtime_commit="a" * 40,
        attempt_root=tmp_path / mode,
        logical_pass_id=f"{mode}-1",
        attempt_id="attempt-1",
        input_gate_receipt=(
            tmp_path / "receipt.json" if mode in {"stage-r", "primary-pass"} else None
        ),
        input_revalidation_workers=worker_count,
    )
    return source, backend, result, calls


def test_stage_r_revalidates_exactly_14_consumed_inputs() -> None:
    source = _Source(science.STAGE_R_FRAMES)
    consumed, evidence = science._revalidate_stage_r(source)
    assert tuple(consumed) == science.stage_r_condition_ids()
    assert len(source.pair_calls) == 7
    assert evidence["conditions_exact"] == 14


def test_full_revalidation_remains_856_for_canonical_passes() -> None:
    source = _Source(_canonical_frames())
    assert science._revalidate_all(source) == {
        "H10_exact": 428,
        "H5_exact": 428,
        "conditions_exact": 856,
    }
    assert len(source.pair_calls) == 428


@pytest.mark.parametrize("mode", ["stage-r", "primary-pass"])
def test_missing_or_invalid_receipt_precedes_backend(
    mode: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    constructed = False

    def construct(**kwargs: object) -> object:
        nonlocal constructed
        constructed = True
        return object()

    monkeypatch.setattr(science.DsvtBackend, "from_environment", construct)
    common = {
        "mode": mode,
        "repository_root": tmp_path,
        "full_ledger": tmp_path / "ledger.json",
        "date_root": tmp_path,
        "runtime_commit": "a" * 40,
        "attempt_root": tmp_path / "attempt",
        "logical_pass_id": f"{mode}-1",
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


def test_parallel_and_serial_revalidation_have_identical_canonical_evidence() -> None:
    frames = _canonical_frames()
    serial = revalidate_primary_inputs(_Source(frames), worker_count=1)
    parallel = revalidate_primary_inputs(
        _Source(frames, delays={frame: 0.0005 for frame in frames[:16]}),
        worker_count=4,
    )
    canonical_fields = (
        "frames_exact",
        "H10_exact",
        "H5_exact",
        "conditions_exact",
        "mismatch_count",
        "condition_order_exact",
        "condition_ids",
        "records",
        "canonical_input_sha256",
    )
    assert {field: serial[field] for field in canonical_fields} == {
        field: parallel[field] for field in canonical_fields
    }
    assert serial["worker_count"] == 1
    assert parallel["worker_count"] == 4
    assert serial["execution_model"] == "serial"
    assert parallel["execution_model"] == "spawned-processes"


def test_worker_scheduling_does_not_change_result_hash() -> None:
    frames = _canonical_frames()
    early_slow = {frame: 0.0005 for frame in frames[:24]}
    late_slow = {frame: 0.0005 for frame in frames[-24:]}
    first = revalidate_primary_inputs(_Source(frames, delays=early_slow), worker_count=4)
    second = revalidate_primary_inputs(_Source(frames, delays=late_slow), worker_count=4)
    assert first["result_sha256"] == second["result_sha256"]
    assert first == second


def test_bounded_workers_use_isolated_mutable_source_state() -> None:
    frames = _canonical_frames()
    source = _Source(frames, delays={frame: 0.0005 for frame in frames[:32]})
    evidence = revalidate_primary_inputs(source, worker_count=4)
    assert evidence["conditions_exact"] == 856
    assert source.shared["all_calls"] == []
    assert len(source.shared["clones"]) == 4
    assert len({id(clone) for clone in source.shared["clones"]}) == 4
    assert source.shared["source_max_active"] == {}
    assert source.pair_calls == []


def test_late_source_corruption_fails_before_gt_backend_and_detector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frames = _canonical_frames()
    source = _Source(frames, corrupt_frame=frames[-1])
    backend, calls = _install_mock_runtime(monkeypatch, source)
    with pytest.raises(ValueError, match="preflight input identity changed"):
        science.run_scientific_attempt(
            mode="primary-pass",
            repository_root=tmp_path,
            full_ledger=tmp_path / "ledger.json",
            date_root=tmp_path,
            runtime_commit="a" * 40,
            attempt_root=tmp_path / "primary-pass",
            logical_pass_id="primary-pass-1",
            attempt_id="attempt-1",
            input_gate_receipt=tmp_path / "receipt.json",
            input_revalidation_workers=4,
        )
    assert calls["backend_constructions"] == 0
    assert calls["gt_loads"] == 0
    assert calls["gate_completions"] == 0
    assert backend.calls == []


def test_valid_receipt_reaches_stage_r_without_full_corpus_revalidation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        science,
        "_revalidate_all",
        lambda source: (_ for _ in ()).throw(AssertionError("full replay called")),
    )
    source, backend, result, calls = _run_mock_attempt(monkeypatch, tmp_path, "stage-r")
    assert len(source.pair_calls) == 7
    assert backend.calls == list(science.stage_r_condition_ids())
    assert calls["backend_constructions"] == 1
    assert calls["gt_loads"] == 1
    assert calls["gate_completions"] == 0
    bindings = result["final_manifest"]["evidence_bindings"]
    assert bindings["input_gate_receipt_sha256"] == "f" * 64
    assert bindings["stage_r_freshly_revalidated_conditions"] == 14


@pytest.mark.parametrize("worker_count", [1, 4])
def test_primary_completes_full_gate_then_reuses_one_pair_per_inference_frame(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, worker_count: int
) -> None:
    source, backend, result, calls = _run_mock_attempt(
        monkeypatch,
        tmp_path,
        "primary-pass",
        worker_count=worker_count,
    )
    expected_conditions = list(canonical_condition_ids())
    expected_frames = list(_canonical_frames())
    expected_total_calls = 856 if worker_count == 1 else 428
    assert len(source.shared["all_calls"]) == expected_total_calls
    assert source.shared["all_calls"][-428:] == expected_frames
    expected_source_calls = expected_frames * 2 if worker_count == 1 else expected_frames
    assert source.pair_calls == expected_source_calls
    assert backend.calls == expected_conditions
    assert calls["backend_constructions"] == 1
    assert calls["gt_loads"] == 1
    assert calls["gate_completions"] == 1
    expected_parent_gate_calls = 428 if worker_count == 1 else 0
    assert calls["pair_calls_before_gt"] == expected_parent_gate_calls
    assert calls["pair_calls_before_backend"] == expected_parent_gate_calls
    bindings = result["final_manifest"]["evidence_bindings"]
    assert bindings["input_gate_receipt_sha256"] == "f" * 64
    assert bindings["primary_preinference_revalidation_workers"] == worker_count
    assert bindings["primary_preinference_revalidated_conditions"] == 856
    assert bindings["primary_fresh_pair_reconstructions"] == 428
    assert bindings["primary_freshly_verified_conditions"] == 856
    assert len(bindings["primary_preinference_revalidation_sha256"]) == 64
    assert len(bindings["primary_consumed_input_verification_sha256"]) == 64


def test_primary_revalidation_evidence_is_complete_and_deterministic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first_root = tmp_path / "first"
    _, _, _, _ = _run_mock_attempt(monkeypatch, first_root, "primary-pass", worker_count=4)
    first_path = first_root / "primary-pass/primary_preinference_revalidation.json"
    first = first_path.read_bytes()

    second_root = tmp_path / "second"
    _, _, _, _ = _run_mock_attempt(monkeypatch, second_root, "primary-pass", worker_count=4)
    second = (second_root / "primary-pass/primary_preinference_revalidation.json").read_bytes()
    assert first == second
    payload = json.loads(first)
    assert payload["worker_count"] == 4
    assert payload["execution_model"] == "spawned-processes"
    assert payload["frames_exact"] == 428
    assert payload["conditions_exact"] == 856
    assert payload["H10_exact"] == payload["H5_exact"] == 428
    assert payload["mismatch_count"] == 0
    assert payload["condition_order_exact"] is True
    assert payload["condition_ids"] == list(canonical_condition_ids())
    assert len(payload["records"]) == 856
    identity = payload.pop("result_sha256")
    assert science.canonical_json_sha256(payload) == identity


def test_primary_consumed_input_evidence_is_complete_and_deterministic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first_root = tmp_path / "first"
    source, _, _, _ = _run_mock_attempt(monkeypatch, first_root, "primary-pass")
    first = (first_root / "primary-pass/primary_consumed_input_verification.json").read_bytes()
    assert len(source.pair_calls) == 856

    second_root = tmp_path / "second"
    _, _, _, _ = _run_mock_attempt(monkeypatch, second_root, "primary-pass")
    second = (second_root / "primary-pass/primary_consumed_input_verification.json").read_bytes()
    assert first == second
    payload = json.loads(first)
    assert payload["pair_reconstructions_exact"] == 428
    assert payload["conditions_exact"] == 856
    assert payload["H10_exact"] == payload["H5_exact"] == 428
    assert payload["condition_order_exact"] is True
    assert payload["condition_ids"] == list(canonical_condition_ids())
    assert len(payload["records"]) == 856
    identity = payload.pop("result_sha256")
    assert science.canonical_json_sha256(payload) == identity


def test_zero_intensity_retains_full_revalidation_and_existing_pair_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    full_revalidations = 0

    def full(source: _Source) -> dict[str, object]:
        nonlocal full_revalidations
        full_revalidations += 1
        return {"H10_exact": 428, "H5_exact": 428, "conditions_exact": 856}

    monkeypatch.setattr(science, "_revalidate_all", full)
    source, backend, _, calls = _run_mock_attempt(monkeypatch, tmp_path, "zero-intensity-pass")
    assert full_revalidations == 1
    assert len(source.pair_calls) == 856
    assert backend.calls == list(canonical_condition_ids())
    assert calls["backend_constructions"] == 1
    assert calls["gt_loads"] == 1
