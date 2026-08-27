"""Integrated future M7 corpus runner with ledger-bound detector inputs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from benchmarks.m7.detector import (
    build_canonical_m7_detector,
    require_detector_runtime_identity,
)
from benchmarks.m7.evidence import StrictInputLedger, load_strict_input_ledger
from benchmarks.m7.execution import (
    CheckpointIdentity,
    DetectorObservation,
    ExecutionIdentity,
    M7CheckpointStore,
    ObservationHashes,
    RuntimeArtifacts,
    load_inference_authorization,
    repeatability_condition,
    require_actual_input,
)
from benchmarks.m7.prepare_inputs import (
    CanonicalM7SourceAdapter,
    GeneratedCondition,
    generate_canonical_frame,
)
from benchmarks.m7.protocol import (
    M6B_RESULT_FULL_BYTES,
    M6B_RESULT_FULL_SHA256,
    REPEATABILITY_REPETITIONS,
    SENTINEL_FRAMES,
    Arm,
    ProtocolViolation,
    canonical_condition_ids,
)
from benchmarks.m7.provenance import atomic_write_json, verify_external_asset


class M7Detector(Protocol):
    """Fixed inference interface; the runner passes the exact already-hashed array."""

    def infer(self, points: np.ndarray, *, condition_id: str) -> DetectorObservation:
        """Run the unchanged frozen detector on the provided exact model-ready input."""


BOUND_RECORD_FIELDS = (
    "condition_id",
    "drive_id",
    "frame_index",
    "arm",
    "generation_commit",
    "source_a_sha256",
    "source_e_sha256",
    "point_count",
    "xyz_sha256",
    "model_ready_sha256",
    "selected_row_sha256",
    "lag_bit_patterns",
    "lag_support_count",
    "lag_span_seconds",
    "sweep_ids",
    "per_sweep_point_counts",
    "provenance_schema",
    "rank_source_identities",
    "rank_to_lag_bit_pattern",
    "pillar_structure",
    "lag_scale_provenance",
    "quota_provenance",
    "seed_provenance",
    "f_history_ranks",
)


@dataclass(frozen=True, slots=True)
class CorpusRunSummary:
    """Bounded run bookkeeping; scientific aggregation remains separately frozen."""

    condition_count: int
    checkpoint_reused_count: int
    inference_call_count: int
    repeatability_call_count: int

    def to_dict(self) -> dict[str, int | str]:
        """Return a canonical summary record."""

        return {
            "schema_version": "laserperception.m7.corpus-run-summary.v1",
            "condition_count": self.condition_count,
            "checkpoint_reused_count": self.checkpoint_reused_count,
            "inference_call_count": self.inference_call_count,
            "repeatability_call_count": self.repeatability_call_count,
        }


class M7CorpusRunner:
    """Own canonical order, regeneration, ledger binding, inference, repeatability, and resume."""

    def __init__(
        self,
        *,
        ledger: StrictInputLedger,
        source_adapter: CanonicalM7SourceAdapter,
        detector: M7Detector,
        checkpoint_store: M7CheckpointStore,
        implementation_commit: str,
    ) -> None:
        if checkpoint_store.condition_ids != canonical_condition_ids():
            raise ProtocolViolation("canonical M7 corpus runner cannot use a shortened corpus")
        self.ledger = ledger
        self.source_adapter = source_adapter
        self.detector = detector
        self.checkpoint_store = checkpoint_store
        self.implementation_commit = implementation_commit
        self.inference_call_count = 0
        self.repeatability_call_count = 0
        self.checkpoint_reused_count = 0

    @staticmethod
    def _frame_and_arm(condition: str) -> tuple[str, Arm]:
        frame_id, arm_text = condition.split("|", 1)
        try:
            return frame_id, Arm(arm_text)
        except ValueError as error:
            raise ProtocolViolation(f"invalid canonical M7 condition: {condition}") from error

    @staticmethod
    def _bind_record(
        generated: GeneratedCondition,
        authorized: Mapping[str, object],
    ) -> str:
        for field in BOUND_RECORD_FIELDS:
            if generated.record.get(field) != authorized.get(field):
                raise ProtocolViolation(
                    f"M7 regenerated condition differs from authorized ledger field: {field}"
                )
        expected_sha = authorized.get("model_ready_sha256")
        if not isinstance(expected_sha, str):
            raise ProtocolViolation("authorized M7 condition lacks model_ready_sha256")
        return require_actual_input(generated.points, expected_sha)

    def _generated_condition(self, condition: str) -> GeneratedCondition:
        frame_id, arm = self._frame_and_arm(condition)
        generated = generate_canonical_frame(
            self.source_adapter,
            frame_id,
            implementation_commit=self.implementation_commit,
        ).condition(arm)
        authorized = self.ledger.condition(condition)
        self._bind_record(generated, authorized)
        return generated

    def _invoke(
        self,
        condition: str,
        generated: GeneratedCondition,
        *,
        repeatability: bool,
    ) -> DetectorObservation:
        authorized = self.ledger.condition(condition)
        self._bind_record(generated, authorized)
        observation = self.detector.infer(generated.points, condition_id=condition)
        self.inference_call_count += 1
        if repeatability:
            self.repeatability_call_count += 1
        return observation

    def _run_repeatability(self) -> None:
        for frame_id in SENTINEL_FRAMES:
            for arm in (Arm.B, Arm.C, Arm.D, Arm.F):
                condition = f"{frame_id}|{arm.value}"
                expected_sha = self.ledger.condition(condition)["model_ready_sha256"]
                assert isinstance(expected_sha, str)
                complete = self.checkpoint_store.load_complete(
                    condition,
                    expected_input_sha256=expected_sha,
                )
                if complete is not None:
                    payload = complete.get("payload")
                    if (
                        not isinstance(payload, Mapping)
                        or payload.get("repeatability_repetitions") != REPEATABILITY_REPETITIONS
                    ):
                        raise ProtocolViolation(
                            f"sentinel checkpoint lacks exact repeatability proof: {condition}"
                        )
                    self.checkpoint_reused_count += 1
                    continue
                generated = self._generated_condition(condition)
                first: DetectorObservation | None = None

                def execute(
                    condition_id: str = condition,
                    condition_input: GeneratedCondition = generated,
                ) -> ObservationHashes:
                    nonlocal first
                    observation = self._invoke(
                        condition_id,
                        condition_input,
                        repeatability=True,
                    )
                    if first is None:
                        first = observation
                    return observation.hashes()

                hashes = repeatability_condition(execute)
                assert first is not None
                self.checkpoint_store.save_complete(
                    condition,
                    input_sha256=expected_sha,
                    hashes=hashes,
                    payload={
                        **dict(first.payload),
                        "repeatability_repetitions": REPEATABILITY_REPETITIONS,
                        "canonical_repeat": 1,
                    },
                )

    def run(self) -> CorpusRunSummary:
        """Run exact sentinel repeats then every remaining condition in canonical order."""

        self._run_repeatability()
        for condition in canonical_condition_ids():
            expected_sha = self.ledger.condition(condition)["model_ready_sha256"]
            assert isinstance(expected_sha, str)
            complete = self.checkpoint_store.load_complete(
                condition,
                expected_input_sha256=expected_sha,
            )
            if complete is not None:
                self.checkpoint_reused_count += 1
                continue
            generated = self._generated_condition(condition)
            observation = self._invoke(condition, generated, repeatability=False)
            self.checkpoint_store.save_complete(
                condition,
                input_sha256=expected_sha,
                hashes=observation.hashes(),
                payload=observation.payload,
            )
        return CorpusRunSummary(
            condition_count=len(canonical_condition_ids()),
            checkpoint_reused_count=self.checkpoint_reused_count,
            inference_call_count=self.inference_call_count,
            repeatability_call_count=self.repeatability_call_count,
        )


def run_measurement(
    authorization_path: str | Path,
    expected: ExecutionIdentity,
    artifacts: RuntimeArtifacts,
    *,
    dataset_root: str | Path,
    m6b_input_asset: str | Path,
    m6b_result_asset: str | Path,
    checkpoint_root: str | Path,
    output_root: str | Path,
) -> CorpusRunSummary:
    """Run the authorized corpus using only the canonical artifact-bound M6b detector."""

    return _run_measurement_internal(
        authorization_path,
        expected,
        artifacts,
        dataset_root=dataset_root,
        m6b_input_asset=m6b_input_asset,
        m6b_result_asset=m6b_result_asset,
        checkpoint_root=checkpoint_root,
        output_root=output_root,
        _detector_builder=build_canonical_m7_detector,
    )


def _run_measurement_internal(
    authorization_path: str | Path,
    expected: ExecutionIdentity,
    artifacts: RuntimeArtifacts,
    *,
    dataset_root: str | Path,
    m6b_input_asset: str | Path,
    m6b_result_asset: str | Path,
    checkpoint_root: str | Path,
    output_root: str | Path,
    _detector_builder: Callable[[RuntimeArtifacts, ExecutionIdentity], M7Detector],
) -> CorpusRunSummary:
    """Private CPU-test seam; public execution always supplies the canonical builder."""

    load_inference_authorization(authorization_path, expected)
    artifacts.verify_input_ledger(expected)
    ledger = load_strict_input_ledger(
        artifacts.input_ledger,
        expected_implementation_commit=expected.implementation_commit,
    )
    artifacts.verify_runtime(expected)
    verify_external_asset(
        m6b_result_asset,
        expected_bytes=M6B_RESULT_FULL_BYTES,
        expected_sha256=M6B_RESULT_FULL_SHA256,
    )
    source_adapter = CanonicalM7SourceAdapter(dataset_root, m6b_input_asset)
    detector = _detector_builder(artifacts, expected)
    require_detector_runtime_identity(detector, artifacts, expected)
    checkpoint_store = M7CheckpointStore(
        checkpoint_root,
        CheckpointIdentity(
            implementation_commit=expected.implementation_commit,
            input_ledger_sha256=expected.input_ledger_sha256,
            engine_sha256=expected.engine_sha256,
            checkpoint_sha256=expected.checkpoint_sha256,
            onnx_sha256=expected.onnx_sha256,
            evaluator_identity=expected.evaluator_identity,
            protocol_commit=expected.protocol_commit,
        ),
    )
    runner = M7CorpusRunner(
        ledger=ledger,
        source_adapter=source_adapter,
        detector=detector,
        checkpoint_store=checkpoint_store,
        implementation_commit=expected.implementation_commit,
    )
    summary = runner.run()
    atomic_write_json(Path(output_root) / "run_summary.json", summary.to_dict())
    return summary
