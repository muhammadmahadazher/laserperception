"""Future-authorized S2 scientific process; never import before CLI authorization.

This module deliberately imports the detector and real KITTI evaluator. Local
CPU validation must test the authorization boundary without importing it.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from laserperception.detection.m8_backend import DsvtBackend
from laserperception.detection.m8_s1_runtime import CANDIDATE_MANIFEST_PATH, atomic_write_json
from laserperception.detection.m8_s2_reconstruction import verified_inputs
from laserperception.detection.m8_s2_runtime import (
    AtomicAttempt,
    AttemptIdentity,
    M8S2ProtocolViolation,
    condition_ids,
)
from laserperception.detection.measurement_telemetry import (
    NvidiaSmiSampler,
    summarize_gpu_telemetry,
)
from laserperception.evaluation.m8_s1_science import _condition_evidence, _eligible_poses, _load_gt


def run_scientific_attempt(
    *,
    mode: str,
    repository_root: Path,
    date_root: Path,
    m6_ledger: Path,
    execution_commit: str,
    runtime_policy_sha256: str,
    input_gate_receipt_sha256: str,
    attempt_root: Path,
    logical_pass_id: str,
    attempt_id: str,
) -> dict[str, object]:
    """Run one fresh future-authorized process with no canonical partial resume."""

    identity = AttemptIdentity(
        mode=mode,
        logical_pass_id=logical_pass_id,
        attempt_id=attempt_id,
        process_uuid=str(uuid.uuid4()),
        process_id=os.getpid(),
        execution_commit=execution_commit,
        runtime_policy_sha256=runtime_policy_sha256,
        input_gate_receipt_sha256=input_gate_receipt_sha256,
    )
    attempt = AtomicAttempt(attempt_root, identity)
    sampler: NvidiaSmiSampler | None = None
    condition_active = False
    try:
        camera, poses_by_frame = _load_gt(date_root)
        backend = DsvtBackend.from_environment(
            manifest_path=repository_root / CANDIDATE_MANIFEST_PATH
        )
        atomic_write_json(attempt_root / "runtime_state.json", backend.runtime_state())
        sampler = NvidiaSmiSampler(interval_seconds=1.0)
        sampler.start()
        sampler.begin_block("s2_scientific_attempt")
        observed: list[str] = []
        for condition_id, points, consumed in verified_inputs(
            mode=mode,
            repository_root=repository_root,
            date_root=date_root,
            m6_ledger=m6_ledger,
        ):
            # No second reconstruction is performed. This is the verified array.
            condition_active = True
            frame = backend.infer(points, sample_id=condition_id)
            frame_id, arm = condition_id.rsplit("/", 1)
            poses = _eligible_poses(poses_by_frame.get(frame_id, ()), camera)
            payload = _condition_evidence(
                frame,
                frame_id=frame_id,
                history=arm,
                input_sha256=str(consumed["input_sha256"]),
                poses=poses,
                camera=camera,
            )
            payload["arm"] = payload.pop("history")
            payload.update(consumed)
            payload["condition_id"] = condition_id
            attempt.record(condition_id, payload)
            condition_active = False
            observed.append(condition_id)
        if tuple(observed) != condition_ids(mode):
            raise M8S2ProtocolViolation("S2 consumed condition order is incomplete")
        sampler.end_block("s2_scientific_attempt")
        sampler.stop()
        atomic_write_json(attempt_root / "telemetry.json", summarize_gpu_telemetry(sampler.samples))
        sampler = None
        return attempt.finalize()
    except Exception as error:
        if sampler is not None:
            sampler.stop()
        attempt.fail(f"{type(error).__name__}: {error}", count_failed_call=condition_active)
        raise
