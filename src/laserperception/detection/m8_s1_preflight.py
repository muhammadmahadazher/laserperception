"""GT-blind M8 S1 runtime-sizing helpers.

The import graph is intentionally limited to M8 input reconstruction,
candidate execution, input-only capacity metadata, telemetry, and timing. It
must never import KITTI tracklets or ``laserperception.evaluation``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import numpy as np

from laserperception.datasets.kitti_raw import KittiRawSequence
from laserperception.detection.m8_backend import DsvtBackend
from laserperception.detection.m8_input import M8MultiSweepBuilder
from laserperception.detection.m8_s1_runtime import (
    CANDIDATE_MANIFEST_PATH,
    INPUT_LEDGER_PATH,
    INPUT_LEDGER_SHA256,
    PROTOCOL_FREEZE_COMMIT,
    PROTOCOL_JSON_SHA256,
    STAGE_R_FRAMES,
    atomic_write_json,
    canonical_frame_ids,
    sha256_file,
    verify_static_bindings,
)
from laserperception.detection.measurement_telemetry import (
    NvidiaSmiSampler,
    summarize_gpu_telemetry,
    summarize_telemetry_by_block,
)
from laserperception.detection.multisweep import (
    HistoricalSweep,
    MultiSweepBuilderConfig,
    SweepTransform,
)

PREFLIGHT_QUANTILES = (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)


def _load_mapping(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def select_preflight_frames(census: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    """Select ten non-sentinel frames by input-only nearest ordered ranks."""

    untyped = census.get("records")
    if not isinstance(untyped, list):
        raise ValueError("M8 H10 census records are missing")
    canonical = canonical_frame_ids()
    ordinal = {frame_id: index for index, frame_id in enumerate(canonical)}
    candidates: list[dict[str, object]] = []
    for record in untyped:
        if not isinstance(record, Mapping):
            raise ValueError("M8 H10 census record is malformed")
        condition_id = record.get("condition_id")
        pillars = record.get("candidate_dynamic_pillars")
        if not isinstance(condition_id, str) or not condition_id.endswith("/H10"):
            raise ValueError("M8 H10 census condition identity is malformed")
        frame_id = condition_id.removesuffix("/H10")
        if frame_id not in ordinal or isinstance(pillars, bool) or not isinstance(pillars, int):
            raise ValueError("M8 H10 census frame/count is malformed")
        if frame_id not in STAGE_R_FRAMES:
            candidates.append(
                {
                    "frame_id": frame_id,
                    "frozen_corpus_ordinal_1_based": ordinal[frame_id] + 1,
                    "H10_candidate_dynamic_pillars": pillars,
                }
            )
    if len(candidates) != 421:
        raise ValueError("preflight candidates must exclude exactly seven sentinels")
    ranked = sorted(
        candidates,
        key=lambda value: (
            cast(int, value["H10_candidate_dynamic_pillars"]),
            cast(int, value["frozen_corpus_ordinal_1_based"]),
        ),
    )
    selected = []
    for quantile in PREFLIGHT_QUANTILES:
        rank = int(np.floor(quantile * (len(ranked) - 1) + 0.5))
        selected.append({**ranked[rank], "target_quantile": quantile, "ordered_rank_0_based": rank})
    if len({item["frame_id"] for item in selected}) != 10:
        raise ValueError("input-only preflight quantiles did not select ten unique frames")
    return tuple(selected)


def select_warmup_frame(selected: Sequence[Mapping[str, object]]) -> str:
    """Choose the earliest non-sentinel corpus frame outside the measured set."""

    excluded = {*STAGE_R_FRAMES, *(str(item["frame_id"]) for item in selected)}
    return next(frame_id for frame_id in canonical_frame_ids() if frame_id not in excluded)


def _historical_sweeps(
    sequence: KittiRawSequence,
    current_index: int,
    records: Sequence[Mapping[str, object]],
) -> tuple[HistoricalSweep, ...]:
    current = sequence.frame(current_index).to_raw_sweep()
    expected = tuple(range(current_index - 1, max(-1, current_index - 11), -1))
    if len(records) != len(expected):
        raise ValueError("frozen transform count does not match available history")
    result = []
    for expected_index, record in zip(expected, records, strict=True):
        if record.get("source_index") != expected_index:
            raise ValueError("frozen transform order mismatch")
        source = sequence.frame(expected_index).to_raw_sweep()
        matrix = np.asarray(record.get("lidar2sensor"), dtype=np.float32)
        if _sha256_array(matrix) != record.get("lidar2sensor_sha256"):
            raise ValueError("frozen transform identity mismatch")
        result.append(
            HistoricalSweep(source, SweepTransform(matrix, source.source_id, current.source_id))
        )
    return tuple(result)


@dataclass(slots=True)
class FrozenInputSource:
    """Reconstruct only requested inputs while verifying the accepted ledger."""

    date_root: Path
    frames: Mapping[str, Mapping[str, object]]
    accepted: Mapping[str, Mapping[str, object]]
    sequences: dict[str, KittiRawSequence]

    @classmethod
    def load(
        cls, *, date_root: Path, full_ledger: Path, accepted_ledger: Path
    ) -> FrozenInputSource:
        """Load frozen identities without loading GT, tracklets, or predictions."""

        if sha256_file(accepted_ledger) != INPUT_LEDGER_SHA256:
            raise ValueError("accepted M8 input ledger identity changed")
        source_payload = _load_mapping(full_ledger)
        accepted_payload = _load_mapping(accepted_ledger)
        source_frames = source_payload.get("frames")
        accepted_records = accepted_payload.get("records")
        if not isinstance(source_frames, list) or not isinstance(accepted_records, list):
            raise ValueError("frozen M6b/M8 ledger schema changed")
        frames = {
            str(frame["frame_id"]): frame
            for frame in source_frames
            if isinstance(frame, Mapping) and isinstance(frame.get("frame_id"), str)
        }
        accepted = {
            str(record["condition_id"]): record
            for record in accepted_records
            if isinstance(record, Mapping) and isinstance(record.get("condition_id"), str)
        }
        if tuple(frames) != canonical_frame_ids() or len(accepted) != 856:
            raise ValueError("frozen M8 input order or count changed")
        return cls(date_root, frames, accepted, {})

    def pair(self, frame_id: str) -> tuple[tuple[np.ndarray, dict[str, object]], ...]:
        """Reconstruct and hash one verified H10/H5 pair."""

        frame = self.frames.get(frame_id)
        if frame is None:
            raise ValueError(f"frame is outside the frozen corpus: {frame_id}")
        raw_index = frame.get("frame_index")
        transforms = frame.get("frozen_sweep_transforms")
        if (
            isinstance(raw_index, bool)
            or not isinstance(raw_index, int)
            or not isinstance(transforms, list)
        ):
            raise ValueError("frozen source frame is malformed")
        drive_id = frame_id.split("/", 1)[0]
        if drive_id not in self.sequences:
            self.sequences[drive_id] = KittiRawSequence(
                self.date_root, self.date_root / f"{drive_id}_sync"
            )
        sequence = self.sequences[drive_id]
        current = sequence.frame(raw_index).to_raw_sweep()
        historical = _historical_sweeps(sequence, raw_index, transforms)
        result = []
        for history, depth in (("H10", 10), ("H5", 5)):
            points = (
                M8MultiSweepBuilder(MultiSweepBuilderConfig(max_historical_sweeps=depth))
                .build(current, historical)
                .points
            )
            condition_id = f"{frame_id}/{history}"
            frozen = self.accepted.get(condition_id)
            identity = _sha256_array(points)
            if frozen is None or frozen.get("full_M8_XYZIT_sha256") != identity:
                raise ValueError(f"preflight input identity changed: {condition_id}")
            if frozen.get("point_count") != points.shape[0]:
                raise ValueError(f"preflight input count changed: {condition_id}")
            result.append(
                (
                    points,
                    {
                        "condition_id": condition_id,
                        "history": history,
                        "input_point_count": int(points.shape[0]),
                        "input_sha256": identity,
                    },
                )
            )
        return tuple(result)


def _host_memory() -> dict[str, object]:
    memory: dict[str, int] = {}
    memory_path = Path("/proc/meminfo")
    if memory_path.is_file():
        for line in memory_path.read_text(encoding="utf-8").splitlines():
            name, _, raw = line.partition(":")
            if name in {"MemTotal", "MemAvailable"}:
                memory[name] = int(raw.strip().split()[0]) * 1024
    status: dict[str, int] = {}
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            name, _, raw = line.partition(":")
            if name in {"VmHWM", "VmRSS"}:
                status[name] = int(raw.strip().split()[0]) * 1024
    return {
        "total_bytes": memory.get("MemTotal"),
        "available_bytes_at_capture": memory.get("MemAvailable"),
        "process_max_rss_bytes": status.get("VmHWM", status.get("VmRSS")),
    }


def run_sizing_worker(
    *,
    repository_root: Path,
    full_ledger: Path,
    date_root: Path,
    census_path: Path,
    runtime_commit: str,
    worker_index: int,
    output: Path,
    telemetry_interval_seconds: float = 0.25,
) -> dict[str, object]:
    """Run one fresh 2-warmup/20-measured GT-blind sizing process."""

    manifest_path = repository_root / CANDIDATE_MANIFEST_PATH
    manifest = _load_mapping(manifest_path)
    environment = manifest.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError("candidate environment binding is malformed")
    upstream_name = environment.get("upstream_root_variable")
    checkpoint_name = environment.get("checkpoint_variable")
    if not isinstance(upstream_name, str) or not isinstance(checkpoint_name, str):
        raise ValueError("candidate environment variable names are malformed")
    upstream_root = os.environ.get(upstream_name)
    checkpoint_path = os.environ.get(checkpoint_name)
    if not upstream_root or not checkpoint_path:
        raise RuntimeError(f"set {upstream_name} and {checkpoint_name} before preflight")
    binding = verify_static_bindings(
        repository_root,
        upstream_root=upstream_root,
        checkpoint_path=checkpoint_path,
    )
    if binding.repository_head != runtime_commit:
        raise RuntimeError("preflight process HEAD differs from the bound runtime commit")
    selected = select_preflight_frames(_load_mapping(census_path))
    warmup_frame = select_warmup_frame(selected)
    source = FrozenInputSource.load(
        date_root=date_root,
        full_ledger=full_ledger,
        accepted_ledger=repository_root / INPUT_LEDGER_PATH,
    )

    initialization_start = time.monotonic_ns()
    backend = DsvtBackend.from_environment(manifest_path=manifest_path)
    backend.synchronize()
    initialization_seconds = (time.monotonic_ns() - initialization_start) / 1_000_000_000
    runtime_state = backend.runtime_state()
    sampler = NvidiaSmiSampler(interval_seconds=telemetry_interval_seconds)
    sampler.start()
    calls: list[dict[str, object]] = []
    try:
        sampler.begin_block("warmup")
        for points, _ in source.pair(warmup_frame):
            backend.run_gt_blind_timing_call(points)
        backend.synchronize()
        sampler.end_block("warmup")

        sampler.begin_block("measured")
        for selection in selected:
            frame_id = str(selection["frame_id"])
            pair_calls = []
            for points, identity in source.pair(frame_id):
                pillar_count = backend.candidate_pillar_count(points)
                backend.synchronize()
                start = time.monotonic_ns()
                backend.run_gt_blind_timing_call(points)
                backend.synchronize()
                elapsed = (time.monotonic_ns() - start) / 1_000_000_000
                pair_calls.append(
                    {
                        **identity,
                        "candidate_dynamic_pillars": pillar_count,
                        "wall_seconds": elapsed,
                        "process_id": os.getpid(),
                    }
                )
            calls.extend(pair_calls)
        sampler.end_block("measured")
    finally:
        sampler.stop()
    if len(calls) != 20:
        raise RuntimeError("GT-blind sizing worker did not complete exactly 20 measured calls")
    result = {
        "schema_version": "laserperception.m8.s1.preflight-worker.v1",
        "status": "COMPLETE",
        "worker_index": worker_index,
        "process_uuid": str(uuid.uuid4()),
        "process_id": os.getpid(),
        "started_in_fresh_python_process": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_commit": runtime_commit,
        "python": platform.python_version(),
        "initialization_seconds": initialization_seconds,
        "runtime_state": runtime_state,
        "warmup": {
            "frame_id": warmup_frame,
            "condition_order": [f"{warmup_frame}/H10", f"{warmup_frame}/H5"],
            "call_count": 2,
            "excluded_from_measurement": True,
        },
        "measured_calls": calls,
        "measured_call_count": 20,
        "ground_truth_loaded": False,
        "evaluator_loaded": False,
        "semantic_output_retained": False,
        "prediction_count_observed_or_stored": False,
        "telemetry": {
            "summary": summarize_gpu_telemetry(sampler.samples),
            "by_block": summarize_telemetry_by_block(sampler.samples),
        },
        "host_memory": _host_memory(),
        "resource_after": backend.runtime_state()["cuda_memory"],
    }
    atomic_write_json(output, result)
    return result


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("preflight timing summary cannot be empty")
    return {
        "count": len(values),
        "minimum_seconds": min(values),
        "median_seconds": statistics.median(values),
        "maximum_seconds": max(values),
    }


def _duration(central: float, lower: float, upper: float) -> dict[str, object]:
    return {
        "central_seconds": central,
        "conservative_observed_rate_envelope_seconds": [lower, upper],
    }


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("preflight numeric field is malformed")
    return float(value)


def combine_sizing_workers(
    workers: Sequence[Mapping[str, object]],
    *,
    runtime_commit: str,
    census_path: Path,
) -> dict[str, object]:
    """Combine exactly two workers without retaining detector semantics."""

    if len(workers) != 2 or len({worker.get("process_uuid") for worker in workers}) != 2:
        raise ValueError("preflight requires exactly two fresh process identities")
    if len({worker.get("process_id") for worker in workers}) != 2:
        raise ValueError("preflight workers did not use distinct operating-system processes")
    if any(
        worker.get("ground_truth_loaded") is not False
        or worker.get("evaluator_loaded") is not False
        or worker.get("semantic_output_retained") is not False
        for worker in workers
    ):
        raise ValueError("preflight worker crossed a GT-blind boundary")
    calls = [
        cast(Mapping[str, object], call)
        for worker in workers
        for call in cast(Sequence[object], worker["measured_calls"])
    ]
    if len(calls) != 40:
        raise ValueError("preflight must contain exactly 40 measured calls")
    h10 = [_number(call["wall_seconds"]) for call in calls if call["history"] == "H10"]
    h5 = [_number(call["wall_seconds"]) for call in calls if call["history"] == "H5"]
    pairs = [h10[index] + h5[index] for index in range(20)]
    initialization = [_number(worker["initialization_seconds"]) for worker in workers]
    pair_summary = _summary(pairs)
    init_summary = _summary(initialization)
    pair_central = _number(pair_summary["median_seconds"])
    pair_low = _number(pair_summary["minimum_seconds"])
    pair_high = _number(pair_summary["maximum_seconds"])
    init_central = _number(init_summary["median_seconds"])
    init_low = _number(init_summary["minimum_seconds"])
    init_high = _number(init_summary["maximum_seconds"])

    def estimate(pair_count: int, process_count: int) -> dict[str, object]:
        return _duration(
            process_count * init_central + pair_count * pair_central,
            process_count * init_low + pair_count * pair_low,
            process_count * init_high + pair_count * pair_high,
        )

    one_pass = estimate(428, 1)
    stage_r = estimate(70, 10)
    three_passes = estimate(1_284, 3)
    full = estimate(2_638, 16)
    cuda_ratios = []
    host_ratios = []
    for worker in workers:
        cuda = cast(Mapping[str, object], worker["resource_after"])
        cuda_ratios.append(
            _number(cuda["max_reserved_bytes"]) / _number(cuda["device_total_bytes"])
        )
        host = cast(Mapping[str, object], worker["host_memory"])
        host_ratios.append(_number(host["process_max_rss_bytes"]) / _number(host["total_bytes"]))
    comfortable = max(cuda_ratios) < 0.90 and max(host_ratios) < 0.90
    selection = select_preflight_frames(_load_mapping(census_path))
    return {
        "schema_version": "laserperception.m8.s1.runtime-sizing.v1",
        "status": "M8_P1_S1_GT_BLIND_RUNTIME_SIZING_COMPLETE",
        "tier": "engineering_preflight",
        "protocol_freeze_commit": PROTOCOL_FREEZE_COMMIT,
        "protocol_json_sha256": PROTOCOL_JSON_SHA256,
        "runtime_implementation_head": runtime_commit,
        "input_selection": {
            "source": "selected-candidate H10 input-only dynamic-pillar census",
            "census_sha256": sha256_file(census_path),
            "target_quantiles": list(PREFLIGHT_QUANTILES),
            "nearest_rank_rule": "round(q * (N - 1)); ties by earliest frozen corpus order",
            "sentinels_excluded": list(STAGE_R_FRAMES),
            "selected_frames": list(selection),
            "selected_before_model_output": True,
        },
        "workers": [dict(worker) for worker in workers],
        "process_count": 2,
        "warmup_call_count": 4,
        "measured_call_count": 40,
        "scientific_campaign_call_count": 0,
        "ground_truth_loaded": False,
        "evaluator_loaded": False,
        "semantic_output_retained": False,
        "prediction_count_observed_or_stored": False,
        "timing_method": (
            "CUDA synchronize, monotonic wall timer, candidate inference, CUDA synchronize"
        ),
        "timing": {
            "initialization": init_summary,
            "H10": _summary(h10),
            "H5": _summary(h5),
            "H10_H5_pair": pair_summary,
        },
        "engineering_duration_estimates": {
            "method": (
                "observed median/min/max H10+H5 pair rate plus observed model initialization "
                "per required fresh process; stratified engineering sizing, not confidence bounds"
            ),
            "one_856_condition_pass": one_pass,
            "stage_r_10_processes_140_calls": stage_r,
            "primary_3_passes": three_passes,
            "zero_intensity_3_passes": three_passes,
            "full_5276_accepted_call_campaign": full,
        },
        "resources": {
            "maximum_cuda_reserved_fraction": max(cuda_ratios),
            "maximum_process_rss_fraction_of_host_total": max(host_ratios),
            "capacity_assessment_rule": (
                "engineering preflight classification requires both observed peak fractions "
                "to remain below 0.90; this is not a scientific acceptance gate"
            ),
            "comfortably_below_available_capacity": comfortable,
            "workers": [
                {
                    "process_uuid": worker["process_uuid"],
                    "initialization_cuda_memory": cast(
                        Mapping[str, object], worker["runtime_state"]
                    )["cuda_memory"],
                    "post_sizing_cuda_memory": worker["resource_after"],
                    "host_memory": worker["host_memory"],
                }
                for worker in workers
            ],
        },
        "telemetry": [
            {
                "process_uuid": worker["process_uuid"],
                "telemetry": worker["telemetry"],
            }
            for worker in workers
        ],
        "operational_feasibility": {
            "classification": (
                "OPERATIONALLY PLAUSIBLE"
                if comfortable
                else "OWNER REVIEW REQUIRED FOR OPERATIONAL FEASIBILITY"
            ),
            "reason": (
                "observed sizing completed in two fresh processes and measured memory stayed "
                "below available device/host capacity"
                if comfortable
                else "observed sizing approached device or host capacity"
            ),
            "protocol_unchanged": True,
        },
    }
