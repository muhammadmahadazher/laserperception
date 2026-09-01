"""CPU-testable NVIDIA telemetry helpers for diagnostic measurement sessions."""

from __future__ import annotations

import csv
import subprocess
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from io import StringIO
from statistics import fmean, median

NVIDIA_SMI_FIELDS = (
    "name",
    "driver_version",
    "pstate",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "clocks.sm",
    "clocks.mem",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
)
_NUMERIC_FIELDS = frozenset(
    {
        "temperature.gpu",
        "power.draw",
        "power.limit",
        "clocks.sm",
        "clocks.mem",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
    }
)


def parse_nvidia_smi_row(text: str) -> dict[str, object]:
    """Parse one unit-free nvidia-smi CSV row without requiring NVIDIA software."""

    rows = list(csv.reader(StringIO(text.strip())))
    if len(rows) != 1 or len(rows[0]) != len(NVIDIA_SMI_FIELDS):
        raise ValueError("nvidia-smi telemetry must contain exactly one complete GPU row")
    result: dict[str, object] = {"available": True}
    for field, raw_value in zip(NVIDIA_SMI_FIELDS, rows[0], strict=True):
        value = raw_value.strip()
        if field in _NUMERIC_FIELDS:
            result[field] = _optional_float(value)
        else:
            result[field] = value
    return result


def query_nvidia_smi() -> dict[str, object]:
    """Capture one GPU-0 state sample, returning an explicit unavailable record on failure."""

    try:
        process = subprocess.run(
            [
                "nvidia-smi",
                "--id=0",
                f"--query-gpu={','.join(NVIDIA_SMI_FIELDS)}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "error": str(error)}
    if process.returncode != 0 or not process.stdout.strip():
        return {
            "available": False,
            "error": process.stderr.strip() or "nvidia-smi returned no telemetry",
        }
    try:
        return parse_nvidia_smi_row(process.stdout.splitlines()[0])
    except ValueError as error:
        return {"available": False, "error": str(error), "raw": process.stdout.strip()}


def nvidia_clock_capability() -> dict[str, object]:
    """Record supported-clock visibility without changing or locking any clock."""

    try:
        process = subprocess.run(
            ["nvidia-smi", "--id=0", "-q", "-d", "SUPPORTED_CLOCKS"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "query_available": False,
            "error": str(error),
            "clock_locking_attempted": False,
        }
    output = process.stdout.strip()
    try:
        application = subprocess.run(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=clocks.applications.graphics,clocks.applications.memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        application_record: dict[str, object] = {
            "application_clocks_query_returncode": application.returncode,
            "application_clocks_setting": application.stdout.strip(),
            "application_clocks_query_error": application.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        application_record = {
            "application_clocks_query_returncode": None,
            "application_clocks_setting": "",
            "application_clocks_query_error": str(error),
        }
    return {
        "query_available": process.returncode == 0 and bool(output),
        "query_returncode": process.returncode,
        "supported_clocks_reported": "Supported Clocks" in output,
        "query_output": output,
        "query_error": process.stderr.strip(),
        **application_record,
        "clock_locking_attempted": False,
        "clock_setting_changed": False,
        "clock_locking_not_attempted_reason": (
            "visibility of supported clocks does not establish a safe WSL2 laptop lock"
        ),
    }


class NvidiaSmiSampler:
    """Periodically sample GPU 0 and label samples by the active measured block."""

    def __init__(
        self,
        *,
        interval_seconds: float,
        query: Callable[[], dict[str, object]] = query_nvidia_smi,
    ) -> None:
        if interval_seconds <= 0.0:
            raise ValueError("telemetry interval must be positive")
        self.interval_seconds = float(interval_seconds)
        self._query = query
        self._lock = threading.Lock()
        self._query_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._block: str | None = None
        self._samples: list[dict[str, object]] = []

    def start(self) -> None:
        """Start periodic sampling after one synchronous session snapshot."""

        if self._thread is not None:
            raise RuntimeError("telemetry sampler has already been started")
        self._capture()
        self._thread = threading.Thread(target=self._run, name="nvidia-smi-sampler", daemon=True)
        self._thread.start()

    def begin_block(self, label: str) -> None:
        """Label telemetry and capture a synchronous pre-block sample."""

        if not label.strip():
            raise ValueError("telemetry block label must be non-empty")
        with self._lock:
            if self._block is not None:
                raise RuntimeError("a telemetry block is already active")
            self._block = label
        self._capture()

    def end_block(self, label: str) -> None:
        """Capture a synchronous post-block sample and clear its label."""

        with self._lock:
            if self._block != label:
                raise RuntimeError("telemetry block end does not match the active block")
        self._capture()
        with self._lock:
            self._block = None

    def stop(self) -> None:
        """Stop sampling and capture one final unlabeled session snapshot."""

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 3.0))
            if self._thread.is_alive():
                raise RuntimeError("telemetry sampler thread did not stop")
        self._capture()

    @property
    def samples(self) -> tuple[dict[str, object], ...]:
        """Return a defensive copy of every raw telemetry sample."""

        with self._lock:
            return tuple(dict(sample) for sample in self._samples)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._capture()

    def _capture(self) -> None:
        with self._lock:
            block = self._block
        with self._query_lock:
            measurement = self._query()
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "monotonic_ns": time.monotonic_ns(),
            **measurement,
            "block": block,
        }
        with self._lock:
            self._samples.append(record)


def summarize_gpu_telemetry(samples: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Summarize raw telemetry while retaining categorical performance-state counts."""

    available = [sample for sample in samples if sample.get("available") is True]
    result: dict[str, object] = {
        "sample_count": len(samples),
        "available_sample_count": len(available),
        "unavailable_sample_count": len(samples) - len(available),
    }
    if not available:
        result["available"] = False
        return result
    result.update(
        {
            "available": True,
            "gpu_names": sorted({str(sample.get("name", "")) for sample in available}),
            "driver_versions": sorted(
                {str(sample.get("driver_version", "")) for sample in available}
            ),
            "pstate_counts": dict(
                sorted(Counter(str(sample.get("pstate", "")) for sample in available).items())
            ),
            "numeric": {
                field: _numeric_summary(available, field) for field in sorted(_NUMERIC_FIELDS)
            },
        }
    )
    return result


def summarize_telemetry_by_block(
    samples: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Group and summarize telemetry under each non-empty measured-block label."""

    labels = sorted({str(sample["block"]) for sample in samples if sample.get("block") is not None})
    return {
        label: summarize_gpu_telemetry(
            [sample for sample in samples if str(sample.get("block")) == label]
        )
        for label in labels
    }


def paired_gpu_state_eligibility(
    samples: Sequence[Mapping[str, object]],
    pairs: Sequence[tuple[str, str, str]],
) -> dict[str, object]:
    """Reject only obvious paired-state separation; define no universal clock threshold."""

    pair_records: list[dict[str, object]] = []
    rejection_reasons: list[str] = []
    for pair_name, reference_label, candidate_label in pairs:
        reference = [
            sample
            for sample in samples
            if sample.get("available") is True and sample.get("block") == reference_label
        ]
        candidate = [
            sample
            for sample in samples
            if sample.get("available") is True and sample.get("block") == candidate_label
        ]
        reasons: list[str] = []
        clock_range_overlap: dict[str, bool | None] = {}
        assessable = bool(reference and candidate)
        pstate_overlap: bool | None = None
        if assessable:
            reference_pstates = {str(sample.get("pstate")) for sample in reference}
            candidate_pstates = {str(sample.get("pstate")) for sample in candidate}
            pstate_overlap = bool(reference_pstates.intersection(candidate_pstates))
            for field in ("clocks.sm", "clocks.mem"):
                reference_range = _numeric_range(reference, field)
                candidate_range = _numeric_range(candidate, field)
                clock_range_overlap[field] = (
                    None
                    if reference_range is None or candidate_range is None
                    else _ranges_overlap(reference_range, candidate_range)
                )
            if pstate_overlap is False and any(
                overlap is False for overlap in clock_range_overlap.values()
            ):
                reasons.append("disjoint_performance_states_with_disjoint_clock_ranges")
        record = {
            "pair": pair_name,
            "reference_block": reference_label,
            "candidate_block": candidate_label,
            "reference": summarize_gpu_telemetry(reference),
            "candidate": summarize_gpu_telemetry(candidate),
            "assessable": assessable,
            "performance_state_overlap": pstate_overlap,
            "clock_range_overlap": clock_range_overlap,
            "obvious_material_state_mismatch": bool(reasons),
            "reasons": reasons,
        }
        pair_records.append(record)
        rejection_reasons.extend(f"{pair_name}:{reason}" for reason in reasons)
    available_count = sum(sample.get("available") is True for sample in samples)
    return {
        "policy": (
            "reject only disjoint performance states corroborated by disjoint clock ranges; "
            "no universal MHz threshold"
        ),
        "telemetry_available": available_count > 0,
        "eligible": not rejection_reasons,
        "assessment_limited_by_unavailable_telemetry": available_count == 0,
        "rejection_reasons": rejection_reasons,
        "pairs": pair_records,
    }


def _optional_float(value: str) -> float | None:
    normalized = value.strip().lower()
    if normalized in {"", "n/a", "[not supported]", "not supported"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _numeric_values(samples: Sequence[Mapping[str, object]], field: str) -> list[float]:
    return [
        float(value)
        for sample in samples
        if isinstance((value := sample.get(field)), (int, float)) and not isinstance(value, bool)
    ]


def _numeric_range(
    samples: Sequence[Mapping[str, object]], field: str
) -> tuple[float, float] | None:
    values = _numeric_values(samples, field)
    return None if not values else (min(values), max(values))


def _numeric_summary(
    samples: Sequence[Mapping[str, object]], field: str
) -> dict[str, float | int | None]:
    values = _numeric_values(samples, field)
    if not values:
        return {"count": 0, "minimum": None, "median": None, "mean": None, "maximum": None}
    return {
        "count": len(values),
        "minimum": min(values),
        "median": median(values),
        "mean": fmean(values),
        "maximum": max(values),
    }


def _ranges_overlap(first: tuple[float, float], second: tuple[float, float]) -> bool:
    return max(first[0], second[0]) <= min(first[1], second[1])
