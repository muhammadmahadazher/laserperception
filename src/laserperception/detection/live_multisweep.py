"""ROS-independent live history and transform adapters for M4.5b."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np

from laserperception.detection.multisweep import NUSCENES_LOAD_DIM, RawSweep, SweepTransform
from laserperception.detection.ros2_contract import TimeStamp

HistoryResetReason = Literal["time_regression", "gap"]


def stamp_nanoseconds(stamp: TimeStamp) -> int:
    """Return one exact integer nanosecond acquisition time."""

    return stamp.sec * 1_000_000_000 + stamp.nanosec


def stamp_microseconds(stamp: TimeStamp) -> int:
    """Quantize a ROS stamp down to the preceding integer microsecond.

    nuScenes replay stamps are exact multiples of 1,000 nanoseconds, so this
    adaptation preserves the frozen M4.5a integer-microsecond arithmetic.
    """

    return stamp_nanoseconds(stamp) // 1_000


def acquisition_identity(frame_id: str, stamp: TimeStamp) -> str:
    """Identify an acquisition without changing its ROS frame name."""

    normalized = frame_id.strip()
    if not normalized:
        raise ValueError("frame_id must be a non-empty string")
    return f"{normalized}@{stamp.sec}.{stamp.nanosec:09d}"


@dataclass(frozen=True, slots=True)
class LiveRawSweep:
    """A raw acquisition with its exact ROS stamp and physical frame name."""

    sweep: RawSweep
    frame_id: str
    stamp: TimeStamp

    def __post_init__(self) -> None:
        frame_id = self.frame_id.strip()
        if not frame_id:
            raise ValueError("frame_id must be a non-empty string")
        expected_identity = acquisition_identity(frame_id, self.stamp)
        if self.sweep.source_id != expected_identity:
            raise ValueError("raw sweep source_id must match frame and acquisition timestamp")
        if self.sweep.timestamp_microseconds != stamp_microseconds(self.stamp):
            raise ValueError("raw sweep timestamp must match quantized ROS acquisition time")
        object.__setattr__(self, "frame_id", frame_id)

    @property
    def stamp_ns(self) -> int:
        """Return the exact ROS acquisition time as integer nanoseconds."""

        return stamp_nanoseconds(self.stamp)


def live_raw_sweep_from_xyz(
    points_xyz: np.ndarray,
    *,
    frame_id: str,
    stamp: TimeStamp,
) -> LiveRawSweep:
    """Adapt finite XYZ into the frozen five-column M4.5a raw representation."""

    points = np.asarray(points_xyz)
    if points.dtype != np.dtype(np.float32):
        raise TypeError("raw XYZ points must have dtype float32")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("raw XYZ points must have shape (N, 3)")
    if len(points) == 0:
        raise ValueError("raw XYZ points must contain at least one valid row")
    if not np.isfinite(points).all():
        raise ValueError("raw XYZ points must contain only finite values")
    raw = np.zeros((len(points), NUSCENES_LOAD_DIM), dtype=np.float32)
    raw[:, :3] = points
    identity = acquisition_identity(frame_id, stamp)
    return LiveRawSweep(
        RawSweep(raw, stamp_microseconds(stamp), identity),
        frame_id=frame_id,
        stamp=stamp,
    )


@dataclass(frozen=True, slots=True)
class HistorySelection:
    """Previous acquisitions selected nearest-to-farthest for one current sweep."""

    historical: tuple[LiveRawSweep, ...]
    reset_reason: HistoryResetReason | None


class LiveSweepHistory:
    """Bounded acquisition history with explicit clock-reset behavior."""

    def __init__(
        self,
        *,
        max_historical_sweeps: int = 10,
        reset_gap_sec: float = 0.0,
    ) -> None:
        if isinstance(max_historical_sweeps, bool) or not isinstance(max_historical_sweeps, int):
            raise TypeError("max_historical_sweeps must be an integer")
        if max_historical_sweeps <= 0:
            raise ValueError("max_historical_sweeps must be positive")
        gap = float(reset_gap_sec)
        if not isfinite(gap) or gap < 0.0:
            raise ValueError("reset_gap_sec must be finite and non-negative")
        self.max_historical_sweeps = max_historical_sweeps
        self.reset_gap_sec = gap
        self._sweeps: deque[LiveRawSweep] = deque()
        self._last_stamp_ns: int | None = None
        self._pending_identity: str | None = None

    @property
    def depth(self) -> int:
        """Return the number of buffered historical acquisitions."""

        return len(self._sweeps)

    def select_for_current(self, current: LiveRawSweep) -> HistorySelection:
        """Reset if needed and select previous sweeps nearest-to-farthest."""

        if self._pending_identity is not None:
            raise RuntimeError("the previous current sweep has not been stored")
        reset_reason: HistoryResetReason | None = None
        current_ns = current.stamp_ns
        if self._last_stamp_ns is not None:
            if current_ns <= self._last_stamp_ns:
                reset_reason = "time_regression"
            elif self.reset_gap_sec > 0.0:
                gap_ns = current_ns - self._last_stamp_ns
                if gap_ns > int(self.reset_gap_sec * 1_000_000_000):
                    reset_reason = "gap"
        if reset_reason is not None:
            self._sweeps.clear()
        self._last_stamp_ns = current_ns
        self._pending_identity = current.sweep.source_id
        selected = tuple(reversed(self._sweeps))[: self.max_historical_sweeps]
        return HistorySelection(selected, reset_reason)

    def store_current(self, current: LiveRawSweep) -> None:
        """Store a valid current acquisition after its build attempt finishes."""

        if self._pending_identity != current.sweep.source_id:
            raise RuntimeError("current sweep does not match the pending history selection")
        self._sweeps.append(current)
        while len(self._sweeps) > self.max_historical_sweeps:
            self._sweeps.popleft()
        self._pending_identity = None


def sweep_transform_from_ros(
    *,
    translation_xyz: Sequence[float],
    quaternion_xyzw: Sequence[float],
    source_id: str,
    target_id: str,
) -> SweepTransform:
    """Encode a ROS source-to-target transform for the M4.5a row-vector builder.

    ROS uses column vectors: ``p_target = R @ p_source + t``. The accepted
    builder applies ``p_source_row @ A - b``. Therefore this adapter stores
    ``A = R.T`` and ``b = -t`` before the single required float32 cast.
    """

    translation = _finite_float64_vector(translation_xyz, 3, "translation_xyz")
    quaternion = _finite_float64_vector(quaternion_xyzw, 4, "quaternion_xyzw")
    x, y, z, w = (float(value) for value in quaternion)
    norm_squared = x * x + y * y + z * z + w * w
    if norm_squared <= 0.0:
        raise ValueError("quaternion_xyzw must have non-zero norm")
    if abs(norm_squared - 1.0) > 1e-6:
        raise ValueError("quaternion_xyzw must be normalized")
    scale = 2.0 / norm_squared
    rotation = np.array(
        [
            [1.0 - scale * (y * y + z * z), scale * (x * y - z * w), scale * (x * z + y * w)],
            [scale * (x * y + z * w), 1.0 - scale * (x * x + z * z), scale * (y * z - x * w)],
            [scale * (x * z - y * w), scale * (y * z + x * w), 1.0 - scale * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation.T
    matrix[:3, 3] = -translation
    return SweepTransform(matrix.astype(np.float32), source_id=source_id, target_id=target_id)


def _finite_float64_vector(value: Sequence[float], length: int, name: str) -> np.ndarray:
    array = np.asarray(tuple(value), dtype=np.float64)
    if array.shape != (length,):
        raise ValueError(f"{name} must contain exactly {length} values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array
