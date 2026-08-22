"""Portable reproduction of frozen M6b model-ready input oracles.

The M6a reconstruction contract intentionally preserves upstream NumPy
arithmetic and float32 transform serialization. Tiny BLAS/LAPACK differences
can therefore change a serialized transform by one float32 ULP across host
platforms. M6b freezes those already-computed float32 transforms before
detector inference, then continues to use the unchanged ``MultiSweepBuilder``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np

from laserperception.datasets.kitti_raw import KittiRawSequence, KittiReconstructionResult
from laserperception.detection.multisweep import (
    HistoricalSweep,
    MultiSweepBuilder,
    SweepTransform,
)


def _matrix_sha256(matrix: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(matrix).tobytes(order="C")).hexdigest()


def freeze_sweep_transforms(
    sequence: KittiRawSequence,
    current_index: int,
) -> tuple[dict[str, object], ...]:
    """Serialize the canonical float32 transforms for one reconstruction."""

    current = sequence.frame(current_index).to_raw_sweep()
    current_pose = sequence.lidar_pose(current_index)
    records: list[dict[str, object]] = []
    for source_index in range(current_index - 1, max(-1, current_index - 11), -1):
        source = sequence.frame(source_index).to_raw_sweep()
        transform = SweepTransform.from_poses(
            source_id=source.source_id,
            target_id=current.source_id,
            sweep_pose=sequence.lidar_pose(source_index),
            current_pose=current_pose,
        )
        records.append(
            {
                "source_index": source_index,
                "source_id": source.source_id,
                "target_id": current.source_id,
                "lidar2sensor": transform.lidar2sensor.tolist(),
                "lidar2sensor_sha256": _matrix_sha256(transform.lidar2sensor),
            }
        )
    return tuple(records)


def reconstruct_from_frozen_transforms(
    sequence: KittiRawSequence,
    current_index: int,
    transform_records: Sequence[Mapping[str, object]],
    *,
    builder: MultiSweepBuilder,
) -> KittiReconstructionResult:
    """Reconstruct with frozen transforms and the unchanged production builder."""

    expected_indices = tuple(range(current_index - 1, max(-1, current_index - 11), -1))
    if len(transform_records) != len(expected_indices):
        raise ValueError("frozen transform count does not match the available history")

    current_frame = sequence.frame(current_index)
    current = current_frame.to_raw_sweep()
    historical: list[HistoricalSweep] = []
    source_counts = [current_frame.points_xyzi.shape[0]]
    selected_indices = [current_index]
    for expected_index, record in zip(expected_indices, transform_records, strict=True):
        source_index = record.get("source_index")
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise TypeError("frozen transform source_index must be an integer")
        if source_index != expected_index:
            raise ValueError("frozen transforms are not in nearest-to-farthest source order")
        source_frame = sequence.frame(source_index)
        source = source_frame.to_raw_sweep()
        source_id = record.get("source_id")
        target_id = record.get("target_id")
        if source_id != source.source_id or target_id != current.source_id:
            raise ValueError("frozen transform source/target identity mismatch")
        matrix = np.asarray(record.get("lidar2sensor"), dtype=np.float32)
        transform = SweepTransform(matrix, source.source_id, current.source_id)
        expected_sha256 = record.get("lidar2sensor_sha256")
        if not isinstance(expected_sha256, str) or _matrix_sha256(matrix) != expected_sha256:
            raise ValueError("frozen transform SHA256 mismatch")
        historical.append(HistoricalSweep(source, transform))
        selected_indices.append(source_index)
        source_counts.append(source_frame.points_xyzi.shape[0])

    point_cloud = builder.build(current, historical)
    return KittiReconstructionResult(
        current_index=current_index,
        selected_indices=tuple(selected_indices),
        source_counts=tuple(source_counts),
        point_cloud=point_cloud,
    )
