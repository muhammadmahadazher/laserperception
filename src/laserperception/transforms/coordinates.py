"""Coordinate transforms that are separate from file loading."""

from __future__ import annotations

from copy import deepcopy

from laserperception.core import PointCloud


def normalize_coordinates(cloud: PointCloud, mode: str = "min_xyz") -> PointCloud:
    """Return a normalized copy of ``cloud``.

    ``min_xyz`` subtracts the per-axis minimum so the transformed bounding-box minimum is zero.
    The original cloud is not mutated. No other modes are implemented until an experiment defines
    and validates them.
    """
    if mode != "min_xyz":
        raise ValueError(f"unsupported coordinate normalization mode: {mode!r}")
    if len(cloud) == 0:
        raise ValueError("cannot normalize an empty point cloud")

    source_origin = cloud.xyz.min(axis=0)
    normalized_xyz = cloud.xyz - source_origin
    metadata = deepcopy(cloud.metadata)
    transform_record = {
        "mode": mode,
        "source_origin_xyz": source_origin.tolist(),
        "applied_translation_xyz": (-source_origin).tolist(),
    }
    history = list(metadata.get("coordinate_transforms", []))
    history.append(transform_record)
    metadata["coordinate_transforms"] = history
    metadata["coordinate_normalization"] = transform_record
    metadata["coordinates_normalized"] = True

    return PointCloud(
        xyz=normalized_xyz,
        labels=cloud.labels,
        attributes=cloud.attributes,
        metadata=metadata,
    )
