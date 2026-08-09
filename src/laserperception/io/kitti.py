"""KITTI Velodyne and SemanticKITTI label I/O.

The format follows the official SemanticKITTI documentation:
https://semantic-kitti.org/dataset.html
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from laserperception.core import PointCloud

_FLOAT32_BYTES = np.dtype(np.float32).itemsize
_UINT32_BYTES = np.dtype(np.uint32).itemsize
_KITTI_FIELDS = 4
_SEMANTIC_MASK = np.uint32(0xFFFF)
_INSTANCE_SHIFT = np.uint32(16)


@dataclass(frozen=True)
class SemanticKITTILabels:
    """Decoded SemanticKITTI point labels."""

    semantic_ids: np.ndarray
    instance_ids: np.ndarray
    packed: np.ndarray


def load_semantic_kitti_labels(
    path: str | Path, *, expected_points: int | None = None
) -> SemanticKITTILabels:
    """Read packed SemanticKITTI labels and split semantic and instance IDs.

    Each record is an unsigned 32-bit integer. The lower 16 bits contain the semantic ID and the
    upper 16 bits contain the instance ID, as specified by the dataset maintainers.
    """
    label_path = Path(path)
    byte_count = label_path.stat().st_size
    if byte_count % _UINT32_BYTES != 0:
        raise ValueError(
            f"SemanticKITTI label file size must be divisible by 4 bytes; received {byte_count}"
        )

    packed = np.fromfile(label_path, dtype="<u4")
    if expected_points is not None and packed.size != expected_points:
        raise ValueError(
            "SemanticKITTI label count must match point count: "
            f"received {packed.size} labels for {expected_points} points"
        )

    semantic_ids = (packed & _SEMANTIC_MASK).astype(np.uint16, copy=False)
    instance_ids = (packed >> _INSTANCE_SHIFT).astype(np.uint16, copy=False)
    return SemanticKITTILabels(
        semantic_ids=np.array(semantic_ids, copy=True),
        instance_ids=np.array(instance_ids, copy=True),
        packed=np.array(packed, copy=True),
    )


def load_kitti_bin(path: str | Path, *, label_path: str | Path | None = None) -> PointCloud:
    """Load a KITTI/SemanticKITTI Velodyne ``.bin`` scan.

    Records are little-endian float32 values in ``[x, y, z, remission]`` order. Coordinates are
    returned without normalization. When ``label_path`` is supplied, decoded semantic IDs become
    ``PointCloud.labels`` and instance IDs are retained as a point attribute.
    """
    scan_path = Path(path)
    byte_count = scan_path.stat().st_size
    record_bytes = _KITTI_FIELDS * _FLOAT32_BYTES
    if byte_count % record_bytes != 0:
        raise ValueError(
            "KITTI Velodyne file size must be divisible by 16 bytes "
            f"(four float32 values per point); received {byte_count}"
        )

    records = np.fromfile(scan_path, dtype="<f4").reshape(-1, _KITTI_FIELDS)
    labels: np.ndarray | None = None
    attributes: dict[str, np.ndarray] = {"remission": records[:, 3]}
    metadata: dict[str, object] = {
        "format": "kitti_velodyne_float32_xyzi",
        "source_file": scan_path.name,
        "coordinate_frame": "sensor",
        "coordinates_normalized": False,
    }

    if label_path is not None:
        decoded = load_semantic_kitti_labels(label_path, expected_points=records.shape[0])
        labels = decoded.semantic_ids
        attributes["instance_id"] = decoded.instance_ids
        metadata["label_file"] = Path(label_path).name
        metadata["label_encoding"] = "uint32: semantic=lower16, instance=upper16"

    return PointCloud(xyz=records[:, :3], labels=labels, attributes=attributes, metadata=metadata)


def write_kitti_bin(cloud: PointCloud, path: str | Path) -> None:
    """Write KITTI-compatible float32 ``[x, y, z, remission]`` point records.

    This writes one scan only; it does not reproduce a KITTI or SemanticKITTI dataset hierarchy.
    """
    if "remission" not in cloud.attributes:
        raise ValueError("KITTI output requires a 'remission' point attribute")
    remission = np.asarray(cloud.attributes["remission"])
    if remission.ndim != 1 or remission.shape[0] != len(cloud):
        raise ValueError("'remission' must have shape (N,) for KITTI output")
    records = np.column_stack((cloud.xyz, remission)).astype("<f4", copy=False)
    records.tofile(Path(path))
