"""Canonical in-memory point-cloud representation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PointCloud:
    """A validated point cloud with geometry, optional labels, and point attributes.

    Arrays are defensively copied during construction. Coordinates are canonicalized to
    ``float32``; label and attribute dtypes are preserved. Loading a file does not normalize or
    otherwise translate its coordinates.
    """

    xyz: np.ndarray
    labels: np.ndarray | None = None
    attributes: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and defensively copy all mutable inputs."""
        xyz = np.asarray(self.xyz)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"xyz must have shape (N, 3); received {xyz.shape}")
        if not (np.issubdtype(xyz.dtype, np.integer) or np.issubdtype(xyz.dtype, np.floating)):
            raise TypeError(f"xyz must have a real numeric dtype; received {xyz.dtype}")
        self.xyz = xyz.astype(np.float32, copy=True)

        point_count = self.xyz.shape[0]
        if self.labels is not None:
            labels = np.asarray(self.labels)
            if labels.ndim != 1:
                raise ValueError(f"labels must have shape (N,); received {labels.shape}")
            if labels.shape[0] != point_count:
                raise ValueError(
                    "labels length must match xyz: "
                    f"received {labels.shape[0]} labels for {point_count} points"
                )
            self.labels = np.array(labels, copy=True)

        copied_attributes: dict[str, np.ndarray] = {}
        for name, values in self.attributes.items():
            if not isinstance(name, str) or not name:
                raise ValueError("attribute names must be non-empty strings")
            array = np.asarray(values)
            if array.ndim == 0:
                raise ValueError(f"attribute {name!r} must have a point dimension")
            if array.shape[0] != point_count:
                raise ValueError(
                    f"attribute {name!r} length must match xyz: "
                    f"received {array.shape[0]} values for {point_count} points"
                )
            copied_attributes[name] = np.array(array, copy=True)
        self.attributes = copied_attributes

        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")
        self.metadata = deepcopy(self.metadata)

    def __len__(self) -> int:
        """Return the number of points."""
        return int(self.xyz.shape[0])

    def copy(self) -> PointCloud:
        """Return a deep, independently mutable copy of the cloud."""
        return PointCloud(
            xyz=self.xyz,
            labels=self.labels,
            attributes=self.attributes,
            metadata=self.metadata,
        )
