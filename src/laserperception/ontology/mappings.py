"""Mappings into the Experiment 001 six-class ontology.

Source IDs were verified against:

* SemanticKITTI's official ``semantic-kitti.yaml`` configuration:
  https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml
* The DALES dataset paper, which defines unknown=0 and classes 1 through 8:
  https://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Varney_DALES_A_Large-Scale_Aerial_LiDAR_Data_Set_for_Semantic_Segmentation_CVPRW_2020_paper.html

The grouping itself is LaserPerception's explicit Experiment 001 policy, not an upstream dataset
mapping. Unlisted source IDs map to ``IGNORE_ID``.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum
from types import MappingProxyType

import numpy as np

IGNORE_ID = -1


class SharedClass(IntEnum):
    """Contiguous class IDs used by the shared geometry-only ontology."""

    GROUND = 0
    BUILDING = 1
    NATURAL = 2
    VEHICLE = 3
    POLE = 4
    FENCE = 5


CLASS_NAMES = ("Ground", "Building", "Natural", "Vehicle", "Pole", "Fence")

SEMANTICKITTI_TO_SHARED: Mapping[int, int] = MappingProxyType(
    {
        10: SharedClass.VEHICLE,  # car
        13: SharedClass.VEHICLE,  # bus
        16: SharedClass.VEHICLE,  # on-rails
        18: SharedClass.VEHICLE,  # truck
        20: SharedClass.VEHICLE,  # other-vehicle
        40: SharedClass.GROUND,  # road
        44: SharedClass.GROUND,  # parking
        48: SharedClass.GROUND,  # sidewalk
        49: SharedClass.GROUND,  # other-ground
        50: SharedClass.BUILDING,
        51: SharedClass.FENCE,
        60: SharedClass.GROUND,  # lane-marking; official map merges this into road
        70: SharedClass.NATURAL,  # vegetation
        71: SharedClass.NATURAL,  # trunk
        72: SharedClass.NATURAL,  # terrain
        80: SharedClass.POLE,
        252: SharedClass.VEHICLE,  # moving-car
        256: SharedClass.VEHICLE,  # moving-on-rails
        257: SharedClass.VEHICLE,  # moving-bus
        258: SharedClass.VEHICLE,  # moving-truck
        259: SharedClass.VEHICLE,  # moving-other-vehicle
    }
)

DALES_TO_SHARED: Mapping[int, int] = MappingProxyType(
    {
        1: SharedClass.GROUND,
        2: SharedClass.NATURAL,  # vegetation
        3: SharedClass.VEHICLE,  # cars
        4: SharedClass.VEHICLE,  # trucks
        # 5 is power lines and is outside the six-class ontology.
        6: SharedClass.FENCE,
        7: SharedClass.POLE,
        8: SharedClass.BUILDING,
    }
)


def map_labels(
    source_labels: np.ndarray,
    mapping: Mapping[int, int],
    *,
    ignore_id: int = IGNORE_ID,
) -> np.ndarray:
    """Map one-dimensional integer source labels using an explicit ID mapping."""
    labels = np.asarray(source_labels)
    if labels.ndim != 1:
        raise ValueError(f"source_labels must have shape (N,); received {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"source_labels must have an integer dtype; received {labels.dtype}")

    mapped = np.full(labels.shape, ignore_id, dtype=np.int16)
    for source_id, target_id in mapping.items():
        mapped[labels == source_id] = int(target_id)
    return mapped


def map_semantickitti_labels(source_labels: np.ndarray) -> np.ndarray:
    """Map official SemanticKITTI semantic IDs into the shared ontology."""
    return map_labels(source_labels, SEMANTICKITTI_TO_SHARED)


def map_dales_labels(source_labels: np.ndarray) -> np.ndarray:
    """Map official DALES classification IDs into the shared ontology."""
    return map_labels(source_labels, DALES_TO_SHARED)
