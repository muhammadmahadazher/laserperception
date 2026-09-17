"""Framework-independent input metadata and zero-copy payload envelopes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .contracts import CoordinateContract, FeatureSpec, TemporalContract
from .serialization import JsonRecord


@dataclass(frozen=True)
class InputDescription(JsonRecord):
    """Metadata needed to assess a LiDAR input without loading point bytes."""

    schema_version: Literal["1.0"]
    sample_id: str
    frame_id: str
    payload_kind: Literal["point_cloud", "model_ready_xyzt", "m8_xyzit"]
    features: tuple[FeatureSpec, ...]
    coordinates: CoordinateContract
    temporal: TemporalContract
    source: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.features or len({feature.name for feature in self.features}) != len(
            self.features
        ):
            raise ValueError("input features must be non-empty and unique")
        positions = tuple(feature.position for feature in self.features)
        if any(position is None for position in positions):
            raise ValueError("executable input features require explicit positions")
        if positions != tuple(range(len(self.features))):
            raise ValueError("input features must be ordered by contiguous position")


@dataclass(frozen=True)
class PerceptionInput:
    """Reference an existing point payload without copying its arrays."""

    description: InputDescription
    payload: object

    def __post_init__(self) -> None:
        if not isinstance(self.description, InputDescription):
            raise TypeError("description must be InputDescription")
        if self.payload is None:
            raise ValueError("payload must be supplied for execution")
