"""Explicit CPU baseline policy; distances are meters in a fixed common frame."""

from dataclasses import dataclass

from laserperception.perception.serialization import JsonRecord


@dataclass(frozen=True)
class TrackerConfig(JsonRecord):
    min_hits: int = 3
    max_missed: int = 2
    association_radius_m: float = 3.0
    class_aware: bool = True
    velocity_smoothing: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.min_hits < 1 or self.max_missed < 0:
            raise ValueError("min_hits must be positive and max_missed non-negative")
        if self.association_radius_m <= 0:
            raise ValueError("association radius must be positive")
        if not 0 <= self.velocity_smoothing <= 1:
            raise ValueError("velocity_smoothing must be in [0, 1]")
