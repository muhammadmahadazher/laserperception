"""Metadata contracts describe capabilities; they never authorize execution."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from .serialization import JsonRecord


class PerceptionTask(str, Enum):
    DETECTION_3D = "detection_3d"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    TRACKING = "tracking"
    EMBEDDING = "embedding"
    SCENE_UNDERSTANDING = "scene_understanding"


@dataclass(frozen=True)
class FeatureSpec(JsonRecord):
    name: str
    dtype: Literal["float32", "float64", "int32", "uint16"]
    required: bool
    unit: str | None
    semantics: str
    position: int | None
    missing_policy: Literal["reject", "omit"]
    source: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.name.strip() or not self.semantics.strip() or not self.source.strip():
            raise ValueError("feature name, semantics and source must be non-empty")
        if self.position is not None and self.position < 0:
            raise ValueError("feature position must be non-negative")
        if self.required and self.missing_policy != "reject":
            raise ValueError("required features cannot be synthesized or omitted")


@dataclass(frozen=True)
class CoordinateContract(JsonRecord):
    frame_role: str
    handedness: Literal["right", "left"]
    axes: tuple[str, str, str]
    distance_unit: str
    box_center: str
    dimension_order: tuple[str, str, str]
    yaw: str
    reference: str


@dataclass(frozen=True)
class TemporalContract(JsonRecord):
    mode: Literal["single_scan", "multi_sweep"]
    min_history: int
    max_history: int
    timestamp_unit: str
    reference: str
    elapsed_feature: str | None
    sweep_order: str
    motion_compensation: bool

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 <= self.min_history <= self.max_history:
            raise ValueError("invalid historical sweep range")
        if self.mode == "single_scan" and self.max_history != 0:
            raise ValueError("single scan cannot declare history")


@dataclass(frozen=True)
class Capabilities(JsonRecord):
    tasks: tuple[PerceptionTask, ...]
    temporal_support: bool
    velocity_output: bool
    runtime_targets: tuple[str, ...]
    deployment_maturity: str
    training_support: bool
    output_schema: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ValueError("capability tasks must be non-empty and unique")
        if not self.runtime_targets or len(set(self.runtime_targets)) != len(self.runtime_targets):
            raise ValueError("runtime targets must be non-empty and unique")
