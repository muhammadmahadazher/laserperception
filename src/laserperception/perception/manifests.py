"""Reviewed model identities, independent of optional detector installations."""

import re
from dataclasses import dataclass
from typing import Literal

from .contracts import (
    Capabilities,
    CoordinateContract,
    FeatureSpec,
    PerceptionTask,
    TemporalContract,
)
from .serialization import JsonRecord


@dataclass(frozen=True)
class ArtifactReference(JsonRecord):
    role: str
    name: str
    byte_size: int | None
    sha256: str | None
    provenance: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.role or not self.name or not self.provenance:
            raise ValueError("artifact role, name and provenance are required")
        if self.byte_size is not None and self.byte_size < 0:
            raise ValueError("artifact size must be non-negative")
        if self.sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("artifact SHA256 must be 64 lowercase hexadecimal characters")


@dataclass(frozen=True)
class ModelManifest(JsonRecord):
    schema_version: Literal["1.0"]
    model_id: str
    display_name: str
    model_version: str
    status: Literal["released_historical", "research"]
    family: str
    upstream: str
    capabilities: Capabilities
    input_features: tuple[FeatureSpec, ...]
    coordinates: CoordinateContract
    temporal: TemporalContract
    class_names: tuple[str, ...]
    runtime_requirements: tuple[str, ...]
    artifacts: tuple[ArtifactReference, ...]
    limitations: tuple[str, ...]
    scientific_status: str
    authorization_required: bool

    def __post_init__(self) -> None:
        super().__post_init__()
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", self.model_id):
            raise ValueError("model_id must be a stable lowercase identifier")
        for name in ("display_name", "model_version", "family", "upstream", "scientific_status"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        names = [f.name for f in self.input_features]
        positions = [f.position for f in self.input_features if f.position is not None]
        if not names or len(set(names)) != len(names) or len(set(positions)) != len(positions):
            raise ValueError("features and positions must be non-empty/unique")
        if positions != list(range(len(positions))):
            raise ValueError("ordered model feature positions must be contiguous")
        if not self.class_names or len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class taxonomy must be non-empty and unique")
        if self.temporal.elapsed_feature is not None and self.temporal.elapsed_feature not in names:
            raise ValueError("temporal feature is absent from feature contract")
        if self.capabilities.temporal_support != (self.temporal.mode == "multi_sweep"):
            raise ValueError("temporal capability and contract disagree")

    @property
    def tasks(self) -> tuple[PerceptionTask, ...]:
        return self.capabilities.tasks
