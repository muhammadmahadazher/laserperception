"""Lightweight perception metadata. Model availability is not execution permission."""

from .contracts import (
    Capabilities,
    CoordinateContract,
    FeatureSpec,
    PerceptionTask,
    TemporalContract,
)
from .manifests import ArtifactReference, ModelManifest
from .registry import ModelRegistry, builtin_registry
from .validation import ValidationReport, validate_input

registry = builtin_registry()

__all__ = [
    "ArtifactReference",
    "Capabilities",
    "CoordinateContract",
    "FeatureSpec",
    "ModelManifest",
    "ModelRegistry",
    "PerceptionTask",
    "TemporalContract",
    "ValidationReport",
    "registry",
    "validate_input",
]
