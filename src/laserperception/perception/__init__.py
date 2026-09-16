"""Lightweight perception metadata, planning, and explicitly gated APIs."""

from .api import LoadedModel, load_model
from .backends import (
    BackendDescription,
    BackendUnavailableError,
    DetectionBackend,
    backend_description,
    list_backend_descriptions,
)
from .contracts import (
    Capabilities,
    CoordinateContract,
    FeatureSpec,
    PerceptionTask,
    TemporalContract,
)
from .execution import (
    ExecutionAuthorization,
    ExecutionContext,
    ModelResources,
    Precision,
    RuntimeTarget,
)
from .inputs import InputDescription, PerceptionInput
from .manifests import ArtifactReference, ModelManifest
from .pipeline import DetectionPipeline
from .planning import ExecutionPlan
from .registry import ModelRegistry, builtin_registry
from .validation import ValidationReport, validate_input

registry = builtin_registry()

__all__ = [
    "ArtifactReference",
    "BackendDescription",
    "BackendUnavailableError",
    "Capabilities",
    "CoordinateContract",
    "DetectionBackend",
    "DetectionPipeline",
    "ExecutionAuthorization",
    "ExecutionContext",
    "ExecutionPlan",
    "FeatureSpec",
    "InputDescription",
    "LoadedModel",
    "ModelManifest",
    "ModelRegistry",
    "ModelResources",
    "PerceptionInput",
    "PerceptionTask",
    "Precision",
    "RuntimeTarget",
    "TemporalContract",
    "ValidationReport",
    "backend_description",
    "list_backend_descriptions",
    "load_model",
    "registry",
    "validate_input",
]
