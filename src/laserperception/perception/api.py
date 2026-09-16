"""Developer-facing metadata, planning, and explicitly gated prediction API."""

from __future__ import annotations

from laserperception.detection.types import DetectionFrame

from .backends.catalog import BackendDescription, backend_description
from .backends.factory import backend_for
from .execution import ExecutionContext, ModelResources
from .inputs import InputDescription, PerceptionInput
from .manifests import ModelManifest
from .pipeline import DetectionPipeline
from .planning import ExecutionPlan, plan_execution
from .registry import ModelRegistry, builtin_registry
from .validation import ValidationReport, validate_execution_input


class LoadedModel:
    """A lightweight model handle that loads a detector only inside ``predict``."""

    def __init__(self, registry: ModelRegistry, model_id: str) -> None:
        self._registry = registry
        self._manifest = registry.get(model_id)
        self._backend_description = backend_description(model_id)
        self._pipeline: DetectionPipeline | None = None
        self._binding: tuple[ExecutionContext, ModelResources] | None = None
        self._closed = False

    @property
    def manifest(self) -> ModelManifest:
        return self._manifest

    def describe(self) -> ModelManifest:
        return self._manifest

    def describe_backend(self) -> BackendDescription:
        return self._backend_description

    def validate(self, value: InputDescription | PerceptionInput) -> ValidationReport:
        description = value.description if isinstance(value, PerceptionInput) else value
        if not isinstance(description, InputDescription):
            raise TypeError("validate requires InputDescription or PerceptionInput")
        return validate_execution_input(
            self._manifest,
            description.features,
            coordinates=description.coordinates,
            temporal=description.temporal,
            payload_kind=description.payload_kind,
            expected_payload_kind=self._backend_description.payload_kind,
        )

    def plan(
        self,
        description: InputDescription,
        context: ExecutionContext,
        resources: ModelResources,
    ) -> ExecutionPlan:
        if not isinstance(description, InputDescription):
            raise TypeError("plan requires InputDescription")
        return plan_execution(
            self._registry,
            self._manifest.model_id,
            description,
            context,
            resources,
        )

    def predict(
        self,
        value: PerceptionInput,
        *,
        context: ExecutionContext,
        resources: ModelResources,
    ) -> DetectionFrame:
        """Run a selected backend only after all static and authorization gates."""

        if self._closed:
            raise RuntimeError("model handle is closed")
        if not isinstance(value, PerceptionInput):
            raise TypeError("predict requires PerceptionInput")
        binding = (context, resources)
        if self._pipeline is None:
            backend = backend_for(
                self._registry,
                self._manifest.model_id,
                value,
                context,
                resources,
            )
            self._pipeline = DetectionPipeline(backend)
            self._binding = binding
        elif binding != self._binding:
            raise ValueError(
                "an initialized model handle cannot change execution context or resources"
            )
        try:
            return self._pipeline.predict(value)
        except Exception:
            self._closed = True
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pipeline is not None:
            self._pipeline.close()

    def __enter__(self) -> LoadedModel:
        if self._closed:
            raise RuntimeError("model handle is closed")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


def load_model(model_id: str, *, registry: ModelRegistry | None = None) -> LoadedModel:
    """Resolve reviewed metadata without importing or initializing a backend."""

    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")
    return LoadedModel(builtin_registry() if registry is None else registry, model_id)
