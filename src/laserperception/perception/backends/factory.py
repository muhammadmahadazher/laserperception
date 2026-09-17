"""Validated, authorization-first lazy construction of detection adapters."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import cast

from ..execution import (
    ExecutionContext,
    ModelResources,
    require_canonical_execution_manifest,
    verify_generic_prediction_authorization,
)
from ..inputs import PerceptionInput
from ..planning import plan_execution
from ..registry import ModelRegistry
from .base import BackendUnavailableError, DetectionBackend
from .catalog import backend_description

BackendConstructor = Callable[[object, ExecutionContext, ModelResources], object]


def _import_adapter(name: str) -> object:
    return importlib.import_module(name)


def _blocked_message(plan_errors: tuple[str, ...], validation_errors: tuple[str, ...]) -> str:
    details = (*validation_errors, *plan_errors)
    return "; ".join(details) if details else "execution plan is blocked"


def backend_for(
    registry: ModelRegistry,
    model_id: str,
    value: PerceptionInput,
    context: ExecutionContext,
    resources: ModelResources,
) -> DetectionBackend:
    """Resolve a backend only after metadata and authorization gates pass."""

    plan = plan_execution(registry, model_id, value.description, context, resources)
    if not plan.ready_for_guarded_initialization:
        raise ValueError(_blocked_message(plan.context_errors, plan.validation.errors))
    manifest = registry.get(model_id)
    require_canonical_execution_manifest(manifest)
    description = backend_description(model_id)

    if model_id == "pointpillars-nuscenes-v0.3":
        from laserperception.detection.ros2_contract import ModelReadyPointCloud

        if not isinstance(value.payload, ModelReadyPointCloud):
            raise TypeError("PointPillars execution requires ModelReadyPointCloud")
        verify_generic_prediction_authorization(
            context,
            manifest,
            resources,
            value.description,
            input_sha256=value.payload.sha256,
        )
    else:
        raise BackendUnavailableError(
            "M8 generic execution remains owned by the frozen S1 runner and accounted session"
        )

    try:
        module = _import_adapter(description.adapter_module)
        factory_object = getattr(module, description.adapter_factory)
    except (AttributeError, ImportError, OSError) as error:
        raise BackendUnavailableError(
            f"backend adapter {description.backend_id!r} is unavailable"
        ) from error
    if not callable(factory_object):
        raise BackendUnavailableError(
            f"backend adapter factory {description.adapter_factory!r} is not callable"
        )
    factory = cast(BackendConstructor, factory_object)
    backend = factory(manifest, context, resources)
    if not isinstance(backend, DetectionBackend):
        raise TypeError("backend factory did not return the DetectionBackend protocol")
    return backend
