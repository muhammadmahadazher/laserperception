"""Dry-run plans inspect metadata without importing a backend or probing hardware."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .backends.catalog import BackendDescription, backend_description
from .execution import ExecutionContext, ModelResources, Precision, execution_context_errors
from .inputs import InputDescription
from .manifests import ArtifactReference
from .registry import ModelRegistry
from .serialization import JsonRecord
from .validation import ValidationReport, validate_execution_input


@dataclass(frozen=True)
class ExecutionPlan(JsonRecord):
    """Deterministic metadata preflight; it never asserts live executability."""

    schema_version: Literal["1.0"]
    model_id: str
    backend_id: str
    task: Literal["detection_3d"]
    sample_id: str
    payload_kind: str
    runtime_target: str
    precision: str
    device_target: str
    runtime_id: str
    task_id: str
    execution_commit: str
    deterministic: bool
    validation: ValidationReport
    context_errors: tuple[str, ...]
    context_warnings: tuple[str, ...]
    runtime_requirements: tuple[str, ...]
    dependency_expectations: tuple[str, ...]
    dependency_state: Literal["not_imported_not_verified"]
    artifacts: tuple[ArtifactReference, ...]
    artifact_verification_state: Literal["not_checked_dry_run"]
    authorization_state: str
    authorization_verified: Literal[False]
    ready_for_guarded_initialization: bool
    backend_load_required: Literal[True]
    hardware_probed: Literal[False]
    executable: Literal[False]


def _resource_errors(
    description: BackendDescription,
    resources: ModelResources,
) -> list[str]:
    errors: list[str] = []
    if description.model_id == "pointpillars-nuscenes-v0.3":
        if resources.config_path is None:
            errors.append("PointPillars config path is required")
        if resources.checkpoint_path is None:
            errors.append("PointPillars checkpoint path is required")
        if resources.upstream_root is None:
            errors.append("PointPillars pinned upstream checkout root is required")
    else:
        if resources.candidate_manifest_path is None:
            errors.append("M8 candidate manifest path is required")
    return errors


def _authorization_findings(
    description: BackendDescription,
    context: ExecutionContext,
) -> tuple[list[str], str]:
    reference = context.authorization
    if description.model_id == "dsvt-pillar-transfusion-m8":
        errors = [
            "generic DSVT execution is unavailable until the frozen S1 runner supplies an "
            "authorization-, policy-, input-, and accounting-bound session"
        ]
        if reference is None or reference.kind != "m8_s1":
            errors.insert(0, "M8 requires its existing mode/pass-scoped S1 authorization artifact")
            return errors, "missing_m8_s1_authorization"
        return errors, "reference_supplied_live_and_accounting_checks_pending"
    if reference is None or reference.kind != "generic_prediction":
        return (
            ["PointPillars prediction requires a hashed owner authorization artifact"],
            "missing_generic_prediction_authorization",
        )
    return [], "reference_supplied_exact_file_and_resource_checks_pending"


def plan_execution(
    registry: ModelRegistry,
    model_id: str,
    description: InputDescription,
    context: ExecutionContext,
    resources: ModelResources,
) -> ExecutionPlan:
    """Create a stable, CPU-only plan without file hashing or runtime discovery."""

    manifest = registry.get(model_id)
    backend = backend_description(model_id)
    report = validate_execution_input(
        manifest,
        description.features,
        coordinates=description.coordinates,
        temporal=description.temporal,
        payload_kind=description.payload_kind,
        expected_payload_kind=backend.payload_kind,
    )
    precisions = tuple(Precision(value) for value in backend.supported_precisions)
    errors = list(
        execution_context_errors(
            manifest,
            context,
            supported_precisions=precisions,
            required_device="cuda:0",
        )
    )
    errors.extend(_resource_errors(backend, resources))
    authorization_errors, authorization_state = _authorization_findings(backend, context)
    errors.extend(authorization_errors)
    warnings = (
        "dry run did not read or hash authorization and model artifacts",
        "dry run did not import or verify optional runtime dependencies",
        "hardware and live runtime policy were not inspected",
    )
    ready = report.valid and not errors
    return ExecutionPlan(
        "1.0",
        manifest.model_id,
        backend.backend_id,
        "detection_3d",
        description.sample_id,
        description.payload_kind,
        context.runtime_target.value,
        context.precision.value,
        context.device_target,
        context.runtime_id,
        context.task_id,
        context.execution_commit,
        context.deterministic,
        report,
        tuple(errors),
        warnings,
        manifest.runtime_requirements,
        backend.dependency_expectations,
        "not_imported_not_verified",
        manifest.artifacts,
        "not_checked_dry_run",
        authorization_state,
        False,
        ready,
        True,
        False,
        False,
    )
