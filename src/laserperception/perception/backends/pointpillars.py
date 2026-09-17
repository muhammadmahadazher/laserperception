"""Thin, authorized delegate to the existing MMDetection3D PointPillars path."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Protocol, cast

from laserperception.detection.ros2_contract import ModelReadyPointCloud
from laserperception.detection.types import DetectionFrame

from ..execution import (
    POINTPILLARS_CHECKPOINT_SHA256,
    ExecutionContext,
    ModelResources,
    Precision,
    execution_context_errors,
    require_canonical_execution_manifest,
    verify_generic_prediction_authorization,
)
from ..inputs import PerceptionInput
from ..manifests import ModelManifest
from ..validation import ValidationReport, validate_execution_input
from .base import BackendUnavailableError, PreparedInput, require_prepared_input


class PointPillarsDelegate(Protocol):
    def prepare_model_ready_points(
        self, points: object, *, sample_id: str, coordinate_frame: str
    ) -> object: ...

    def run_prepared(self, sample: object) -> DetectionFrame: ...


class PointPillarsAdapter:
    """Preserve the historical eager FP32 prepare-and-run path exactly."""

    def __init__(
        self,
        manifest: ModelManifest,
        context: ExecutionContext,
        resources: ModelResources,
    ) -> None:
        self._manifest = manifest
        self._context = context
        self._resources = resources
        self._delegate: PointPillarsDelegate | None = None
        self._owner_token = object()
        self._closed = False

    def describe(self) -> ModelManifest:
        return self._manifest

    def validate_input(self, value: PerceptionInput) -> ValidationReport:
        return validate_execution_input(
            self._manifest,
            value.description.features,
            coordinates=value.description.coordinates,
            temporal=value.description.temporal,
            payload_kind=value.description.payload_kind,
            expected_payload_kind="model_ready_xyzt",
        )

    def _require_static_preconditions(self, value: PerceptionInput) -> ModelReadyPointCloud:
        require_canonical_execution_manifest(self._manifest)
        report = self.validate_input(value)
        if not report.valid:
            raise ValueError("PointPillars input is incompatible: " + "; ".join(report.errors))
        if not isinstance(value.payload, ModelReadyPointCloud):
            raise TypeError("PointPillars execution requires ModelReadyPointCloud")
        context_errors = execution_context_errors(
            self._manifest,
            self._context,
            supported_precisions=(Precision.FP32,),
            required_device="cuda:0",
        )
        if context_errors:
            raise ValueError(
                "PointPillars execution context is incompatible: " + "; ".join(context_errors)
            )
        verify_generic_prediction_authorization(
            self._context,
            self._manifest,
            self._resources,
            value.description,
            input_sha256=value.payload.sha256,
        )
        return value.payload

    def prepare(self, value: PerceptionInput) -> PreparedInput:
        if self._closed:
            raise RuntimeError("backend is closed")
        payload = self._require_static_preconditions(value)
        if self._delegate is None:
            self._delegate = _load_existing_backend(self._context, self._resources)
        prepared = self._delegate.prepare_model_ready_points(
            payload,
            sample_id=value.description.sample_id,
            coordinate_frame=value.description.frame_id,
        )
        return PreparedInput(self._manifest.model_id, value, prepared, self._owner_token)

    def predict(self, value: PreparedInput) -> DetectionFrame:
        if self._closed or self._delegate is None:
            raise RuntimeError("backend must be prepared before prediction")
        require_prepared_input(
            value,
            backend_id=self._manifest.model_id,
            owner_token=self._owner_token,
        )
        result = self._delegate.run_prepared(value.payload)
        if not isinstance(result, DetectionFrame):
            raise TypeError("PointPillars delegate did not return DetectionFrame")
        return result

    def close(self) -> None:
        if self._closed:
            return
        delegate = self._delegate
        self._delegate = None
        self._closed = True
        if delegate is not None:
            close = getattr(delegate, "close", None)
            if callable(close):
                cast(Callable[[], None], close)()


def _load_existing_backend(
    context: ExecutionContext,
    resources: ModelResources,
) -> PointPillarsDelegate:
    if resources.config_path is None or resources.checkpoint_path is None:
        raise BackendUnavailableError("PointPillars config and checkpoint paths are required")
    try:
        module = importlib.import_module("laserperception.detection.mmdet3d_backend")
        backend_type = module.Mmdet3dBackend
    except (AttributeError, ImportError, OSError) as error:
        raise BackendUnavailableError(
            "the historical MMDetection3D PointPillars backend is unavailable"
        ) from error
    return cast(
        PointPillarsDelegate,
        backend_type(
            resources.config_path,
            resources.checkpoint_path,
            checkpoint_sha256=POINTPILLARS_CHECKPOINT_SHA256,
            device=context.device_target,
        ),
    )
