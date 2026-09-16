"""Obvious non-scientific CPU fake used only by tests and examples."""

from __future__ import annotations

from laserperception.detection.types import DetectionFrame

from ..inputs import PerceptionInput
from ..manifests import ModelManifest
from ..validation import ValidationReport, validate_execution_input
from .base import PreparedInput, require_prepared_input


class FakeDetectionBackend:
    """Return an empty marked frame while exercising the generic lifecycle."""

    def __init__(self, manifest: ModelManifest, *, payload_kind: str = "point_cloud") -> None:
        self._manifest = manifest
        self._payload_kind = payload_kind
        self._owner_token = object()
        self._closed = False
        self._events: list[str] = []

    @property
    def events(self) -> tuple[str, ...]:
        return tuple(self._events)

    def describe(self) -> ModelManifest:
        return self._manifest

    def validate_input(self, value: PerceptionInput) -> ValidationReport:
        return validate_execution_input(
            self._manifest,
            value.description.features,
            coordinates=value.description.coordinates,
            temporal=value.description.temporal,
            payload_kind=value.description.payload_kind,
            expected_payload_kind=self._payload_kind,
        )

    def prepare(self, value: PerceptionInput) -> PreparedInput:
        if self._closed:
            raise RuntimeError("backend is closed")
        report = self.validate_input(value)
        if not report.valid:
            raise ValueError("fake input is incompatible: " + "; ".join(report.errors))
        self._events.append("prepare")
        return PreparedInput(self._manifest.model_id, value, value.payload, self._owner_token)

    def predict(self, value: PreparedInput) -> DetectionFrame:
        if self._closed:
            raise RuntimeError("backend is closed")
        require_prepared_input(
            value,
            backend_id=self._manifest.model_id,
            owner_token=self._owner_token,
        )
        self._events.append("predict")
        return DetectionFrame(
            detections=(),
            sample_id=value.source.description.sample_id,
            coordinate_frame=value.source.description.frame_id,
            metadata={"backend": "fake", "non_scientific": True},
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._events.append("close")
