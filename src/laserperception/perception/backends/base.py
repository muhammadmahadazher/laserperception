"""Narrow detection backend contract shared by the generic pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from laserperception.detection.types import DetectionFrame

from ..inputs import PerceptionInput
from ..manifests import ModelManifest
from ..validation import ValidationReport


@dataclass(frozen=True)
class PreparedInput:
    """Opaque prepared value minted by one backend instance."""

    backend_id: str
    source: PerceptionInput
    payload: object
    _owner_token: object = field(repr=False, compare=False)


@runtime_checkable
class DetectionBackend(Protocol):
    def describe(self) -> ModelManifest: ...

    def validate_input(self, value: PerceptionInput) -> ValidationReport: ...

    def prepare(self, value: PerceptionInput) -> PreparedInput: ...

    def predict(self, value: PreparedInput) -> DetectionFrame: ...

    def close(self) -> None: ...


class BackendUnavailableError(RuntimeError):
    """The explicitly selected backend cannot run in the supplied context."""


def require_prepared_input(
    value: PreparedInput,
    *,
    backend_id: str,
    owner_token: object,
) -> None:
    """Reject handles not minted by this exact backend instance."""

    if not isinstance(value, PreparedInput):
        raise TypeError("predict requires a PreparedInput")
    if value.backend_id != backend_id or value._owner_token is not owner_token:
        raise ValueError("prepared input was not created by this backend instance")
