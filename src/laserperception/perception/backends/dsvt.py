"""M8 adapter accepts only a session minted by the frozen accounted runner."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Protocol

from laserperception.detection.m8_input import M8PointCloud
from laserperception.detection.types import DetectionFrame

from ..execution import (
    ExecutionContext,
    ModelResources,
    Precision,
    execution_context_errors,
    file_sha256,
    require_canonical_execution_manifest,
)
from ..inputs import PerceptionInput
from ..manifests import ModelManifest
from ..validation import ValidationReport, validate_execution_input
from .base import BackendUnavailableError, PreparedInput, require_prepared_input

M8_ACCOUNTING_IDENTITY = "laserperception.m8.s1.accounted-session.v1"


def _import_m8_runtime() -> ModuleType:
    return importlib.import_module("laserperception.detection.m8_s1_runtime")


@dataclass(frozen=True)
class AccountedSessionBinding:
    """Bindings asserted by a session created inside the frozen S1 runner."""

    accounting_identity: str
    model_id: str
    runtime_id: str
    task_id: str
    execution_commit: str
    mode: str
    logical_pass_id: str
    authorization_sha256: str
    runtime_policy_sha256: str
    candidate_manifest_sha256: str


class AccountedDsvtSession(Protocol):
    @property
    def binding(self) -> AccountedSessionBinding: ...

    def predict_accounted(self, value: PerceptionInput) -> DetectionFrame: ...

    def close(self) -> None: ...


class DsvtAdapter:
    """Delegate only to an authorization-, input-, and call-accounted S1 session."""

    def __init__(
        self,
        manifest: ModelManifest,
        context: ExecutionContext,
        resources: ModelResources,
    ) -> None:
        self._manifest = manifest
        self._context = context
        self._resources = resources
        self._session: AccountedDsvtSession | None = None
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
            expected_payload_kind="m8_xyzit",
        )

    def _require_static_preconditions(self, value: PerceptionInput) -> M8PointCloud:
        require_canonical_execution_manifest(self._manifest)
        report = self.validate_input(value)
        if not report.valid:
            raise ValueError("M8 input is incompatible: " + "; ".join(report.errors))
        if not isinstance(value.payload, M8PointCloud):
            raise TypeError("M8 execution requires M8PointCloud")
        context_errors = execution_context_errors(
            self._manifest,
            self._context,
            supported_precisions=(Precision.FP32,),
            required_device="cuda:0",
        )
        if context_errors:
            raise ValueError("M8 execution context is incompatible: " + "; ".join(context_errors))
        return value.payload

    def _verified_authorization(self) -> tuple[dict[str, object], str, str]:
        reference = self._context.authorization
        if reference is None or reference.kind != "m8_s1":
            raise ValueError("M8 requires its existing mode/pass-scoped S1 authorization")
        if not reference.path.is_file() or file_sha256(reference.path) != reference.sha256:
            raise ValueError("M8 authorization file identity mismatch")
        if reference.mode is None or reference.logical_pass_id is None:
            raise AssertionError("validated M8 authorization lost its mode or pass")
        if reference.runtime_policy_path is None:
            raise AssertionError("validated M8 authorization lost its runtime policy path")
        candidate_path = self._resources.candidate_manifest_path
        if candidate_path is None or not candidate_path.is_file():
            raise ValueError("M8 candidate manifest is missing")

        runtime = _import_m8_runtime()
        candidate_sha256 = runtime.CANDIDATE_MANIFEST_SHA256
        if file_sha256(candidate_path) != candidate_sha256:
            raise ValueError("M8 candidate manifest SHA256 mismatch")
        payload_object: object = runtime.require_scientific_authorization(
            reference.mode,
            reference.logical_pass_id,
            reference.path,
            runtime.AuthorizationIdentity(self._context.execution_commit),
        )
        if not isinstance(payload_object, Mapping):
            raise TypeError("M8 authorization verifier returned a non-mapping")
        payload = dict(payload_object)
        runtime_policy_sha256 = payload.get("runtime_policy_binding_sha256")
        if not isinstance(runtime_policy_sha256, str):
            raise ValueError("M8 authorization lacks its runtime-policy binding SHA256")
        if (
            not reference.runtime_policy_path.is_file()
            or file_sha256(reference.runtime_policy_path) != runtime_policy_sha256
        ):
            raise ValueError("M8 runtime-policy binding file identity mismatch")
        return payload, runtime_policy_sha256, candidate_sha256

    def prepare(self, value: PerceptionInput) -> PreparedInput:
        if self._closed:
            raise RuntimeError("backend is closed")
        self._require_static_preconditions(value)
        authorization, runtime_policy_sha256, candidate_sha256 = self._verified_authorization()
        reference = self._context.authorization
        assert reference is not None
        assert reference.mode is not None
        assert reference.logical_pass_id is not None
        if self._session is None:
            session = _open_accounted_session(
                self._context,
                self._resources,
                authorization,
            )
            expected = AccountedSessionBinding(
                M8_ACCOUNTING_IDENTITY,
                self._manifest.model_id,
                self._context.runtime_id,
                self._context.task_id,
                self._context.execution_commit,
                reference.mode,
                reference.logical_pass_id,
                reference.sha256,
                runtime_policy_sha256,
                candidate_sha256,
            )
            if session.binding != expected:
                session.close()
                raise ValueError("M8 session bindings do not match this exact request")
            self._session = session
        return PreparedInput(self._manifest.model_id, value, value.payload, self._owner_token)

    def predict(self, value: PreparedInput) -> DetectionFrame:
        if self._closed or self._session is None:
            raise RuntimeError("backend must be prepared before prediction")
        require_prepared_input(
            value,
            backend_id=self._manifest.model_id,
            owner_token=self._owner_token,
        )
        if value.source.payload is not value.payload:
            raise ValueError("M8 prepared payload identity changed")
        result = self._session.predict_accounted(value.source)
        if not isinstance(result, DetectionFrame):
            raise TypeError("M8 accounted session did not return DetectionFrame")
        return result

    def close(self) -> None:
        if self._closed:
            return
        session = self._session
        self._session = None
        self._closed = True
        if session is not None:
            session.close()


def _open_accounted_session(
    context: ExecutionContext,
    resources: ModelResources,
    authorization: Mapping[str, object],
) -> AccountedDsvtSession:
    """Fail closed until the frozen S1 runner owns this session bridge."""

    raise BackendUnavailableError(
        "M8 generic prediction is disabled: only the frozen S1 runner may supply a session after "
        "static binding, exact owner authorization, live runtime-policy verification, canonical "
        "input selection, and AtomicAttempt accounting"
    )
