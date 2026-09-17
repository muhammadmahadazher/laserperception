"""Static backend descriptions; this module never imports detector adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..serialization import JsonRecord


@dataclass(frozen=True)
class BackendDescription(JsonRecord):
    schema_version: Literal["1.0"]
    backend_id: str
    model_id: str
    adapter_module: str
    adapter_factory: str
    payload_kind: Literal["model_ready_xyzt", "m8_xyzit"]
    supported_precisions: tuple[Literal["fp32", "fp16"], ...]
    dependency_expectations: tuple[str, ...]
    execution_policy: str


_BACKENDS = (
    BackendDescription(
        "1.0",
        "mmdetection3d-pointpillars-fp32",
        "pointpillars-nuscenes-v0.3",
        "laserperception.perception.backends.pointpillars",
        "PointPillarsAdapter",
        "model_ready_xyzt",
        ("fp32",),
        ("MMDetection3D 1.4.0", "PyTorch 2.1.0+cu118", "CUDA 11.8"),
        "hashed owner authorization before lazy backend import",
    ),
    BackendDescription(
        "1.0",
        "openpcdet-dsvt-m8-accounted",
        "dsvt-pillar-transfusion-m8",
        "laserperception.perception.backends.dsvt",
        "DsvtAdapter",
        "m8_xyzit",
        ("fp32",),
        (
            "OpenPCDet/DSVT 0.6.0+8cfc2a6",
            "PyTorch 2.1.0+cu118",
            "CUDA 11.8",
            "spconv 2.3.8",
            "torch-scatter 2.1.2+pt21cu118",
        ),
        "frozen M8 runner authorization, live policy, input identity, and call accounting",
    ),
)


def list_backend_descriptions() -> tuple[BackendDescription, ...]:
    """List reviewed adapters in deterministic model-ID order."""

    return tuple(sorted(_BACKENDS, key=lambda item: item.model_id))


def backend_description(model_id: str) -> BackendDescription:
    """Return static adapter metadata without importing its implementation."""

    for description in _BACKENDS:
        if description.model_id == model_id:
            return description
    raise ValueError(f"no detection backend is registered for model {model_id!r}")
