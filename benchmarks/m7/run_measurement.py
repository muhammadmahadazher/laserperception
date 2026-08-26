"""Future M7 measurement entrypoint with an owner-authorization hard barrier."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from benchmarks.m7.execution import (
    ExecutionIdentity,
    RuntimeArtifacts,
    load_inference_authorization,
    run_authorized,
)

TDetector = TypeVar("TDetector")
TResult = TypeVar("TResult")


def run_measurement(
    authorization_path: str | Path,
    expected: ExecutionIdentity,
    artifacts: RuntimeArtifacts,
    detector_factory: Callable[[], TDetector],
    execute: Callable[[TDetector], TResult],
) -> TResult:
    """Verify owner authorization before lazily reaching any detector/runtime factory."""

    authorization = load_inference_authorization(authorization_path, expected)
    artifacts.verify(expected)
    return run_authorized(authorization, expected, detector_factory, execute)
