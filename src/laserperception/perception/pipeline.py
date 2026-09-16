"""Task-specific detection pipeline with framework-independent output."""

from __future__ import annotations

from laserperception.detection.types import DetectionFrame

from .backends.base import DetectionBackend
from .inputs import PerceptionInput


class DetectionPipeline:
    """Validate, prepare, predict, and own one backend lifecycle."""

    def __init__(self, backend: DetectionBackend) -> None:
        if not isinstance(backend, DetectionBackend):
            raise TypeError("DetectionPipeline requires a DetectionBackend")
        self._backend = backend
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def predict(self, value: PerceptionInput) -> DetectionFrame:
        if self._closed:
            raise RuntimeError("pipeline is closed")
        try:
            report = self._backend.validate_input(value)
            if not report.valid:
                raise ValueError("detection input is incompatible: " + "; ".join(report.errors))
            prepared = self._backend.prepare(value)
            result = self._backend.predict(prepared)
            if not isinstance(result, DetectionFrame):
                raise TypeError("detection backend did not return DetectionFrame")
            return result
        except Exception as error:
            try:
                self.close()
            except Exception as cleanup_error:
                raise error from cleanup_error
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._backend.close()

    def __enter__(self) -> DetectionPipeline:
        if self._closed:
            raise RuntimeError("pipeline is closed")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
