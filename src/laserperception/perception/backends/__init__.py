"""Lightweight backend contracts and static discovery."""

from .base import BackendUnavailableError, DetectionBackend
from .catalog import BackendDescription, backend_description, list_backend_descriptions

__all__ = [
    "BackendDescription",
    "BackendUnavailableError",
    "DetectionBackend",
    "backend_description",
    "list_backend_descriptions",
]
