"""Unified discovery and CPU ingestion wrappers for implemented readers."""

from .contracts import (
    DataAdapter,
    DataAdapterManifest,
    DatasetSequence,
    InputInspection,
    InputOptions,
    LoadedInput,
    PointAttributeInfo,
    SampleRef,
)
from .inspection import inspect_input, inspect_loaded_input
from .io import FileDataAdapter, load_input, load_pointcloud2_xyz, select_adapter
from .registry import DataAdapterRegistry, registry

__all__ = [
    "DataAdapter",
    "DataAdapterManifest",
    "DatasetSequence",
    "InputInspection",
    "InputOptions",
    "LoadedInput",
    "PointAttributeInfo",
    "SampleRef",
    "FileDataAdapter",
    "load_input",
    "load_pointcloud2_xyz",
    "select_adapter",
    "inspect_input",
    "inspect_loaded_input",
    "DataAdapterRegistry",
    "registry",
]
