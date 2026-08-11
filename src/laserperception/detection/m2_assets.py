"""Portable cache-path resolution for external M2 deployment assets."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

M2_CACHE_ENVIRONMENT_VARIABLE = "LASERPERCEPTION_M2_CACHE"


@dataclass(frozen=True, slots=True)
class M2AssetPaths:
    """Resolved external paths for the MMDeploy checkout and generated artifacts."""

    cache_root: Path
    mmdeploy_root: Path
    artifact_directory: Path
    engine_directory: Path


def resolve_m2_asset_paths(
    manifest: Mapping[str, object],
    *,
    environ: Mapping[str, str] | None = None,
) -> M2AssetPaths:
    """Resolve M2 cache paths without placing generated binaries in the repository."""

    cache = _required_mapping(manifest, "cache")
    environment_variable = _required_string(cache, "root_environment_variable")
    if environment_variable != M2_CACHE_ENVIRONMENT_VARIABLE:
        raise ValueError(
            f"M2 manifest cache.root_environment_variable must be {M2_CACHE_ENVIRONMENT_VARIABLE}"
        )

    environment = os.environ if environ is None else environ
    configured_root = environment.get(M2_CACHE_ENVIRONMENT_VARIABLE)
    root_value = (
        configured_root
        if configured_root is not None and configured_root.strip()
        else _required_string(cache, "default_root")
    )
    cache_root = Path(root_value).expanduser().resolve()
    artifact_directory = cache_root / _required_relative_path(cache, "artifact_directory_relative")
    return M2AssetPaths(
        cache_root=cache_root,
        mmdeploy_root=cache_root / _required_relative_path(cache, "mmdeploy_checkout_relative"),
        artifact_directory=artifact_directory,
        engine_directory=artifact_directory
        / _required_relative_path(cache, "engine_directory_relative"),
    )


def _required_mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"M2 manifest {key} must be a mapping")
    return value


def _required_string(parent: Mapping[str, object], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"M2 manifest {key} must be a non-empty string")
    return value


def _required_relative_path(parent: Mapping[str, object], key: str) -> Path:
    value = _required_string(parent, key)
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"M2 manifest {key} must be a safe relative path")
    return path
