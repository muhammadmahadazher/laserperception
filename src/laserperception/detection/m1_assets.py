"""Portable cache-path resolution for the pinned M1 assets."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

M1_CACHE_ENVIRONMENT_VARIABLE = "LASERPERCEPTION_M1_CACHE"


@dataclass(frozen=True, slots=True)
class M1AssetPaths:
    """Resolved external paths for the M1 framework checkout and checkpoint."""

    cache_root: Path
    mmdet3d_root: Path
    checkpoint_directory: Path
    checkpoint_path: Path


def resolve_m1_asset_paths(
    manifest: Mapping[str, object],
    *,
    environ: Mapping[str, str] | None = None,
) -> M1AssetPaths:
    """Resolve M1 cache assets from the environment and relative manifest entries.

    ``LASERPERCEPTION_M1_CACHE`` takes precedence. An unset or empty variable uses
    the manifest's portable ``~/.cache/laserperception`` default.
    """

    cache = _required_mapping(manifest, "cache")
    model = _required_mapping(manifest, "model")
    checkpoint = _required_mapping(model, "checkpoint")
    environment_variable = _required_string(cache, "root_environment_variable")
    if environment_variable != M1_CACHE_ENVIRONMENT_VARIABLE:
        raise ValueError(
            f"M1 manifest cache.root_environment_variable must be {M1_CACHE_ENVIRONMENT_VARIABLE}"
        )

    environment = os.environ if environ is None else environ
    configured_root = environment.get(M1_CACHE_ENVIRONMENT_VARIABLE)
    root_value = (
        configured_root
        if configured_root is not None and configured_root.strip()
        else _required_string(cache, "default_root")
    )
    cache_root = Path(root_value).expanduser().resolve()
    mmdet3d_relative = _required_relative_path(cache, "mmdet3d_checkout_relative")
    checkpoint_directory_relative = _required_relative_path(cache, "checkpoint_directory_relative")
    checkpoint_filename = _required_string(checkpoint, "filename")
    if Path(checkpoint_filename).name != checkpoint_filename:
        raise ValueError("M1 checkpoint filename must not contain directory components")

    checkpoint_directory = cache_root / checkpoint_directory_relative
    return M1AssetPaths(
        cache_root=cache_root,
        mmdet3d_root=cache_root / mmdet3d_relative,
        checkpoint_directory=checkpoint_directory,
        checkpoint_path=checkpoint_directory / checkpoint_filename,
    )


def _required_mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"M1 manifest {key} must be a mapping")
    return value


def _required_string(parent: Mapping[str, object], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"M1 manifest {key} must be a non-empty string")
    return value


def _required_relative_path(parent: Mapping[str, object], key: str) -> Path:
    value = _required_string(parent, key)
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"M1 manifest {key} must be a safe relative path")
    return path
