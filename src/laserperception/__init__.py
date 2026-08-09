"""Public package interface for LaserPerception."""

from importlib.metadata import PackageNotFoundError, version

from laserperception.core import PointCloud

try:
    __version__ = version("laserperception")
except PackageNotFoundError:  # pragma: no cover - only occurs for an uninstalled source tree
    __version__ = "0.1.0.dev0"

__all__ = ["PointCloud", "__version__"]
