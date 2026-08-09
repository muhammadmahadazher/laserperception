"""LAS and optional LAZ loading through laspy."""

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np

from laserperception.core import PointCloud

_GEOMETRY_DIMENSIONS = {"X", "Y", "Z"}


def load_las(path: str | Path) -> PointCloud:
    """Load a LAS/LAZ file without coordinate normalization.

    Scaled LAS coordinates become canonical ``float32`` geometry. Classification is exposed as
    ``PointCloud.labels``; other stored point dimensions remain separate point attributes. LAZ
    requires an installed laspy backend such as the ``laserperception[laz]`` extra.
    """
    las_path = Path(path)
    if las_path.suffix.lower() not in {".las", ".laz"}:
        raise ValueError(f"expected a .las or .laz file; received {las_path.name!r}")

    las = laspy.read(las_path)
    xyz = np.column_stack(
        (
            np.asarray(las.x, dtype=np.float64),
            np.asarray(las.y, dtype=np.float64),
            np.asarray(las.z, dtype=np.float64),
        )
    )

    dimension_names = tuple(str(name) for name in las.point_format.dimension_names)
    attributes: dict[str, np.ndarray] = {}
    for name in dimension_names:
        if name in _GEOMETRY_DIMENSIONS or name == "classification":
            continue
        attributes[name] = np.array(las[name], copy=True)

    labels = (
        np.array(las.classification, copy=True) if "classification" in dimension_names else None
    )
    crs = las.header.parse_crs()
    metadata: dict[str, object] = {
        "format": las_path.suffix.lower().lstrip("."),
        "source_file": las_path.name,
        "las_version": str(las.header.version),
        "point_format_id": int(las.header.point_format.id),
        "scales": tuple(float(value) for value in las.header.scales),
        "offsets": tuple(float(value) for value in las.header.offsets),
        "available_dimensions": dimension_names,
        "coordinates_normalized": False,
    }
    if crs is not None:
        metadata["crs_wkt"] = crs.to_wkt()

    return PointCloud(xyz=xyz, labels=labels, attributes=attributes, metadata=metadata)
