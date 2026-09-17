"""Metadata compatibility reports; never initialize a detector or synthesize data."""

from pathlib import Path

from laserperception.perception import registry as model_registry
from laserperception.perception.contracts import FeatureSpec, TemporalContract
from laserperception.perception.validation import validate_input

from .contracts import InputInspection, InputOptions, LoadedInput, PointAttributeInfo
from .io import load_input


def inspect_loaded_input(loaded: LoadedInput, *, model_id: str | None = None) -> InputInspection:
    cloud = loaded.cloud
    attributes = (PointAttributeInfo("xyz", str(cloud.xyz.dtype), cloud.xyz.shape),) + tuple(
        PointAttributeInfo(name, str(values.dtype), values.shape)
        for name, values in sorted(cloud.attributes.items())
    )
    specs = []
    lidar = loaded.adapter.adapter_id in (
        "kitti-velodyne",
        "semantickitti",
        "kitti-raw",
        "nuscenes-lidar-top",
    )
    message = loaded.adapter.adapter_id == "pointcloud2-xyz"
    for i, (name, axis) in enumerate(zip(("x", "y", "z"), ("forward", "left", "up"), strict=True)):
        specs.append(
            FeatureSpec(
                name,
                "float32",
                True,
                "metre" if lidar else None,
                f"{axis} coordinate"
                if lidar
                else "message coordinate; axes and units unverified"
                if message
                else "stored scaled coordinate; axes unverified",
                i,
                "reject",
                "raw acquisition" if lidar else "message acquisition" if message else "stored file",
            )
        )
    limitations = list(loaded.adapter.limitations)
    for name, values in sorted(cloud.attributes.items()):
        dtype = str(values.dtype)
        if dtype not in ("float32", "float64", "int32", "uint16") or values.ndim != 1:
            limitations.append(
                f"attribute {name} retained but not representable as a model FeatureSpec"
            )
            continue
        # Actual semantics are retained; no model manifest is copied to manufacture compatibility.
        semantics = (
            "raw reflectance; no normalization or synthesis"
            if name == "intensity" and loaded.adapter.adapter_id == "nuscenes-lidar-top"
            else f"stored {name}; instrument interpretation unverified"
        )
        specs.append(
            FeatureSpec.from_dict(
                {
                    "name": name,
                    "dtype": dtype,
                    "required": False,
                    "unit": None,
                    "semantics": semantics,
                    "position": 3 if lidar and name in ("intensity", "remission") else None,
                    "missing_policy": "omit",
                    "source": "raw acquisition"
                    if lidar
                    else "message acquisition"
                    if message
                    else "stored file",
                }
            )
        )
    temporal = TemporalContract(
        "single_scan",
        0,
        0,
        "nanosecond" if loaded.timestamp_ns is not None else "unavailable",
        "raw single acquisition",
        None,
        "no history",
        False,
    )
    report = (
        validate_input(model_registry.get(model_id), tuple(specs), temporal=temporal)
        if model_id
        else None
    )
    return InputInspection(
        loaded.adapter,
        loaded.sample_id,
        len(cloud),
        attributes,
        cloud.labels is not None,
        str(cloud.labels.dtype) if cloud.labels is not None else None,
        loaded.adapter.coordinate_semantics,
        loaded.timestamp_ns,
        model_id,
        report,
        False,
        tuple(limitations) + ("Complete model coordinate contract has not been verified.",),
    )


def inspect_input(
    path: str | Path,
    *,
    adapter_id: str | None = None,
    options: InputOptions | None = None,
    model_id: str | None = None,
) -> InputInspection:
    return inspect_loaded_input(
        load_input(path, adapter_id=adapter_id, options=options), model_id=model_id
    )
