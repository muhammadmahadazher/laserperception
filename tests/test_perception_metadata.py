"""Synthetic CPU contracts and discovery; no detector initialization."""

import importlib
import importlib.abc
import json
import sys
from dataclasses import FrozenInstanceError, replace

import pytest

from laserperception.cli import main
from laserperception.perception import (
    ModelManifest,
    ModelRegistry,
    PerceptionTask,
    registry,
    validate_input,
)


@pytest.mark.parametrize("manifest", registry.list_models(), ids=lambda m: m.model_id)
def test_manifest_roundtrip_and_immutability(manifest):
    assert ModelManifest.from_json(manifest.to_json()) == manifest
    assert manifest.to_json() == ModelManifest.from_dict(manifest.to_dict()).to_json()
    with pytest.raises(FrozenInstanceError):
        manifest.model_id = "changed"
    assert validate_input(
        manifest,
        manifest.input_features,
        coordinates=manifest.coordinates,
        temporal=manifest.temporal,
    ).valid


@pytest.mark.parametrize("mutation", ["missing", "extra", "schema", "bool_size", "hash", "feature"])
def test_invalid_manifest(mutation):
    raw = registry.list_models()[0].to_dict()
    if mutation == "missing":
        del raw["upstream"]
    elif mutation == "extra":
        raw["surprise"] = True
    elif mutation == "schema":
        raw["schema_version"] = "2.0"
    elif mutation == "bool_size":
        raw["artifacts"][0]["byte_size"] = True
    elif mutation == "hash":
        raw["artifacts"][0]["sha256"] = "bad"
    else:
        raw["input_features"][0]["position"] = -1
    with pytest.raises(ValueError):
        ModelManifest.from_dict(raw)


def test_registry_lookup_filter_order_and_duplicates():
    models = registry.list_models()
    assert models == ModelRegistry(reversed(models)).list_models()
    assert list(m.model_id for m in models) == sorted(m.model_id for m in models)
    assert registry.filter_by_task(PerceptionTask.DETECTION_3D) == models
    assert registry.filter_by_task(PerceptionTask.TRACKING) == ()
    assert registry.filter_by_runtime("cpu") == ()
    assert registry.inspect(models[0].model_id) == models[0].to_dict()
    with pytest.raises(ValueError, match="duplicate"):
        ModelRegistry([models[0], models[0]])
    with pytest.raises(ValueError, match="unknown model"):
        registry.get("unknown")


@pytest.mark.parametrize("missing", ["intensity", "time_lag"])
def test_missing_m8_features(missing):
    m = registry.get("dsvt-pillar-transfusion-m8")
    report = validate_input(m, tuple(f for f in m.input_features if f.name != missing))
    assert not report.valid
    assert report.missing_required_features == (missing,)


def test_coordinate_temporal_and_feature_mismatch():
    m = registry.list_models()[0]
    report = validate_input(
        m,
        (replace(m.input_features[0], dtype="float64"), *m.input_features[1:]),
        coordinates=replace(m.coordinates, handedness="left"),
        temporal=replace(m.temporal, max_history=20),
    )
    assert not report.valid
    assert report.coordinate_incompatibilities == ("handedness",)
    assert report.temporal_incompatibilities == ("history_range",)
    assert any("dtype" in e for e in report.errors)


def test_duplicate_json_key_and_nonfinite():
    with pytest.raises(ValueError, match="duplicate"):
        ModelManifest.from_json('{"model_id":"a","model_id":"b"}')
    raw = registry.list_models()[0].to_dict()
    raw["artifacts"][0]["byte_size"] = float("nan")
    with pytest.raises(ValueError):
        ModelManifest.from_dict(raw)


@pytest.mark.parametrize(
    "argv",
    [
        ["models", "list"],
        ["models", "list", "--json"],
        ["models", "inspect", "dsvt-pillar-transfusion-m8"],
        ["models", "inspect", "pointpillars-nuscenes-v0.3", "--json"],
    ],
)
def test_cli(argv, capsys):
    assert main(argv) == 0
    output = capsys.readouterr().out
    assert "detection_3d" in output
    if "--json" in argv:
        assert json.loads(output)


def test_discovery_import_guard(monkeypatch):
    forbidden = {
        "torch",
        "mmdet",
        "mmdet3d",
        "mmengine",
        "tensorrt",
        "rclpy",
        "pcdet",
        "spconv",
        "torch_scatter",
        "pycuda",
        "cupy",
    }

    class Guard(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in forbidden:
                raise AssertionError(f"metadata attempted optional import: {fullname}")

    for name in tuple(sys.modules):
        if name.startswith("laserperception.perception"):
            monkeypatch.delitem(sys.modules, name)
    guard = Guard()
    sys.meta_path.insert(0, guard)
    try:
        package = importlib.import_module("laserperception.perception")
        assert len(package.registry.list_models()) == 2
        assert package.registry.inspect("dsvt-pillar-transfusion-m8")
    finally:
        sys.meta_path.remove(guard)
