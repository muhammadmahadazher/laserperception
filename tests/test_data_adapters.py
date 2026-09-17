"""Synthetic CPU ingestion/discovery and truthful compatibility regressions."""

import builtins
import importlib
import json
import subprocess
import sys
from dataclasses import replace

import laspy
import numpy as np
import pytest

from laserperception.cli import main
from laserperception.data import (
    DataAdapterRegistry,
    FileDataAdapter,
    InputOptions,
    inspect_input,
    load_input,
    load_pointcloud2_xyz,
    registry,
    select_adapter,
)
from laserperception.data.sequences import (
    DalesSequence,
    KittiRawDataSequence,
    SemanticKITTISequence,
)
from laserperception.datasets.dales import DalesDataset
from laserperception.datasets.semantickitti import SemanticKITTIDataset
from laserperception.detection.ros2_contract import (
    PointCloud2Layout,
    PointFieldLayout,
    SourceHeader,
    TimeStamp,
)
from laserperception.perception.contracts import CoordinateContract, PerceptionTask
from laserperception.semantic import EXPERIMENT_001_TAXONOMY, evaluate_semantics
from laserperception.semantic.datasets import ground_truth_from_point_cloud


def las_fixture(path):
    las = laspy.LasData(laspy.LasHeader(point_format=3, version="1.2"))
    las.x = [3, 1, 2]
    las.y = [0, 1, 2]
    las.z = [2, 3, 4]
    las.classification = [1, 8, 5]
    las.intensity = [100, 200, 300]
    las.write(path)
    return path


def sk_fixture(root):
    scan = root / "sequences" / "08" / "velodyne"
    label = root / "sequences" / "08" / "labels"
    scan.mkdir(parents=True)
    label.mkdir()
    for i in (1, 0):
        np.array([[1, 2, 3, 0.25], [4, 5, 6, 0.75]], dtype="<f4").tofile(scan / f"{i:06d}.bin")
        np.array([40, 10], dtype="<u4").tofile(label / f"{i:06d}.label")
    return root


def test_deterministic_metadata_and_duplicate_filter():
    manifests = registry.list_adapters()
    ids = tuple(m.adapter_id for m in manifests)
    assert ids == tuple(sorted(ids)) and len(ids) == 8
    reversed_registry = DataAdapterRegistry(reversed(manifests))
    assert reversed_registry.list_adapters() == manifests
    with pytest.raises(ValueError, match="duplicate"):
        reversed_registry.register(manifests[0])
    assert registry.filter_by_feature("ring_index")[0].adapter_id == "nuscenes-lidar-top"
    assert registry.filter_by_task(PerceptionTask.SEMANTIC_SEGMENTATION)
    for m in manifests:
        assert type(m).from_json(m.to_json()) == m
    with pytest.raises(ValueError):
        registry.get("invented")


def test_las_inspection_preserves_dimensions_and_unknown_coordinates(tmp_path):
    path = las_fixture(tmp_path / "tile.las")
    loaded = load_input(path)
    assert loaded.cloud.xyz[:, 0].tolist() == [3, 1, 2]
    assert loaded.cloud.labels.tolist() == [1, 8, 5]
    assert loaded.cloud.attributes["intensity"].dtype == np.uint16
    result = inspect_input(path, model_id="dsvt-pillar-transfusion-m8")
    assert result.adapter.adapter_id == "las" and result.point_count == 3
    assert result.timestamp_ns is None and not result.compatibility_verified
    assert not result.compatibility.valid
    assert "time_lag" in result.compatibility.missing_required_features
    assert "coordinate compatibility not checked" in result.compatibility.warnings
    assert any("not representable" in v for v in result.limitations)
    assert type(result).from_json(result.to_json()) == result


@pytest.mark.parametrize("suffix,expected", [(".las", "las"), (".LAZ", "laz")])
def test_unambiguous_generic_format_only(suffix, expected):
    assert select_adapter("dales_tile" + suffix).adapter_id == expected


@pytest.mark.parametrize("filename", ["sample.bin", "arbitrary.pcd", "arbitrary.csv", "dataset"])
def test_ambiguous_and_unsupported_require_explicit(filename):
    with pytest.raises(ValueError):
        select_adapter(filename)


def test_kitti_labels_and_remission_no_fabrication(tmp_path):
    path = tmp_path / "scan.bin"
    labels = tmp_path / "scan.label"
    values = np.array([[1, 2, 3, 0.25], [4, 5, 6, 0.75]], dtype="<f4")
    values.tofile(path)
    np.array([40 | (5 << 16), 10], dtype="<u4").tofile(labels)
    loaded = FileDataAdapter("kitti-velodyne").load(path, InputOptions(label_path=labels))
    assert np.array_equal(loaded.cloud.xyz, values[:, :3])
    assert loaded.cloud.attributes["remission"].tolist() == [0.25, 0.75]
    assert loaded.cloud.attributes["instance_id"].tolist() == [5, 0]
    assert "intensity" not in loaded.cloud.attributes and "time_lag" not in loaded.cloud.attributes
    result = inspect_input(path, adapter_id="kitti-velodyne", model_id="dsvt-pillar-transfusion-m8")
    assert set(result.compatibility.missing_required_features) == {"intensity", "time_lag"}
    assert loaded.timestamp_ns is None


def test_semantickitti_sequence_lazy_numeric_order_and_directory_inspect(tmp_path, monkeypatch):
    root = sk_fixture(tmp_path)
    dataset = SemanticKITTIDataset(root, split="valid")
    sequence = SemanticKITTISequence(dataset)
    original = dataset.load
    monkeypatch.setattr(
        dataset, "load", lambda i: (_ for _ in ()).throw(AssertionError("loaded during refs"))
    )
    refs = list(sequence)
    assert [r.sample_id for r in refs] == ["semantickitti:08:000000", "semantickitti:08:000001"]
    assert all(r.timestamp_ns is None for r in refs)
    monkeypatch.setattr(dataset, "load", original)
    assert sequence.load(1).labels.tolist() == [40, 10]
    result = inspect_input(
        root, adapter_id="semantickitti", options=InputOptions(split="valid", index=1)
    )
    assert result.sample_id == refs[1].sample_id and result.labels_available
    with pytest.raises(ValueError):
        sequence.sample_info(-1)
    with pytest.raises(ValueError):
        sequence.load(2)


def test_dales_full_tiles_lazy_refs_and_chunk_offsets(tmp_path, monkeypatch):
    directory = tmp_path / "train"
    directory.mkdir()
    las_fixture(directory / "b.las")
    las_fixture(directory / "a.las")
    dataset = DalesDataset(tmp_path, split="train")
    sequence = DalesSequence(dataset)
    assert [r.sample_id for r in sequence] == ["dales:train:a.las", "dales:train:b.las"]
    full = sequence.load(0)
    assert "intensity" in full.attributes
    chunks = list(sequence.iter_chunks(0, chunk_size=2))
    assert [len(c) for c in chunks] == [2, 1]
    assert [c.metadata["source_row_start"] for c in chunks] == [0, 2]
    assert [c.metadata["source_row_stop"] for c in chunks] == [2, 3]
    assert all(not c.attributes for c in chunks)
    result = inspect_input(tmp_path, adapter_id="dales", options=InputOptions(split="train"))
    assert result.point_count == 3 and result.sample_id == "dales:train:a.las"


def test_kitti_raw_native_points_and_exact_timestamp_lazy_refs(tmp_path, monkeypatch):
    from test_kitti_raw_dataset import _synthetic_sequence

    sequence = _synthetic_sequence(tmp_path, frame_count=2)
    wrapper = KittiRawDataSequence(sequence)
    original = sequence.frame
    monkeypatch.setattr(
        sequence, "frame", lambda i: (_ for _ in ()).throw(AssertionError("point load during refs"))
    )
    refs = list(wrapper)
    assert refs[1].timestamp_ns - refs[0].timestamp_ns == 100_000_000
    monkeypatch.setattr(sequence, "frame", original)
    cloud = wrapper.load(0)
    assert cloud.xyz[0].tolist() == [1, 0, 0]
    assert cloud.attributes["remission"].tolist() == [0.25, 0.75]
    result = inspect_input(
        sequence.drive_root,
        adapter_id="kitti-raw",
        options=InputOptions(date_root=sequence.date_root, index=1),
    )
    assert result.timestamp_ns == refs[1].timestamp_ns


def test_nuscenes_explicit_time_and_ring_not_time_lag(tmp_path):
    path = tmp_path / "scan.bin"
    np.array([[1, 2, 3, 0.5, 12], [4, 5, 6, 0.8, 13]], dtype=np.float32).tofile(path)
    with pytest.raises(ValueError, match="explicit"):
        load_input(path, adapter_id="nuscenes-lidar-top")
    result = load_input(
        path,
        adapter_id="nuscenes-lidar-top",
        options=InputOptions(timestamp_microseconds=123456, sample_id="nu:scan"),
    )
    assert result.timestamp_ns == 123456000
    assert result.cloud.attributes["ring_index"].tolist() == [12, 13]
    assert "time_lag" not in result.cloud.attributes


def test_pointcloud2_header_and_filter_mapping_disclosed():
    points = np.array([[1, 2, 3], [np.nan, 2, 3], [4, 5, 6]], dtype=np.float32)
    layout = PointCloud2Layout(
        1,
        3,
        tuple(PointFieldLayout(name, i * 4, 7, 1) for i, name in enumerate(("x", "y", "z"))),
        False,
        12,
        36,
        points.tobytes(),
    )
    loaded = load_pointcloud2_xyz(layout, SourceHeader("lidar", TimeStamp(10, 123)))
    assert loaded.timestamp_ns == 10_000_000_123
    assert loaded.cloud.xyz.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert loaded.cloud.metadata["source_point_count"] == 3
    assert loaded.cloud.metadata["original_source_row_mapping"].startswith("unavailable")


@pytest.mark.parametrize("adapter", ["semantickitti", "dales", "kitti-raw", "pointcloud2-xyz"])
def test_required_explicit_directory_context(adapter, tmp_path):
    with pytest.raises(ValueError):
        load_input(tmp_path, adapter_id=adapter)


@pytest.mark.parametrize(
    "options",
    [
        {"index": -1},
        {"index": True},
        {"timestamp_microseconds": True},
        {"timestamp_microseconds": -1},
        {"sample_id": " "},
    ],
)
def test_invalid_options(options):
    with pytest.raises(ValueError):
        InputOptions(**options)


def test_missing_optional_laz_backend_actionable(tmp_path, monkeypatch):
    path = tmp_path / "tile.laz"
    path.write_bytes(b"fixture")
    module = importlib.import_module("laserperception.io.las")

    def missing(path):
        raise laspy.errors.LaspyException("No LazBackend selected")

    monkeypatch.setattr(module, "load_las", missing)
    with pytest.raises(RuntimeError, match=r"laserperception\[laz\]"):
        load_input(path)


def test_adapter_semantic_composition(tmp_path):
    path = las_fixture(tmp_path / "dales.las")
    loaded = load_input(path, adapter_id="dales")
    coords = CoordinateContract(
        "synthetic_map",
        "right",
        ("east", "north", "up"),
        "metre",
        "not_applicable",
        ("not_applicable",) * 3,
        "not_applicable",
        "synthetic fixture",
    )
    gt = ground_truth_from_point_cloud(
        loaded.cloud,
        sample_id=loaded.sample_id,
        frame_id="map",
        coordinates=coords,
        taxonomy=EXPERIMENT_001_TAXONOMY,
        mapping="dales",
    )
    prediction = replace(
        gt, provenance=replace(gt.provenance, operation="synthetic fixture prediction")
    )
    assert evaluate_semantics(prediction, gt).overall_accuracy == 1


def test_cli_listing_manifest_and_input(tmp_path, capsys):
    assert main(["data", "adapters", "list", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 8
    assert main(["data", "adapters", "inspect", "kitti-raw", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["adapter_id"] == "kitti-raw"
    path = las_fixture(tmp_path / "input.las")
    assert (
        main(["data", "inspect", str(path), "--model", "dsvt-pillar-transfusion-m8", "--json"]) == 0
    )
    assert not json.loads(capsys.readouterr().out)["compatibility_verified"]


def test_metadata_discovery_imports_no_reader_gpu_or_process(monkeypatch):
    parent = sys.modules["laserperception"]
    original_package = parent.data
    prefix = "laserperception.data"
    saved = {k: v for k, v in sys.modules.items() if k == prefix or k.startswith(prefix + ".")}
    for name in saved:
        del sys.modules[name]
    original = builtins.__import__
    forbidden = {
        "torch",
        "pcdet",
        "spconv",
        "torch_scatter",
        "tensorrt",
        "cupy",
        "pynvml",
        "rclpy",
        "laspy",
        "scipy",
    }

    def guard(name, *args, **kwargs):
        assert name.split(".")[0] not in forbidden
        assert not name.startswith("laserperception.datasets")
        return original(name, *args, **kwargs)

    def no_process(*args, **kwargs):
        raise AssertionError("discovery must not start a process")

    monkeypatch.setattr(builtins, "__import__", guard)
    monkeypatch.setattr(subprocess, "run", no_process)
    try:
        assert len(importlib.import_module(prefix).registry.list_adapters()) == 8
    finally:
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                del sys.modules[name]
        sys.modules.update(saved)
        parent.data = original_package


@pytest.mark.parametrize(
    "suffix,extra", [(".bin", b"x"), (".dat", b""), (".bin", b"xx"), (".bin", b"xxx")]
)
def test_nuscenes_complete_records_and_extension(tmp_path, suffix, extra):
    path = tmp_path / ("scan" + suffix)
    path.write_bytes(np.array([1, 2, 3, 0.5, 12], dtype=np.float32).tobytes() + extra)
    with pytest.raises(ValueError):
        load_input(
            path,
            adapter_id="nuscenes-lidar-top",
            options=InputOptions(timestamp_microseconds=0, sample_id="scan"),
        )


@pytest.mark.parametrize("index", [-1, True, 1])
def test_chunk_index_rejected_before_reader_call(tmp_path, monkeypatch, index):
    directory = tmp_path / "train"
    directory.mkdir()
    las_fixture(directory / "a.las")
    dataset = DalesDataset(tmp_path, split="train")
    sequence = DalesSequence(dataset)
    calls = []

    def read(*args, **kwargs):
        calls.append(1)
        return iter(())

    monkeypatch.setattr(dataset, "iter_tile_chunks", read)
    with pytest.raises(ValueError):
        list(sequence.iter_chunks(index))
    assert calls == []


def test_sequence_loaded_identity_matches_refs(tmp_path):
    from test_kitti_raw_dataset import _synthetic_sequence

    raw = KittiRawDataSequence(_synthetic_sequence(tmp_path / "raw", frame_count=1))
    sk = SemanticKITTISequence(SemanticKITTIDataset(sk_fixture(tmp_path / "sk"), split="valid"))
    directory = tmp_path / "dales" / "train"
    directory.mkdir(parents=True)
    las_fixture(directory / "a.las")
    dales = DalesSequence(DalesDataset(directory.parent, split="train"))
    for sequence in (raw, sk, dales):
        assert sequence.load(0).metadata["sample_id"] == sequence.sample_info(0).sample_id


def test_inspection_feature_positions_are_truthful(tmp_path, monkeypatch):
    import laserperception.data.inspection as module

    captured = []
    original = module.validate_input

    def capture(manifest, features, **kwargs):
        captured.extend(features)
        return original(manifest, features, **kwargs)

    monkeypatch.setattr(module, "validate_input", capture)
    path = las_fixture(tmp_path / "a.las")
    inspect_input(path, model_id="dsvt-pillar-transfusion-m8")
    assert next(f for f in captured if f.name == "intensity").position is None
    captured.clear()
    points = np.array([[1, 2, 3]], dtype=np.float32)
    layout = PointCloud2Layout(
        1,
        1,
        tuple(PointFieldLayout(n, i * 4, 7, 1) for i, n in enumerate(("x", "y", "z"))),
        False,
        12,
        12,
        points.tobytes(),
    )
    loaded = load_pointcloud2_xyz(layout, SourceHeader("lidar", TimeStamp(1, 0)))
    module.inspect_loaded_input(loaded, model_id="dsvt-pillar-transfusion-m8")
    assert all(f.source == "message acquisition" for f in captured)
    assert all("message coordinate" in f.semantics for f in captured)


def test_complete_cpu_journey_outputs_reusable_cli_inputs(tmp_path, monkeypatch, capsys):
    import runpy
    from pathlib import Path

    output = tmp_path / "journey"
    script = Path(__file__).resolve().parents[1] / "examples/perception_cpu_journey.py"
    monkeypatch.setattr(sys, "argv", [str(script), str(output)])
    runpy.run_path(str(script), run_name="__main__")
    assert (output / "detections.jsonl").is_file()
    assert len((output / "detections.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    capsys.readouterr()
    assert (
        main(
            [
                "track",
                str(output / "detections.jsonl"),
                "--sequence-id",
                "synthetic-journey",
                "--json",
            ]
        )
        == 0
    )
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3
    assert (
        main(
            [
                "semantic",
                "evaluate",
                str(output / "prediction.json"),
                str(output / "ground-truth.json"),
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["evaluated_points"] == 3
