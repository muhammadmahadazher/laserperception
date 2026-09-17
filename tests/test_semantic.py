"""Synthetic CPU semantic integrity and exact metric regressions."""

import hashlib
import json
from dataclasses import FrozenInstanceError, replace

import laspy
import numpy as np
import pytest

from laserperception.cli import main
from laserperception.core import PointCloud
from laserperception.datasets.semantickitti import SemanticKITTIDataset
from laserperception.perception.contracts import CoordinateContract
from laserperception.semantic import (
    EXPERIMENT_001_TAXONOMY,
    SemanticClass,
    SemanticPointFrame,
    SemanticProvenance,
    SemanticTaxonomy,
    evaluate_semantics,
    load_semantic_frame,
    save_semantic_frame,
)
from laserperception.semantic.datasets import (
    dales_ground_truth,
    ground_truth_from_point_cloud,
    semantickitti_ground_truth,
)

COORDS = CoordinateContract(
    "test_sensor",
    "right",
    ("forward", "left", "up"),
    "m",
    "not_applicable",
    ("not_applicable",) * 3,
    "not_applicable",
    "synthetic",
)
TAXONOMY = SemanticTaxonomy(
    "synthetic.native",
    "1",
    (
        SemanticClass(10, "A"),
        SemanticClass(20, "B"),
        SemanticClass(30, "C"),
        SemanticClass(-1, "Ignored", True),
    ),
)
PROVENANCE = SemanticProvenance("synthetic", "fixture", "test")


def frame(values, *, rows=None, source_count=None, taxonomy=TAXONOMY):
    return SemanticPointFrame(
        "sample",
        "sensor",
        COORDS,
        len(values) if source_count is None else source_count,
        np.asarray(values, dtype=np.int64),
        taxonomy,
        PROVENANCE,
        None if rows is None else np.asarray(rows, dtype=np.int64),
    )


def test_perfect_and_excluded_absent():
    result = evaluate_semantics(frame([10, 20, 10]), frame([10, 20, 10]))
    assert result.confusion_matrix == ((2, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0))
    assert result.mean_iou == result.overall_accuracy == 1
    assert result.excluded_mean_iou_class_ids == (30,)
    assert result.class_metrics[2].iou is None
    assert type(result).from_json(result.to_json()) == result


def test_exact_mixed_metrics_unknown_and_ignored():
    result = evaluate_semantics(frame([10, 20, 99, 30, 99]), frame([10, 10, 20, 20, -1]))
    assert result.confusion_matrix == ((1, 1, 0, 0), (0, 0, 1, 1), (0, 0, 0, 0))
    assert result.evaluated_points == 4 and result.ignored_ground_truth_points == 1
    assert result.unknown_prediction_points == 1
    assert result.overall_accuracy == 0.25
    assert result.mean_iou == pytest.approx(1 / 6)
    a, b, c = result.class_metrics
    assert (a.true_positive, a.false_positive, a.false_negative, a.support, a.iou) == (
        1,
        0,
        1,
        2,
        0.5,
    )
    assert (b.true_positive, b.false_positive, b.false_negative, b.support, b.iou) == (
        0,
        1,
        2,
        2,
        0,
    )
    assert c.support == 0 and c.false_positive == 1 and c.iou == 0
    assert result.excluded_mean_iou_class_ids == ()


def test_all_wrong_and_ignored_prediction():
    result = evaluate_semantics(frame([20, 10, -1]), frame([10, 20, 10]))
    assert result.mean_iou == result.overall_accuracy == 0
    assert result.unknown_prediction_points == 1


@pytest.mark.parametrize("values", [[], [-1, -1]])
def test_empty_or_ignored_has_no_fabricated_metrics(values):
    result = evaluate_semantics(frame(values), frame(values))
    assert result.evaluated_points == 0
    assert result.mean_iou is None and result.overall_accuracy is None
    assert result.excluded_mean_iou_class_ids == (10, 20, 30)


@pytest.mark.parametrize(
    "changes",
    [
        {"sample_id": "other"},
        {"frame_id": "other"},
        {"coordinates": replace(COORDS, reference="other")},
        {"taxonomy": replace(TAXONOMY, version="2")},
        {
            "taxonomy": replace(
                TAXONOMY,
                classes=(
                    SemanticClass(10, "Changed"),
                    SemanticClass(20, "B"),
                    SemanticClass(30, "C"),
                    SemanticClass(-1, "Ignored", True),
                ),
            )
        },
        {"source_point_count": 4},
        {"source_rows": np.array([1, 2], dtype=np.int64)},
    ],
)
def test_identity_alignment_rejects(changes):
    gt = frame([10, 20], rows=[0, 2], source_count=3)
    prediction = replace(gt, **changes)
    with pytest.raises(ValueError, match="alignment"):
        evaluate_semantics(prediction, gt)


def test_alignment_length_and_unknown_gt_reject():
    with pytest.raises(ValueError, match="alignment"):
        evaluate_semantics(frame([10], rows=[0], source_count=2), frame([10, 20]))
    with pytest.raises(ValueError, match="ground truth"):
        evaluate_semantics(frame([10]), frame([99]))


@pytest.mark.parametrize(
    "labels",
    [
        np.array([1.0]),
        np.array([True]),
        np.array([[1]]),
        np.array([2**63], dtype=np.uint64),
        np.array(["10"], dtype=object),
    ],
)
def test_invalid_label_vectors(labels):
    with pytest.raises(ValueError):
        SemanticPointFrame("s", "f", COORDS, 1, labels, TAXONOMY, PROVENANCE)


@pytest.mark.parametrize("rows", [[0, 0], [1, 0], [-1, 1], [0, 3], [0]])
def test_invalid_source_rows(rows):
    with pytest.raises(ValueError):
        frame([10, 20], rows=rows, source_count=3)


def test_filter_needs_map_and_integer_count():
    with pytest.raises(ValueError, match="source_rows"):
        frame([10], source_count=2)
    with pytest.raises(ValueError):
        frame([10], source_count=True)


@pytest.mark.parametrize(
    "confidence",
    [
        np.array([np.nan]),
        np.array([np.inf]),
        np.array([-0.1]),
        np.array([1.1]),
        np.array([1]),
        np.array([0.2, 0.3]),
    ],
)
def test_confidence_invalid(confidence):
    with pytest.raises(ValueError):
        replace(frame([10]), confidence=confidence)


def test_immutable_copy_confidence_and_inline_roundtrip():
    source = np.array([10, 20])
    result = replace(frame(source), confidence=np.array([0.2, 0.9]))
    source[0] = 99
    for array in (result.class_ids, result.source_rows, result.confidence):
        with pytest.raises(ValueError):
            array.flags.writeable = True
    with pytest.raises(FrozenInstanceError):
        result.sample_id = "changed"
    restored = SemanticPointFrame.from_json(result.to_json())
    assert np.array_equal(restored.class_ids, result.class_ids)
    assert np.array_equal(restored.source_rows, result.source_rows)
    assert np.array_equal(restored.confidence, result.confidence)
    assert restored.taxonomy == result.taxonomy
    assert restored.to_json() == result.to_json()


def test_sidecar_roundtrip_large_and_no_overwrite(tmp_path):
    result = frame([10] * 4100)
    path = tmp_path / "result.json"
    with pytest.raises(ValueError, match="sidecars"):
        result.to_json()
    save_semantic_frame(result, path)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    assert envelope["class_ids"]["values"] is None
    assert envelope["class_ids"]["dtype"] == "int64"
    assert envelope["class_ids"]["shape"] == [4100]
    assert np.array_equal(load_semantic_frame(path).class_ids, result.class_ids)
    with pytest.raises(FileExistsError):
        save_semantic_frame(result, path)
    assert not any("pcdet" in p.name for p in tmp_path.iterdir())


@pytest.mark.parametrize("mutation", ["hash", "shape", "dtype", "traversal", "size", "role"])
def test_sidecar_identity_rejected(tmp_path, mutation):
    path = tmp_path / "result.json"
    save_semantic_frame(frame([10]), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    payload = data["class_ids"]
    if mutation == "hash":
        payload["sha256"] = "0" * 64
    elif mutation == "shape":
        payload["shape"] = [2]
    elif mutation == "dtype":
        side = tmp_path / payload["path"]
        np.save(side, np.array([10], dtype=np.int32), allow_pickle=False)
        raw = side.read_bytes()
        payload["size_bytes"] = len(raw)
        payload["sha256"] = hashlib.sha256(raw).hexdigest()
    elif mutation == "traversal":
        payload["path"] = "../outside.npy"
    elif mutation == "size":
        payload["size_bytes"] += 1
    else:
        payload["dtype"] = "float64"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_semantic_frame(path)


@pytest.mark.parametrize("bad", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}'])
def test_strict_json(bad):
    with pytest.raises(ValueError):
        SemanticPointFrame.from_json(bad)


def test_taxonomy_duplicate_boolean_no_classes():
    for classes in (
        (SemanticClass(10, "A"), SemanticClass(10, "B")),
        (SemanticClass(-1, "I", True),),
    ):
        with pytest.raises(ValueError):
            SemanticTaxonomy("t", "1", classes)
    with pytest.raises(ValueError):
        SemanticClass(True, "A")


def test_explicit_dataset_mapping_preserves_raw_labels():
    cloud = PointCloud(np.zeros((4, 3)), np.array([40, 50, 10, 999], dtype=np.uint16))
    original = cloud.labels.copy()
    result = ground_truth_from_point_cloud(
        cloud,
        sample_id="s",
        frame_id="f",
        coordinates=COORDS,
        taxonomy=EXPERIMENT_001_TAXONOMY,
        mapping="semantickitti",
    )
    assert result.class_ids.tolist() == [0, 1, 3, -1]
    assert np.array_equal(cloud.labels, original)
    with pytest.raises(ValueError):
        ground_truth_from_point_cloud(
            cloud,
            sample_id="s",
            frame_id="f",
            coordinates=COORDS,
            taxonomy=TAXONOMY,
            mapping="dales",
        )


def test_existing_semantickitti_reader_fixture(tmp_path):
    sequence = tmp_path / "sequences" / "08"
    scan = sequence / "velodyne"
    labels = sequence / "labels"
    scan.mkdir(parents=True)
    labels.mkdir()
    np.array([[1, 2, 3, 0.5], [4, 5, 6, 0.9]], dtype=np.float32).tofile(scan / "000000.bin")
    np.array([40, 10], dtype=np.uint32).tofile(labels / "000000.label")
    dataset = SemanticKITTIDataset(tmp_path, split="valid")
    result = semantickitti_ground_truth(dataset, 0, frame_id="lidar", coordinates=COORDS)
    assert result.sample_id == "semantickitti:08:000000"
    assert result.class_ids.tolist() == [0, 3]


def test_existing_dales_las_fixture(tmp_path):
    las = laspy.LasData(laspy.LasHeader(point_format=3, version="1.2"))
    las.x = [1, 2, 3]
    las.y = [0, 0, 0]
    las.z = [0, 0, 0]
    las.classification = [1, 8, 5]
    path = tmp_path / "tile.las"
    las.write(path)
    result = dales_ground_truth(path, sample_id="dales:tile", frame_id="map", coordinates=COORDS)
    assert result.class_ids.tolist() == [0, 1, -1]
    assert result.source_rows.tolist() == [0, 1, 2]


def test_cli_evaluate_and_inspect(tmp_path, capsys):
    path = tmp_path / "result.json"
    save_semantic_frame(frame([10, 20]), path, inline=True)
    assert main(["semantic", "evaluate", str(path), str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["mean_iou"] == 1
    assert main(["semantic", "inspect", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["result_rows"] == 2


def test_metadata_imports_do_not_discover_hardware(monkeypatch):
    import builtins
    import importlib
    import subprocess
    import sys

    prefix = "laserperception.semantic"
    saved = {k: v for k, v in sys.modules.items() if k == prefix or k.startswith(prefix + ".")}
    for name in saved:
        del sys.modules[name]
    original = builtins.__import__
    forbidden = {"torch", "pcdet", "spconv", "torch_scatter", "tensorrt", "cupy", "pynvml", "scipy"}

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in forbidden
        return original(name, *args, **kwargs)

    def no_process(*args, **kwargs):
        raise AssertionError("semantic metadata must not launch a process")

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(subprocess, "run", no_process)
    try:
        assert importlib.import_module(prefix).SemanticPointFrame
    finally:
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                del sys.modules[name]
        sys.modules.update(saved)


def test_inline_integer_bounds_and_wrong_schema():
    data = json.loads(frame([10]).to_json())
    data["class_ids"]["values"] = [2**63]
    with pytest.raises(ValueError, match="signed range"):
        SemanticPointFrame.from_json(json.dumps(data))
    data = json.loads(frame([10]).to_json())
    data["schema_version"] = "future"
    with pytest.raises(ValueError):
        SemanticPointFrame.from_json(json.dumps(data))


@pytest.mark.parametrize("kind", ["npz", "huge_header", "trailing"])
def test_malformed_binary_sidecar_before_array_allocation(tmp_path, kind):
    import io

    path = tmp_path / "result.json"
    save_semantic_frame(frame([10]), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    payload = data["class_ids"]
    stream = io.BytesIO()
    if kind == "npz":
        np.savez(stream, labels=np.array([10]))
    elif kind == "huge_header":
        np.lib.format.write_array_header_1_0(
            stream, {"descr": "<i8", "fortran_order": False, "shape": (10**12,)}
        )
    else:
        np.save(stream, np.array([10], dtype=np.int64), allow_pickle=False)
        stream.write(b"trailing")
    raw = stream.getvalue()
    (tmp_path / payload["path"]).write_bytes(raw)
    payload["sha256"] = hashlib.sha256(raw).hexdigest()
    payload["size_bytes"] = len(raw)
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_semantic_frame(path)


def test_portable_output_name_rejects_before_creation(tmp_path):
    with pytest.raises(ValueError):
        save_semantic_frame(frame([10]), tmp_path / "CON.json")
    assert list(tmp_path.iterdir()) == []


def test_taxonomy_signed_bound():
    with pytest.raises(ValueError):
        SemanticClass(2**63, "too large")
    with pytest.raises(ValueError):
        replace(TAXONOMY, unknown_id=2**63)
