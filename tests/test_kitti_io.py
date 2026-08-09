from pathlib import Path

import numpy as np
import pytest

from laserperception.io import (
    load_kitti_bin,
    load_semantic_kitti_labels,
    write_kitti_bin,
)


def test_kitti_synthetic_read_and_round_trip(tmp_path: Path) -> None:
    records = np.array(
        [[1.0, 2.0, 3.0, 0.1], [-4.0, 5.5, 6.0, 0.9]],
        dtype="<f4",
    )
    source = tmp_path / "scan.bin"
    records.tofile(source)

    cloud = load_kitti_bin(source)
    assert np.array_equal(cloud.xyz, records[:, :3])
    assert np.array_equal(cloud.attributes["remission"], records[:, 3])
    assert cloud.metadata["coordinates_normalized"] is False

    output = tmp_path / "roundtrip.bin"
    write_kitti_bin(cloud, output)
    reloaded = load_kitti_bin(output)
    assert np.array_equal(reloaded.xyz, cloud.xyz)
    assert np.array_equal(reloaded.attributes["remission"], cloud.attributes["remission"])


def test_malformed_kitti_file_raises(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.bin"
    malformed.write_bytes(b"\x00" * 15)
    with pytest.raises(ValueError, match="divisible by 16"):
        load_kitti_bin(malformed)


def test_semantic_kitti_labels_decode_and_attach(tmp_path: Path) -> None:
    scan = tmp_path / "scan.bin"
    np.zeros((2, 4), dtype="<f4").tofile(scan)

    semantic = np.array([10, 70], dtype=np.uint32)
    instance = np.array([3, 65535], dtype=np.uint32)
    packed = (instance << np.uint32(16)) | semantic
    label_path = tmp_path / "scan.label"
    packed.astype("<u4").tofile(label_path)

    decoded = load_semantic_kitti_labels(label_path, expected_points=2)
    assert np.array_equal(decoded.semantic_ids, semantic.astype(np.uint16))
    assert np.array_equal(decoded.instance_ids, instance.astype(np.uint16))
    assert np.array_equal(decoded.packed, packed)

    cloud = load_kitti_bin(scan, label_path=label_path)
    assert cloud.labels is not None
    assert np.array_equal(cloud.labels, decoded.semantic_ids)
    assert np.array_equal(cloud.attributes["instance_id"], decoded.instance_ids)


def test_semantic_kitti_label_count_must_match_scan(tmp_path: Path) -> None:
    scan = tmp_path / "scan.bin"
    labels = tmp_path / "scan.label"
    np.zeros((2, 4), dtype="<f4").tofile(scan)
    np.zeros((1,), dtype="<u4").tofile(labels)
    with pytest.raises(ValueError, match="label count"):
        load_kitti_bin(scan, label_path=labels)
