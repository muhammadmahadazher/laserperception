from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from laserperception.detection.mmdet3d_backend import (
    DetectionEnvironmentError,
    _load_mmdet3d_runtime,
    convert_mmdet3d_prediction,
    sha256_file,
)


class _FakeBoxes:
    def __init__(self) -> None:
        self.gravity_center = np.array([[5.0, 1.0, 2.0], [2.0, 3.0, 1.0]])
        self.dims = np.array([[4.0, 2.0, 1.5], [0.8, 0.7, 1.8]])
        self.yaw = np.array([0.25, -0.5])
        # Tensor Z deliberately represents bottom center; conversion must use gravity_center.
        self.tensor = np.array(
            [
                [5.0, 1.0, 1.25, 4.0, 2.0, 1.5, 0.25, 1.5, -0.2],
                [2.0, 3.0, 0.1, 0.8, 0.7, 1.8, -0.5, 0.0, 0.1],
            ]
        )


def _prediction(labels: np.ndarray | None = None) -> object:
    return SimpleNamespace(
        pred_instances_3d=SimpleNamespace(
            bboxes_3d=_FakeBoxes(),
            scores_3d=np.array([0.7, 0.9]),
            labels_3d=np.array([0, 7]) if labels is None else labels,
        )
    )


def test_conversion_uses_geometric_center_lwh_velocity_and_upstream_names() -> None:
    classes = (
        "car",
        "truck",
        "trailer",
        "bus",
        "construction_vehicle",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "barrier",
    )
    frame = convert_mmdet3d_prediction(
        _prediction(), class_names=classes, sample_id="token", metadata={"sample_index": 4}
    )

    assert [item.class_name for item in frame.detections] == ["pedestrian", "car"]
    assert frame.detections[0].center_xyz == (2.0, 3.0, 1.0)
    assert frame.detections[1].center_xyz == (5.0, 1.0, 2.0)
    assert frame.detections[1].size_lwh == (4.0, 2.0, 1.5)
    assert frame.detections[1].velocity_xy == (1.5, -0.2)
    assert frame.metadata["raw_detection_count"] == 2
    assert frame.metadata["sample_index"] == 4


def test_conversion_rejects_labels_outside_upstream_taxonomy() -> None:
    with pytest.raises(ValueError, match="outside the upstream class taxonomy"):
        convert_mmdet3d_prediction(
            _prediction(labels=np.array([0, 99])), class_names=("car",), sample_id="token"
        )


def test_sha256_file_streams_known_content(tmp_path: Path) -> None:
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"laserperception")
    assert sha256_file(asset, chunk_size=3) == (
        "c7740add14b6bcaf2e755fc040e18425cad1ec04f13e90017b29ba1cc1bb702b"
    )


def test_missing_optional_environment_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module

    def fail_torch(name: str, package: str | None = None) -> object:
        if name == "torch":
            raise ImportError("synthetic missing torch")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fail_torch)
    with pytest.raises(DetectionEnvironmentError, match="setup_detection_m1.sh"):
        _load_mmdet3d_runtime()
