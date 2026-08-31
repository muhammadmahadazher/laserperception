from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from laserperception.detection.geometry import bev_corners
from laserperception.detection.m8_backend import (
    M8_CLASS_NAMES,
    M8_SCIENTIFIC_SCORE_THRESHOLD,
    DsvtBackend,
    dsvt_predictions_to_detection_frame,
    load_m8_candidate_manifest,
    map_m8_class_to_primary,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/m8/dsvt_nuscenes_pillar.json"


def test_backend_module_import_is_lightweight_and_lazy() -> None:
    before = set(sys.modules)
    importlib.reload(sys.modules["laserperception.detection.m8_backend"])
    newly_imported = set(sys.modules) - before

    assert "pcdet" not in newly_imported
    assert "torch" not in newly_imported
    assert "spconv" not in newly_imported


def test_manifest_binds_candidate_artifacts_and_score_contract() -> None:
    manifest = load_m8_candidate_manifest(MANIFEST)
    checkpoint = manifest["checkpoint"]
    output = manifest["output_contract"]
    assert isinstance(checkpoint, dict)
    assert isinstance(output, dict)

    assert checkpoint["sha256"] == (
        "a675149d095eef8ddc0c137ae46eeac075ccc504c7608162c71e7adf318793fb"
    )
    assert checkpoint["bytes"] == 28_665_215
    assert output["internal_score_threshold"] <= M8_SCIENTIFIC_SCORE_THRESHOLD
    assert output["class_names"] == list(M8_CLASS_NAMES)


def test_unsupported_environment_fails_before_optional_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LASERPERCEPTION_M8_DSVT_ROOT", raising=False)
    monkeypatch.delenv("LASERPERCEPTION_M8_DSVT_CHECKPOINT", raising=False)

    with pytest.raises(RuntimeError, match="set LASERPERCEPTION_M8_DSVT_ROOT"):
        DsvtBackend.from_environment(manifest_path=MANIFEST)


def test_class_mapping_is_prospective_and_does_not_merge_vehicle_classes() -> None:
    assert map_m8_class_to_primary("car") == "car"
    assert map_m8_class_to_primary("pedestrian") == "pedestrian"
    assert map_m8_class_to_primary("truck") is None
    assert map_m8_class_to_primary("bus") is None


def test_native_box_conversion_preserves_geometry_scores_and_velocity() -> None:
    boxes = np.array(
        [[1.0, 2.0, 3.0, 4.0, 2.0, 1.5, math.pi / 2.0, 0.5, -0.25]],
        dtype=np.float32,
    )
    scores = np.array([0.75], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)

    frame = dsvt_predictions_to_detection_frame(boxes, scores, labels, sample_id="analytic")
    detection = frame.detections[0]

    assert detection.center_xyz == pytest.approx((1.0, 2.0, 3.0))
    assert detection.size_lwh == pytest.approx((4.0, 2.0, 1.5))
    assert detection.yaw_rad == pytest.approx(math.pi / 2.0)
    assert detection.velocity_xy == pytest.approx((0.5, -0.25))
    assert detection.score == pytest.approx(0.75)
    assert detection.class_name == "car"
    corners = bev_corners(detection)
    assert np.ptp(corners[:, 0]) == pytest.approx(2.0)
    assert np.ptp(corners[:, 1]) == pytest.approx(4.0)


def test_conversion_rejects_invalid_native_outputs() -> None:
    boxes = np.ones((1, 9), dtype=np.float32)
    with pytest.raises(ValueError, match="outside"):
        dsvt_predictions_to_detection_frame(
            boxes,
            np.array([0.5], dtype=np.float32),
            np.array([11], dtype=np.int32),
            sample_id="bad",
        )


def test_manifest_is_valid_json_without_optional_yaml_dependency() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["feature_contract"]["columns"] == [
        "x",
        "y",
        "z",
        "intensity",
        "time_lag",
    ]
