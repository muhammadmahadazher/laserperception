from pathlib import Path

import numpy as np
import pytest

from laserperception.detection import Detection3D, DetectionFrame
from laserperception.detection.visualization import prepare_bev_render_data, render_bev


def _frame() -> DetectionFrame:
    return DetectionFrame(
        detections=(
            Detection3D((1, 2, 0), (4, 2, 1.5), 0.2, 0.8, 0, "car"),
            Detection3D((2, 3, 0), (0.8, 0.7, 1.8), -0.1, 0.2, 7, "pedestrian"),
        ),
        sample_id="synthetic-token",
        coordinate_frame="nuscenes_lidar_top",
    )


def test_prepare_bev_data_crops_subsamples_and_filters() -> None:
    points = np.array([[-2, 0, 0], [-1, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    data = prepare_bev_render_data(
        points,
        _frame(),
        min_score=0.5,
        x_limits=(-1.5, 1.5),
        y_limits=(-1, 1),
        max_points=2,
    )

    np.testing.assert_array_equal(data.points_xy, [[-1, 0], [1, 0]])
    assert [detection.class_name for detection in data.detections] == ["car"]


def test_prepare_bev_data_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError, match="increasing"):
        prepare_bev_render_data(
            np.zeros((1, 3)),
            _frame(),
            min_score=0.5,
            x_limits=(1, 1),
            y_limits=(-1, 1),
            max_points=1,
        )


def test_render_bev_writes_headless_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    output = tmp_path / "bev.png"
    result = render_bev(
        np.array([[0, 0, 0], [1, 1, 0]], dtype=np.float32),
        _frame(),
        output,
        max_points=10,
    )

    assert result == output
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
