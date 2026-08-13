from __future__ import annotations

from math import pi, sqrt

import numpy as np
import pytest

from laserperception.detection.ros2_contract import (
    POINT_FIELD_FLOAT32,
    ModelReadyPointCloud,
    PointCloud2Layout,
    PointFieldLayout,
    SourceHeader,
    TimeStamp,
    decode_model_ready_pointcloud,
    detection_frame_to_records,
    model_ready_pointcloud_layout,
    yaw_to_quaternion,
)
from laserperception.detection.types import Detection3D, DetectionFrame


def _frame(*, yaw: float = 0.0) -> DetectionFrame:
    return DetectionFrame(
        detections=(
            Detection3D(
                center_xyz=(4.0, -2.0, 1.25),
                size_lwh=(4.5, 1.8, 1.6),
                yaw_rad=yaw,
                score=0.875,
                class_id=0,
                class_name="car",
                velocity_xy=(1.0, 0.0),
            ),
        ),
        sample_id="sample",
        coordinate_frame="lidar_current",
    )


def test_model_ready_round_trip_is_exact_float32() -> None:
    points = np.array([[1.0, 2.0, 3.0, 0.0], [-4.5, 5.25, -0.5, 0.45]], dtype=np.float32)
    source = ModelReadyPointCloud(points)
    decoded = decode_model_ready_pointcloud(model_ready_pointcloud_layout(source))

    assert decoded.points_xyzt.dtype == np.float32
    assert np.array_equal(decoded.points_xyzt, source.points_xyzt)
    assert decoded.sha256 == source.sha256


def test_pointcloud_fields_may_be_arbitrarily_ordered_with_extras() -> None:
    dtype = np.dtype(
        {
            "names": ["intensity", "time_lag", "x", "z", "y"],
            "formats": ["<f4"] * 5,
            "offsets": [0, 4, 8, 12, 16],
            "itemsize": 20,
        }
    )
    structured = np.zeros(2, dtype=dtype)
    structured["intensity"] = [99.0, 100.0]
    structured["time_lag"] = [0.0, 0.5]
    structured["x"] = [1.0, 2.0]
    structured["y"] = [3.0, 4.0]
    structured["z"] = [5.0, 6.0]
    layout = PointCloud2Layout(
        height=1,
        width=2,
        fields=tuple(
            PointFieldLayout(name, dtype.fields[name][1], POINT_FIELD_FLOAT32)
            for name in dtype.names or ()
        ),
        is_bigendian=False,
        point_step=20,
        row_step=40,
        data=structured.tobytes(),
    )

    decoded = decode_model_ready_pointcloud(layout)
    assert np.array_equal(
        decoded.points_xyzt,
        np.array([[1.0, 3.0, 5.0, 0.0], [2.0, 4.0, 6.0, 0.5]], dtype=np.float32),
    )


def test_organized_cloud_with_row_padding_is_supported() -> None:
    source = ModelReadyPointCloud(np.arange(24, dtype=np.float32).reshape(6, 4))
    flat = model_ready_pointcloud_layout(source)
    row_step = flat.point_step * 3 + 8
    data = bytearray(row_step * 2)
    source_bytes = flat.data
    per_row = flat.point_step * 3
    data[:per_row] = source_bytes[:per_row]
    data[row_step : row_step + per_row] = source_bytes[per_row:]
    organized = PointCloud2Layout(
        height=2,
        width=3,
        fields=flat.fields,
        is_bigendian=False,
        point_step=flat.point_step,
        row_step=row_step,
        data=bytes(data),
    )

    assert np.array_equal(decode_model_ready_pointcloud(organized).points_xyzt, source.points_xyzt)


def test_missing_time_lag_is_rejected_without_intensity_substitution() -> None:
    layout = model_ready_pointcloud_layout(np.ones((1, 4), dtype=np.float32))
    missing = PointCloud2Layout(
        height=layout.height,
        width=layout.width,
        fields=tuple(field for field in layout.fields if field.name != "time_lag"),
        is_bigendian=layout.is_bigendian,
        point_step=layout.point_step,
        row_step=layout.row_step,
        data=layout.data,
    )
    with pytest.raises(ValueError, match="missing required field.*time_lag"):
        decode_model_ready_pointcloud(missing)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"row_step": 1}, "row_step"),
        ({"data": b""}, "data length"),
        ({"width": 0}, "width"),
    ],
)
def test_malformed_pointcloud_layout_is_rejected(change: dict[str, object], message: str) -> None:
    layout = model_ready_pointcloud_layout(np.ones((1, 4), dtype=np.float32))
    values = {
        "height": layout.height,
        "width": layout.width,
        "fields": layout.fields,
        "is_bigendian": layout.is_bigendian,
        "point_step": layout.point_step,
        "row_step": layout.row_step,
        "data": layout.data,
        **change,
    }
    with pytest.raises(ValueError, match=message):
        decode_model_ready_pointcloud(PointCloud2Layout(**values))


def test_non_float_or_nonfinite_required_points_are_rejected() -> None:
    layout = model_ready_pointcloud_layout(np.ones((1, 4), dtype=np.float32))
    wrong_type = PointCloud2Layout(
        height=layout.height,
        width=layout.width,
        fields=tuple(
            PointFieldLayout(field.name, field.offset, 6 if field.name == "x" else field.datatype)
            for field in layout.fields
        ),
        is_bigendian=False,
        point_step=layout.point_step,
        row_step=layout.row_step,
        data=layout.data,
    )
    with pytest.raises(ValueError, match="x must be one float32"):
        decode_model_ready_pointcloud(wrong_type)
    nonfinite = np.ones((1, 4), dtype=np.float32)
    nonfinite[0, 3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ModelReadyPointCloud(nonfinite)


def test_detection_mapping_copies_geometric_center_lwh_class_score_id_and_header() -> None:
    header = SourceHeader("current_lidar", TimeStamp(123, 456))
    records = detection_frame_to_records(_frame(), header)
    record = records.detections[0]

    assert records.header == header
    assert record.header == header
    assert record.center.position_xyz == (4.0, -2.0, 1.25)
    assert record.center.position_xyz[2] == 1.25  # no height/2 shift
    assert record.size_xyz == (4.5, 1.8, 1.6)
    assert record.results[0].class_id == "car"
    assert record.results[0].score == 0.875
    assert record.results[0].pose == record.center
    assert record.tracking_id == ""


@pytest.mark.parametrize(
    ("yaw", "expected_z", "expected_w"),
    [
        (0.0, 0.0, 1.0),
        (pi / 2.0, sqrt(0.5), sqrt(0.5)),
        (-pi / 2.0, -sqrt(0.5), sqrt(0.5)),
    ],
)
def test_yaw_quaternion_convention_and_normalization(
    yaw: float, expected_z: float, expected_w: float
) -> None:
    quaternion = yaw_to_quaternion(yaw)
    assert quaternion.x == 0.0
    assert quaternion.y == 0.0
    assert quaternion.z == pytest.approx(expected_z)
    assert quaternion.w == pytest.approx(expected_w)
    assert sum(value**2 for value in (quaternion.x, quaternion.y, quaternion.z, quaternion.w)) == (
        pytest.approx(1.0)
    )


def test_empty_detection_frame_maps_to_header_only_array() -> None:
    frame = DetectionFrame(detections=(), sample_id="empty", coordinate_frame="current_lidar")
    header = SourceHeader("current_lidar", TimeStamp(1, 2))
    records = detection_frame_to_records(frame, header)
    assert records.header == header
    assert records.detections == ()
