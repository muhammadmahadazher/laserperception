from __future__ import annotations

import numpy as np
import pytest

from laserperception.detection.live_multisweep import live_raw_sweep_from_xyz
from laserperception.detection.ros2_contract import (
    POINT_FIELD_FLOAT32,
    PointCloud2Layout,
    PointFieldLayout,
    TimeStamp,
    decode_raw_xyz_pointcloud,
)


def test_raw_xyz_accepts_arbitrary_order_extras_organized_rows_and_padding() -> None:
    dtype = np.dtype(
        {
            "names": ["intensity", "z", "x", "ring", "y"],
            "formats": ["<f4", "<f4", "<f4", "<u2", "<f4"],
            "offsets": [0, 4, 8, 12, 16],
            "itemsize": 20,
        }
    )
    structured = np.zeros(4, dtype=dtype)
    structured["intensity"] = [10.0, 20.0, 30.0, 40.0]
    structured["ring"] = [1, 2, 3, 4]
    structured["x"] = [1.0, np.nan, 3.0, 4.0]
    structured["y"] = [5.0, 6.0, 7.0, 8.0]
    structured["z"] = [9.0, 10.0, 11.0, 12.0]
    row_step = 2 * dtype.itemsize + 8
    data = bytearray(row_step * 2)
    raw = structured.tobytes()
    data[: 2 * dtype.itemsize] = raw[: 2 * dtype.itemsize]
    data[row_step : row_step + 2 * dtype.itemsize] = raw[2 * dtype.itemsize :]
    layout = PointCloud2Layout(
        height=2,
        width=2,
        fields=tuple(
            PointFieldLayout(
                name=name,
                offset=dtype.fields[name][1],
                datatype=4 if name == "ring" else POINT_FIELD_FLOAT32,
            )
            for name in dtype.names or ()
        ),
        is_bigendian=False,
        point_step=dtype.itemsize,
        row_step=row_step,
        data=bytes(data),
    )

    decoded = decode_raw_xyz_pointcloud(layout)
    assert decoded.source_point_count == 4
    assert decoded.invalid_point_count == 1
    assert decoded.points_xyz.flags.c_contiguous
    assert np.array_equal(
        decoded.points_xyz,
        np.array([[1.0, 5.0, 9.0], [3.0, 7.0, 11.0], [4.0, 8.0, 12.0]], np.float32),
    )


def test_raw_xyz_handles_big_endian_float32() -> None:
    points = np.array([[1.25, -2.5, 3.75]], dtype=">f4")
    layout = PointCloud2Layout(
        height=1,
        width=1,
        fields=tuple(
            PointFieldLayout(name, index * 4, POINT_FIELD_FLOAT32)
            for index, name in enumerate(("x", "y", "z"))
        ),
        is_bigendian=True,
        point_step=12,
        row_step=12,
        data=points.tobytes(),
    )
    decoded = decode_raw_xyz_pointcloud(layout)
    assert decoded.points_xyz.dtype == np.float32
    assert decoded.points_xyz.tolist() == [[1.25, -2.5, 3.75]]


def test_all_nonfinite_rows_are_counted_then_rejected_before_history() -> None:
    points = np.array([[np.nan, 0.0, 0.0], [0.0, np.inf, 0.0]], dtype=np.float32)
    layout = PointCloud2Layout(
        height=1,
        width=2,
        fields=tuple(
            PointFieldLayout(name, index * 4, POINT_FIELD_FLOAT32)
            for index, name in enumerate(("x", "y", "z"))
        ),
        is_bigendian=False,
        point_step=12,
        row_step=24,
        data=points.tobytes(),
    )
    decoded = decode_raw_xyz_pointcloud(layout)
    assert decoded.source_point_count == 2
    assert decoded.invalid_point_count == 2
    assert decoded.points_xyz.shape == (0, 3)
    with pytest.raises(ValueError, match="at least one"):
        live_raw_sweep_from_xyz(
            decoded.points_xyz,
            frame_id="lidar",
            stamp=TimeStamp(sec=1, nanosec=0),
        )


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (
            (
                PointFieldLayout("x", 0, POINT_FIELD_FLOAT32),
                PointFieldLayout("y", 4, POINT_FIELD_FLOAT32),
            ),
            "missing required field.*z",
        ),
        (
            (
                PointFieldLayout("x", 0, 6),
                PointFieldLayout("y", 4, POINT_FIELD_FLOAT32),
                PointFieldLayout("z", 8, POINT_FIELD_FLOAT32),
            ),
            "x must be one float32",
        ),
        (
            (
                PointFieldLayout("x", 0, POINT_FIELD_FLOAT32, count=2),
                PointFieldLayout("y", 4, POINT_FIELD_FLOAT32),
                PointFieldLayout("z", 8, POINT_FIELD_FLOAT32),
            ),
            "x must be one float32",
        ),
    ],
)
def test_raw_xyz_rejects_missing_or_malformed_required_fields(
    fields: tuple[PointFieldLayout, ...], message: str
) -> None:
    layout = PointCloud2Layout(
        height=1,
        width=1,
        fields=fields,
        is_bigendian=False,
        point_step=12,
        row_step=12,
        data=np.zeros((1, 3), dtype=np.float32).tobytes(),
    )
    with pytest.raises(ValueError, match=message):
        decode_raw_xyz_pointcloud(layout)
