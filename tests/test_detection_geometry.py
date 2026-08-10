from math import pi

import numpy as np

from laserperception.detection import Detection3D, bev_corners


def _box(yaw: float) -> Detection3D:
    return Detection3D(
        center_xyz=(10.0, 20.0, 1.0),
        size_lwh=(4.0, 2.0, 1.5),
        yaw_rad=yaw,
        score=0.9,
        class_id=0,
        class_name="car",
    )


def test_bev_corners_preserve_length_width_order_at_zero_yaw() -> None:
    corners = bev_corners(_box(0.0))

    np.testing.assert_allclose(
        corners,
        np.array([[12.0, 21.0], [8.0, 21.0], [8.0, 19.0], [12.0, 19.0]]),
    )


def test_bev_corners_rotate_counter_clockwise() -> None:
    corners = bev_corners(_box(pi / 2.0))

    np.testing.assert_allclose(
        corners,
        np.array([[9.0, 22.0], [9.0, 18.0], [11.0, 18.0], [11.0, 22.0]]),
        atol=1e-12,
    )
    np.testing.assert_allclose(corners.mean(axis=0), np.array([10.0, 20.0]))
