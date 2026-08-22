import math

import numpy as np
import pytest

from laserperception.evaluation.kitti_m6b import (
    KittiTrackletPose,
    convert_tracklet_pose,
    normalize_angle,
)


def _pose(yaw: float) -> KittiTrackletPose:
    return KittiTrackletPose(
        track_id=4,
        frame_index=15,
        object_type="Car",
        height=1.8,
        width=1.9,
        length=4.2,
        translation_xyz=(4.0, -3.0, -1.5),
        rotation_xyz=(0.0, 0.0, yaw),
        state=2,
        occlusion=0,
        truncation=0,
    )


@pytest.mark.parametrize(
    ("source_yaw", "expected_yaw"),
    [
        (0.0, math.pi / 2.0),
        (math.pi / 2.0, -math.pi),
        (-math.pi / 2.0, 0.0),
        (0.37, 0.37 + math.pi / 2.0),
    ],
)
def test_fail_first_yaw_fixtures_reject_zero_or_minus_pi_over_two_rules(
    source_yaw: float, expected_yaw: float
) -> None:
    converted = convert_tracklet_pose(_pose(source_yaw))

    assert converted.yaw_rad == pytest.approx(normalize_angle(expected_yaw))
    if source_yaw == 0.0:
        assert converted.yaw_rad != pytest.approx(source_yaw)
        assert converted.yaw_rad != pytest.approx(normalize_angle(source_yaw - math.pi / 2.0))


def test_inverse_basis_and_centre_shift_recover_source_bottom_centre() -> None:
    source = _pose(0.37)
    converted = convert_tracklet_pose(source)
    centre_model = np.asarray(converted.center_xyz)
    basis = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    centre_kitti = basis.T @ centre_model
    recovered_bottom = centre_kitti - np.array([0.0, 0.0, source.height / 2.0])

    assert recovered_bottom == pytest.approx(source.translation_xyz)
    assert converted.size_lwh == (source.length, source.width, source.height)
    assert normalize_angle(converted.yaw_rad - math.pi / 2.0) == pytest.approx(
        source.rotation_xyz[2]
    )
