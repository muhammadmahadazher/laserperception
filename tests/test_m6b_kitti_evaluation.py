import math
from pathlib import Path

import numpy as np
import pytest

from laserperception.detection.types import Detection3D
from laserperception.evaluation.kitti_m6b import (
    KittiReferenceCamera,
    KittiTrackletPose,
    M6bGroundTruthBox,
    bev_iou,
    convert_tracklet_pose,
    match_detections,
    normalize_angle,
    parse_kitti_tracklets,
    visible_in_reference_camera,
)


def _pose(**overrides: object) -> KittiTrackletPose:
    values: dict[str, object] = {
        "track_id": 3,
        "frame_index": 12,
        "object_type": "Car",
        "height": 2.0,
        "width": 2.0,
        "length": 4.0,
        "translation_xyz": (1.0, 2.0, 3.0),
        "rotation_xyz": (0.0, 0.0, 0.25),
        "state": 2,
        "occlusion": 1,
        "truncation": 0,
    }
    values.update(overrides)
    return KittiTrackletPose(**values)  # type: ignore[arg-type]


def _gt(
    *,
    track_id: int = 1,
    role: str = "target",
    class_name: str = "car",
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> M6bGroundTruthBox:
    return M6bGroundTruthBox(
        track_id=track_id,
        frame_index=10,
        source_type="Car" if role == "target" else "Van",
        evaluation_role=role,
        class_name=class_name,
        center_xyz=center,
        size_lwh=(4.0, 2.0, 2.0),
        yaw_rad=0.0,
    )


def _prediction(
    *,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    score: float = 0.9,
) -> Detection3D:
    return Detection3D(
        center_xyz=center,
        size_lwh=(4.0, 2.0, 2.0),
        yaw_rad=0.0,
        score=score,
        class_id=0,
        class_name="car",
    )


def test_fail_first_tracklet_conversion_uses_bottom_centre_and_frozen_basis() -> None:
    converted = convert_tracklet_pose(_pose())

    assert converted.center_xyz == pytest.approx((-2.0, 1.0, 4.0))
    assert converted.size_lwh == (4.0, 2.0, 2.0)
    assert converted.yaw_rad == pytest.approx(0.25 + math.pi / 2.0)
    assert converted.class_name == "car"
    assert converted.evaluation_role == "target"


def test_tracklet_label_predicate_and_neighbour_mapping_are_explicit() -> None:
    assert _pose().valid_labelled_pose
    assert not _pose(state=1).valid_labelled_pose
    assert not _pose(occlusion=2).valid_labelled_pose
    assert not _pose(truncation=99).valid_labelled_pose
    sitting = _pose(object_type="Person (sitting)")
    truck = _pose(object_type="Truck")
    assert (sitting.evaluation_role, sitting.evaluation_class) == (
        "neighbour_ignore",
        "pedestrian",
    )
    assert (truck.evaluation_role, truck.evaluation_class) == ("unmapped", None)


def test_parse_tracklets_preserves_first_frame_offset(tmp_path: Path) -> None:
    path = tmp_path / "tracklet_labels.xml"
    path.write_text(
        """<?xml version='1.0'?>
<boost_serialization><tracklets><count>1</count><item>
<objectType>Pedestrian</objectType><h>1.7</h><w>0.6</w><l>0.8</l><first_frame>7</first_frame>
<poses><count>1</count><item><tx>1</tx><ty>2</ty><tz>3</tz><rx>0</rx><ry>0</ry>
<rz>0</rz><state>2</state><occlusion>0</occlusion><truncation>0</truncation>
</item></poses></item></tracklets></boost_serialization>""",
        encoding="utf-8",
    )

    poses = parse_kitti_tracklets(path)

    assert len(poses) == 1
    assert poses[0].frame_index == 7
    assert poses[0].object_type == "Pedestrian"


def test_reference_camera_visibility_clips_near_plane_and_intersects_extent() -> None:
    camera = KittiReferenceCamera(
        projection=np.array(
            [[100.0, 0.0, 50.0, 0.0], [0.0, 100.0, 40.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        ),
        rectified_camera_from_velodyne=np.eye(4),
        image_size_wh=(100, 80),
    )
    visible = np.array(
        [
            [-0.1, -0.1, -0.1],
            [0.1, -0.1, -0.1],
            [0.1, 0.1, -0.1],
            [-0.1, 0.1, -0.1],
            [-0.1, -0.1, 2.0],
            [0.1, -0.1, 2.0],
            [0.1, 0.1, 2.0],
            [-0.1, 0.1, 2.0],
        ]
    )
    behind = visible.copy()
    behind[:, 2] = -2.0

    assert visible_in_reference_camera(visible, camera)
    assert not visible_in_reference_camera(behind, camera)


def test_bev_iou_handles_identical_disjoint_and_rotated_boxes() -> None:
    first = _gt()
    assert bev_iou(first, _prediction()) == pytest.approx(1.0)
    assert bev_iou(first, _prediction(center=(10.0, 0.0, 0.0))) == 0.0
    rotated = Detection3D(
        center_xyz=(0.0, 0.0, 0.0),
        size_lwh=(4.0, 2.0, 2.0),
        yaw_rad=math.pi / 2.0,
        score=0.9,
        class_id=0,
        class_name="car",
    )
    assert bev_iou(first, rotated) == pytest.approx(1.0 / 3.0)


def test_matching_is_score_ordered_one_to_one_and_applies_neighbour_ignore() -> None:
    targets = (_gt(track_id=2),)
    ignores = (_gt(track_id=8, role="neighbour_ignore", center=(10.0, 0.0, 0.0)),)
    predictions = (
        _prediction(score=0.8),
        _prediction(score=0.9),
        _prediction(center=(10.0, 0.0, 0.0), score=0.7),
        _prediction(center=(30.0, 0.0, 0.0), score=0.6),
        _prediction(center=(0.0, 0.0, 0.0), score=0.2),
    )

    summary = match_detections(
        predictions,
        targets,
        ignores,
        class_name="car",
        iou_threshold=0.5,
    )

    assert (summary.true_positives, summary.false_positives, summary.false_negatives) == (1, 2, 0)
    assert summary.ignored_predictions == 1
    assert [record.prediction_index for record in summary.records] == [1, 0, 2, 3]
    assert [record.disposition for record in summary.records] == [
        "true_positive",
        "false_positive",
        "ignored_neighbour",
        "false_positive",
    ]


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(0.0, 0.0), (math.pi, -math.pi), (-math.pi, -math.pi), (3 * math.pi, -math.pi)],
)
def test_normalize_angle_uses_half_open_interval(angle: float, expected: float) -> None:
    assert normalize_angle(angle) == pytest.approx(expected)
