from math import nan

import pytest

from laserperception.detection import Detection3D, DetectionFrame


def _detection(*, score: float = 0.8, class_id: int = 0, center_x: float = 1.0) -> Detection3D:
    return Detection3D(
        center_xyz=(center_x, 2.0, 0.5),
        size_lwh=(4.0, 2.0, 1.5),
        yaw_rad=0.25,
        score=score,
        class_id=class_id,
        class_name="car" if class_id == 0 else "pedestrian",
        velocity_xy=(1.0, -0.5),
    )


def test_detection_is_validated_and_json_compatible() -> None:
    detection = _detection()

    assert detection.center_xyz == (1.0, 2.0, 0.5)
    assert detection.size_lwh == (4.0, 2.0, 1.5)
    assert detection.to_dict()["velocity_xy"] == [1.0, -0.5]


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("center_xyz", (0.0, nan, 0.0), "finite"),
        ("size_lwh", (1.0, 0.0, 1.0), "positive"),
        ("yaw_rad", nan, "finite"),
        ("score", 1.1, "between 0 and 1"),
        ("class_id", -1, "non-negative"),
        ("class_name", "", "non-empty"),
    ],
)
def test_detection_rejects_invalid_values(field: str, value: object, error: str) -> None:
    values: dict[str, object] = {
        "center_xyz": (0.0, 0.0, 0.0),
        "size_lwh": (1.0, 1.0, 1.0),
        "yaw_rad": 0.0,
        "score": 0.5,
        "class_id": 0,
        "class_name": "car",
    }
    values[field] = value
    with pytest.raises((TypeError, ValueError), match=error):
        Detection3D(**values)  # type: ignore[arg-type]


def test_frame_orders_deterministically_and_filters_without_mutation() -> None:
    low = _detection(score=0.3, class_id=1, center_x=2.0)
    tied_later = _detection(score=0.8, center_x=4.0)
    tied_first = _detection(score=0.8, center_x=1.0)
    metadata = {"source": {"backend": "synthetic"}}
    frame = DetectionFrame(
        detections=(low, tied_later, tied_first),
        sample_id="sample-token",
        coordinate_frame="nuscenes_lidar_top",
        metadata=metadata,
    )
    metadata["source"] = {"backend": "changed"}

    assert frame.detections == (tied_first, tied_later, low)
    assert frame.metadata["source"] == {"backend": "synthetic"}

    filtered = frame.filtered(0.5)
    assert filtered.detections == (tied_first, tied_later)
    assert len(frame.detections) == 3
    assert filtered.to_dict()["schema_version"] == "1.0"


def test_frame_rejects_invalid_threshold() -> None:
    frame = DetectionFrame(detections=(), sample_id="sample", coordinate_frame="nuscenes_lidar_top")
    with pytest.raises(ValueError, match="between 0 and 1"):
        frame.filtered(-0.1)
