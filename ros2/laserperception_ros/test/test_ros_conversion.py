from __future__ import annotations

from math import pi, sqrt

import numpy as np
import pytest
import rclpy
from builtin_interfaces.msg import Time
from laserperception_ros.conversion import (
    detection_frame_to_message,
    model_ready_to_pointcloud2,
    pointcloud2_to_model_ready,
)
from laserperception_ros.detector_node import LaserPerceptionDetectorNode
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from laserperception.detection.ros2_contract import ModelReadyPointCloud
from laserperception.detection.types import Detection3D, DetectionFrame


def _header() -> Header:
    return Header(stamp=Time(sec=42, nanosec=123_456_789), frame_id="current_lidar")


def _frame() -> DetectionFrame:
    return DetectionFrame(
        detections=(
            Detection3D(
                center_xyz=(1.0, 2.0, 3.0),
                size_lwh=(4.0, 2.0, 1.5),
                yaw_rad=pi / 2.0,
                score=0.75,
                class_id=7,
                class_name="pedestrian",
                velocity_xy=(0.2, -0.1),
            ),
        ),
        sample_id="sample",
        coordinate_frame="current_lidar",
    )


def test_detection_message_preserves_headers_and_exact_box_contract() -> None:
    message = detection_frame_to_message(_frame(), _header())
    detection = message.detections[0]

    assert message.header.frame_id == "current_lidar"
    assert message.header.stamp.sec == 42
    assert message.header.stamp.nanosec == 123_456_789
    assert detection.header == message.header
    assert detection.bbox.center.position.x == 1.0
    assert detection.bbox.center.position.y == 2.0
    assert detection.bbox.center.position.z == 3.0
    assert detection.bbox.size.x == 4.0
    assert detection.bbox.size.y == 2.0
    assert detection.bbox.size.z == 1.5
    assert detection.bbox.center.orientation.z == pytest.approx(sqrt(0.5))
    assert detection.bbox.center.orientation.w == pytest.approx(sqrt(0.5))
    assert detection.results[0].hypothesis.class_id == "pedestrian"
    assert detection.results[0].hypothesis.score == 0.75
    assert detection.results[0].pose.pose == detection.bbox.center
    assert detection.id == ""


def test_model_ready_ros_message_round_trip_is_exact() -> None:
    source = ModelReadyPointCloud(
        np.array([[1.0, 2.0, 3.0, 0.0], [4.0, 5.0, 6.0, 0.5]], dtype=np.float32)
    )
    message = model_ready_to_pointcloud2(source, _header())
    result = pointcloud2_to_model_ready(message)
    assert result.sha256 == source.sha256
    assert np.array_equal(result.points_xyzt, source.points_xyzt)


def test_exact_subscriber_conversion_accepts_extra_and_reordered_fields() -> None:
    values = np.array(
        [[100.0, 0.2, 1.0, 3.0, 2.0], [200.0, 0.0, 4.0, 6.0, 5.0]],
        dtype="<f4",
    )
    message = PointCloud2(
        header=_header(),
        height=1,
        width=2,
        fields=[
            PointField(name="intensity", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="time_lag", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="x", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=16, datatype=PointField.FLOAT32, count=1),
        ],
        is_bigendian=False,
        point_step=20,
        row_step=40,
        data=values.tobytes(),
        is_dense=True,
    )
    assert np.array_equal(
        pointcloud2_to_model_ready(message).points_xyzt,
        np.array([[1.0, 2.0, 3.0, 0.2], [4.0, 5.0, 6.0, 0.0]], dtype=np.float32),
    )


class _RuntimeThatMustNotRun:
    engine_sha256 = "test"

    def infer(self, *_args: object, **_kwargs: object) -> DetectionFrame:
        raise AssertionError("invalid messages must not reach inference")


def test_detector_rejects_missing_time_lag_and_publishes_nothing() -> None:
    rclpy.init()
    node = LaserPerceptionDetectorNode(runtime=_RuntimeThatMustNotRun())
    try:
        message = PointCloud2(
            header=_header(),
            height=1,
            width=1,
            fields=[
                PointField(name=name, offset=index * 4, datatype=PointField.FLOAT32, count=1)
                for index, name in enumerate(("x", "y", "z"))
            ],
            point_step=12,
            row_step=12,
            data=np.ones((1, 3), dtype=np.float32).tobytes(),
            is_dense=True,
        )
        node._on_points(message)
        assert node.received_count == 1
        assert node.accepted_count == 0
        assert node.rejected_count == 1
        assert node.published_count == 0
    finally:
        node.destroy_node()
        rclpy.shutdown()
