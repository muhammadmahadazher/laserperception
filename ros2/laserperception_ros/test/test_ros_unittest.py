from __future__ import annotations

import math
import unittest

import numpy as np
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
                yaw_rad=math.pi / 2.0,
                score=0.75,
                class_id=7,
                class_name="pedestrian",
            ),
        ),
        sample_id="sample",
        coordinate_frame="current_lidar",
    )


class _RuntimeThatMustNotRun:
    engine_sha256 = "test"

    def infer(self, *_args: object, **_kwargs: object) -> DetectionFrame:
        raise AssertionError("invalid messages must not reach inference")


class RosConversionTest(unittest.TestCase):
    def test_detection_message_contract(self) -> None:
        message = detection_frame_to_message(_frame(), _header())
        detection = message.detections[0]
        self.assertEqual(message.header.frame_id, "current_lidar")
        self.assertEqual(message.header.stamp.sec, 42)
        self.assertEqual(detection.header, message.header)
        self.assertEqual(detection.bbox.center.position.z, 3.0)
        self.assertEqual(
            (detection.bbox.size.x, detection.bbox.size.y, detection.bbox.size.z),
            (4.0, 2.0, 1.5),
        )
        self.assertAlmostEqual(detection.bbox.center.orientation.z, math.sqrt(0.5))
        self.assertAlmostEqual(detection.bbox.center.orientation.w, math.sqrt(0.5))
        self.assertEqual(detection.results[0].hypothesis.class_id, "pedestrian")
        self.assertEqual(detection.results[0].hypothesis.score, 0.75)
        self.assertEqual(detection.results[0].pose.pose, detection.bbox.center)
        self.assertEqual(detection.id, "")

    def test_model_ready_message_round_trip(self) -> None:
        source = ModelReadyPointCloud(
            np.array([[1.0, 2.0, 3.0, 0.0], [4.0, 5.0, 6.0, 0.5]], dtype=np.float32)
        )
        result = pointcloud2_to_model_ready(model_ready_to_pointcloud2(source, _header()))
        self.assertEqual(result.sha256, source.sha256)
        self.assertTrue(np.array_equal(result.points_xyzt, source.points_xyzt))

    def test_extra_reordered_fields(self) -> None:
        values = np.array([[100.0, 0.2, 1.0, 3.0, 2.0]], dtype="<f4")
        message = PointCloud2(
            header=_header(),
            height=1,
            width=1,
            fields=[
                PointField(name=name, offset=index * 4, datatype=PointField.FLOAT32, count=1)
                for index, name in enumerate(("intensity", "time_lag", "x", "z", "y"))
            ],
            point_step=20,
            row_step=20,
            data=values.tobytes(),
            is_dense=True,
        )
        expected = np.array([[1.0, 2.0, 3.0, 0.2]], dtype=np.float32)
        self.assertTrue(np.array_equal(pointcloud2_to_model_ready(message).points_xyzt, expected))

    def test_missing_time_lag_is_rejected_before_inference(self) -> None:
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
            self.assertEqual(node.received_count, 1)
            self.assertEqual(node.accepted_count, 0)
            self.assertEqual(node.rejected_count, 1)
            self.assertEqual(node.published_count, 0)
        finally:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    unittest.main()
