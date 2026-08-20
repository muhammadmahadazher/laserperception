from __future__ import annotations

import threading
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMessage
from geometry_msgs.msg import TransformStamped
from laserperception_ros.conversion import pointcloud2_to_model_ready, pointcloud2_to_raw_xyz
from laserperception_ros.multisweep_node import LaserPerceptionMultiSweepNode
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformBroadcaster, TransformException


def _stamp(sec: int, nanosec: int = 0) -> TimeMessage:
    return TimeMessage(sec=sec, nanosec=nanosec)


def _raw_message(
    *,
    sec: int,
    points: np.ndarray | None = None,
    frame_id: str = "lidar",
) -> PointCloud2:
    values = np.array([[1.0, 2.0, 0.5]], dtype=np.float32) if points is None else points
    return PointCloud2(
        header=Header(stamp=_stamp(sec), frame_id=frame_id),
        height=1,
        width=len(values),
        fields=[
            PointField(name=name, offset=index * 4, datatype=PointField.FLOAT32, count=1)
            for index, name in enumerate(("x", "y", "z"))
        ],
        is_bigendian=False,
        point_step=12,
        row_step=12 * len(values),
        data=np.asarray(values, dtype=np.float32).tobytes(),
        is_dense=bool(np.isfinite(values).all()),
    )


def _map_to_lidar(sec: int, x: float) -> TransformStamped:
    transform = TransformStamped()
    transform.header = Header(stamp=_stamp(sec), frame_id="map")
    transform.child_frame_id = "lidar"
    transform.transform.translation.x = x
    transform.transform.rotation.w = 1.0
    return transform


def _parameters(timeout: float = 0.1) -> list[Parameter]:
    return [
        Parameter("fixed_frame", value="map"),
        Parameter("transform_timeout_sec", value=timeout),
        Parameter("tf_cache_time_sec", value=20.0),
    ]


class _CapturePublisher:
    def __init__(self) -> None:
        self.messages: list[PointCloud2] = []

    def publish(self, message: PointCloud2) -> None:
        self.messages.append(message)


class _RecordingBuffer:
    def __init__(self) -> None:
        self.fail = False
        self.calls: list[tuple[str, int, str, int, str]] = []

    def lookup_transform_full(
        self,
        target_frame: str,
        target_time: Time,
        source_frame: str,
        source_time: Time,
        fixed_frame: str,
        *,
        timeout: object,
    ) -> TransformStamped:
        del timeout
        self.calls.append(
            (
                target_frame,
                target_time.nanoseconds,
                source_frame,
                source_time.nanoseconds,
                fixed_frame,
            )
        )
        if self.fail:
            raise TransformException("missing test transform")
        transform = TransformStamped()
        transform.header.frame_id = target_frame
        transform.child_frame_id = source_frame
        transform.transform.rotation.w = 1.0
        return transform


def test_raw_ros_conversion_filters_nonfinite_rows_and_keeps_order() -> None:
    message = _raw_message(
        sec=1,
        points=np.array(
            [[1.0, 2.0, 3.0], [np.nan, 4.0, 5.0], [6.0, 7.0, 8.0]],
            dtype=np.float32,
        ),
    )
    decoded = pointcloud2_to_raw_xyz(message)
    assert decoded.invalid_point_count == 1
    assert decoded.points_xyz.tolist() == [[1.0, 2.0, 3.0], [6.0, 7.0, 8.0]]


def test_installed_tf2_same_frame_different_time_is_not_identity() -> None:
    buffer = Buffer()
    buffer.set_transform(_map_to_lidar(1, 0.0), "test")
    buffer.set_transform(_map_to_lidar(2, 1.0), "test")

    transform = buffer.lookup_transform_full(
        "lidar",
        Time(seconds=2),
        "lidar",
        Time(seconds=1),
        "map",
    )
    assert transform.transform.translation.x == -1.0
    assert transform.transform.translation.y == 0.0
    assert transform.transform.rotation.w == 1.0


def test_node_history_fail_closed_retains_valid_acquisition_and_recovers() -> None:
    rclpy.init()
    buffer = _RecordingBuffer()
    node = LaserPerceptionMultiSweepNode(
        tf_buffer=buffer,
        parameter_overrides=_parameters(),
    )
    capture = _CapturePublisher()
    node._publisher = capture
    try:
        node._on_raw_points(_raw_message(sec=1))
        assert node.model_ready_frames_published == 1
        assert pointcloud2_to_model_ready(capture.messages[-1]).points_xyzt.shape == (1, 4)

        buffer.fail = True
        node._on_raw_points(_raw_message(sec=2))
        assert node.tf_failures == 1
        assert node.rejected_frames == 1
        assert node.model_ready_frames_published == 1

        buffer.fail = False
        node._on_raw_points(_raw_message(sec=3))
        assert node.current_history_depth == 2
        assert node.model_ready_frames_published == 2
        assert pointcloud2_to_model_ready(capture.messages[-1]).points_xyzt.shape == (3, 4)
        assert [call[3] for call in buffer.calls[-2:]] == [2_000_000_000, 1_000_000_000]
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_dedicated_listener_processes_delayed_tf_while_raw_callback_waits() -> None:
    rclpy.init()
    node = LaserPerceptionMultiSweepNode(parameter_overrides=_parameters(timeout=0.2))
    broadcaster_node = Node("m45b_test_tf_broadcaster")
    broadcaster = TransformBroadcaster(broadcaster_node)
    try:
        node._on_raw_points(_raw_message(sec=1))
        assert node.model_ready_frames_published == 1

        callback = threading.Thread(target=node._on_raw_points, args=(_raw_message(sec=2),))
        callback.start()
        time.sleep(0.03)
        broadcaster.sendTransform([_map_to_lidar(1, 0.0), _map_to_lidar(2, 1.0)])
        callback.join(timeout=1.0)
        assert not callback.is_alive()
        assert node.tf_failures == 0
        assert node.model_ready_frames_published == 2

        node._on_raw_points(_raw_message(sec=3))
        assert node.tf_failures == 1
        assert node.model_ready_frames_published == 2

        broadcaster.sendTransform([_map_to_lidar(3, 2.0), _map_to_lidar(4, 3.0)])
        time.sleep(0.05)
        node._on_raw_points(_raw_message(sec=4))
        assert node.tf_failures == 1
        assert node.current_history_depth == 3
        assert node.model_ready_frames_published == 3
    finally:
        broadcaster_node.destroy_node()
        node.destroy_node()
        assert not node._listener_thread_alive_after_shutdown
        rclpy.shutdown()


def test_node_rejects_all_invalid_without_buffering_or_publication() -> None:
    rclpy.init()
    node = LaserPerceptionMultiSweepNode(
        tf_buffer=_RecordingBuffer(),
        parameter_overrides=_parameters(),
    )
    capture = _CapturePublisher()
    node._publisher = capture
    try:
        node._on_raw_points(
            _raw_message(
                sec=1,
                points=np.array([[np.nan, 0.0, 0.0], [0.0, np.inf, 0.0]], np.float32),
            )
        )
        assert node.raw_frames_received == 1
        assert node.valid_raw_frames == 0
        assert node.invalid_points_filtered == 2
        assert node.rejected_frames == 1
        assert node.current_history_depth == 0
        assert capture.messages == []
    finally:
        node.destroy_node()
        rclpy.shutdown()
