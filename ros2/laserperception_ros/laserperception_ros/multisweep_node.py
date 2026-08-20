"""Live raw PointCloud2 plus time-aware TF multi-sweep builder node."""

from __future__ import annotations

from typing import Any

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformException, TransformListener

from laserperception.detection.live_multisweep import (
    LiveRawSweep,
    LiveSweepHistory,
    live_raw_sweep_from_xyz,
    sweep_transform_from_ros,
)
from laserperception.detection.multisweep import (
    HistoricalSweep,
    MultiSweepBuilder,
    MultiSweepBuilderConfig,
)
from laserperception_ros.conversion import (
    copy_header,
    model_ready_to_pointcloud2,
    pointcloud2_to_raw_xyz,
    source_header_record,
)


class LaserPerceptionMultiSweepNode(Node):
    """Build exact model-ready history from raw sweeps and cross-time TF."""

    def __init__(
        self,
        *,
        tf_buffer: Any | None = None,
        tf_listener: Any | None = None,
        parameter_overrides: list[Any] | None = None,
    ) -> None:
        super().__init__(
            "laserperception_multisweep_builder",
            parameter_overrides=parameter_overrides,
        )
        self.declare_parameter("raw_points_topic", "/laserperception/points_raw")
        self.declare_parameter("model_ready_topic", "/laserperception/points_model_ready")
        self.declare_parameter("fixed_frame", "")
        self.declare_parameter("target_frame", "")
        self.declare_parameter("max_historical_sweeps", 10)
        self.declare_parameter("transform_timeout_sec", 0.2)
        self.declare_parameter("history_reset_gap_sec", 0.0)
        self.declare_parameter("tf_cache_time_sec", 10.0)
        self.declare_parameter("raw_qos_depth", 5)
        self.declare_parameter("model_ready_qos_depth", 1)

        self._fixed_frame = str(self.get_parameter("fixed_frame").value).strip()
        self._target_frame = str(self.get_parameter("target_frame").value).strip()
        self._transform_timeout_sec = float(self.get_parameter("transform_timeout_sec").value)
        max_history = int(self.get_parameter("max_historical_sweeps").value)
        reset_gap = float(self.get_parameter("history_reset_gap_sec").value)
        cache_time = float(self.get_parameter("tf_cache_time_sec").value)
        raw_depth = int(self.get_parameter("raw_qos_depth").value)
        model_ready_depth = int(self.get_parameter("model_ready_qos_depth").value)
        if not self._fixed_frame:
            raise ValueError("fixed_frame is required for time-aware history")
        if self._transform_timeout_sec <= 0.0:
            raise ValueError("transform_timeout_sec must be positive")
        if cache_time <= 0.0:
            raise ValueError("tf_cache_time_sec must be positive")

        self._history = LiveSweepHistory(
            max_historical_sweeps=max_history,
            reset_gap_sec=reset_gap,
        )
        self._builder = MultiSweepBuilder(
            MultiSweepBuilderConfig(max_historical_sweeps=max_history)
        )
        self._owns_tf_listener = tf_buffer is None
        if tf_buffer is None:
            self._tf_buffer = Buffer(cache_time=Duration(seconds=cache_time))
            self._tf_listener = TransformListener(
                self._tf_buffer,
                None,
                spin_thread=True,
            )
        else:
            self._tf_buffer = tf_buffer
            self._tf_listener = tf_listener

        self.raw_frames_received = 0
        self.valid_raw_frames = 0
        self.invalid_points_filtered = 0
        self.model_ready_frames_published = 0
        self.rejected_frames = 0
        self.tf_failures = 0
        self.history_resets = 0
        self.current_history_depth = 0
        self._listener_thread_alive_after_shutdown = False

        raw_qos = _qos(depth=raw_depth, reliability=ReliabilityPolicy.BEST_EFFORT)
        model_ready_qos = _qos(
            depth=model_ready_depth,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self._publisher = self.create_publisher(
            PointCloud2,
            str(self.get_parameter("model_ready_topic").value),
            model_ready_qos,
        )
        self._subscription = self.create_subscription(
            PointCloud2,
            str(self.get_parameter("raw_points_topic").value),
            self._on_raw_points,
            raw_qos,
        )
        self.get_logger().info(
            "M4.5b raw multi-sweep builder ready; "
            f"fixed_frame={self._fixed_frame}, target_frame={self._target_frame or '<current>'}, "
            f"max_history={max_history}, tf_listener=dedicated_thread"
        )

    def _on_raw_points(self, message: PointCloud2) -> None:
        self.raw_frames_received += 1
        current: LiveRawSweep | None = None
        history_selected = False
        try:
            decoded = pointcloud2_to_raw_xyz(message)
            self.invalid_points_filtered += decoded.invalid_point_count
            header = source_header_record(message.header)
            current = live_raw_sweep_from_xyz(
                decoded.points_xyz,
                frame_id=header.frame_id,
                stamp=header.stamp,
            )
            self.valid_raw_frames += 1
            target_frame = self._target_frame or current.frame_id
            if target_frame != current.frame_id:
                raise ValueError(
                    "target_frame must be empty or equal the current raw frame; "
                    "cross-frame current-sweep conversion is not supported"
                )

            selection = self._history.select_for_current(current)
            history_selected = True
            if selection.reset_reason is not None:
                self.history_resets += 1
            self.current_history_depth = len(selection.historical)
            historical = tuple(
                self._historical_sweep(item, current, target_frame) for item in selection.historical
            )
            model_ready = self._builder.build(current.sweep, historical)
            output_header = copy_header(message.header)
            output_header.frame_id = target_frame
            self._publisher.publish(model_ready_to_pointcloud2(model_ready, output_header))
            self.model_ready_frames_published += 1
        except TransformException as error:
            self.tf_failures += 1
            self.rejected_frames += 1
            self.get_logger().warning(
                f"rejected raw frame because required cross-time TF is unavailable: {error}",
                throttle_duration_sec=5.0,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            self.rejected_frames += 1
            self.get_logger().warning(
                f"rejected raw PointCloud2: {error}",
                throttle_duration_sec=5.0,
            )
        finally:
            if current is not None and history_selected:
                self._history.store_current(current)

    def _historical_sweep(
        self,
        historical: LiveRawSweep,
        current: LiveRawSweep,
        target_frame: str,
    ) -> HistoricalSweep:
        transform = self._tf_buffer.lookup_transform_full(
            target_frame,
            Time.from_msg(_time_message(current)),
            historical.frame_id,
            Time.from_msg(_time_message(historical)),
            self._fixed_frame,
            timeout=Duration(seconds=self._transform_timeout_sec),
        )
        encoded = sweep_transform_from_ros(
            translation_xyz=(
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ),
            quaternion_xyzw=(
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ),
            source_id=historical.sweep.source_id,
            target_id=current.sweep.source_id,
        )
        return HistoricalSweep(historical.sweep, encoded)

    def destroy_node(self) -> bool:
        """Stop the owned listener executor before destroying the raw node."""

        if self._owns_tf_listener and self._tf_listener is not None:
            listener = self._tf_listener
            listener.executor.shutdown()
            listener.dedicated_listener_thread.join(timeout=2.0)
            self._listener_thread_alive_after_shutdown = (
                listener.dedicated_listener_thread.is_alive()
            )
            listener.unregister()
            listener.unregister = lambda: None
            listener.node.destroy_node()
            self._tf_listener = None
        return super().destroy_node()


def _time_message(sweep: LiveRawSweep) -> Any:
    message = Time(nanoseconds=sweep.stamp_ns).to_msg()
    return message


def _qos(*, depth: int, reliability: ReliabilityPolicy) -> QoSProfile:
    if depth <= 0:
        raise ValueError("QoS depth must be positive")
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=reliability,
        durability=DurabilityPolicy.VOLATILE,
    )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LaserPerceptionMultiSweepNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
