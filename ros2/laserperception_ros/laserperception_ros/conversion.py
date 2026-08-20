"""Exact ROS-message conversion for the M3A public interface."""

from __future__ import annotations

from builtin_interfaces.msg import Time
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from laserperception.detection.ros2_contract import (
    ModelReadyPointCloud,
    PointCloud2Layout,
    PointFieldLayout,
    RawPointCloudXYZ,
    SourceHeader,
    TimeStamp,
    decode_model_ready_pointcloud,
    decode_raw_xyz_pointcloud,
    detection_frame_to_records,
    model_ready_pointcloud_layout,
)
from laserperception.detection.types import DetectionFrame


def pointcloud2_to_model_ready(message: PointCloud2) -> ModelReadyPointCloud:
    """Run the exact M3 subscriber conversion without assuming field order."""

    return decode_model_ready_pointcloud(pointcloud2_layout(message))


def pointcloud2_to_raw_xyz(message: PointCloud2) -> RawPointCloudXYZ:
    """Decode and filter an ordinary single-sweep float32 XYZ message."""

    return decode_raw_xyz_pointcloud(pointcloud2_layout(message))


def pointcloud2_layout(message: PointCloud2) -> PointCloud2Layout:
    """Copy the byte-layout portion of a ROS PointCloud2 message."""

    return PointCloud2Layout(
        height=int(message.height),
        width=int(message.width),
        fields=tuple(
            PointFieldLayout(
                name=str(field.name),
                offset=int(field.offset),
                datatype=int(field.datatype),
                count=int(field.count),
            )
            for field in message.fields
        ),
        is_bigendian=bool(message.is_bigendian),
        point_step=int(message.point_step),
        row_step=int(message.row_step),
        data=bytes(message.data),
    )


def model_ready_to_pointcloud2(
    points: ModelReadyPointCloud,
    header: Header,
) -> PointCloud2:
    """Serialize replay points to the canonical four-field M3 PointCloud2."""

    layout = model_ready_pointcloud_layout(points)
    message = PointCloud2()
    message.header = copy_header(header)
    message.height = layout.height
    message.width = layout.width
    message.fields = [
        PointField(
            name=field.name,
            offset=field.offset,
            datatype=field.datatype,
            count=field.count,
        )
        for field in layout.fields
    ]
    message.is_bigendian = layout.is_bigendian
    message.point_step = layout.point_step
    message.row_step = layout.row_step
    message.data = layout.data
    message.is_dense = True
    return message


def detection_frame_to_message(
    frame: DetectionFrame,
    source_header: Header,
) -> Detection3DArray:
    """Map geometric-center boxes and preserve the exact input header."""

    records = detection_frame_to_records(frame, source_header_record(source_header))
    message = Detection3DArray()
    message.header = copy_header(source_header)
    message.detections = [_detection_message(record) for record in records.detections]
    return message


def source_header_record(header: Header) -> SourceHeader:
    """Convert a ROS header without changing its stamp or frame."""

    return SourceHeader(
        frame_id=str(header.frame_id),
        stamp=TimeStamp(sec=int(header.stamp.sec), nanosec=int(header.stamp.nanosec)),
    )


def copy_header(source: Header) -> Header:
    """Copy a source header rather than substituting inference-completion time."""

    return Header(
        stamp=Time(sec=int(source.stamp.sec), nanosec=int(source.stamp.nanosec)),
        frame_id=str(source.frame_id),
    )


def _detection_message(record: object) -> Detection3D:
    message = Detection3D()
    message.header = Header(
        stamp=Time(sec=record.header.stamp.sec, nanosec=record.header.stamp.nanosec),
        frame_id=record.header.frame_id,
    )
    message.bbox.center.position.x = record.center.position_xyz[0]
    message.bbox.center.position.y = record.center.position_xyz[1]
    message.bbox.center.position.z = record.center.position_xyz[2]
    message.bbox.center.orientation.x = record.center.orientation.x
    message.bbox.center.orientation.y = record.center.orientation.y
    message.bbox.center.orientation.z = record.center.orientation.z
    message.bbox.center.orientation.w = record.center.orientation.w
    message.bbox.size.x = record.size_xyz[0]
    message.bbox.size.y = record.size_xyz[1]
    message.bbox.size.z = record.size_xyz[2]
    message.id = record.tracking_id
    message.results = [_hypothesis_message(hypothesis) for hypothesis in record.results]
    return message


def _hypothesis_message(record: object) -> ObjectHypothesisWithPose:
    message = ObjectHypothesisWithPose()
    message.hypothesis.class_id = record.class_id
    message.hypothesis.score = record.score
    message.pose.pose.position.x = record.pose.position_xyz[0]
    message.pose.pose.position.y = record.pose.position_xyz[1]
    message.pose.pose.position.z = record.pose.position_xyz[2]
    message.pose.pose.orientation.x = record.pose.orientation.x
    message.pose.pose.orientation.y = record.pose.orientation.y
    message.pose.pose.orientation.z = record.pose.orientation.z
    message.pose.pose.orientation.w = record.pose.orientation.w
    return message
