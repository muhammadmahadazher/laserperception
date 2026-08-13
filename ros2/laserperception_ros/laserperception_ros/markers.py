"""Optional per-frame RViz/Foxglove markers derived from a DetectionFrame."""

from __future__ import annotations

from builtin_interfaces.msg import Duration
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

from laserperception.detection.ros2_contract import yaw_to_quaternion
from laserperception.detection.types import DetectionFrame
from laserperception_ros.conversion import copy_header


def detection_frame_to_markers(frame: DetectionFrame, source_header: Header) -> MarkerArray:
    """Clear previous markers and recreate non-tracking box/label markers."""

    clear = Marker()
    clear.header = copy_header(source_header)
    clear.action = Marker.DELETEALL
    markers = [clear]
    for index, detection in enumerate(frame.detections):
        box = Marker()
        box.header = copy_header(source_header)
        box.ns = "laserperception_boxes_per_frame"
        box.id = index
        box.type = Marker.CUBE
        box.action = Marker.ADD
        box.pose.position.x, box.pose.position.y, box.pose.position.z = detection.center_xyz
        quaternion = yaw_to_quaternion(detection.yaw_rad)
        box.pose.orientation.x = quaternion.x
        box.pose.orientation.y = quaternion.y
        box.pose.orientation.z = quaternion.z
        box.pose.orientation.w = quaternion.w
        box.scale.x, box.scale.y, box.scale.z = detection.size_lwh
        box.color.r, box.color.g, box.color.b, box.color.a = _class_color(detection.class_id)
        box.lifetime = Duration(sec=0, nanosec=200_000_000)
        markers.append(box)

        label = Marker()
        label.header = copy_header(source_header)
        label.ns = "laserperception_labels_per_frame"
        label.id = index
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = detection.center_xyz[0]
        label.pose.position.y = detection.center_xyz[1]
        label.pose.position.z = detection.center_xyz[2] + detection.size_lwh[2] / 2.0 + 0.3
        label.pose.orientation.w = 1.0
        label.scale.z = 0.55
        label.color.r = 1.0
        label.color.g = 1.0
        label.color.b = 1.0
        label.color.a = 0.95
        label.text = f"{detection.class_name} {detection.score:.2f}"
        label.lifetime = Duration(sec=0, nanosec=200_000_000)
        markers.append(label)
    return MarkerArray(markers=markers)


def _class_color(class_id: int) -> tuple[float, float, float, float]:
    palette = (
        (0.15, 0.80, 1.00, 0.55),
        (1.00, 0.55, 0.10, 0.55),
        (0.65, 0.35, 1.00, 0.55),
        (0.20, 1.00, 0.40, 0.55),
        (0.95, 0.25, 0.25, 0.55),
        (1.00, 0.85, 0.15, 0.55),
        (0.95, 0.40, 0.75, 0.55),
        (0.45, 1.00, 0.85, 0.55),
        (0.90, 0.90, 0.90, 0.55),
        (0.60, 0.60, 0.60, 0.55),
    )
    return palette[class_id % len(palette)]
