"""Launch deterministic KITTI Raw replay through the existing live builder."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory("laserperception_ros"))
    default_config = str(share / "config/m6c_kitti_ros_exactness.yaml")
    config = LaunchConfiguration("config")
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            Node(
                package="laserperception_ros",
                executable="laserperception_kitti_raw_replay",
                name="laserperception_kitti_raw_replay",
                parameters=[config],
                output="screen",
            ),
            Node(
                package="laserperception_ros",
                executable="laserperception_multisweep_builder",
                name="laserperception_multisweep_builder",
                parameters=[config],
                output="screen",
            ),
        ]
    )
