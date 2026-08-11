from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory("laserperception_ros"))
    parameters = str(share / "config/m3_ros2.yaml")
    rviz_config = str(share / "config/m3_demo.rviz")
    return LaunchDescription(
        [
            DeclareLaunchArgument("run_replay", default_value="true"),
            DeclareLaunchArgument("run_rviz", default_value="true"),
            Node(
                package="laserperception_ros",
                executable="laserperception_detector",
                name="laserperception_detector",
                output="screen",
                parameters=[parameters],
            ),
            Node(
                package="laserperception_ros",
                executable="laserperception_replay",
                name="laserperception_replay",
                output="screen",
                parameters=[parameters],
                condition=IfCondition(LaunchConfiguration("run_replay")),
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", rviz_config],
                condition=IfCondition(LaunchConfiguration("run_rviz")),
            ),
        ]
    )
