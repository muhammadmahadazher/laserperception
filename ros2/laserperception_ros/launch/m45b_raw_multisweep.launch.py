from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory("laserperception_ros"))
    parameters = str(share / "config/m45b_multisweep.yaml")
    rviz_config = str(share / "config/m3_demo.rviz")
    return LaunchDescription(
        [
            DeclareLaunchArgument("run_raw_replay", default_value="true"),
            DeclareLaunchArgument("run_detector", default_value="true"),
            DeclareLaunchArgument("run_rviz", default_value="true"),
            Node(
                package="laserperception_ros",
                executable="laserperception_multisweep_builder",
                name="laserperception_multisweep_builder",
                output="screen",
                parameters=[parameters],
            ),
            Node(
                package="laserperception_ros",
                executable="laserperception_detector",
                name="laserperception_detector",
                output="screen",
                parameters=[parameters],
                condition=IfCondition(LaunchConfiguration("run_detector")),
            ),
            Node(
                package="laserperception_ros",
                executable="laserperception_nuscenes_raw_replay",
                name="laserperception_nuscenes_raw_replay",
                output="screen",
                parameters=[parameters],
                condition=IfCondition(LaunchConfiguration("run_raw_replay")),
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", rviz_config],
                condition=IfCondition(LaunchConfiguration("run_rviz")),
            ),
        ]
    )
