from setuptools import find_packages, setup

PACKAGE_NAME = "laserperception_ros"

setup(
    name=PACKAGE_NAME,
    version="0.2.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml"]),
        (
            f"share/{PACKAGE_NAME}/launch",
            ["launch/m3_demo.launch.py", "launch/m45b_raw_multisweep.launch.py"],
        ),
        (
            f"share/{PACKAGE_NAME}/config",
            ["config/m3_ros2.yaml", "config/m3_demo.rviz", "config/m45b_multisweep.yaml"],
        ),
        (
            f"share/{PACKAGE_NAME}/config/detection",
            [
                "../../configs/detection/m1_pointpillars_nuscenes.yaml",
                "../../configs/detection/m2_pointpillars_tensorrt.yaml",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="Mahad Azher",
    maintainer_email="muhammadmahadazher@users.noreply.github.com",
    description=(
        "ROS 2 Humble interface for LaserPerception detection and raw multi-sweep ingestion."
    ),
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "laserperception_detector = laserperception_ros.detector_node:main",
            "laserperception_multisweep_builder = laserperception_ros.multisweep_node:main",
            "laserperception_nuscenes_raw_replay = laserperception_ros.raw_replay_node:main",
            "laserperception_replay = laserperception_ros.replay_node:main",
        ],
    },
)
