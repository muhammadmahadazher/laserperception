from setuptools import find_packages, setup

PACKAGE_NAME = "laserperception_ros"

setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml"]),
        (f"share/{PACKAGE_NAME}/launch", ["launch/m3_demo.launch.py"]),
        (f"share/{PACKAGE_NAME}/config", ["config/m3_ros2.yaml", "config/m3_demo.rviz"]),
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
    description="ROS 2 Humble interface for the LaserPerception M3 deployment path.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "laserperception_detector = laserperception_ros.detector_node:main",
            "laserperception_replay = laserperception_ros.replay_node:main",
        ],
    },
)
