# Third-party notices

LaserPerception's own source code is licensed under Apache-2.0. The following software, model
artifact, datasets, and specifications remain third-party material under their own licenses and
terms. They are not vendored or redistributed by this repository unless explicitly stated.

## Core Python dependencies

The Python distribution declares NumPy and laspy, with lazrs as an optional LAZ backend and
Matplotlib as an optional headless-visualization backend. These packages are installed from their
upstream distributions and remain governed by their upstream license and notice files.

## M1 OpenMMLab and PyTorch environment

M1 wraps installed copies of:

- [MMDetection3D 1.4.0](https://github.com/open-mmlab/mmdetection3d/tree/v1.4.0), commit
  `fe25f7a51d36e3702f961e198894580d83c4387b`, Apache-2.0;
- [MMDetection 3.2.0](https://github.com/open-mmlab/mmdetection/tree/v3.2.0), Apache-2.0;
- [MMCV 2.1.0](https://github.com/open-mmlab/mmcv/tree/v2.1.0), Apache-2.0;
- [MMEngine 0.10.7](https://github.com/open-mmlab/mmengine/tree/v0.10.7), Apache-2.0; and
- [PyTorch 2.1.0 and torchvision 0.16.0](https://github.com/pytorch/pytorch/tree/v2.1.0),
  under their upstream BSD-style licenses and notices.

No OpenMMLab or PyTorch source is copied into LaserPerception. The setup script clones/installs the
official packages into an external user cache and environment.

## M2 deployment environment

M2 additionally installs external copies of:

- [MMDeploy 1.3.1](https://github.com/open-mmlab/mmdeploy/tree/v1.3.1), commit
  `bc75c9d6c8940aa03d0e1e5b5962bd930478ba77`, Apache-2.0;
- [ONNX 1.14.1](https://github.com/onnx/onnx/tree/v1.14.1), Apache-2.0; and
- NVIDIA CUDA 11.8 libraries, cuDNN 8.9.7, and
  [TensorRT 8.6.1](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/pdf/TensorRT-SLA.pdf),
  governed by NVIDIA's applicable software license terms.

These packages, the generated ONNX file, and serialized TensorRT engines are not vendored or
redistributed by LaserPerception. The setup script installs them into the isolated WSL environment
and external cache. A TensorRT engine is environment-specific and is not covered by the
repository's Apache-2.0 license.

## M3 ROS 2 and middleware environment

M3 installs ROS 2 Humble packages from the official ROS apt distribution. LaserPerception wraps but
does not vendor these external components:

- [rclpy](https://github.com/ros2/rclpy), the ROS 2 Python client library, Apache-2.0;
- [ROS 2 common interfaces](https://github.com/ros2/common_interfaces), including `geometry_msgs`,
  `sensor_msgs`, `std_msgs`, and `visualization_msgs`, under their recorded Apache-2.0 package
  licenses; the installed Humble `sensor_msgs_py` package declares BSD;
- [tf2_ros / geometry2](https://github.com/ros2/geometry2), whose installed Humble `tf2_ros`
  package declares BSD;
- [vision_msgs](https://github.com/ros-perception/vision_msgs), Apache-2.0;
- [rmw_fastrtps](https://github.com/ros2/rmw_fastrtps) and
  [eProsima Fast DDS](https://github.com/eProsima/Fast-DDS), Apache-2.0; and
- [RViz](https://github.com/ros2/rviz), BSD-3-Clause-Clear.

These components and their transitive dependencies remain under their own upstream terms. ROS 2,
tf2, its message definitions, Fast DDS, and RViz are installed externally and are not relicensed by
LaserPerception. The recorded M3/M4.5b system used `rmw_fastrtps_cpp`; the repository does not
redistribute that middleware. M4.5b adds no non-ROS Python distribution dependency.

## Upstream pretrained PointPillars checkpoint

M1 uses the official MMDetection3D nuScenes PointPillars checkpoint
`hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth`, obtained from the
[OpenMMLab model distribution](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth).
Its verified SHA256 is
`f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0`.

The checkpoint is downloaded to an ignored external cache and is not distributed in this
repository. LaserPerception did not train it, does not claim ownership of it, and does not relicense
it. Users are responsible for reviewing the upstream software/model terms and the nuScenes dataset
terms that apply to the pretrained artifact and its use.

## Datasets and specifications

nuScenes, SemanticKITTI, KITTI, DALES, CVGC, their papers, datasets, development kits, and model
weights are not included and are not relicensed by LaserPerception. nuScenes v1.0-mini is obtained
only from the [official nuScenes distribution](https://www.nuscenes.org/nuscenes), remains subject to
the [nuScenes terms of use](https://www.nuscenes.org/terms-of-use), and is not redistributed. No
endorsement by Motional, nuScenes, or any dataset maintainer is implied.

Dataset formats and source IDs in the parked segmentation layer were implemented from public
specifications rather than copied third-party source code.

## Community documents

The project Code of Conduct uses the Contributor Covenant, version 2.1, available at
<https://www.contributor-covenant.org/version/2/1/code_of_conduct/> under Creative Commons
Attribution 4.0.

If third-party source or assets are added later, record the component, source, version/commit,
license, modifications, and required attribution here before distribution.
