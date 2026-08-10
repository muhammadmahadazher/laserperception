# Vision — evidence before deployment claims

LaserPerception aims to become an open-source 3D LiDAR perception toolkit for reproducible,
real-time object detection and deployment engineering. The near-term path is deliberately narrow:
reproduce one official pretrained PointPillars model on nuScenes, measure it honestly on available
hardware, then evaluate TensorRT FP16 and ROS 2 in later gated milestones.

The project favors small framework-independent result contracts, explicit coordinate and box
conventions, isolated optional GPU dependencies, headless visual evidence, and benchmark records
whose provenance can be audited. It is a research, benchmarking, and demonstration toolkit—not a
safety-certified perception system or a certified system for operation around people.

## Active sequence

1. M1: pretrained PointPillars, nuScenes v1.0-mini, RTX 4060 FP32 measurements, and BEV output.
2. M2: ONNX and TensorRT FP16, only after M1 review.
3. M3: ROS 2 integration, only after M2 review.
4. M4: v0.1 with evidence-backed capabilities and limitations.
5. M5: Jetson measurements only if physical hardware is available.

Training, additional detector architectures, INT8, camera fusion, foundation models, custom CUDA
kernels, and Jetson tuning are outside the pre-v0.1 plan unless the owner revises scope.

## Parked semantic-transfer research

The SemanticKITTI-to-DALES Experiment 001 infrastructure remains valuable, tested, and supported as
existing code. Its cross-view semantic-segmentation model and benchmark remain unimplemented and
all results remain `Pending measurement`. It is not the active development line before detection
v0.1 and may be revisited through a separately reviewed roadmap.
