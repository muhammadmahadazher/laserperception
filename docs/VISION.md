# Vision — evidence before deployment claims

LaserPerception is an open-source 3D LiDAR object-detection and deployment-engineering toolkit. Its
v0.1 path deliberately freezes one official pretrained PointPillars detector, verifies TensorRT
FP16 deployment, preserves deterministic voxel semantics, and exposes the result through ROS 2.
The project values auditable correctness, honest hardware-specific measurements, and visible
failure records over broad or unqualified performance claims.

The accepted release is a research, benchmarking, and demonstration toolkit—not a novel detector,
a trained LaserPerception model, a production-ready autonomy stack, or a safety-certified system.

## Evidence-gated sequence

1. M0: project direction and governance transition — complete.
2. M1: pretrained PointPillars, nuScenes v1.0-mini, RTX 4060 FP32 evidence, and BEV output —
   complete.
3. M2: ONNX/TensorRT FP16, parity/fidelity evidence, and repaired performance baseline — complete.
4. M3: ROS 2 interface, exact-fast deployment, correctness gates, and representative W1 rate
   evidence — complete.
5. M4: v0.1.0 release engineering — active until review, merge, and tag.
6. M5: physical Jetson measurements only if hardware is actually available.

Future post-v0.1 work is separately scoped. Training, additional detectors, INT8, tracking, camera
fusion, custom CUDA, and ROS/DDS optimization are not part of v0.1.0.

## Parked semantic-transfer research

The SemanticKITTI-to-DALES Experiment 001 infrastructure remains valuable, tested, and supported.
Its semantic-segmentation model and benchmark remain unimplemented, and all result fields remain
`Pending measurement`. It is not the active v0.1 product line.