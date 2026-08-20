# Vision — evidence before deployment claims

LaserPerception is an open-source 3D LiDAR object-detection and deployment-engineering toolkit. Its
accepted path deliberately freezes one official pretrained PointPillars detector, verifies
TensorRT FP16 deployment, preserves deterministic voxel semantics, exposes detections through ROS
2, and can reconstruct model-ready temporal input from compatible raw XYZ PointCloud2 plus
time-aware TF. The project values auditable correctness, honest hardware-specific measurements,
and visible failure records over broad or unqualified performance claims.

The accepted release line is a research, benchmarking, and demonstration toolkit—not a novel
detector, a trained LaserPerception model, a production-ready autonomy stack, or a safety-certified
system.

## Evidence-gated sequence

1. M0: project direction and governance transition — complete.
2. M1: pretrained PointPillars, nuScenes v1.0-mini, RTX 4060 FP32 evidence, and BEV output —
   complete.
3. M2: ONNX/TensorRT FP16, parity/fidelity evidence, and repaired performance baseline — complete.
4. M3: ROS 2 interface, exact-fast deployment, correctness gates, and representative W1 rate
   evidence — complete.
5. M4: v0.1.0 release engineering — complete.
6. M4.5: offline and live raw-sweep reconstruction, time-aware TF, and exact detector-chain
   evidence — complete.
7. M4.6: v0.2.0 release engineering for the accepted M4.5 capability — active and release-only.
8. M5: conditional physical Jetson measurements only if hardware is available and the owner
   activates the milestone; currently inactive.

Training, additional detectors, INT8, tracking, camera fusion, custom CUDA, localization, sensor
calibration automation, and ROS/DDS optimization are not part of v0.2.0.

## Parked semantic-transfer research

The SemanticKITTI-to-DALES Experiment 001 infrastructure remains valuable, tested, and supported.
Its semantic-segmentation model and benchmark remain unimplemented, and all result fields remain
`Pending measurement`. It is not the active detection release line.
