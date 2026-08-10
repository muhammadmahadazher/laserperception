# Roadmap

Progress is evidence-gated. Dates and capabilities are not promised before prerequisites pass.

## M0 — project transition

- [x] Position LaserPerception around reproducible real-time 3D LiDAR object detection.
- [x] Preserve the existing SemanticKITTI/DALES segmentation and audit infrastructure as parked,
  supported work.
- [x] Define an optional GPU dependency policy that keeps the core package and CI CPU-testable.

## M1 — PointPillars first sight

- [x] Reproduce the isolated official MMDetection3D stack and initialize the pinned pretrained
  PointPillars checkpoint on the RTX 4060 Laptop GPU.
- [x] Export framework-independent 3D detections with documented box conventions and upstream class
  names.
- [x] Implement original headless BEV rendering and bounded qualifying-sample discovery.
- [x] Implement synchronized, two-boundary FP32 latency and CUDA-memory measurement.
- [ ] Run real inference against prepared nuScenes v1.0-mini and inspect its converted output.
- [ ] Produce and review real-prediction BEV output, including an honest pedestrian prediction if a
  bounded 0.25-threshold scan finds one.
- [ ] Record a sanitized real RTX 4060 FP32 benchmark and promote measured values.

M1 is **PARTIAL** until all data-dependent unchecked items complete. It is inference-only and does
not include training, a second detector, model conversion, ROS 2, or edge deployment.

## M2 — TensorRT FP16

Convert and benchmark the verified M1 model with ONNX and TensorRT FP16. This work begins only after
M1 review.

## M3 — ROS 2

Add ROS 2 integration around a verified detector runtime. This work begins only after M2 review.

## M4 — v0.1

Publish an evidence-backed open-source release with reproducible setup, measurements, limitations,
and safety wording.

## M5 — conditional Jetson measurements

Measure and tune for Jetson only if the target physical hardware becomes available. No hardware
figures will be estimated or simulated.

## Parked experimental infrastructure

The earlier Experiment 001 data foundation—`PointCloud`, I/O, SemanticKITTI and DALES adapters,
explicit normalization, ontology mappings, and dataset audits—remains tested and supported. Its
modeling, training, and zero-shot semantic-segmentation evaluation are inactive before detection
v0.1. See [VISION.md](VISION.md).
