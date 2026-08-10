# Roadmap

Progress is evidence-gated. Dates and capabilities are not promised before prerequisites pass.

## M0 — project transition

- [x] Position LaserPerception around reproducible real-time 3D LiDAR object detection.
- [x] Preserve the existing SemanticKITTI/DALES segmentation and audit infrastructure as parked,
  supported work.
- [x] Define an optional GPU dependency policy that keeps the core package and CI CPU-testable.

## M1 — PointPillars first sight

- [ ] Reproduce an official pretrained MMDetection3D PointPillars model on nuScenes v1.0-mini.
- [ ] Export framework-independent 3D detections with documented box conventions.
- [ ] Produce original, headless BEV visualizations of real predictions.
- [ ] Measure honest FP32 latency and peak GPU memory on an RTX 4060 Laptop GPU.

M1 is inference-only. It does not include training, a second detector, model conversion, ROS 2, or
edge deployment.

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
