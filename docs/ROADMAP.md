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
- [x] Prepare official nuScenes v1.0-mini metadata with the ten-sweep upstream converter (323 train,
  81 validation samples observed).
- [x] Run real FP32 inference and inspect the converted output.
- [x] Produce original headless BEV output with genuine model-predicted pedestrian detections at the
  fixed 0.25 threshold.
- [x] Measure and promote a sanitized, two-boundary RTX 4060 FP32 latency and CUDA-memory result.

M1 is complete and merged. It remains inference-only and does not include training, a second
detector, model conversion, ROS 2, or edge deployment.

## M2 — TensorRT FP16

- [x] Freeze the exact M1 asset, official MMDeploy v1.3.1 commit, deployment boundary, 20-sample
  parity set, and acceptance tolerances before engine evidence.
- [x] Pass the standalone TensorRT 8.6.x FP16 build/serialize/execute smoke gate.
- [x] Measure all 81 `mini_val` voxel shapes and justify the final optimization profile.
- [x] Export and validate the pinned PointPillars ONNX graph and build the external FP16 engine.
- [x] Pass versioned final-box parity v2 on all frozen samples with shared preprocessing/postprocessing.
- [ ] Remeasure the deployable PyTorch FP32 and TensorRT FP16 paths in the same session.

M2 is partial. Gate 0, the 81-sample profile, ONNX checking, and FP16 engine build pass. Parity v1
remains failed; after architecture review, the separately preregistered v2 Stage 1 passed on the
unchanged engine and samples. Same-session benchmarking remains unrun pending reviewer
authorization, and all benchmark fields remain `Pending measurement`.

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
