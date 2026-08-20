# Changelog

All notable changes to LaserPerception are documented here. The project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and semantic versioning.

## [Unreleased]

### Added

- A ROS-independent M4.5a `MultiSweepBuilder` that reconstructs the pinned float32 XYZT PointPillars
  input from raw nuScenes sweeps, integer-microsecond timestamps, calibration, and ego poses without
  a production MMDetection3D dependency.
- M4.5b compatible raw float32 XYZ PointCloud2 decoding, bounded live history, time-aware
  `tf2_ros.Buffer.lookup_transform_full` transforms through a fixed frame, model-ready PointCloud2
  publication, nuScenes raw replay, launch/config installation, and ROS-native regressions.
- Sanitized exactness evidence: M4.5a matched 81/81 mini-val matrices and all frozen detector
  outputs; the M4.5b raw ROS path matched 20/20 model-ready inputs, voxel tensors, raw TensorRT
  outputs, DetectionFrames, and Detection3DArray semantics.

### Changed

- Corrected the ROS-to-`SweepTransform` inverse-translation storage from `-t` to `-R.T @ t` while
  preserving the accepted M4.5a accumulation core and every frozen detector/runtime artifact.
- Marked M4.5a, M4.5b, and M4.5 overall complete. No subsequent technical milestone is active;
  post-M4.5 work requires an owner decision.

### Scientific chronology

- The first full-history W1 ROS exactness run failed and remains preserved.
- A transform ledger localized the discrepancy to the ROS/tf2 adapter boundary.
- A fail-first rotation-plus-translation regression exposed the inverse-translation error; the
  minimal repair restored exact scene-start, W1, and rotation-stratified sentinel results.
- Final validation kept that failure history visible and passed the unchanged frozen 20-sample
  detector chain exactly. The older model-ready M3 smoke also remained valid.

### Scope

- M4.5 is post-v0.1 development; the v0.1.0 release notes and version remain unchanged.
- No checkpoint, model, ONNX, engine, exact-fast, threshold, class mapping, voxel geometry, or
  historical M1/M2/M3/M4.5a evidence changed. No performance campaign was run.

## [0.1.0] - 2026-08-13

### Added

- A lightweight Python 3.10–3.13 core with point-cloud I/O, explicit transforms, dataset adapters,
  ontology/audit tooling, and framework-independent `Detection3D`/`DetectionFrame` contracts.
- Official pretrained MMDetection3D PointPillars inference on nuScenes v1.0-mini, original BEV
  rendering, and a sanitized RTX 4060 Laptop FP32 measurement (M1).
- Pinned MMDeploy ONNX and TensorRT FP16 deployment, parity/fidelity tools, external artifact-hash
  validation, and a repaired native-PyTorch-versus-TensorRT benchmark (M2).
- A ROS 2 Humble package with a strict model-ready multi-sweep `PointCloud2` contract,
  `Detection3DArray` output, bounded QoS, replay, RViz/Foxglove markers, and ROS-native tests (M3).
- The supported LaserPerception `exact_fast` deterministic voxelizer and explicit `official` /
  `exact_fast` plus `full` / `live` policies.
- Reproducibility records, release quickstart, validated demo wrapper, release notes, and packaging
  metadata for v0.1.0.

### Scientific chronology

- M1 established real FP32 inference and a warm-cache, repeated scene-start benchmark. Later sweep
  auditing confirmed that its `mini_val` index 0 input has zero available history; the configured
  ten-sweep pipeline itself is intact.
- M2 parity v1 failed and remains failed. The separately preregistered parity v2 passed on the
  unchanged frozen engine.
- The first M2 benchmark at `e2f9b6b…` was rejected because MMDeploy-rewritten eager PyTorch was an
  invalid performance denominator. The retained diagnostic then proved native/rewrite fidelity;
  the repaired canonical benchmark used native MMDetection3D PyTorch FP32 and measured a 1.2991×
  end-to-end median TensorRT speedup on scene-start index 0.
- Sweep-history verification qualified M1/M2 scene-start performance separately from the 19
  full-history cases in the frozen 20-sample correctness suite.
- M3A preserved its failed scene-start 20 Hz stress result rather than presenting it as success.
- M3B-V1 rejected upstream `deterministic=False`: it changed saturated retained-point subsets and
  observable repeatability.
- M3B-V2 replaced that candidate with `exact_fast`, which matched all 81 validation voxel outputs
  bit-for-bit and retained exact frozen raw detector outputs and final detections.
- Final M3 production correctness passed, then representative W1 (10 historical sweeps plus current,
  354,182 points) sustained 10 Hz cleanly; 15 Hz and 20 Hz were not sustained.

### Changed

- The historical/evidence default remains official deterministic voxelization with full provenance.
  ROS deployment explicitly selects exact-fast voxelization with live provenance and fails closed.
- Release-facing metadata now describes 3D object detection, TensorRT deployment, deterministic
  voxelization, and ROS 2 rather than the parked semantic-segmentation research direction.

### Limitations

- nuScenes, the pretrained checkpoint, ONNX, TensorRT engine, GPU environment, and ROS 2 stack are
  external and are not included in the core wheel.
- v0.1.0 requires model-ready multi-sweep `PointCloud2`; it does not build sweep history from a raw
  physical LiDAR stream.
- Training, tracking, camera fusion, INT8, additional detectors, and Jetson measurements are not
  included. The project is not safety certified.
