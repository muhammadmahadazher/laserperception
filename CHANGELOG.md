# Changelog

All notable changes to LaserPerception are documented here. The project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and semantic versioning.

## [Unreleased]

## [0.3.0] - Release date pending

### Added

- Official KITTI Raw decoding, Raw-devkit pose/calibration handling, a frozen KITTI-to-model basis,
  deterministic offline multi-sweep reconstruction, and ROS replay/integration tooling.
- Frozen-detector cross-domain characterization across 428 current KITTI Raw frames and 856 paired
  H10/H5 conditions, with Raw-tracklet metrics, capacity diagnostics, and real visualizations.
- A complete M6 documentation index and technical note connecting the failed gates, diagnoses,
  prospective protocols, accepted evidence, and final claim boundaries.

### Changed

- Widened the M6 TensorRT structural input profile from the historical 30,000-voxel maximum to the
  upstream preprocessing contract's 40,000 maximum without changing the ONNX, model, precision,
  thresholds, voxel geometry, or postprocessing. This is not a latency or performance claim.
- Prepared Python, ROS, citation, changelog, roadmap, and release-facing metadata for v0.3.0.

### Validation / Evidence

- M6a prospective Protocol R2 passed 271/271 exact Raw-devkit pose comparisons, a separate 108/108
  canonical-drive transfer check, exact decoding, and 24/24 deterministic offline reconstructions.
- M6b completed all 856 frozen H10/H5 conditions. At score 0.25 and oriented BEV IoU 0.50, H10
  Car/Pedestrian recall was 0.242/0.553 and H5 was 0.727/0.677. This is a compound history ablation;
  temporal span alone was not isolated as the cause.
- M6c final R3 validation passed 24/24 Gate 1A, 856/856 Gate 1B, and 860/860 unique projected live
  conditions byte-for-byte. The detector envelope retained 113/113 detections, 81/81
  high-confidence coverage, 81/81 passes for every continuous metric, zero class mismatches, zero
  continuous outliers, and 10/10 exact `Detection3DArray` conversions.

### Preserved failures / limitations

- The initial M6a odometry-equality gate remains FAIL; R1 diagnosed distinct official timing/data
  products before prospective R2 was frozen. The original M6b run remains a structural stop because
  the 30k engine rejected valid input before network execution.
- M6c R2 remains failed at the original-M6a byte boundary. D1 identified the unit-quaternion/SO(3)
  representation boundary, and prospective R3 used same-platform projected references without
  weakening R2 or adopting a tolerance.
- The detector remains the unchanged official pretrained nuScenes PointPillars checkpoint. No
  training or fine-tuning, official KITTI benchmark AP, physical-LiDAR validation, real-time ROS
  measurement, production-readiness claim, or safety certification was added.

## [0.2.0] - 2026-08-20

### Added

- Raw ROS 2 LiDAR ingestion for compatible scalar float32 XYZ `PointCloud2`, time-aware tf2 lookup
  through a fixed frame, bounded current-plus-ten-sweep history, and publication into the existing
  model-ready detector boundary.
- The ROS-independent M4.5a `MultiSweepBuilder`, nuScenes raw replay, launch/config installation,
  fail-closed input and TF handling, ROS-native regressions, and sanitized exactness evidence.

### Changed

- Corrected the ROS-to-`SweepTransform` inverse-translation adapter from `-t` to `-R.T @ t` while
  preserving the accepted accumulation core and every frozen detector/runtime artifact.
- Updated current Python, ROS, citation, quickstart, release-note, and repository metadata for the
  v0.2.0 release.

### Scientific chronology

- M4.5a matched 81/81 mini-val model-ready matrices byte-for-byte against the pinned official
  preparation path: two current-only scene starts and 79 ten-history samples. Its frozen 20-sample
  detector outputs also remained exact.
- The first M4.5b full-history W1 raw ROS run failed at 354,184 points versus 354,182 expected. The
  failed record remains preserved.
- A transform ledger localized the discrepancy to the ROS/tf2 adapter. A fail-first
  rotation-plus-translation regression exposed the incorrect `-t` storage rule.
- The minimal `-R.T @ t` repair restored exact scene-start, W1, and rotation-stratified sentinel
  reconstruction. Final validation passed 20/20 model-ready inputs, voxel tensors, raw TensorRT
  outputs, DetectionFrames, and Detection3DArray semantic/geometric content exactly. The older
  model-ready M3 smoke also passed.

### Scope

- M4.5/M4.5b was correctness and integration work, not a new performance campaign. Historical
  model-ready M3 10/15/20 Hz evidence is not a raw-ingestion throughput claim.
- No checkpoint, model, ONNX, engine, exact-fast implementation, threshold, class mapping, voxel
  geometry, or historical M1/M2/M3/M4.5 evidence changed.

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
