# Changelog

All notable changes to LaserPerception will be documented here. The project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and intends to use semantic versioning once
releases begin.

## [Unreleased]

### Changed

- Preserved failed M2 parity protocol v1 as historical evidence and preregistered a separate parity
  v2 protocol with unchanged samples, artifacts, thresholds, and numerical tolerances plus
  per-detection 99% acceptance and explicit direction diagnostics. Stage 1 passed every gate on the
  unchanged 20-sample suite and existing FP16 engine, then passed again at the exact benchmark
  implementation commit.
- Tightened benchmark promotion to require protocol-v2, passing, non-diagnostic, exact-commit
  parity evidence for the frozen 20 samples and current ONNX/engine hashes; v1 and malformed
  evidence are rejected by CPU tests.
- Rejected the M2 benchmark measured at e2f9b6b after sanity review, preserved it as explicit
  non-canonical diagnostic history, and separated the MMDeploy-rewritten parity reference from the
  native MMDetection3D PyTorch performance baseline.
- Added fail-closed CUDA tensor assertions, a native raw-network path, 20-sample export-rewrite
  fidelity diagnostics, component profiling, and benchmark review flags.

- Centralized M1 external cache resolution around `LASERPERCEPTION_M1_CACHE` with a portable default,
  and made setup accept any usable CUDA GPU while retaining the RTX 4060 Laptop GPU as the canonical
  measured reference hardware.
- Transitioned the active roadmap to reproducible real-time 3D LiDAR object detection: M1 uses an
  official pretrained PointPillars model on nuScenes v1.0-mini with honest RTX 4060 FP32
  measurements and headless BEV visualization; TensorRT FP16 and ROS 2 remain later milestones.
- Parked, rather than removed, the tested SemanticKITTI/DALES semantic-segmentation and audit
  infrastructure while keeping the core package and standard CI lightweight and CPU-testable.

### Added

- Frozen M2 MMDeploy/TensorRT deployment boundary, official upstream pins, fixed 20-sample parity
  set, immutable acceptance tolerances, shape-profile policy, and same-session benchmark protocol.
- Retained the rejected M2 timing record in benchmarks/m2/diagnostics without presenting its
  latency or speedup values as accepted evidence.
- Pinned isolated TensorRT 8.6.1/MMDeploy 1.3.1 setup, executable Gate 0, complete 81-sample voxel
  profiler, official ONNX export and engine builder, shared deployment runtime, deterministic parity
  diagnostics, benchmark promotion guards, sanitized artifact provenance, and synthetic CPU tests.
- Real official nuScenes v1.0-mini evidence: 323/81 prepared train/validation samples, successful
  FP32 detection with genuine pedestrian predictions, original BEV output, and a sanitized 50-run
  two-boundary RTX 4060 Laptop GPU benchmark.
- Framework-independent, validated `Detection3D` and `DetectionFrame` contracts with documented
  nuScenes LiDAR axes, length-width-height dimensions, yaw, class names, filtering, and JSON export.
- Lazy MMDetection3D 1.4.0 PointPillars backend that validates the official checkpoint SHA256,
  preserves the upstream ten-sweep nuScenes pipeline, and runs explicit FP32 evaluation on `cuda:0`.
- Reproducible pinned WSL2 setup and nuScenes v1.0-mini preparation, inference, and sample-discovery
  commands without making PyTorch or OpenMMLab core dependencies.
- Original deterministic headless BEV renderer for LiDAR points and oriented model-predicted boxes.
- Two-boundary FP32 benchmark protocol with CUDA events, synchronized end-to-end timing, complete
  latency statistics, CUDA memory counters, sanitized metadata, and fail-closed output behavior.
- Synthetic CPU tests for detection types, geometry, conversion, lazy dependency diagnostics,
  visualization, and benchmark statistics.
- Professional open-source repository foundation and CPU CI.
- Canonical validated `PointCloud` representation.
- KITTI/SemanticKITTI scan I/O and packed label decoding.
- LAS/optional LAZ loading with metadata and attribute preservation.
- Explicit `min_xyz` coordinate normalization.
- Verified six-class SemanticKITTI and DALES ontology mappings.
- Experiment 001 configuration, synthetic tests, and research documentation, including a pinned
  CVGC reference investigation.
- Directory-level SemanticKITTI adapter with pinned official splits and scan/label validation.
- Memory-conscious DALES adapter with selective chunked reading and deterministic grid patches.
- CPU-only dataset audit CLI with redacted JSON reports and ontology-coverage statistics.
- Synthetic adapter, boundary, conservation, memory-path, and audit regression tests.
