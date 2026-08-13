# Changelog

All notable changes to LaserPerception will be documented here. The project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and intends to use semantic versioning once
releases begin.

## [Unreleased]

### Changed

- Completed M3 at exact measurement commit
  a129b3507597b25f44ab1a833562f68883ebe8ce. Production exact-fast/live preserved 81/81
  official voxel outputs and all frozen 20 raw TensorRT outputs, final DetectionFrames, and ROS
  round trips exactly. The representative full-history W1 ROS test did not sustain 20 Hz:
  callback/loopback medians were 75.701/134.250 ms, effective output was 10.825 Hz, and 159/359
  measured inputs dropped with first-to-second-half deterioration. Bounded characterization
  sustained 10 Hz and did not sustain 15 Hz. This honest failure is the canonical M3 result.

- Accepted M3B-V2 and integrated it through a separate production path without relabeling its
  diagnostic timings. The direct W1 live median remains 43.168 ms direct-runtime evidence rather
  than ROS callback evidence; historical M2 official/full behavior remains unchanged.
- Added an explicit voxel-provenance policy for TensorRT output metadata: **full** remains the
  historical default with exact tensor hashes, while opt-in ROS **live** mode records only
  lightweight semantic metadata and deliberately omits the hashing cost.

- Qualified the frozen detection evidence by actual sweep history: M1 and canonical M2 performance
  repeatedly use scene-start `mini_val` index 0 with zero historical sweeps; M2 parity v2 and M3
  round-trip correctness use 19 full-history samples plus that scene-start sample; and the failed
  M3A synthetic 20 Hz replay also uses scene-start index 0. The 81-sample validation split contains
  79 full 10-history-plus-current inputs and two scene starts, so the multi-sweep dataset pipeline
  is intact.
- Advanced to M3 only after PR #3 merged and preserved every frozen M2 artifact. Exact-commit
  20-sample PointCloud2 fidelity passed, but the M3A 20 Hz diagnostic measured 238.255 ms callback
  median, 303.283 ms loopback median, 3.990 Hz output, and 875 bounded-QoS input drops.
- Diagnosed official hard voxelization in authorized diagnostic-only M3B-V1. The in-memory
  nondeterministic candidate greatly reduced direct voxelization time but did not demonstrate 20 Hz
  end to end and failed W2 repeatability against the existing detector yardstick, so it was not
  adopted. No model, engine, ONNX, postprocess, ROS/DDS production path, or M4 work changed.
- Added a strict model-ready multi-sweep PointCloud2 interface that rejects missing `time_lag`,
  performs no TF/history reconstruction, and preserves source headers and detector box semantics.

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
- Recorded the exact-commit M2 diagnosis: native and rewritten FP32 outputs were element-exact on
  all 20 samples; native/TensorRT raw medians were 20.800/6.917 ms; current postprocess was
  24.093 ms. These diagnostic values are not a replacement canonical benchmark.
- Reconfirmed parity v2 and native/rewrite fidelity at exact measurement commit 3f240d6, then
  promoted the repaired RTX 4060 benchmark: native PyTorch FP32 and TensorRT FP16 measured 59.289 ms
  and 45.637 ms direct end-to-end medians (1.2991× headline speedup), while the corresponding
  network-only medians were 19.189 ms and 6.126 ms (3.1326× secondary speedup). No benchmark review
  flags triggered, and the rejected e2f9b6b values remain non-canonical.

- Centralized M1 external cache resolution around `LASERPERCEPTION_M1_CACHE` with a portable default,
  and made setup accept any usable CUDA GPU while retaining the RTX 4060 Laptop GPU as the canonical
  measured reference hardware.
- Transitioned the active roadmap to reproducible real-time 3D LiDAR object detection: M1 uses an
  official pretrained PointPillars model on nuScenes v1.0-mini with honest RTX 4060 FP32
  measurements and headless BEV visualization; TensorRT FP16 and ROS 2 remain later milestones.
- Parked, rather than removed, the tested SemanticKITTI/DALES semantic-segmentation and audit
  infrastructure while keeping the core package and standard CI lightweight and CPU-testable.

### Added

- Promoted the validated M3B-V2 voxelizer to the supported `exact_fast` LaserPerception
  deployment policy, with fail-closed initialization and explicit ROS exact-fast/live configuration.
  The historical/default M2 evidence policy remains official/full.
- Added a sanitized canonical M3 result with the full callback/loopback distributions, offered and
  effective rates, drops, first/second-half backlog behavior, eligible GPU telemetry, and the
  strictly bounded 10/15 Hz characterization.
- Isolated ROS 2 Humble Python package with one-time TensorRT initialization, bounded QoS,
  Detection3DArray conversion, nuScenes replay, per-frame visualization markers, launch/config,
  official setup, and ROS-native tests without adding ROS to the core wheel.
- ROS-independent PointCloud2 layout and detection-message contracts with CPU regression tests, plus
  all-20-sample exact transport-fidelity and two-boundary ROS latency tooling.

- Frozen M2 MMDeploy/TensorRT deployment boundary, official upstream pins, fixed 20-sample parity
  set, immutable acceptance tolerances, shape-profile policy, and same-session benchmark protocol.
- Retained the rejected M2 timing record in benchmarks/m2/diagnostics without presenting its
  latency or speedup values as accepted evidence.
- Sanitized canonical M2 result at
  `benchmarks/m2/results/rtx4060_pytorch_fp32_vs_tensorrt_fp16.json`, including exact-commit parity
  and fidelity hashes, complete latency distributions, environment provenance, limitations, memory
  accounting methods, and empty review flags.
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
