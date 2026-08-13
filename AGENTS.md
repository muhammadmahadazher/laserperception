# AGENTS.md — authoritative repository instructions

This file is the source of truth for AI coding agents working on LaserPerception. Read it before
modifying the repository. User instructions take precedence when they explicitly change scope.

## Active project and current milestone

LaserPerception is an open-source 3D LiDAR object-detection and deployment-engineering toolkit. The
current milestone is **M4 — v0.1.0 release engineering**. M0, M1, M2, and M3 are complete. M4 may
prepare, validate, document, and package the accepted implementation; it must not expand product or
performance scope.

The accepted v0.1 path uses an official pretrained MMDetection3D PointPillars checkpoint on
nuScenes, TensorRT FP16, the LaserPerception `exact_fast` deterministic deployment voxelizer, and a
ROS 2 Humble interface. The earlier SemanticKITTI-to-DALES adapters, ontology, configuration, and
audit pipeline remain tested, supported, parked experimental infrastructure and must not be
deleted.

## Branch ownership and collaboration

- One coding implementer works on a feature or release branch at a time; the owner assigns that
  implementer.
- Other AI systems are review and specification participants unless the owner explicitly assigns
  them as the implementer.
- Never use two coding agents simultaneously on the same branch.
- Review comments may guide the assigned implementer, but reviewers must not make uncoordinated
  edits to that branch.

## Roadmap and scope

- M0: project direction and governance transition — complete.
- M1: official pretrained PointPillars, nuScenes v1.0-mini, FP32 CUDA inference, BEV visualization,
  and RTX 4060 measurements — complete.
- M2: official MMDeploy ONNX/TensorRT FP16 path, parity/fidelity evidence, and repaired same-session
  performance comparison — complete.
- M3: ROS 2 Humble interface, exact deterministic deployment voxelization, correctness evidence,
  and representative full-history ROS measurement — complete.
- M4: evidence-backed v0.1.0 release — active release-engineering scope only.
- M5: physical Jetson measurements only if target hardware is actually available.

Do not add training, a second detector, INT8, tracking, camera fusion, a raw single-sweep history
builder, custom CUDA, Jetson tuning without hardware, or unrelated features unless the owner
explicitly changes scope. During M4, do not optimize postprocessing, DDS, executors, voxelization,
or any measured runtime path.

## Detection and deployment architecture

- Use the official pretrained MMDetection3D PointPillars model and pinned nuScenes preprocessing.
  LaserPerception did not train the detector and must not claim that it did.
- Keep the framework-independent `DetectionFrame` contract small and explicit. Document coordinate
  frame, axes, length-width-height order, yaw convention, classes, scores, and optional velocity;
  never silently swap length and width.
- Preserve official nuScenes class names in raw converted results. Future taxonomy changes must be
  explicit, versioned, and evidence-gated.
- Keep export and visualization filtering separate from model execution. Display thresholds must
  not redefine benchmarked inference.
- Preserve the official multi-sweep nuScenes path. Do not force it through the parked single-scan
  `PointCloud` abstraction.
- The historical/core evidence voxelization default is `official` with `full` provenance.
- The ROS deployment policy is explicitly `exact_fast` with `live` provenance. `exact_fast` is a
  LaserPerception implementation proven bit-exact against the pinned official deterministic hard
  voxelization by the accepted 81-sample and frozen-detector gates.
- The upstream `deterministic=False` shortcut remains rejected because it changed saturated
  retained-point subsets and observable detections. Never silently fall back to it or substitute
  another semantics-changing voxelizer without a new explicit evidence gate.
- Do not duplicate the detector, voxel geometry, NMS, postprocessing, or the validated exact-fast
  algorithm.

## Dependency and environment policy

- The core wheel remains lightweight, CPU-testable, and importable without GPU or ROS libraries.
- PyTorch, CUDA, MMDetection3D, MMDeploy, ONNX, TensorRT, and ROS 2 remain optional, isolated
  deployment dependencies. Standard GitHub CI must run without them.
- Heavy environments, datasets, checkpoints, ONNX files, TensorRT engines, caches, logs, and
  generated outputs stay outside the repository, preferably on the WSL ext4 filesystem.
- GPU and ROS integration tests are manual/local and must skip cleanly when their environment is
  absent. Setup failures must be actionable and fail closed.

## Dataset and asset rules

- Never commit datasets, point-cloud tiles, archives, checkpoints, weights, ONNX files, TensorRT
  engines, caches, virtual environments, generated raw logs, or unreviewed visualizations.
- Use environment variables or config for dataset/cache roots; never hard-code private machine
  paths.
- v0.1 detection evidence uses nuScenes v1.0-mini. Respect its terms and never redistribute it.
- Download checkpoints only from the recorded official upstream source, store them externally, and
  verify the recorded SHA256.
- Apache-2.0 does not relicense nuScenes, SemanticKITTI, KITTI, DALES, external weights, engines,
  papers, or third-party software.

## Scientific integrity and reproducibility

- Never fabricate detections, accuracy, parity, latency, throughput, memory, dataset statistics,
  hardware data, citations, authors, DOIs, or novelty claims. Use `Pending measurement` for
  genuinely unmeasured benchmark fields.
- Preserve failed and rejected evidence with its status. Do not promote diagnostics or compare
  uncontrolled sessions as though they were same-session measurements.
- Record commit SHA, config, upstream versions, artifact hashes, dataset/split/sample, sweep history,
  precision, thresholds, warmups, measurements, timing boundaries, environment, hardware,
  timestamp, statistics, and memory method for measured runs.
- M2 parity reference: MMDeploy-rewritten PyTorch FP32 versus TensorRT FP16.
- M2 performance baseline: native MMDetection3D PyTorch FP32 versus TensorRT FP16. Rewritten eager
  PyTorch is not the performance denominator.
- Correctness evidence and one-system performance measurements are distinct. Do not present RTX
  4060 Laptop/WSL2 timings as portable hardware guarantees.
- Do not claim SOTA, universality, production readiness, deployment safety, or autonomous-driving
  certification.

## Parked segmentation architecture

- Keep the Python `src` layout and public package import `laserperception`.
- Keep `PointCloud` as float32 `(N, 3)` geometry with optional labels, separate attributes, and
  metadata. Readers preserve data and do not normalize, crop, voxelize, or augment.
- `min_xyz` remains an explicit, non-mutating transform recorded in metadata. LAS remains storage
  and interchange rather than a required neural representation.
- Experiment 001 retains its geometry-only six-class ontology and `Pending measurement` model and
  result fields.

## Code, tests, packaging, and release discipline

- Use type hints, focused docstrings, deterministic behavior, defensive validation, and clear
  exceptions.
- Keep synthetic CPU tests free of downloads and heavy optional dependencies. Add regression tests
  for release metadata or wrappers where they materially prevent drift.
- Run `ruff check .`, `ruff format --check .`, `mypy src`, `python -m pytest`, `python -m build`, and
  `git diff --check` before a major push. Run clean colcon and ROS-native smoke/tests for ROS release
  validation.
- Audit wheel and sdist separately. The wheel contains only the lightweight Python package; the
  sdist may retain reviewed, sanitized benchmark evidence.
- Preserve `LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES.md`; verify rather than invent third-party
  terms. Do not add a DOI unless one actually exists.
- Prefer Conventional Commit messages, inspect staged content for secrets and large files, and keep
  `CHANGELOG.md` current.