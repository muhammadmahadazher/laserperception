# AGENTS.md — authoritative repository instructions

This file is the source of truth for AI coding agents working on LaserPerception. Read it before
modifying the repository. User instructions take precedence when they explicitly change scope.

## Active project and current milestone

LaserPerception is an open-source 3D LiDAR perception toolkit focused on reproducible real-time
object detection and deployment engineering.

The current milestone is **M2 — TensorRT FP16 deployment**: deploy the exact verified M1
PointPillars checkpoint through the pinned official MMDeploy ONNX/TensorRT path, prove final-box
parity against frozen tolerances, and measure same-session PyTorch FP32 versus TensorRT FP16
performance on the available NVIDIA GeForce RTX 4060 Laptop GPU. M1 is complete and merged.

Existing SemanticKITTI-to-DALES semantic-segmentation adapters, ontology, configuration, and audit
pipeline remain built, tested, documented, and supported as parked experimental infrastructure.
They are not the active development line before detection v0.1 and must not be deleted.

## Roadmap scope

- M0: transition project governance and documentation to the detection direction.
- M1: official pretrained PointPillars, nuScenes v1.0-mini, FP32 CUDA inference, original BEV
  visualization, and real RTX 4060 measurements.
- M2 only: ONNX and TensorRT FP16 conversion and benchmarking.
- M3 only: ROS 2 integration.
- M4: evidence-backed v0.1 release.
- M5 only if physical hardware is available: Jetson measurements and tuning.

Before v0.1, do not add training, a second detector architecture, INT8 optimization, camera fusion,
foundation models, custom CUDA kernels, Jetson-specific tuning, or unrelated features unless the
owner explicitly changes scope. Do not begin M3 work during M2.

## Detection architecture rules

- Use the official pretrained MMDetection3D PointPillars implementation and nuScenes pipeline. Do
  not reimplement PointPillars, voxelization, NMS, CUDA operations, or training infrastructure.
- Keep the LaserPerception-owned detection output contract small and independent of MMDetection3D
  types. Document coordinate frame, axes, dimension order, and yaw convention; never silently swap
  length and width.
- Preserve official nuScenes class names in raw converted results. Any future taxonomy mapping must
  be explicit and versioned.
- Keep model output separate from export and visualization filtering. A display threshold must not
  alter the model execution used for latency measurement.
- nuScenes inference must preserve the official multi-sweep MMDetection3D preprocessing pipeline.
  Do not force it through the existing single-scan `PointCloud` abstraction.
- Generated visualizations and raw benchmark outputs belong in ignored artifact directories. Only
  a reviewed, sanitized, real benchmark result may be committed.

## Dependency and environment policy

- The core package must remain lightweight, CPU-testable, and importable without GPU libraries.
- PyTorch, CUDA, MMDetection3D, MMDeploy, ONNX, and TensorRT are permitted for M1/M2 only as
  optional, isolated detection/deployment dependencies.
- Heavy GPU dependencies must not become core requirements or be imported by core modules.
- Standard GitHub CI must continue to run without a GPU or detection dependencies. GPU integration
  tests are manual/local and must skip cleanly when their environment is absent.
- Put CUDA-specific installation in dedicated setup and environment documentation. Keep the heavy
  environment, datasets, checkpoints, caches, logs, and generated outputs outside the Google Drive
  repository, preferably on the WSL ext4 filesystem.
- ONNX and TensorRT are permitted starting in M2 only. ROS 2 is permitted starting in M3 only.

## Parked segmentation architecture

- Use the Python `src` layout and keep the public package import as `laserperception`.
- Keep `PointCloud` simple: float32 `(N, 3)` geometry, optional point labels, separate attributes,
  and metadata. Avoid inheritance and speculative abstraction.
- File readers decode and preserve data. They must not normalize, crop, voxelize, or augment.
- Coordinate normalization is an explicit, non-mutating transform. `min_xyz` means
  `xyz - xyz.min(axis=0)` and must record its parameters.
- Keep LAS as storage/interchange, not a required neural representation.
- Experiment 001 retains its geometry-only six-class ontology: Ground, Building, Natural, Vehicle,
  Pole, Fence. Its input policy is `x`, `y`, `z`; `min_xyz`; 0.30 m reference voxel size; and mIoU
  plus per-class IoU. Its model and every unmeasured result remain `Pending measurement`.

## Dataset and asset rules

- Never commit datasets, point-cloud tiles, downloaded archives, checkpoints, weights, caches,
  training outputs, generated visualizations, raw benchmark outputs, or logs.
- Use environment/configuration variables for dataset roots; do not hard-code machine paths.
- M1 and M2 use nuScenes v1.0-mini only. Respect its official access terms and never redistribute
  it.
- Download checkpoints only from an official upstream source, keep them outside the repository,
  and record the source, version/commit, license note, and SHA256 after download.
- Do not copy third-party dataset or model tooling without verifying license and attribution.
- Verify source class IDs from authoritative dataset material before changing mappings. Cite the
  source and test ignored/unmapped behavior.
- Treat ontology or detector-class mapping changes as scientifically material preprocessing changes.
- Apache-2.0 does not relicense nuScenes, SemanticKITTI, KITTI, DALES, external weights, papers, or
  third-party assets.

## Scientific integrity and reproducibility

- Never fabricate detections, accuracy, latency, throughput, memory, dataset statistics, hardware
  data, citations, authors, DOIs, or novelty claims.
- Use the exact text `Pending measurement` for unmeasured benchmark fields.
- Record commit SHA, manifest/config, exact upstream versions/commits, checkpoint checksum, dataset
  version/split, sample selection, precision, thresholds, warmup/measured counts, timing boundaries,
  environment, hardware, timestamp, metrics, and memory method for measured runs.
- FP32 M1 benchmarking must explicitly disable autocast and use correct CUDA synchronization/events.
- M2 speedup must use a same-session MMDeploy-rewritten PyTorch FP32 baseline and TensorRT FP16
  runtime with common voxelization and postprocessing; the historical M1 result is context only.
- Do not claim SOTA, universality, production readiness, deployment suitability, or safety around
  people without evidence and certification.

## Code, tests, and dependencies

- Use type hints, focused docstrings, deterministic behavior, defensive validation, and clear
  exceptions.
- Add synthetic CPU tests for the detection contract, conversion, geometry, parity, visualization
  helpers, lazy optional-dependency failures, artifact metadata, and benchmark statistics. Tests
  must never download datasets or checkpoints.
- Run `ruff check .`, `ruff format --check .`, `mypy src`, `python -m pytest`, and
  `python -m build` before a major push.
- Keep base dependencies minimal and CPU-testable. Document optional detection dependency behavior
  and skip optional-backend tests cleanly.

## Licensing, citations, and contributions

- Original LaserPerception code is Apache-2.0. Preserve `LICENSE` and `NOTICE`.
- Record incorporated third-party material and required attribution in `THIRD_PARTY_NOTICES.md`.
- Cite authoritative file specifications, framework documentation, model sources, and scientific
  sources; do not invent bibliographic data.
- Prefer Conventional Commit messages such as `feat(detection): ...`, `test: ...`, and `docs: ...`.
- Keep commits logical, inspect staged content for secrets and large files, and update
  `CHANGELOG.md` for user-visible changes.
