# AGENTS.md — authoritative repository instructions

This file is the source of truth for AI coding agents working on LaserPerception. Read it before
modifying the repository. User instructions take precedence when they explicitly change scope.

## Mission and current experiment

LaserPerception studies cross-view and cross-domain semantic understanding of heterogeneous 3D
LiDAR point clouds. Experiment 001 asks how SemanticKITTI automotive-source knowledge transfers
zero-shot to DALES airborne LiDAR under a geometry-only six-class ontology: Ground, Building,
Natural, Vehicle, Pole, Fence.

The current input policy is `x`, `y`, `z`; `min_xyz` normalization; 0.30 m reference voxel size; and
mIoU plus per-class IoU. The model is not implemented and every result is `Pending measurement`.

## Scope and non-goals

Current work is 3D LiDAR semantic segmentation research. Do not implement 2D LiDAR, detection,
tracking, ROS2, a C++ SDK, TensorRT, Jetson optimization, language models, multimodal fusion,
embeddings, streaming, or foundation models unless the project scope is explicitly revised. Keep
aspirational material in `docs/VISION.md`, not as README functionality.

## Architecture rules

- Use the Python `src` layout and keep the public package import as `laserperception`.
- Keep `PointCloud` simple: float32 `(N, 3)` geometry, optional point labels, separate attributes,
  and metadata. Avoid inheritance and speculative abstraction.
- File readers decode and preserve data. They must not normalize, crop, voxelize, or augment.
- Coordinate normalization is an explicit, non-mutating transform. `min_xyz` means
  `xyz - xyz.min(axis=0)` and must record its parameters.
- Keep LAS as storage/interchange, not a required neural representation.
- Keep model/training dependencies out of the core until corresponding functionality exists.

## Dataset and ontology rules

- Never commit datasets, point-cloud tiles, downloaded archives, checkpoints, weights, caches,
  training outputs, or logs.
- Use environment/configuration variables for dataset roots; do not hard-code machine paths.
- Do not copy third-party dataset tooling without verifying license and attribution obligations.
- Verify source class IDs from authoritative dataset material before changing mappings. Cite the
  source and test ignored/unmapped behavior.
- Treat ontology changes as scientifically material preprocessing changes.
- The Apache-2.0 project license does not relicense SemanticKITTI, KITTI, DALES, external weights,
  papers, or third-party assets.

## Scientific integrity and reproducibility

- Never fabricate accuracy, mIoU, IoU, dataset statistics, hardware data, timing, VRAM, citations,
  authors, DOIs, or novelty claims.
- Use the exact text `Pending measurement` for unmeasured benchmark fields.
- Make every preprocessing decision explicit in configuration.
- Record commit SHA, config, dataset version/split, preprocessing/ontology versions, seeds,
  environment, hardware, metrics, wall-clock boundaries, and memory method for measured runs.
- Do not claim SOTA, universality, production readiness, or deployment suitability without evidence.

## Code, tests, and dependencies

- Use type hints, focused docstrings, deterministic behavior, defensive validation, and clear
  exceptions.
- Add synthetic tests for new I/O; tests must never download public datasets.
- Run `ruff check .`, `ruff format --check .`, `mypy src`, `python -m pytest`, and
  `python -m build` before a major push.
- Keep base dependencies minimal and CPU-testable. Do not add Torch, spconv, TorchSparse, Open3D,
  CUDA tooling, or another heavy framework for scaffolding.
- Document optional dependency behavior and skip optional-backend tests cleanly.

## Licensing, citations, and contributions

- Original LaserPerception code is Apache-2.0. Preserve `LICENSE` and `NOTICE`.
- Record incorporated third-party material and required attribution in `THIRD_PARTY_NOTICES.md`.
- Cite authoritative file specifications and scientific sources; do not invent bibliographic data.
- Prefer Conventional Commit messages such as `feat(io): ...`, `test: ...`, and `docs: ...`.
- Keep commits logical, inspect staged content for secrets and large files, and update
  `CHANGELOG.md` for user-visible changes.
