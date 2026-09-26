# AGENTS.md — authoritative repository instructions

This file is the source of truth for AI coding agents working on LaserPerception. Read it before
modifying the repository. User instructions take precedence when they explicitly change scope.

## Project and milestone state

LaserPerception is an open-source 3D LiDAR object-detection and deployment-engineering toolkit.
The canonical current operational summary is `docs/PROJECT_STATUS.md`; use it instead of
duplicating mutable status across documents.
M1 through M7 are historical and frozen as applicable. **M8 — Detector V2 is active**, with
DSVT-Pillar plus TransFusion selected as the modern detector. M8 P1-E is complete, the S1 protocol
is frozen, and Stage R completed on the retired machine with its raw evidence merged. The retired
machine received a historical authorization for primary A2/E2, but executed zero primary calls.
That authorization is hardware- and runtime-bound and is not portable to a cloud worker, rented
GPU, future laptop, or any other runtime. Later external A40 work completed a fresh 10-process,
140-call Stage R; a primary attempt then executed 779 detector conditions but ended incomplete,
with zero accepted complete processes and zero accepted canonical primary calls. A later authorized
campaign completed three accepted fresh primary processes and 2,568 canonical A2/E2 calls. Its
raw measurement and separate scientific interpretation are published. The interpretation records
the positive descriptive Car direction and severe Pedestrian cross-domain failure without causal,
winner, or significance claims. A separate three-process, 2,568-call zero-intensity measurement
and scientific interpretation are published. It did not recover severe Pedestrian failure, while
the positive H5-over-H10 Car direction persisted; intensity causality is not established. S2 design
may begin, but S2 execution remains blocked and not started, and training has not started. No
scientific inference may run without a fresh, explicit, runtime-specific owner authorization.

Within that frozen history, M6c is complete with a positive final R3 projected-reference ROS
validation result; preserve the original R2 failure and D1 diagnosis. M5 remains conditional and inactive
rather than implicitly activated by later milestones. No technical submilestone is
currently active within the closed M6 milestone; M8 is the separately authorized active milestone.

The accepted v0.2 path uses an official pretrained MMDetection3D PointPillars checkpoint on
nuScenes, TensorRT FP16, the LaserPerception `exact_fast` deterministic deployment voxelizer, a
ROS 2 Humble interface, and compatible raw XYZ PointCloud2 plus time-aware TF multi-sweep
reconstruction. The earlier SemanticKITTI-to-DALES adapters, ontology, configuration, and audit
pipeline remain tested, supported, parked experimental infrastructure and must not be deleted.

## Branch ownership and collaboration

- One coding implementer works on a feature or release branch at a time; the owner assigns that
  implementer.
- Other AI systems are review and specification participants unless the owner explicitly assigns
  them as the implementer.
- Never use two coding agents simultaneously on the same branch.
- Review comments may guide the assigned implementer, but reviewers must not make uncoordinated
  edits to that branch.

## Roadmap and scope

- M1–M7: historical and frozen as applicable; preserve their accepted and failed evidence.
- M8 Detector V2: active. DSVT-Pillar plus TransFusion is selected; P1-E is complete.
- M8 P1-S1: its protocol is frozen. External-runtime qualification, fresh Stage R, and the accepted
  three-pass/2,568-call primary A2/E2 measurement and scientific interpretation completed.
  Preserve the incomplete 779-condition attempt separately; its accepted canonical primary-call
  count is zero. The separately authorized zero-intensity intervention completed three accepted
  processes and 2,568 calls on a newly qualified A40; its raw result and separate scientific
  interpretation are published. S2 additionally requires prospective denominator-stability and
  multi-realization partition rules before B2/C2/D2/F2 execution.
- A new runtime must undergo GT-blind qualification, machine-specific policy binding, a repeated
  Stage R, owner review, and a new authorization before primary inference.
- Zero-intensity measurement and interpretation are published. S2 design may begin;
  B2/C2/D2/F2 execution, S2, and training are not authorized.

Do not add training, another detector, INT8, unrelated tracking research, camera fusion, custom CUDA, Jetson tuning
without hardware, localization, vendor SDK drivers, unrelated optimization, or unrelated features
unless the owner explicitly changes scope. Frozen scientific artifacts and historical failures
must not be modified or reinterpreted.

## Local CPU development and external compute

- Normal development must support CPU-only workstations. Do not assume a GPU exists, probe an
  optional GPU, import a GPU runtime for discovery, or run GPU integration tests without explicit
  owner/runtime authorization. Missing optional dependencies must fail closed before discovery.
- GPU workers are separate, explicitly selected external runtimes. Provider selection is an owner
  decision, not a hard-coded dependency; RunPod has been used operationally, while the tooling
  remains provider-neutral. Codex Cloud is not required or the primary development environment.
  Use local CPU development plus external GPU compute on demand.
- Static GPU-worker editing, syntax checks, and CPU mocks do not authorize execution of the
  worker's GPU paths. GPU discovery and execution belong only inside an explicitly authorized
  GPU runtime. Qualification and scientific execution require their own scoped authorization.
- The retired RTX runtime's policy and primary authorization remain historical and non-portable.
  A new worker must pass the full qualification and authorization sequence in
  `docs/CLOUD_WORKFLOW.md`; transferring verified artifacts does not transfer permission.
- Implement on an owner-assigned feature branch from verified main, commit with the owner's
  configured human identity, push, and open a PR. Never implement/commit/push directly on main,
  merge the PR, or enable auto-merge. Keep fixing the same PR until required CI is green.

## GitHub/Drive persistence and worker lifecycle

- GitHub is authoritative for tracked source, documentation, tests, configs, PRs, releases, and
  compact final evidence suitable for Git.
- The canonical private Google Drive remote is `lpdrive:` with folder ID
  `18Q73IkiVcFT0EXAowIPlOiISNk0mkhHJ`. `_CLOUD_STATE/` holds durable indexes and
  `_CLOUD_WORK/` holds task capsules. Datasets, checkpoints, binaries, large/raw/failed evidence,
  logs, valuable temporary state, and other non-Git project state must be Drive-backed.
- External compute workers are disposable. A task is not complete while unique
  LaserPerception state exists only on a worker. Git-suitable work must be represented in Git/PR;
  non-Git state must be Drive-backed.
- Checkpoint expensive or scientific results after every canonical pass/process before proceeding,
  including failed attempts. Preserve temporary files when they have project, debug, or scientific
  value. Never upload or commit credentials.
- Hydrate only required inputs and verify recorded hashes. Never mirror the whole Drive project,
  recursively hydrate private state, or optimize Git metadata on Drive-backed storage as routine
  cleanup. Preserve private untracked state. Diagnose unexpected tracked changes before proceeding;
  do not hide them with Git configuration or index flags. See `docs/CLOUD_WORKFLOW.md`.

## Detection and deployment architecture

- The historical M1–M7 stack uses the official pretrained MMDetection3D PointPillars model and
  pinned nuScenes preprocessing. Its checkpoint, contracts, and evidence remain frozen and bound
  to PointPillars; do not reinterpret or rerun them without explicit authorization. LaserPerception
  did not train that detector and must not claim that it did.
- The active M8 stack is the DSVT/OpenPCDet-based DSVT-Pillar with TransFusion candidate selected
  in P1-E. Frozen M8 candidate, config, checkpoint, and input identities govern M8; PointPillars is
  the historical comparison baseline, not the active primary M8 detector.
- The M8 feature contract is `[x, y, z, intensity, time_lag]`. Preserve the prospectively frozen
  source-row order and do not introduce random test-time shuffle.
- Keep the framework-independent `DetectionFrame` contract small and explicit. Document coordinate
  frame, axes, length-width-height order, yaw convention, classes, scores, and optional velocity;
  never silently swap length and width.
- Preserve official nuScenes class names in raw converted results. Future taxonomy changes must be
  explicit, versioned, and evidence-gated.
- Keep export and visualization filtering separate from model execution. Display thresholds must
  not redefine benchmarked inference.
- Preserve the official multi-sweep nuScenes path. Do not force it through the parked single-scan
  `PointCloud` abstraction.
- The M4.5a production builder is NumPy-only and returns the existing `ModelReadyPointCloud`.
  MMDetection3D is its manual parity oracle only, not a runtime dependency. Preserve exact sweep
  and source-row order, timestamp arithmetic, transform cast/write-back points, and strict range
  semantics recorded in `docs/m45/UPSTREAM_MULTISWEEP_CONTRACT.md`.
- The M4.5b ROS boundary accepts compatible raw PointCloud2 with scalar float32 XYZ, removes
  non-finite rows without reordering, retains bounded acquisition history, and requires time-aware
  `lookup_transform_full` through a fixed frame. Same frame names at different timestamps are not
  identity. Preserve the accepted ROS column-vector to builder storage mapping
  `rotation = R.T`, `translation = -R.T @ t`; do not reintroduce the rejected `-t` adapter.
- M4.5b authoritative evidence is actual raw nuScenes data through PointCloud2/tf2/builder and the
  unchanged detector, not the synthetic transform fixture alone. Preserve the original W1 failure,
  transform ledger, repair exactness evidence, and final 20-sample canonical record.
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
  historical reproduction or deployment dependencies. OpenPCDet/DSVT, PyTorch, CUDA, spconv, and
  torch-scatter are likewise optional, isolated dependencies for the active M8 scientific runtime;
  none are core-wheel dependencies. Standard GitHub CI must run without them.
- Heavy environments, datasets, checkpoints, ONNX files, TensorRT engines, caches, logs, and
  generated outputs stay outside the repository and must be persisted to the canonical Drive when
  they are not reconstructible or have project value.
- GPU integration tests are manual and confined to authorized external GPU runtimes. ROS tests
  require a separately provisioned environment. CPU development must skip these integrations
  before optional hardware discovery; setup failures must be actionable and fail closed.

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
