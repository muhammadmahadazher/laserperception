# Roadmap

Progress is evidence-gated. Dates and capabilities are not promised before prerequisites pass.

## Completed foundations

- [x] **M0 — project transition:** position LaserPerception around reproducible 3D LiDAR detection
  and deployment while preserving the parked SemanticKITTI/DALES infrastructure.
- [x] **M1 — PointPillars first sight:** pinned official pretrained PointPillars, nuScenes
  v1.0-mini preparation, framework-independent detections, original BEV output, and real RTX 4060
  Laptop FP32 evidence.
- [x] **M2 — TensorRT FP16:** pinned MMDeploy export/build path, preserved parity-v1 failure,
  parity-v2 pass, native/rewrite fidelity, rejected first benchmark, and repaired canonical
  native-PyTorch-versus-TensorRT measurement.
- [x] **M3 — ROS 2:** model-ready multi-sweep `PointCloud2`, exact output conversion, bounded QoS,
  replay/visualization, the rejected nondeterministic voxelizer, accepted exact-fast replacement,
  production correctness gates, and representative full-history ROS evidence.

M3 closed honestly: representative W1 (10 historical sweeps plus current, 354,182 points) sustained
10 Hz cleanly; 15 Hz and 20 Hz were not sustained. This result does not authorize additional M3
optimization.

## M4 — v0.1.0 release

The release-candidate work prepares the accepted implementation without changing measured runtime
behavior:

- [x] align version, citation, governance, package metadata, and release history;
- [x] provide stranger-first release documentation and an external-asset-aware demo path;
- [x] preserve and clearly separate M1, M2, M3B-V2, and M3 evidence;
- [ ] merge the reviewed release PR into `main`, then create tag `v0.1.0` from that merged commit.

The tag, GitHub release, and any package publication happen only after final release review. They are
not created from the release branch.

## M5 — conditional physical Jetson measurement

Measure and tune for Jetson only if target hardware is physically available. No Jetson figure will
be estimated, simulated, or inferred from the RTX 4060 Laptop result.

## Post-v0.1 backlog — not started

These are separate future proposals, not v0.1 commitments:

- MMDeploy postprocessing profiling and optimization;
- ROS/DDS/executor profiling and tuning;
- further exact-fast tuning;
- custom CUDA only if later evidence justifies it;
- INT8;
- additional detector architectures; and
- training infrastructure.

## Parked experimental infrastructure

The earlier Experiment 001 foundation—`PointCloud`, I/O, SemanticKITTI and DALES adapters, explicit
normalization, ontology mappings, and dataset audits—remains tested and supported. Its model,
training, and accuracy evaluation remain `Pending measurement` and outside the v0.1 detection line.