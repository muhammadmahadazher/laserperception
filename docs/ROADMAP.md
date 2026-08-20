# Roadmap

Progress is evidence-gated. Dates and capabilities are not promised before prerequisites pass.
No technical milestone is currently active; the next post-M4.5 phase requires an owner decision.

## Completed foundations

- [x] **M0 — project transition:** position LaserPerception around reproducible 3D LiDAR detection
  and deployment while preserving the parked SemanticKITTI/DALES infrastructure.
- [x] **M1 — PointPillars first sight:** pinned official pretrained PointPillars, nuScenes
  v1.0-mini preparation, framework-independent detections, original BEV output, and real RTX 4060
  Laptop FP32 evidence.
- [x] **M2 — TensorRT FP16:** pinned MMDeploy export/build path, preserved parity-v1 failure,
  parity-v2 pass, native/rewrite fidelity, rejected first benchmark, and repaired canonical
  native-PyTorch-versus-TensorRT measurement.
- [x] **M3 — ROS 2:** model-ready multi-sweep PointCloud2, exact output conversion, bounded QoS,
  replay/visualization, rejected nondeterministic voxelization, accepted exact-fast replacement,
  production correctness, and representative full-history ROS evidence.
- [x] **M4 — v0.1.0 release:** release metadata, stranger-first documentation, packaging audits,
  final validation, merged release commit, `v0.1.0` tag, and GitHub release.
- [x] **M4.5a — offline multi-sweep reconstruction:** independent raw-sweep/pose reconstruction,
  81/81 exact official-pipeline parity, and exact frozen 20-sample detector verification.
- [x] **M4.5b — live raw ROS ingestion:** compatible raw XYZ PointCloud2 decoding, time-aware tf2,
  bounded live history, preserved transform-repair chronology, exact model-ready reconstruction,
  and 20/20 exact unchanged detector-chain evidence.
- [x] **M4.5 overall:** offline known-pose reconstruction and live ROS/tf2 ingestion both complete.

M3 closed honestly: representative W1 (10 historical sweeps plus current, 354,182 points) sustained
10 Hz cleanly; 15 Hz and 20 Hz were not sustained. M4.5b ran correctness and integration gates only
and did not reopen performance work.

## Completed M4.5 boundary

```text
M4.5a:
raw sweep + known pose/calibration metadata -> model-ready temporal cloud

M4.5b:
compatible raw PointCloud2 + time-aware TF + bounded live history
    -> same model-ready temporal cloud -> unchanged detector
```

M4.5b consumes an existing valid localization/TF source. It does not add localization, odometry,
calibration automation, per-point deskew, a vendor sensor driver, or a new detector path.

## M5 — conditional physical Jetson measurement

M5 remains conditional and inactive. Measure or tune for Jetson only if target hardware is
physically available and the owner explicitly authorizes the milestone. No Jetson figure will be
estimated, simulated, or inferred from the RTX 4060 Laptop result.

## Post-v0.1 backlog — not started

These are separate future proposals, not current work or commitments:

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
