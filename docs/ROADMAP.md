# Roadmap

Progress is evidence-gated. Dates and capabilities are not promised before prerequisites pass.
M0 through M4.6 are complete and v0.2.0 is released. M6 — Cross-Domain Validation: KITTI Raw is
complete. M6a is complete under prospective Protocol R2; the original Tier-A failure remains
preserved as a failure. M6b is complete under owner-approved Protocol R2 after structural 40k
remediation, the non-evaluation H5 profile-gap parity gate, and the full frozen offline
characterization. M6c is complete with a positive final R3 projected-reference ROS validation;
the original R2 failure and D1 diagnosis remain preserved. No technical submilestone is currently
active, M5 remains conditional and inactive, and any next milestone requires explicit owner
authorization.

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
- [x] **M4.6 — v0.2.0 release:** release metadata, documentation, packaging and validation,
  merged release commit, annotated `v0.2.0` tag, and public GitHub release.

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

## Completed M4.6 — v0.2.0 release engineering

M4.6 packaged and documented the accepted M4.5 raw-ingestion capability as v0.2.0. The release PR
was merged, the annotated `v0.2.0` tag was created, and the public GitHub release was published.
M4.6 did not reopen runtime implementation, correctness protocols, or performance measurement. No
later milestone activates automatically; any next technical work requires explicit owner
authorization.

## M6 — complete

- [x] **M6a — complete under Protocol R2:** authoritative KITTI Raw discovery, dataset contract,
  direct official Raw-devkit pose/calibration verification, model-frame alignment, exact raw
  decoding, and a 24-frame deterministic offline reconstruction oracle.
- [x] **M6b — complete under Protocol R2:** offline frozen-detector execution on 428 verified KITTI
  Raw current frames, Raw-tracklet ground-truth/domain-shift characterization, paired H10/H5
  compound analysis, capacity diagnostics, and deterministic offline visualization. The original
  30k-engine failure remains failed; the prospective structural 40k engine passed nuScenes,
  non-evaluation KITTI, profile-gap, repeatability, and full-corpus execution gates.
- [x] **M6c — complete; positive final R3 result:** KITTI Raw PointCloud2 replay, time-aware tf2,
  24/24 Gate 1A and 856/856 Gate 1B projected-reference exactness (860/860 unique live conditions),
  unchanged parity-v2 Stage 1 PASS on ten frozen detector sentinels, and 10/10 exact
  `Detection3DArray` conversions. R2 remains a preserved original-reference byte-exactness failure;
  D1 remains its post-failure diagnosis.

At measurement commit `ec9e341056807d5549353c8ef362fd109b25f2f2`, 271 mapped frames differed
from the official odometry oracle by as much as 0.0884767 m translation and 0.000416629 rad
rotation, above the frozen numerical-only tolerances. No tolerance was changed and no canonical
reconstruction evidence was generated under Protocol v1. The sanitized failure remains retained
under `benchmarks/m6a/diagnostics/` with status FAIL.

R1 later established that synchronized Raw OXTS and KITTI Odometry are different official timing
products and that the production adapter matched a direct Raw-devkit implementation exactly.
Prospective Protocol R2 was committed only after that diagnosis. Its clean canonical measurement
at `1ab832df89109546abedc9f4e7f21c16c4cd0dca` passed 271/271 exact pose-oracle comparisons on
`2011_09_30_drive_0016`, a separate 108/108 exact transfer check on canonical reconstruction
`2011_09_26_drive_0001`, exact raw decoding, and 24/24 offline reconstruction outputs over ten
repeats each. See `docs/m6/M6A_RESULTS_R2.md`.

M6a remains limited to engineering interoperability. It did not initialize or run the detector on
KITTI, inspect predictions, or implement ROS replay. M6b completed its separately authorized
offline evaluation under frozen Protocol R2 without target-domain tuning. M6c then completed its
separately authorized integration-correctness cycle without tuning or performance measurement.
M6 is closed; no R4 or later technical work starts automatically.

## M5 — conditional physical Jetson measurement

M5 remains conditional and inactive. Measure or tune for Jetson only if target hardware is
physically available and the owner explicitly authorizes the milestone. No Jetson figure will be
estimated, simulated, or inferred from the RTX 4060 Laptop result.

## Post-v0.2 backlog — not started

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
training, and accuracy evaluation remain `Pending measurement` and outside the current detection
release line.
