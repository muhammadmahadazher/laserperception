# Roadmap

Progress is evidence-gated. Dates and capabilities are not promised before prerequisites pass.
M0 through M4.6 are complete and v0.2.0 is released. M6 — Cross-Domain Validation: KITTI Raw is
complete and v0.3.0 is its release boundary. M6a is complete under prospective Protocol R2; the
original Tier-A failure remains preserved as a failure. M6b is complete under owner-approved
Protocol R2 after structural 40k remediation, the non-evaluation H5 profile-gap parity gate, and
the full frozen offline characterization. M6c is complete with a positive final R3
projected-reference ROS validation; the original R2 failure and D1 diagnosis remain preserved.
M7 is complete and frozen. M8 Detector V2 is active, with its scientific S1 execution paused pending
a newly qualified external GPU runtime. M5 remains conditional and inactive. Work beyond the
explicitly authorized M8 scope requires a new owner decision.

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

## v0.3.0 release boundary — complete

v0.3.0 packages the completed M6 repository state, documentation, and reviewed evidence. The
release itself did not reopen M6, start M7, or authorize another experiment. M7 and M8 were
subsequently authorized separately.

## M7 — complete and frozen

The controlled history-mechanism study completed prospective R2 measurement, raw aggregation, and
owner-approved interpretation. Preserve the earlier binding failure, diagnosis, corrected runtime,
raw results, and [frozen interpretation](m7/M7_RESULTS.md). The measured encoded-lag contribution
is class- and corpus-dependent; it does not authorize lag compression as a product default.

## M8 Detector V2 — active; S1 execution paused

| Step | Current state |
|---|---|
| Candidate | DSVT-Pillar plus TransFusion selected on engineering feasibility |
| P1-E | Complete, including the amended H10 structural-capacity gates |
| S1 protocol | Owner-approved and frozen |
| Measurement implementation | Reviewed historical runtime with mode/pass authorization barriers |
| Retired-runtime Stage R | Complete; raw evidence merged and reviewed |
| Retired-runtime primary A2/E2 | Historically authorized on that runtime; **zero calls executed** |
| New external runtime | Provider not selected; complete qualification and new authorization required |
| S1 execution | Paused pending the newly qualified GPU runtime |
| Zero-intensity / S2 / training | Not authorized; not started |

Read the historical [candidate decision](m8/M8_CANDIDATE_DECISION.md),
[P1-E integration](m8/M8_PHASE1_ENGINEERING.md), [S1 protocol](m8/M8_S1_PROTOCOL.md),
[measurement implementation](m8/M8_S1_MEASUREMENT_IMPLEMENTATION.md),
[Stage R raw evidence](m8/M8_S1_STAGE_R_RAW.md), and
[primary authorization](m8/M8_S1_PRIMARY_AUTHORIZATION.md) in chronological order. Their status
statements describe those acts at the time; the retired authorization is not portable.

The [compute workflow](CLOUD_WORKFLOW.md) requires owner-scoped GT-blind qualification, artifact
and capacity verification, machine-specific policy, owner review, fresh Stage-R-only authorization,
repeated Stage R and raw review, then a fresh primary authorization before A2/E2. Verified transfers
and historical capacity measurements grant no execution permission. The scientific comparison
remains a frozen detector-stack comparison, not an architecture-only causal study. DSVT's partial
TensorRT route is not end-to-end deployment parity. No PointPillars rerun is authorized.

## M5 — conditional physical Jetson measurement

M5 remains conditional and inactive. Measure or tune for Jetson only if target hardware is
physically available and the owner explicitly authorizes the milestone. No Jetson figure will be
estimated, simulated, or inferred from the RTX 4060 Laptop result.

## Other engineering backlog — not authorized

These are separate future proposals, not current work or commitments:

- MMDeploy postprocessing profiling and optimization;
- ROS/DDS/executor profiling and tuning;
- further exact-fast tuning;
- custom CUDA only if later evidence justifies it;
- INT8;
- detector architectures beyond the separately selected M8 candidate; and
- training infrastructure.

## Perception platform engineering track

P0 and P1 now provide stable input/model contracts, reviewed manifests, model discovery, an explicit
execution context, static backend descriptions, a generic detection pipeline, `load_model()`, and a
CPU-safe prediction dry run. Thin adapters preserve the existing PointPillars output path and the M8
S1 authorization/accounting boundary; this engineering work made no detector call.

Later targets include richer sensor/temporal ingestion, semantic and instance/panoptic understanding,
embeddings, scene understanding, and eventually language-queryable representations. These
are not current capabilities or active milestones. They do not authorize training, another model,
segmentation, a scientific experiment, or changes to frozen evidence. The owner must select
and scope each increment separately.

## Parked experimental infrastructure

The earlier Experiment 001 foundation—`PointCloud`, I/O, SemanticKITTI and DALES adapters, explicit
normalization, ontology mappings, and dataset audits—remains tested and supported. Its model,
training, and accuracy evaluation remain `Pending measurement` and outside the current detection
release line.

## CPU tracking — implemented P2

`laserperception.tracking` provides immutable Track3D/TrackFrame results, explicit nanosecond timestamps,
constant-XY-velocity prediction, class-aware deterministic global greedy association, and configurable
lifecycle management. `track_sequence()` and `laserperception track` stream precomputed detections.
No detector is executed. See [tracking documentation](TRACKING.md) for coordinate assumptions and
limitations. The synthetic example is not benchmark evidence.

## Semantic results and evaluation — implemented P3

`laserperception.semantic` supplies immutable, row-aligned SemanticPointFrame results and a NumPy
confusion/IoU evaluator. Versioned taxonomy descriptions reuse the Experiment 001 ontology and its
explicit SemanticKITTI/DALES mapping policy. Evaluation requires identical sample, coordinate, taxonomy,
source-count and source-row identities. NPY sidecars bind dtype, shape, size and SHA256; small fixtures
may use bounded inline JSON. `semantic inspect` and `semantic evaluate` run on CPU.
Production segmentation models are not integrated. See [semantic documentation](SEMANTIC_SEGMENTATION.md).

## Data discovery and ingestion — implemented P4

`laserperception.data` adds eight reviewed metadata adapters, canonical PointCloud reader wrappers,
lazy sample references and local input inspection. Compatibility uses existing model metadata and
reports missing preparation and unverified coordinates. `data adapters list/inspect` and `data inspect`
are CPU-safe. Existing readers and scientific multi-sweep paths retain their behavior. See
[data adapters](DATA_ADAPTERS.md) and the expanded [CPU quickstart](QUICKSTART_PERCEPTION.md).
