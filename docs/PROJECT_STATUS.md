# Project status

This page is the canonical current-status summary. Historical protocols and results retain the
status recorded when they were created; this page does not rewrite them.

## Latest public release

LaserPerception v0.4.0 is the current release. It includes the lightweight CPU perception platform,
eight reviewed data adapters, guarded detector planning APIs, deterministic CPU multi-object
tracking, immutable semantic point results and evaluation, provider-neutral worker tooling, and the
historical PointPillars/TensorRT/ROS stack.

## Implemented capabilities

| Area | Current state |
|---|---|
| PointPillars | Official pretrained nuScenes model integrated; historical M1–M7 evidence frozen |
| TensorRT and ROS 2 | Historical deployment path released with explicit parity and rate boundaries |
| Perception platform | Model manifests, registry, validation, exact input contracts, deterministic dry-run planning |
| Tracking | Deterministic class-aware constant-XY-velocity tracking over timed `DetectionFrame` input |
| Semantic results | Immutable row-aligned results, taxonomy descriptions, identity-bound serialization, confusion/IoU evaluation |
| Data adapters | Eight reviewed input paths with CPU-safe discovery and local inspection |
| External workers | Verified artifacts, task manifests, qualification records, persistence helpers, fail-closed authorization boundaries |

Tracking has no claimed end-to-end detector/tracker benchmark. Semantic infrastructure does not
include a production segmentation model.

## Historical frozen milestones

M1–M7 are historical and frozen as applicable. Their accepted, failed, rejected, and diagnostic
records remain authoritative for their exact commits and environments. M6c closed with the
accepted projected-reference R3 result while retaining the original R2 failure. M7 completed the
controlled-history study and retains its preflight failures.

## Active M8 research

M8 selected the official pretrained DSVT-Pillar + TransFusion candidate. P1 engineering and the
frozen S1 protocol are complete. External A40 qualification and the fresh ten-process, 140-call
Stage R campaign are preserved. A later accepted primary campaign completed three fresh processes,
856 A2/E2 conditions per process, and 2,568 accepted canonical calls at frozen execution commit
`6994d72c3e7691a86116d1417ac3ae08256d163f` and receipt file SHA256
`bef4c55575581aefe8f477e32d1b394f40823a0b0858c66c3ac5c1fae141ec4d`.

The M8 primary A2/E2 [raw measurement](m8/M8_S1_MEASUREMENT_RAW.md), compact primary and secondary
results, and measurement manifest are published. Its separate
[scientific interpretation](m8/M8_S1_INTERPRETATION.md) records exact threshold-level agreement
across all three processes. Car recall was 19/66 for A2/H10 and 43/66 for E2/H5, satisfying the
prospective positive-direction criterion. Pedestrian recall was 0/396 and 1/396, a severe
class-specific cross-domain failure. This is a frozen detector-stack comparison: winner selection,
architecture or intensity causality, statistical significance, and production readiness are not
claimed.

The separately authorized zero-intensity intervention completed three accepted fresh A40
processes and 2,568 canonical calls at execution commit
`95fb66ac1f57c41f06f05bd9ef5dac27b1e3ea54`. Its
[raw measurement](m8/M8_S1_ZERO_INTENSITY_RAW.md) and separate
[scientific interpretation](m8/M8_S1_ZERO_INTENSITY_INTERPRETATION.md) are published. The
intervention did not rescue severe Pedestrian failure (H10 0/396 to 0/396; H5 1/396 to 0/396),
while the positive H5-over-H10 Car direction persisted (+24/66 primary; +26/66 zero intensity).
H10 Car changed from 19/66 to 17/66, with its two-TP difference at 0–20 m; H5 Car stayed 43/66.
The intervention supplies no affirmative evidence that raw intensity mismatch alone is sufficient
to explain Pedestrian failure, but all-zero intensity may also be out of distribution, so that
hypothesis is not ruled out. Retaining the original raw values is not necessary for the observed
positive Car direction. Intensity causality or a unique mechanism is not established. Primary and
zero-intensity processes
are separate realizations and are not paired. The prospective
[S2 scientific protocol](m8/M8_S2_PROTOCOL.md), minimum-gap/denominator-stability rule,
normalized-recovery eligibility, multi-realization partition rule, and S1-derived V2 paired
partitions are frozen. S2 execution is not ready: the CPU-only input implementation and ledger,
1,712/1,712 M7 XYZT identity proof, implementation review, runtime binding, and separate inference
authorization remain pending. S2 and training have not started.

The earlier 779-condition primary attempt remains `INCOMPLETE`, with zero accepted complete
processes and zero accepted canonical calls. A separate interrupted 26-condition attempt before the
accepted pass-1 restart is also preserved externally with zero accepted canonical calls. Neither
partial attempt was spliced into the accepted campaign.

Historical incomplete-attempt accounting remains explicit:

- accepted complete primary processes: 0
- accepted canonical primary calls: 0
- pass 2 started: false
- pass 3 started: false

## External GPU operational status

External GPU tooling is provider-neutral. RunPod has been used operationally for M8 qualification
and Stage R. Several bounded A40/A6000 allocation attempts on 2026-09-22 failed before Pod creation;
they produced no detector calls and no new GPU spend. No Pod is active. These are infrastructure
capacity blockers, not scientific failures.

## Independent external evaluation

OmniLink independently evaluated historical v0.3.0 using synthetic OmniSim scenes. Transform and
reconstruction checks passed, while neither sparse nor native authored traffic-cone input produced
a valid intended match at score threshold 0.25. This is a negative domain-gap observation, not a
dataset-level benchmark. See [the detailed record](external/OMNILINK_OMNISIM_EVALUATION.md).

## Pending and future work

- Implement and review the S2 CPU-only input path and complete input ledger, including the exact
  1,712-condition M7 XYZT identity proof, before any B2/C2/D2/F2 execution. Runtime binding and
  separate owner inference authorization are still required. S2 and training have not started.
- Production segmentation, camera fusion, broader learned representations, physical-sensor
  validation, and productionization remain future work.

See [ROADMAP.md](ROADMAP.md) for milestone order, [BENCHMARKS.md](BENCHMARKS.md) for evidence, and
[FAILURE_INDEX.md](FAILURE_INDEX.md) for preserved failures.
