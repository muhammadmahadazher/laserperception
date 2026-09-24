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
frozen S1 protocol are complete. External RunPod A40 qualification and a fresh ten-process Stage R
campaign were completed and preserved. A later primary attempt ended `INCOMPLETE` after 779
attempted detector conditions:

- accepted complete primary processes: 0
- accepted canonical primary calls: 0
- pass 2 started: false
- pass 3 started: false

That partial attempt is not a primary scientific result and cannot be spliced into a future pass.
PRs #41–#45 subsequently established a complete execution-bound CPU receipt, restored the full
856-condition live pre-inference gate, selected four bounded CPU revalidation workers, used
frame-major 428-pair inference, and preserved fail-closed evidence before source loading, gate
execution, and backend construction.

The pending campaign remains frozen to execution commit
`6994d72c3e7691a86116d1417ac3ae08256d163f` and receipt file SHA256
`bef4c55575581aefe8f477e32d1b394f40823a0b0858c66c3ac5c1fae141ec4d`. The v0.4.0 release
does not rebind that campaign. See
[M8 external-runtime status](m8/M8_S1_EXTERNAL_RUNTIME_STATUS.md).

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

- Complete the three-pass M8 primary A2/E2 campaign under fresh runtime-specific authorization.
- Zero-intensity remains unauthorized; S2 and training have not started.
- Production segmentation, camera fusion, broader learned representations, physical-sensor
  validation, and productionization remain future work.

See [ROADMAP.md](ROADMAP.md) for milestone order, [BENCHMARKS.md](BENCHMARKS.md) for evidence, and
[FAILURE_INDEX.md](FAILURE_INDEX.md) for preserved failures.
