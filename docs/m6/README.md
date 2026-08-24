# M6 — Cross-Domain Validation Record

M6 is complete. Historical drafts and failed protocols are intentionally retained because M6 used
prospective revisions instead of rewriting failed gates after observing results.

## Start here

For most readers:

1. [M6 cross-domain technical note](M6_CROSS_DOMAIN_TECHNICAL_NOTE.md) — the complete narrative and
   claim boundary.
2. [M6c final R3 result](M6C_RESULTS_R3.md) — final ROS projected-reference integration evidence.
3. [M6b final result](M6B_RESULTS.md) — frozen-detector KITTI Raw characterization.
4. [M6a final R2 result](M6A_RESULTS_R2.md) — accepted offline pose and reconstruction validation.

The three submilestones separate distinct questions:

- **M6a:** offline KITTI Raw adapter, pose, frame-alignment, and reconstruction validation.
- **M6b:** offline characterization of the frozen nuScenes detector on KITTI Raw.
- **M6c:** ROS 2 projected-reference reconstruction and detector/output-path integration validation.

## Chronology

| Submilestone | Failure → diagnosis → prospective revision → final evidence |
|---|---|
| M6a | [Original frozen protocol](M6A_PROTOCOL.md) → initial Tier-A pose-oracle FAIL → [data-product/timing diagnosis](M6A_POSE_ORACLE_DIAGNOSIS.md) → [prospective R2 protocol](M6A_PROTOCOL_R2.md) → [final R2 PASS](M6A_RESULTS_R2.md) |
| M6b | [Preregistration draft](M6B_PREREGISTRATION_DRAFT.md) → [original frozen protocol](M6B_PROTOCOL.md) → 30k engine structural stop → [prospective engine remediation](M6B_ENGINE_REMEDIATION_PROTOCOL.md) → [R2 draft](M6B_PROTOCOL_R2_DRAFT.md) → [frozen R2 protocol](M6B_PROTOCOL_R2.md) → [final characterization](M6B_RESULTS.md) |
| M6c | [Frozen R2 original-M6a byte-exact protocol](M6C_PROTOCOL.md) → [preserved Gate-A failure](M6C_RESULTS.md) → [preregistered D1 plan](M6C_POST_FAILURE_DIAGNOSTIC_PLAN.md) → [D1 diagnosis](M6C_POST_FAILURE_DIAGNOSIS.md) → [R3 feasibility and draft](M6C_PROTOCOL_R3_DRAFT.md) → [frozen R3 protocol](M6C_PROTOCOL_R3.md) → [positive final R3 result](M6C_RESULTS_R3.md) |

### Find the preserved failures quickly

- **M6a initial pose-oracle route:** [original protocol](M6A_PROTOCOL.md) and
  [post-failure diagnosis](M6A_POSE_ORACLE_DIAGNOSIS.md).
- **M6b original engine-profile stop:** [original protocol](M6B_PROTOCOL.md) and
  [structural remediation protocol](M6B_ENGINE_REMEDIATION_PROTOCOL.md).
- **M6c R2 failure:** [preserved failed result](M6C_RESULTS.md).
- **M6c post-failure D1:** [diagnosis](M6C_POST_FAILURE_DIAGNOSIS.md) and its
  [preregistered plan](M6C_POST_FAILURE_DIAGNOSTIC_PLAN.md).

## File index

“Status when written” records the document's role at its own freeze/publication boundary. “Current
status” classifies its role in the completed M6 record. A superseded document remains part of the
scientific chronology; supersession does not turn an observed failure into a pass.

### M6 overview

| Document | Status when written | Current status | Superseded by → | Purpose |
|---|---|---|---|---|
| [M6_CROSS_DOMAIN_TECHNICAL_NOTE.md](M6_CROSS_DOMAIN_TECHNICAL_NOTE.md) | Final public technical narrative after M6 closure | **START HERE** | — | Connects M6a–M6c results, failures, limitations, and final claim boundary. |

### M6a — offline reconstruction

| Document | Status when written | Current status | Superseded by → | Purpose |
|---|---|---|---|---|
| [KITTI_RAW_CONTRACT.md](KITTI_RAW_CONTRACT.md) | Frozen M6a source/dataset contract before detector use | **SUPPORTING CONTRACT / NOTE** | — | Defines authoritative KITTI Raw products, decoding, timing, calibration, and data-use boundaries. |
| [MODEL_FRAME_ALIGNMENT.md](MODEL_FRAME_ALIGNMENT.md) | Frozen M6a R2 frame-alignment contract | **SUPPORTING CONTRACT / NOTE** | — | Defines KITTI Velodyne to frozen detector-frame axes and box conventions. |
| [M6A_PROTOCOL.md](M6A_PROTOCOL.md) | Frozen original protocol before measurement | **FROZEN PROTOCOL** — historical, superseded but retained | [M6A_PROTOCOL_R2.md](M6A_PROTOCOL_R2.md) | Records the original Tier-A pose-oracle route that subsequently failed. |
| [M6A_POSE_ORACLE_DIAGNOSIS.md](M6A_POSE_ORACLE_DIAGNOSIS.md) | Post-failure diagnostic while M6a remained failed | **DIAGNOSIS** | — | Establishes that Raw OXTS and Odometry were distinct data/timing products and motivates R2. |
| [M6A_PROTOCOL_R2.md](M6A_PROTOCOL_R2.md) | Prospectively frozen after diagnosis, before new measurement | **FROZEN PROTOCOL** | — | Defines the accepted Raw-devkit like-for-like pose and reconstruction oracle. |
| [M6A_RESULTS_R2.md](M6A_RESULTS_R2.md) | Canonical R2 PASS awaiting review | **CURRENT / FINAL** | — | Reports the accepted offline pose, decoding, reconstruction, and repeatability evidence. |

### M6b — frozen detector characterization

| Document | Status when written | Current status | Superseded by → | Purpose |
|---|---|---|---|---|
| [M6B_PREREGISTRATION_DRAFT.md](M6B_PREREGISTRATION_DRAFT.md) | Complete draft; not active and no predictions observed | **SUPERSEDED DRAFT** — retained for chronology | [M6B_PROTOCOL.md](M6B_PROTOCOL.md) | Preserves the early proposed cross-domain questions and guardrails. |
| [M6B_PROTOCOL.md](M6B_PROTOCOL.md) | Frozen original protocol before detector inference | **FROZEN PROTOCOL** — historical, superseded but retained | [M6B_PROTOCOL_R2.md](M6B_PROTOCOL_R2.md) | Defines the initial study and preserves the 30k-profile structural stop boundary. |
| [M6B_ENGINE_REMEDIATION_PROTOCOL.md](M6B_ENGINE_REMEDIATION_PROTOCOL.md) | Prospectively preregistered while M6b remained blocked | **FROZEN PROTOCOL** — historical remediation record | [M6B_PROTOCOL_R2.md](M6B_PROTOCOL_R2.md) | Freezes the one-artifact 40k engine-profile remediation and its gates. |
| [M6B_PROTOCOL_R2_DRAFT.md](M6B_PROTOCOL_R2_DRAFT.md) | Draft structural-engine revision pending the H5 profile-gap gate | **SUPERSEDED DRAFT** — retained for chronology | [M6B_PROTOCOL_R2.md](M6B_PROTOCOL_R2.md) | Records the reviewed proposal that became final R2 after the additional gate passed. |
| [M6B_PROTOCOL_R2.md](M6B_PROTOCOL_R2.md) | Owner-approved and frozen before evaluation predictions | **FROZEN PROTOCOL** | — | Defines the final 40k-engine H10/H5 characterization contract. |
| [M6B_RESULTS.md](M6B_RESULTS.md) | Completed characterization ready for review | **CURRENT / FINAL** | — | Reports the complete frozen-detector cross-domain results and limitations. |

### M6c — ROS integration

| Document | Status when written | Current status | Superseded by → | Purpose |
|---|---|---|---|---|
| [M6C_PROTOCOL.md](M6C_PROTOCOL.md) | Frozen prospective R2 before accepted KITTI ROS output | **FROZEN PROTOCOL** — historical failed route, retained | [M6C_PROTOCOL_R3.md](M6C_PROTOCOL_R3.md) | Defines original-M6a byte-exact ROS gates whose frame-1 comparison failed. |
| [M6C_RESULTS.md](M6C_RESULTS.md) | R2 stopped negative result; M6c not ready | **PRESERVED FAILURE** | [M6C_RESULTS_R3.md](M6C_RESULTS_R3.md) as the final outcome; R2 remains failed | Preserves frame 0 PASS, frame 1 FAIL, and the downstream stop. |
| [M6C_POST_FAILURE_DIAGNOSTIC_PLAN.md](M6C_POST_FAILURE_DIAGNOSTIC_PLAN.md) | Preregistered D1 diagnostic only; R2 unchanged | **DIAGNOSIS** — completed plan retained | [M6C_POST_FAILURE_DIAGNOSIS.md](M6C_POST_FAILURE_DIAGNOSIS.md) | Freezes the T0–T4 representation ladder and one-condition downstream scope. |
| [M6C_POST_FAILURE_DIAGNOSIS.md](M6C_POST_FAILURE_DIAGNOSIS.md) | Diagnostic evidence; R2 still failed and R3 not yet authorized | **DIAGNOSIS** | — | Separates platform arithmetic, unit-quaternion/SO(3), tf2, storage, and downstream effects. |
| [M6C_PROTOCOL_R3_DRAFT.md](M6C_PROTOCOL_R3_DRAFT.md) | Draft final-cycle proposal with bounded feasibility evidence | **SUPERSEDED DRAFT** — retained for chronology | [M6C_PROTOCOL_R3.md](M6C_PROTOCOL_R3.md) | Records projected-reference feasibility and the reviewed prospective gate design. |
| [M6C_PROTOCOL_R3.md](M6C_PROTOCOL_R3.md) | Frozen final execution cycle before canonical live output | **FROZEN PROTOCOL** | — | Defines the 860-condition projected-reference gates, detector sentinels, and claim boundary. |
| [M6C_RESULTS_R3.md](M6C_RESULTS_R3.md) | Final positive projected-reference ROS validation | **CURRENT / FINAL** | — | Reports Gate 1, inherited parity-v2 Gate 2, ROS output-contract results, and M6 closure. |

## What M6 ultimately established

- M6a validated the accepted KITTI Raw offline pose and reconstruction route.
- M6b characterized the frozen nuScenes detector on KITTI Raw without fine-tuning.
- M6c showed that the live ROS chain reproduced 860/860 committed same-platform projected
  references byte-for-byte while ten frozen detector sentinels stayed inside the inherited
  parity-v2 semantic envelope.
- M6 does not claim physical-LiDAR validation, real-time ROS performance, official KITTI benchmark
  AP, or portability of projected byte hashes across arbitrary platforms.

M5 remains conditional/inactive. No technical submilestone is active.
