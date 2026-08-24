# M6c Protocol R3 draft — projected-reference ROS exactness

STATUS:
**DRAFT — NOT FROZEN — OWNER REVIEW REQUIRED**

This document proposes the final M6c execution cycle. It does not authorize, freeze, or start R3.
The only new measurement behind this draft is the bounded CPU/ROS feasibility check described
below; no detector, GPU, TensorRT, Gate A, Gate B, or detector sentinel was run.

## Preserved scientific state

Revision R2 remains **FAILED** at frozen protocol and measurement commit
`0a8419978d265571b51f943ffc797b5fcc78c4ca`. Its original Gate A result remains one PASS, one
FAIL, and 22 pending; Gate B did not start. Post-failure D1 remains diagnostic only and does not
revise R2.

D1 separated the frame-1 transform path into the following ladder:

| Boundary | Result |
|---|---|
| T0 frozen Windows canonical → T1 direct WSL matrix arithmetic | small platform difference |
| T1 → T2 same-platform matrix → unit quaternion → matrix | dominant representation difference |
| T2 → T3 real `lookup_transform_full` | float32 exact |
| T3 → T4 builder storage | float32 exact |

The accepted KITTI rotations are not perfectly orthonormal, while ROS TF carries a unit
quaternion. The conversion therefore projects the serialized rotation onto an orthonormal proper
rotation. D1 showed that the original M6a bytes are not directly representable through that TF
boundary, but suggested that an independently generated TF-representable reference could still
test the ROS integration exactly.

## Claim decomposition and explicit lineage change

The milestones address different claims:

- **M6a** independently validates official KITTI Raw files, OXTS/calibration pose semantics, and
  offline reconstruction arithmetic against its frozen official-reference contract.
- **M6b** freezes and characterizes detector behavior on the accepted offline inputs.
- Proposed **M6c R3** would test whether ROS transport, timestamp handling, time-aware tf2,
  bounded history, and builder integration reproduce an independent ROS-representable offline
  reference byte-for-byte, then test whether the representation-induced input change stays within
  the project's pre-existing detector parity envelope relative to frozen M6b results.

This is an explicit weakening of the attempted R2 lineage. R2 compared the frozen Windows M6a
oracle bytes directly with live ROS output. R3 would compare a same-platform projected reference
with live ROS output. R3 would **not** re-prove the M6a pose derivation byte-for-byte through ROS.
M6a remains the independent pose/reconstruction validation; M6c would begin after the unavoidable
representation conversion required by TF. New projected hashes would be M6c-only artifacts and
would not replace any M6a or M6b identity.

## Projected offline reference

The projected reference is generated on the measurement platform, without ROS or tf2, by this
path:

```text
accepted KITTI absolute pose matrices
  -> frozen matrix-to-unit-quaternion conversion
  -> independent unit-quaternion-to-matrix reconstruction
  -> offline historical-to-current composition
  -> existing MultiSweepBuilder arithmetic
  -> model-ready float32 XYZT
```

The live path under test is:

```text
KITTI Raw
  -> raw PointCloud2
  -> published unit-quaternion TF
  -> lookup_transform_full
  -> live history
  -> LaserPerceptionMultiSweepNode
  -> model-ready PointCloud2
```

| Component | Projected offline reference | Live ROS path | Relationship |
|---|---|---|---|
| KITTI Raw decoding | accepted `KittiRawSequence` semantics | same accepted source decoding and replay | shared source semantics |
| Absolute KITTI poses | accepted OXTS/calibration poses | same accepted poses | shared |
| Matrix → unit quaternion definition | frozen conversion | frozen conversion used for TF publication | shared representation boundary |
| Historical relative transform composition | direct offline matrix composition after projection | `lookup_transform_full` through the fixed frame | independent computation |
| Raw point transport | direct `RawSweep` input | PointCloud2 serialization, transport, and decoding | independent live boundary |
| History selection | offline requested historical set | `LiveSweepHistory` in the ROS node | independent |
| Builder mathematics | existing `MultiSweepBuilder` contract | existing `MultiSweepBuilder` used by the live node | shared arithmetic |
| Model-ready transport | direct in-memory reference | ROS PointCloud2 publication and decoding | independent live boundary |

Both paths share accepted KITTI decoding and source poses, the frozen matrix-to-unit-quaternion
definition, and the `MultiSweepBuilder` mathematical contract. The projected reference does not
obtain transforms from tf2, `LaserPerceptionMultiSweepNode`, or messages produced by the live
path. The live builder node consumes only published PointCloud2 bytes and tf2. The comparison
therefore covers ROS serialization, timestamps, TF transport, cross-time fixed-frame composition,
history selection, and live builder integration.

Gate 1 would validate PointCloud2 transport, timestamp handling, TF publication and transport,
time-aware fixed-frame composition, live history selection, ROS builder orchestration, and
model-ready PointCloud2 serialization. It would not independently revalidate official KITTI pose
derivation or the internal mathematical correctness of `MultiSweepBuilder`; those claims are
already covered by earlier milestones.

## Bounded feasibility result

The feasibility implementation was committed before output measurement. The first invocation at
`2e4810d28cb92f2a935c335977d187fbaff9f821` stopped during ROS node initialization because a
diagnostic topic token began with a number. It published no PointCloud2 and evaluated no
condition. The orchestration-only name was corrected and committed; the exact run occurred at
`cafc67f41e9abc12fa0e9a9e76a2ef6add197bf1`.

The three frozen, non-outcome-selected H10 conditions all passed:

| Drive / current frame | Projected vs tf2 transforms | Model-ready result | Points | Shared SHA256 |
|---|---:|---:|---:|---|
| `2011_09_26_drive_0001/0000000010` | 10/10 exact | byte exact | 1,312,220 | `d96ceec739735bac578ae812108af98892d2939f9ce9821584ebdba31412d3e5` |
| `2011_09_26_drive_0001/0000000107` | 10/10 exact | byte exact | 1,297,870 | `6f7f63b7db7de179db11bad0a4793ab79208e6db009ff76362b2160351eaa1d2` |
| `2011_09_26_drive_0091/0000000010` | 10/10 exact | byte exact | 1,236,530 | `a6021ea7d9bc6e803c79f8f8241a7c0ae99b60dddadb71097b135cc342b2bc3c` |

All 30/30 transform comparisons had exact rotation, translation, and complete float32 builder
storage; zero float32 elements differed and the maximum absolute delta was `0.0`. All 3/3
model-ready results had exact timestamp, depth, point count, shape, dtype, row order, and complete
XYZT bytes.

Because H10 passed completely, the single permitted H5 check ran for
`2011_09_26_drive_0091/0000000010`. Its 5/5 transforms and model-ready output were exact; both
model-ready paths had 676,502 points and SHA256
`3aba6ae5c0fb308a442b7a98e0d772306672b6728eab55a9ea1fc5db0f8d95dc`.

The classification is **`PROJECTED_REFERENCE_BYTE_GATE_FEASIBLE`**. This positive bounded result
supports proposing a byte gate; it is not the canonical R3 corpus measurement.

## Proposed Gate 1 — projected ROS input byte exactness

Gate 1 would compare the projected offline lineage above with the live ROS lineage. Every
condition must have exact timestamp, requested and actual history depth, point count, shape,
float32 dtype, row order, complete XYZT bytes, and model-ready SHA256.

- **Gate 1A:** the existing M6a 24-frame target set at H10; require 24/24 exact.
- **Gate 1B:** the complete frozen M6b input corpus, 428 H10 plus 428 H5 conditions; require
  856/856 exact.

No tolerance, reduced population, or ROS-derived reference is permitted. The complete run should
record wall-clock progress and use the existing resumable ledger if it cannot finish in one
session; the corpus must not be reduced to fit a session.

Before any canonical live Gate 1 measurement, the prospective sequence would be:

1. Verify the frozen M6a/M6b source artifacts and identities.
2. Generate all projected-reference identities on the frozen measurement platform without ROS or
   tf2.
3. Record compact per-condition drive, frame, H10/H5, useful transform identity, point count, and
   model-ready SHA256.
4. Commit those new M6c-only identities.
5. Freeze a separate `docs/m6/M6C_PROTOCOL_R3.md`.
6. Only then run the live ROS Gate 1 campaign against the committed identities.

The reference must never be regenerated from observed live output.

### Original-reference characterization, not a gate

R3 should separately describe projected-reference versus frozen M6a/M6b input differences:
exactness, differing points/values, maximum deltas, and any already measured discrete
voxel-coordinate change, attributing the shift to platform arithmetic plus unit-quaternion/SO(3)
projection. This is descriptive only. Projected references are not required to equal original
M6a/M6b bytes, and their identities do not modify those frozen artifacts.

## Proposed Gate 2 — unchanged parity-v2 detector envelope

Only after each detector condition passes Gate 1 would its projected/live detector result be
compared with the frozen M6b detector result. The existing parity-v2 code and configuration are
reused unchanged. In this new application, parity-v2's historical `pytorch`/reference fields map
to frozen M6b results, while its `tensorrt`/candidate fields map to results from the exact
projected/live ROS inputs. This is a new use of the envelope, not the FP32-versus-FP16 experiment
for which it was originally designed.

Reuse is proposed because these preregistered thresholds already define LaserPerception's
accepted deployment semantic equivalence, predate M6c and the observed frame-10 301-versus-302
diagnostic, and avoid inventing post-hoc M6c tolerances. R3 must not change a threshold, count
guard, confidence boundary, matcher, sample, or class.

Parity-v2 was originally accepted for rewritten/native FP32 versus TensorRT FP16 comparison on
identical model-ready inputs. R3 would apply the same unchanged numerical envelope to a different
source of variation: detector outputs caused by the ROS-representable projected input rather than
FP16 deployment arithmetic alone. The thresholds are inherited as LaserPerception's pre-existing
semantic deployment standard; they were not statistically derived for quaternion-projection input
noise. A Gate 2 pass would mean only that the representation-induced detector change remained
inside that inherited envelope. It would not establish that this perturbation is experimentally
equivalent to the original FP32-versus-FP16 comparison. Reuse remains preferable to inventing an
M6c tolerance after observing M6c outcomes.

The inspected parity-v2 contract is `configs/detection/m2_parity_v2.yaml` at SHA256
`91e7cde19076c6452d9ff8e0fefc893a6d429622ed30c2da88127d29d4418df0`. Its Stage 1 evaluator,
sample analyzer, and matcher are respectively identified by SHA256
`24fd8c7bcf8ee74049682ecd7d93989f4d62736eaeb35033155c0115281c38b4`,
`37652e464a785174170240e99d593cd9d00a8362008537e182ad0e2b0a83d7f0`, and
`1be52b850ba5f41e1abf96e83923c1f4dbe65a5a2c592a4e6bb4185dc7e83c00` in
`parity_v2.py`, `parity_validation.py`, and `parity.py`.

The exact accepted parity-v2 Stage 1 contract is:

- Export detections at score `>= 0.25`; high confidence is score `>= 0.30`.
- Record the inclusive score edge band `[0.20, 0.30]`; edge cases are diagnostic and are not
  excluded from high-confidence coverage.
- Match one-to-one and class-wise. Process reference detections by descending score followed by
  the stable Detection3D sort key, then choose the maximum-BEV-IoU unmatched same-class candidate
  with minimum IoU `0.50`.
- Per condition, exported-count absolute difference must be at most
  `max(1, ceil(0.05 * reference_exported_count))`.
- Across all ten conditions, exported-count absolute difference divided by the total reference
  exported count must be `<= 0.05`.
- Bidirectional high-confidence match coverage must each be `>= 0.99`. A match enters the
  continuous-metric denominator when either member has score `>= 0.30`.
- Each continuous metric independently requires at least a `0.99` per-detection pass fraction:
  XY center displacement `<= 0.25 m`; absolute Z difference `<= 0.25 m`; maximum relative error
  across L/W/H `<= 0.05` (all three dimensions must pass for that detection); absolute score
  difference `<= 0.05`; and box-axis yaw difference modulo pi `<= 5.0°`.
- Full-heading direction agreement must be `>= 0.99`; disagreement is a full circular yaw
  difference greater than `90°`.
- High-confidence class-name mismatches must equal zero.
- The union of continuous outliers is descriptive, not an additional gate. Each exception remains
  in every applicable metric denominator and may fail more than one metric.
- Stage 1 passes only when every check above passes. A failure requires Stage 2 forensics, but
  forensics cannot alter the result, tolerance, matching, or population.

The frozen detector population remains exactly five current frames crossed with H10 and H5:

1. `2011_09_26_drive_0001/0000000010` H10 and H5;
2. `2011_09_26_drive_0001/0000000083` H10 and H5;
3. `2011_09_26_drive_0001/0000000011` H10 and H5;
4. `2011_09_26_drive_0001/0000000015` H10 and H5;
5. `2011_09_26_drive_0091/0000000010` H10 and H5.

The ten-condition campaign must pass the unchanged aggregate Stage 1 contract, and every
condition must pass its per-condition count guard. This draft does not invent a separate
per-sentinel continuous tolerance. Any unchanged Stage 1 failure makes R3 fail; no result may be
relaxed after observation. Detector output byte equality is not required.

## Final-cycle decision policy

R3 is intended to be the final M6c execution cycle:

- If Gate 1 exactness and Gate 2 semantic parity both pass, M6c completes positively.
- If Gate 1 fails, the result is: “Live ROS integration did not reproduce even the
  ROS-representable offline reference exactly.”
- If Gate 1 passes but Gate 2 fails, the result is: “ROS integration reproduced the projected
  input exactly, but the representation-induced input shift changed detector output beyond the
  project's pre-existing deployment parity envelope.”

Either failure closes M6c as a documented negative result. There is no automatic R4,
implementation remediation, tolerance revision, or population change. Only the owner may reopen
the milestone for a newly discovered implementation defect.

On success, the claim would be limited to this decomposition:

1. M6a independently validated KITTI Raw pose/reconstruction semantics against the official Raw
   reference.
2. Mapping accepted poses to ROS TF's unit-quaternion representation introduced a measured
   numerical shift from the original serialized matrix oracle.
3. Given that projected reference, the live ROS transport/tf2/history/builder path reproduced the
   projected model-ready inputs byte-for-byte across the frozen corpus.
4. Detector behavior remained inside LaserPerception's pre-existing deployment parity envelope
   relative to frozen M6b outputs.

It would not claim original M6a bytes through ROS, preservation of a non-orthonormal matrix by
tf2, byte-identical detections, changed M6b measurements or H5/H10 conclusions, physical LiDAR
validation, or real-time performance.

## Scope and future documentation

R3 would not authorize a model, checkpoint, ONNX, TensorRT engine, precision, voxel geometry,
threshold, postprocess, history, ROS contract, performance, optimization, training, tracking,
camera, Jetson, or physical-LiDAR change. This draft task ran no detector or GPU work and did not
rerun Gate A or start Gate B.

After M6c closes, the public M6 technical note should be updated once with the R2 exact-byte
failure, D1 transform ladder, platform contribution, dominant quaternion/SO(3) projection,
float32 tf2 fidelity, frame-1 structural stability, frame-10 accumulation-induced discrete
divergence, and final R3 outcome. It must not be updated before the final outcome.

Compact feasibility evidence:
[`r3_projected_reference_feasibility.json`](../../benchmarks/m6c/diagnostics/r3_projected_reference_feasibility.json).

Current governance is unchanged: M6a complete, M6b complete, M6c **NOT READY**, and M6 in
progress.
