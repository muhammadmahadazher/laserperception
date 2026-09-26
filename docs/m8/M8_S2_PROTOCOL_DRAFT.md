# M8 P1-S2 prospective scientific protocol — draft for owner review

**DRAFT. NOT FROZEN. NO S2 INFERENCE AUTHORIZED.** This document proposes the S2 design and
binds a CPU-only, S1-derived [partition draft](../../benchmarks/m8/preregistration/m8_s2_partitions_draft.json).
Its [machine-readable companion](../../benchmarks/m8/preregistration/m8_s2_protocol_draft.json)
is also a draft. No B2/C2/D2/F2 detector output exists. The existing S1 primary and zero-intensity
results remain bound to different historical execution commits; any future S2 execution will need
its own reviewed implementation, input freeze, runtime binding, and explicit owner authorization.

## Question and inherited evidence

On the frozen 428-frame KITTI Raw corpus, characterize how DSVT's Car H10-to-H5 recall difference
responds to encoded-lag compression, exact H5-total-point-count matching, their combination, and
a natural long-span comparator. This is a deterministic intervention study on a fixed pretrained
DSVT-Pillar/TransFusion stack, not a randomized or population-causal experiment. The primary S2
outcome would be Car recall at score ≥0.25 and oriented BEV IoU ≥0.50 over 66 eligible Car GT.

The [frozen S1 protocol](M8_S1_PROTOCOL.md) and accepted
[primary measurement](M8_S1_MEASUREMENT_RAW.md) supply three complete A2/H10 and E2/H5 processes.
Their scientific execution commit is `6994d72c3e7691a86116d1417ac3ae08256d163f`. Each has
`TP_A2,Car=19` and `TP_E2,Car=43`, hence gaps `D_1=D_2=D_3=24`. The separate
[zero-intensity interpretation](M8_S1_ZERO_INTENSITY_INTERPRETATION.md) belongs to execution
commit `95fb66ac1f57c41f06f05bd9ef5dac27b1e3ea54`; it is closed S1 sensitivity evidence,
not an S2 arm or a tuning signal. It observed no Pedestrian rescue, persistence of the positive
Car H5/H10 direction, and no intensity-causality result. All-zero intensity may itself be out of
distribution.

M7 is the mechanical ancestor, not a detector-outcome reference for S2. Its
[protocol](../m7/M7_PROTOCOL.md), [input freeze](../m7/M7_INPUT_FREEZE.md),
[implementation review](../m7/M7_IMPLEMENTATION_REVIEW.md),
[raw measurement](../m7/M7_MEASUREMENT_RAW.md), and [interpretation](../m7/M7_RESULTS.md)
establish exact input-transformation definitions and claim limits. M8 changes the detector to
DSVT, uses five rather than four point features, consumes raw KITTI intensity, and has small
numerical variation in score-ranked outputs. S1 has three process realizations rather than M7's
single frozen A/E realization. Therefore S2 uses the S1 Car gap 24, not M7's PointPillars gap 32;
does not borrow M7 matched-GT partitions; and does not assume exact floating-point repeatability.
A2/E2 predictions are not casually rerun.

## Three-realization S1 partition proposal

The accepted private **primary** pass archives were read without altering them. Their archive,
raw-pass, final-manifest, and all 856 per-condition file hashes per process were checked against
the [published acceptance manifest](../../benchmarks/m8/results/m8_s1_measurement_manifest.json).
For each eligible pose and arm, the score/IoU 0.50 matched-GT identity set was cross-checked with
the per-condition target observation and TP count. The exact pose key is the M7 convention
`(drive_id, frame_index, gt_track_id)`, sorted lexicographically. Each category's SHA256 is over
M7 canonical JSON: sorted object keys, compact separators, ASCII, and one final LF. The draft
contains every ordered identity, not merely counts.

| Accepted pass | Process UUID | Archive SHA256 | Raw-pass file SHA256 |
|---|---|---|---|
| 1 | `3256b511-921e-4456-92c6-1fd5d5a8c380` | `fab1db337c41788d38bf2560129494be65e2f98a3e153a130bfd03dc915e26d1` | `deb7286ef3857283a362f96523d34e04c0fea422b7e2e41c02c4681d434c74a3` |
| 2 | `21f32e37-58c3-42e2-b9c1-2011b20e47b7` | `7df28ade47aacd3dde1a2ab836c6796522df75a242ec52ba0e5e2ab5a67443ee` | `8a2850aba316112f7ff91552f52d7e8f1f4d0fd9c62263a4216ec7c5096bee7f` |
| 3 | `b9c1ab4b-9ea2-428a-a41e-e70fb8528d87` | `5bf95e23058c4e01bbf79a35682df24454342f39ac9398e2da3b70c4e33e33a1` | `0681de95b077bda1c98f6c7de7f8a38b74431316bc0ebc4140c4ffceede9ad3e` |

For each pose and arm: **stable detected** means detected in all three accepted primary
processes; **stable missed** means missed in all three; **unstable** means detected in only one or
two. A pose enters shared, E2-only, A2-only, or neither only when both arm states are stable.
Any arm instability puts the pose in the explicit unstable bucket. This is a full per-arm
three-process classification, including stable misses; it is not a selected pass, median
prediction, simple intersection, or union. Zero-intensity, Stage R, and incomplete primary
attempts are excluded.

| Class | Shared | E2-only | A2-only | Neither | Unstable | Total |
|---|---:|---:|---:|---:|---:|---:|
| Car | 19 | 24 | 0 | 23 | 0 | 66 |
| Pedestrian | 0 | 1 | 0 | 395 | 0 | 396 |

For Car, `19+0=19` reproduces A2 TP, `19+24=43` reproduces E2 TP, and `24−0=24` reproduces the
gap. All three process-level GT state maps are complete, with no duplicate or missing eligible
identity. Pedestrian partitions are descriptive only; no normalized Pedestrian recovery is
proposed. The empty-list canonical hash is the same for each zero-count bucket.

| Car category | Canonical list SHA256 |
|---|---|
| Shared | `bb010b66448bd735389a1888549205c80a6ffaaf573fabbbf7665639d7fc8ecb` |
| E2-only | `408285adfbf9df5a6f1b58c21bae3b82cc3952a6ce95c7289fc95cce59d2b0f0` |
| A2-only | `37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570` |
| Neither | `e10777605f4421d004149e499b0d9ce0b3480a0c396e78cc6a560b272c28a67f` |
| Unstable | `37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570` |

These are draft partition commitments for owner review, not a final freeze. The
[CPU-only derivation](../../benchmarks/m8/derive_s2_partitions_draft.py) can reproduce them from
the three exact accepted archives. No large private archive enters Git.

## Denominator and eligibility proposal

Let `D_i = TP(E2_i,Car) − TP(A2_i,Car)` for accepted primary process `i`. Normalized recovery
would be eligible only if all three `D_i>0`, all three are exactly equal, the common gap is at
least 20, **and** the Car partition is stable and reconstructs A2=19/E2=43/gap=24 with zero
unstable Car poses. The minimum 20 is a prospective measurement-resolution choice: one TP moves
an unclamped normalized fraction by `1/D≤0.05`. It is not a significance cutoff and was not
chosen from any B2/C2/D2/F2 outcome. Here `D=(24,24,24)`, so `1/D=1/24≈0.041667`; the proposed
denominator and partition gates pass on frozen S1 evidence.

If the owner freezes this proposal, define fixed `A_REF=19`, `E_REF=43`, `D_REF=24` and, for each
future complete S2 process `i` and arm `X`, `G_car(X_i)=(TP_X_i−19)/24`. Do not clamp: negative,
zero, partial, full, and greater-than-one values all remain reportable. Show passes 1/2/3 before
their min/median/max. The processes repeat one fixed corpus; they are not independent dataset
samples. If any denominator or partition gate fails, report raw TP deltas, all paired counts, and
unstable identities, without headlining normalized recovery or selecting a different consensus
rule after the failure.

## Prospective five-feature arms

Every S2 input would be C-contiguous float32 `[x,y,z,intensity,time_lag]`, preserving the frozen
M8 source-row order. **All arms retain raw KITTI reflectance for each selected row byte-for-byte.**
Zero-intensity is excluded. Each transformation inherits the exact M7 mechanical rule but must be
validated on M8's five-feature inputs before detector execution.

| Arm | Proposed input | What it characterizes |
|---|---|---|
| A2 | Existing native H10, no design-stage rerun | Frozen reference |
| E2 | Existing native H5, no design-stage rerun | Frozen positive comparator |
| B2 | Exact A2 rows, sweep identities, row count/order, XYZ, and intensity; change only `time_lag` by M7's T5/T10 scale | Encoded-lag magnitude sensitivity, not physical temporal span |
| C2 | All A2 current rows plus deterministic H10 historical subsets; exact E2 total point count; native retained XYZ/intensity/lag, original global A2 order | Total point count, not spatial density, occupancy, sweep identity, or smear |
| D2 | Reuse C2's exact selected global-row vector; apply B2 lag mapping, never resample | Combined intervention; C2/D2 differ only in lag |
| F2 | Copy complete A2 current sweep and historical ranks 2,4,6,8,10, in original order with native values | Natural unthinned long-span comparator at H5 sweep count, not span isolation |

For B2/D2 select `T10_f32` and `T5_f32` from the existing A2/E2 historical float32 lags;
convert exactly to binary64, divide `s=T5/T10`, then apply one binary64 multiplication and one
final float32 cast per retained historical row. Rewrite current lags as positive float32 zero;
fail on malformed lags, invalid scale, or support collapse. For C2, use M7 integer
largest-remainder quota allocation (descending remainder, lower rank on ties), its exact
`laserperception-m7-c-v1|drive|ten-digit-frame|rank` SHA256 seed and SplitMix64 key, lowest
`(key, ordinal)` selection, and restoration of global A2 row order. D2 reuses that row vector.
F2 retains complete selected sweeps. There is no second seed or outcome-driven alternative.

For each of the 428 frames and B2/C2/D2/F2, the future five-feature input must project columns
`[0,1,2,4]` into C-contiguous little-endian float32 XYZT bytes matching the corresponding frozen
M7 [input-manifest](../../benchmarks/m7/preregistration/m7_input_manifest.json)
`model_ready_sha256` **exactly**: 428 checks per arm, 1,712/1,712 overall. The M7 manifest's
SHA256 is `8d4f74d783950d24956239f3a67a7a58fe10013e0e83a88d0f8b23e3139ffe90`.
This proof has **not** been performed in the design iteration because the future S2 input-only
ledger has not been implemented or frozen. Any mismatch blocks inference. Retained M8 intensity
must correspond to each selected raw source row; projection equality alone does not prove that.

## Input-only implementation and ledger barrier

Before any S2 inference, a separately owner-reviewed CPU input implementation and ledger must
bind per frame/arm: source frame and A2/E2 input identities; row count/order hash; XYZ, intensity,
XYZT projection, and full XYZIT SHA256; sweep membership and per-sweep counts; B2/D2 T10/T5
float32 bits, scale binary64 bits, and cast policy; C2/D2 `N0,N1..N10,H_target,H_total`, integer
products, initial quotas, remainders, increment decisions, final quotas, exact seeds, and selected
global-row SHA; F2 ranks `[2,4,6,8,10]`; candidate/retained pillar counts if applicable; and
structural coordinate/order identities. Validate B2/A2 row and structural-coordinate identity,
C2's current retention and exact E2 total count, D2/C2 row/XYZ/intensity and structural-coordinate
identity, and F2's complete-sweep subset relation. No detector output may be generated for this
ledger. Its schema is proposed here; neither the implementation nor ledger is frozen.

## Proposed repeatability and full-corpus execution

Only after future protocol/implementation/input/runtime freeze and a separate explicit owner
authorization: use the **same final seven S1 sentinel frames** in S1 order. Run ten fresh
repeatability processes, each executing B2,C2,D2,F2 in that order within every sentinel:
`7×4=28` conditions per process, `10×28=280` proposed accepted diagnostic calls. No sentinel
output is reused as a canonical corpus output. At score ≥0.25 and IoU ≥0.50, require exact
agreement across all ten processes for every arm/sentinel/class on thresholded prediction count,
TP, FP, FN, ignored-prediction count, and matched-GT identity set. At IoU 0.30 and 0.70 require
exact TP and matched-GT identity set. Any discrete disagreement blocks the full corpus and returns
to owner review for prospective revision. Raw floating-point tensor byte equality is not the
criterion.

If that gate passes, run **exactly three fresh complete S2 corpus processes**, fixed before any
S2 result. Each processes all 428 frames, each in B2,C2,D2,F2 order: `428×4=1,712` conditions per
process and `3×1,712=5,136` proposed accepted corpus calls. Repeatability never changes this
pass count. Each process initializes its bound detector once and keeps a separate record. Preserve
each failed/incomplete process as evidence, count zero accepted canonical processes from it, and
restart its entire logical pass from frame 1/B2 in a fresh process. Never splice condition outputs
from different processes. The proposed total is `280+5,136=5,416` accepted scientific calls;
**actual S2 calls remain zero**.

## Proposed reporting, paired sets, and interpretation

For Car and Pedestrian report pass 1/2/3 before min/median/max: TP/FP/FN, recall,
annotation-conditioned precision/F1/AP, IoU 0.30/0.50/0.70, 0–20/20–35/35–50 m range slices,
track continuity, prediction population, outside-FOV counts, and neighbour-ignore behavior.
Precision and AP remain conditioned on incomplete KITTI Raw tracklets, not official KITTI AP or
whole-world false-positive performance. Pedestrian remains secondary with no normalized recovery
formula proposed here.

If eligibility holds, report each future Car arm's exact detections among frozen E2-only (24),
shared (19), A2-only (0), and neither (23) poses, including gained/lost identities. Define
`R_gain=detected E2_ONLY/24`, `R_shared=detected SHARED/19`, `R_Aonly=null/not-applicable` because
the A2-only denominator is zero, and `R_neither=detected NEITHER/23`. Aggregate TP equality alone
does not establish the same pose behavior.

The proposed descriptive phrase “substantially accounts for the observed H10-to-H5 Car
improvement” is permitted only if `G_car(X_i)≥0.50`, `R_gain(X_i)≥0.50`, and at most **one** frozen
shared Car positive is lost. To avoid selecting a favorable S2 realization, the draft proposes
requiring **all three** complete S2 processes to meet all three conditions before using that
phrase; otherwise report each pass and which gate failed. This is an explicit owner-review choice,
not a significance test, population causal estimate, proof of mechanism, or production criterion.
It differs from M7's denominator and paired counts because M8's S1 evidence differs.

For scalar Car TP or recall at the frozen operating point, with fixed A2 threshold reference
`Y_A_REF`, report per S2 process:

```text
L_i = ((Y_B2_i - Y_A_REF) + (Y_D2_i - Y_C2_i)) / 2
P_i = ((Y_C2_i - Y_A_REF) + (Y_D2_i - Y_B2_i)) / 2
I_i = Y_D2_i - Y_B2_i - Y_C2_i + Y_A_REF
```

A fixed A2 reference is defensible for these discrete Car outcomes because all three accepted S1
processes agree exactly. F2 is outside the 2×2 factorial. Do **not** construct fixed-A2 AP
factorials or invent process pairing with historical A2 for AP, whose primary values show numerical
spread. Report AP arm distributions descriptively. These factorial contrasts are fixed-corpus
descriptions, not causal percentages or population effects.

## Claim limits and owner decision

B2 can characterize encoded-lag sensitivity; C2 exact total-point-count matching; D2 their
combination; F2 a natural long-span comparator. None isolates physical span or establishes that
time, density, architecture, or raw intensity caused the observed H5/H10 difference. There are no
p-values, confidence intervals, universal model claims, or production-readiness claims. The
three future processes would characterize numerical/runtime variation on one fixed corpus, not
independent dataset samples.

The owner must decide whether to accept the proposed `D≥20` resolution rule, strict zero-unstable
Car partition gate, exact repeatability gate, three-process interpretation rule, and all input
identity requirements. Even if accepted scientifically, a final protocol freeze, implementation
review, 1,712/1,712 XYZT proof, input-only ledger freeze, runtime binding, and separate inference
authorization remain necessary. **This draft itself authorizes none of them.**
