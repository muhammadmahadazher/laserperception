# M7 controlled history-mechanism study — frozen protocol

Status: **OWNER APPROVED AND FROZEN BEFORE M7 INPUT GENERATION.**
**NO M7 INTERVENTION INPUT OR DETECTOR OUTPUT EXISTS.**

This document freezes the owner-approved prospective design from source draft commit
`7700216c234c0c4bf908dba6ab5a7106e730a627`. Owner approval occurred before M7 input generation,
implementation measurement output, or detector inference. The commit adding this file is the
scientific protocol-freeze boundary for later M7 work.

Implementation remains a separate reviewable step. This protocol does not authorize B/C/D/F input
generation, detector initialization or inference, threshold exploration, training, tuning, or a
production-policy change. After implementation approval, an input-only freeze will occur. Only
after owner review of that input-only freeze may a second explicit authorization permit detector
inference.

## Question and claim boundary

M7 asks:

> Which measurable components of the H10 model-input contract explain the large H5 improvement
> observed in frozen M6b?

M7 is not simply a “temporal span versus density” experiment. Changing encoded `time_lag` while
retaining H10 points does not remove the physical consequences of long-history accumulation:
historical XYZ geometry, moving-object smear or displacement, occlusion, duplicated surfaces,
accumulated scene composition, or sweep membership. Matching total point count does not match
spatial density, pillar occupancy, temporal spacing, or retained-pillar selection. Unexplained
difference remains in an explicit **residual history-effect** bucket.

The interventions are deterministic transformations of a fixed corpus. The 2×2 contrasts are
descriptive mechanistic contrasts, not a randomized experiment or population causal estimates.
KITTI Raw tracklets remain incomplete for the full LiDAR world. GT-linked recall is therefore the
primary mechanism metric; precision and AP are annotation-conditioned descriptive quantities, not
physical false-positive rates or official KITTI benchmark results.

## Frozen M6 continuity and evidence

The following records remain authoritative and unchanged:

- [M6b original protocol](../m6/M6B_PROTOCOL.md)
- [M6b owner-approved Protocol R2](../m6/M6B_PROTOCOL_R2.md)
- [M6b results](../m6/M6B_RESULTS.md)
- [M6 cross-domain technical note](../m6/M6_CROSS_DOMAIN_TECHNICAL_NOTE.md)
- [KITTI Raw contract](../m6/KITTI_RAW_CONTRACT.md)
- [model-frame alignment](../m6/MODEL_FRAME_ALIGNMENT.md)
- [M6b configuration](../../configs/m6/kitti_m6b.yaml)
- [compact frozen input ledger](../../benchmarks/m6b/diagnostics/pre_inference_input_ledger.json),
  SHA256 `2413224808b0140856a0e00f884c53bc9b49ec6b545172f5e7fa57d00802dc15`
- [compact frozen result](../../benchmarks/m6b/results/kitti_raw_cross_domain_characterization.json),
  SHA256 `b9d47120aabb38733d79f987f16277ebab626d2b7c13159c40c30a68b76c1d26`
- v0.3.0 full input ledger release asset, 5,837,452 bytes, SHA256
  `e25b3d62113cc7e8c1fcf736caa68b1ab698f965f007c758ff91d3e498ca6caa`
- v0.3.0 full result release asset, 41,987,113 bytes, SHA256
  `87870b2aa0cc2a91d39331afc8154fdad0c8c796f1cabfb4f8530a3eb106de27`

The full assets must be downloaded from the
[v0.3.0 release](https://github.com/muhammadmahadazher/laserperception/releases/tag/v0.3.0)
and verified before any implementation or ledger freeze. They are inputs to M7 provenance, not
new measurements.

The corpus remains exactly:

- `2011_09_26_drive_0001`, frames 10–107 (98 current frames);
- `2011_09_26_drive_0091`, frames 10–339 (330 current frames);
- 428 ordered current frames, ordered-frame SHA256
  `76bd5f7adac3d892ad2fb00cb9cf5f4f73dd475682ee011b7ea9524060c46c95`.

The frozen baseline model-ready commitments remain:

- H10 ordered hashes:
  `63f4bd20d33a62948dc9a2593b57509380848cb48980827d0b0352c47fa37469`;
- H5 ordered hashes:
  `e5f43d6511d96f6db232c880f94b5464ab5d217f5e5bfdf34bd1626ab8ac7f89`.

M7 must load A/E predictions from this accepted evidence and must not rerun either baseline through
the detector. Because the released ledgers commit input identities rather than embedding every
point row, the future input-only implementation may reproduce A/E model-ready arrays from the
frozen transforms and unchanged M6b builder solely as intervention sources. Every reproduced array
must pass its per-condition frozen model-ready hash before use; it does not become a new baseline or
authorize a new A/E prediction.

Frozen M6b operating results at score 0.25 and oriented BEV IoU 0.50 are:

| Arm | Class | TP | FP | FN | Recall |
|---|---|---:|---:|---:|---:|
| A/H10 | Car | 16 | 144 | 50 | 0.242 |
| E/H5 | Car | 48 | 261 | 18 | 0.727 |
| A/H10 | Pedestrian | 219 | 3,831 | 177 | 0.553 |
| E/H5 | Pedestrian | 268 | 3,868 | 128 | 0.677 |

The input shift motivating M7 remains:

| Statistic | A/H10 | E/H5 |
|---|---:|---:|
| Mean points | 1,330,654 | 726,561 |
| Median points | 1,334,420 | 728,279 |
| Median lag span | 1.035418 s | 0.517701 s |
| Distinct lag supports | 11 | 6 |
| Mean candidate pillars | 31,192 | 25,320 |
| Median candidate pillars | 30,699 | 24,792 |
| Overflow frames | 68/428 | 0/428 |
| Maximum retained pillars | 40,000 | 35,157 |

## Frozen detector and evaluator

M7 inherits the complete M6b detector/evaluator boundary:

- checkpoint SHA256:
  `f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0`;
- ONNX SHA256:
  `61ce22a8ca31498675c32576bfb94f0093d31dc95d2762f7254bf915a59ecc16`;
- accepted TensorRT FP16 engine SHA256:
  `2e790b1cdbdc1b88c2aafdc81b5921ebee152edd8408158f88437ae4dd1f3e7f`;
- profile MIN/OPT/MAX: 4,352/18,207/40,000;
- deterministic `exact_fast`, `max_voxels=40000`, full evidence provenance;
- float32 `x, y, z, time_lag` point-feature order;
- voxel geometry, first-touch retention, MMDeploy postprocess, taxonomy, score threshold 0.25,
  reference-camera FOV, neighbour ignores, GT conversion, oriented-BEV matcher, IoU thresholds,
  range bins, track summaries, and visualization rules.

No model, engine, threshold, postprocess, class mapping, voxel rule, or evaluator rule may change.
No M7 outcome may trigger same-milestone tuning.

Before inference, fail-closed verification must establish the exact checkpoint, ONNX, engine,
input, and evaluator identities. An identity mismatch stops execution. The accepted engine is not
rebuilt.

## Arms and controlled design

The primary design is a deterministic 2×2 intervention on frozen H10 input:

| Point population | Native encoded lag | Compressed encoded lag |
|---|---|---|
| Native H10 points | A — `H10_NATIVE` | B — `H10_LAG_COMPRESSED` |
| H5-total-point-count matched | C — `H10_POINT_COUNT_MATCHED` | D — `H10_LAG_COMPRESSED_POINT_COUNT_MATCHED` |

Two comparators are outside that factorial:

- E — `H5_NATIVE`: frozen positive comparator from M6b;
- F — `H10_ALTERNATE_FULL_SPAN`: secondary natural sparse full-span comparator.

Exact definitions:

- **A:** accepted native H10 input and prediction, current plus history ranks 1–10; never rerun.
- **B:** A rows and XYZ unchanged; only encoded `time_lag` magnitude is compressed condition-wise.
- **C:** all current A rows plus deterministic subsets of every H10 historical rank, with total
  rows exactly equal to corresponding E; native A lag is retained.
- **D:** the exact C row identities, with the exact B lag mapping applied; never resampled.
- **E:** accepted native H5 input and prediction, current plus history ranks 1–5; never rerun.
- **F:** complete A rows from current plus history ranks 2, 4, 6, 8, and 10; no thinning inside a
  retained sweep and no lag remapping.

E is a comparator, not ground truth or a universally optimal configuration. F is secondary and is
not included in the factorial equations.

The final study contains exactly A, B, C, D, E, and F. It contains no arm G, `H5_LAG_EXPANDED`,
sentinel-only mirror condition, or result-triggered confirmation arm. A five-sentinel mirror would
not provide evidence comparable to the frozen 428-frame study, and making it conditional on B
would be an adaptive extension. If M7 later strongly implicates encoded lag magnitude, symmetric
H5 lag expansion may be considered only under a separate prospective follow-up protocol after M7.

## Canonical input bytes and provenance

The frozen model-ready dtype is float32 and shape is `(N, 4)` in `x, y, z, time_lag` order.
M7 hashes model-ready arrays as C-contiguous little-endian IEEE-754 float32 bytes. On the pinned
little-endian M6 environment this equals the accepted native-byte M6 hashes. Selected-row identity
vectors use zero-based global A row indices encoded as C-contiguous little-endian uint64 before
SHA256. Text seed material uses exact UTF-8/ASCII bytes with no terminator.

Sweep provenance comes from the frozen source IDs, source-row order, transform ledger, and exact
float32 lag bit patterns. Current rows must have exact positive-zero bits `0x00000000`; no
historical row may have zero lag. History rank 1 is the nearest previous acquisition and rank 10 is
the oldest. Membership, counts, and order must agree across these provenance routes or input
generation stops.

## Arm B: encoded lag compression

For condition `c`, let `lag_A` and `lag_E` be the frozen float32 lag columns and define:

```text
T10_f32(c) = max(abs(lag_A[row])) over non-current A rows
T5_f32(c)  = max(abs(lag_E[row])) over non-current E rows
T10(c)     = exact binary64 conversion of T10_f32(c)
T5(c)      = exact binary64 conversion of T5_f32(c)
s(c)       = binary64 T5(c) / T10(c)
```

The implementation must use round-to-nearest, ties-to-even. `T10_f32` and `T5_f32` are selected
from existing float32 values before exact conversion to binary64. Record each selected float32 bit
pattern, `T10`, `T5`, the binary64 bit pattern and hexadecimal representation of `s`, and the
implementation/library versions.

For every row, compute one binary64 multiplication and one final cast:

```text
scaled64[row] = binary64(lag_A[row]) * s(c)
lag_B[row]    = float32(scaled64[row])
```

Current rows are then explicitly written as float32 positive zero. No intermediate float32
multiplication, fused operation, per-support remapping, or rounding shortcut is allowed. Apply the
same mapping to the rows selected for D.

B must preserve A's sign convention, row count, row order, sweep membership, all 11 support
identities, and every XYZ byte. It changes only the numerical age feature. It does **not** isolate
physical temporal span or remove long-history geometry, smear, occlusion, duplicated surfaces, or
scene composition.

Fail closed on nonfinite/malformed lags, `T10 <= 0`, `T5 <= 0`, a nonpositive or nonfinite scale,
unexpected support counts, current/historical zero disagreement, or source-condition identity
mismatch. If float32 casting collapses distinct historical supports, stop rather than revise the
mapping.

## Arm C: exact total-point-count matching

C matches the corresponding frozen E **total point count** while retaining H10 support identities.
It is not generally “density matched.” It does not match spatial density, pillar occupancy,
retained pillars, temporal spacing, motion smear, occlusion, or exact H5 sweep membership.

For condition `c`:

```text
N5       = exact frozen E model-ready row count
N0       = exact A current-sweep row count
Ni       = exact A row count for history rank i, i in 1..10
H_total  = sum(Ni for i=1..10)
H_target = N5 - N0
```

Require `0 <= H_target <= H_total`, positive `H_total`, and exact agreement with frozen A/E
provenance. Keep every current row.

### Exact quota apportionment

Use unbounded nonnegative integer arithmetic rather than floating-point quota arithmetic:

```text
product_i   = H_target * Ni
q_i         = product_i // H_total
remainder_i = product_i % H_total
remaining   = H_target - sum(q_i)
```

Sort ranks by descending `remainder_i`, then ascending history rank. Add one to the first
`remaining` quotas. This is exact largest-remainder apportionment of
`H_target * Ni / H_total`. Require `sum(q_i) == H_target` and `0 <= q_i <= Ni`. Record the full
`N1..N10`, product, remainder, initial quota, increment decision, and final quota vectors. Record
every zero quota; do not conceal a missing support.

### Exact deterministic point selection

For drive ID `d`, zero-padded ten-digit frame index `f`, and decimal history rank `i`, construct:

```text
laserperception-m7-c-v1|<d>|<f>|<i>
```

Example frame formatting is `0000000010`; rank formatting is `1` through `10` without padding.
Compute SHA256 over the exact UTF-8 bytes. The seed is the first eight digest bytes interpreted as
an unsigned big-endian 64-bit integer.

For each A row in that sweep, `j` is its zero-based ordinal among the sweep's range-filtered A rows
in original A order. With every operation reduced modulo `2^64`, define:

```text
MASK = 0xffffffffffffffff
x = (seed XOR j) AND MASK
z = (x + 0x9e3779b97f4a7c15) AND MASK
z = ((z XOR (z >> 30)) * 0xbf58476d1ce4e5b9) AND MASK
z = ((z XOR (z >> 27)) * 0x94d049bb133111eb) AND MASK
key = (z XOR (z >> 31)) AND MASK
```

Select exactly `q_i` rows with the lowest `(key, j)` lexicographic pairs. This freezes lower
original index as the tie-break. After all per-sweep selections, preserve the original global A
row order; do not hash-sort the model input. The final selected-row identity SHA is computed from
the retained global A row-index vector defined above.

For every C condition require:

- exact total row count equal to E;
- all A current rows retained byte-for-byte;
- every retained historical XYZ and lag value copied directly from A;
- no changed or synthetic XYZ, duplicate row, or reordered retained row;
- all ten historical supports represented unless their preregistered quota is zero;
- native A lag values and native A physical temporal extent for retained rows.

The selection rule is owner-approved and frozen by this protocol. M7 permits no second seed, seed
sweep, or result-informed alternative.

## Arm D: combined intervention

D reuses C's exact global A row-index vector and selected-row identity SHA. It must not select or
sample again. Starting from C, it applies the exact B condition scale and cast policy to the lag
column. Therefore D and C have byte-identical XYZ, row identities, row count, and row order.

D versus C isolates encoded lag compression on the same thinned population. B versus A applies
the same lag contrast on the native population. C versus A applies point-count matching with
native lag. Any D/C voxel-coordinate, candidate-key/order, or retained-pillar difference is a
pre-inference stop because `time_lag` is not part of coordinate construction.

## Arm F: alternate complete sweeps across the full span

F retains exactly:

- current sweep;
- history ranks 2, 4, 6, 8, and 10.

Rank 1 is nearest history and rank 10 is oldest H10 history. F therefore has six temporal supports,
five real unthinned history sweeps, and the oldest H10 acquisition. It copies all A model-ready
rows belonging to each selected sweep in original A order, with no point mutation, synthetic
point, within-sweep thinning, lag remapping, or changed XYZ.

F is **a natural, unthinned, long-span comparator at matched history-sweep count**. F and E both
contain the current acquisition plus five historical acquisitions and therefore six lag supports,
but their comparison is not a pure single-factor temporal-span isolation. They may differ in exact
historical sweep identities, exact temporal spacing, physical historical XYZ geometry,
moving-object displacement or smear, occlusion, accumulated surfaces, total point count, spatial
density, and candidate or retained pillar population. Those properties are measured rather than
assumed equal.

The statements “F isolates physical temporal span” and “F and E differ only by span” are
prohibited. Measured F/E differences may motivate a physical-span hypothesis, but cannot by
themselves establish physical span as the unique cause. F remains a secondary natural-manifold
comparator, not part of the 2×2 factorial and not a replacement for C.

## Input-only freeze and structural stop gates

After a final protocol and implementation are separately approved and committed, but before any
new detector output exists, construct all 428 B/C/D/F inputs and freeze an input-only ledger. The
ledger must record for every condition:

- drive/frame/arm identity and generation commit;
- source A model-ready SHA and source E SHA where relevant;
- point count, XYZ SHA, full XYZT SHA, selected global-row SHA;
- distinct lag float32 values and bit patterns, minimum/maximum lag, and temporal span;
- sweep/support source identities and per-sweep counts;
- candidate/retained/discarded pillar counts, overflow, and coordinate/order hashes;
- B/D `T10`, `T5`, scale bits, and cast policy identity;
- C/D `N0`, `N1..N10`, `H_target`, quota arithmetic, zero quotas, seeds, and selected-row hash;
- F selected ranks exactly `[2, 4, 6, 8, 10]`;
- implementation, NumPy, Python, OS, and source-evidence identities.

Before the ledger is frozen, no new raw detector tensor, DetectionFrame, score, box, preview, or
M7 result may be generated. Existing frozen A/E evidence may be verified and loaded only as source
identity. The ledger and implementation commit must receive owner review and explicit inference
authorization.

Required pre-inference relations:

- **B/A:** equal rows, XYZ, row order, voxel coordinates, candidate keys/order, and retained
  selection; only lag differs.
- **C:** total rows equal E; current exact A; historical rows form a unique ordered subset of A;
  native A lag retained; pillar population measured rather than assumed equal to E.
- **D/C:** equal rows, XYZ, row order, voxel coordinates, candidate keys/order, and retained
  selection; only lag differs.
- **F/A:** strict subset by complete sweep identity; no mutation or synthetic point; oldest H10
  history retained.

Any relation failure stops before inference. A contract defect may motivate a new prospective
protocol cycle; it may not be repaired or tuned under this one.

## Input-only characterization

For each arm report total-point, lag-support, lag-span, candidate-pillar, retained-pillar, and
discarded-pillar mean, median, minimum, and maximum; overflow-frame count; total and maximum
discarded pillars; and the relation to A and E. Structural statistics may be reviewed before
inference because they contain no detector outcome. They cannot be used to tune a valid
intervention. A fundamental contract defect requires a stop and a new prospective cycle.

## Repeatability gate

Only after protocol freeze, implementation freeze, input-ledger freeze, structural PASS, and owner
authorization may detector execution begin. Use the same frozen sentinels:

1. `2011_09_26_drive_0001/0000000010`;
2. `2011_09_26_drive_0001/0000000083`;
3. `2011_09_26_drive_0001/0000000011`;
4. `2011_09_26_drive_0001/0000000015`;
5. `2011_09_26_drive_0091/0000000010`.

For each new arm B/C/D/F, regenerate/verify the model-ready input hash and execute ten exact
detector repetitions. Within each arm/sentinel, require identical `cls_score`, `bbox_pred`,
`dir_cls_pred`, and final DetectionFrame hashes across all ten repetitions. Repeat 1 becomes that
sentinel's canonical corpus output, avoiding an unnecessary eleventh run. This reuse is a
scientific decision, not merely an efficiency choice: a passing exact repeatability gate proves all
ten repetitions identical, so repeat 1 is scientifically exchangeable with any separate corpus
invocation of the same frozen condition. A/E are frozen and are not rerun. Any nondeterminism stops
M7 before the full corpus.

## Measurement scope, order, and checkpointing

After every prior gate passes, run exactly:

- 428 B conditions;
- 428 C conditions;
- 428 D conditions;
- 428 F conditions;
- **1,712 new conditions total**.

Use M6b frame order. Within each frame the order is B, C, D, F. Sentinel repeat 1 counts as its
canonical corpus condition. There is no extra arm, second thinning seed, parameter sweep, or
result-driven rerun.

Each completed condition uses an atomic local checkpoint containing the protocol/implementation,
engine/model/input hashes, raw-output hashes, DetectionFrame hash and payload, and evaluator
identity. Resume requires exact identity equality, refuses duplicate completed inference, and never
deletes a result. A defect after predictions begin marks the current condition failed, preserves all
partial output, and stops the protocol; repair requires a new prospective cycle.

The 1,712-condition corpus must never be shortened, subsampled, or trimmed to fit an execution
session. Session interruption is handled only by identity-checked resume from atomic checkpoints.

## Primary and secondary outcomes

The primary outcome is **Car recall** at score `>= 0.25` and oriented BEV IoU `>= 0.50`, with the
fixed 66 eligible-GT denominator. Car is primary because M6b showed the strongest separation and a
clean paired structure.

Pedestrian recall at the same operating point is secondary. For both classes report TP/FP/FN,
annotation-conditioned precision, F1, Raw-tracklet AP, TP counts at IoU 0.30/0.50/0.70, recall in
0–20/20–35/35–50 m bins, per-track detected-frame continuity, prediction population, outside-FOV
counts, and neighbour-ignore behavior. Precision/AP remain secondary because incomplete Raw
tracklets do not label the full LiDAR world.

## Car gap recovery and paired identities

Frozen A has 16/66 Car TP and E has 48/66, a gap of 32. For any arm X:

```text
G_car(X) = (TP_X - 16) / 32
```

Do not clamp this continuous value. Negative values, partial recovery, equality to E's aggregate
TP count, and overshoot must remain visible.

Before M7 inference, load the exact M6b pose identities from the verified full result asset, sort
them by `(drive_id, frame_index, GT track ID)`, and freeze both the lists and their canonical JSON
SHA256 values in machine-readable preregistration:

- `S_shared`: 16 Car poses detected by A and E;
- `S_E_only`: 32 Car poses detected only by E;
- `S_neither`: 18 Car poses detected by neither;
- A-only: empty.

For every new arm X report:

```text
R_gain(X)   = detected members of S_E_only / 32
R_shared(X) = detected members of S_shared / 16
R_novel(X)  = detected members of S_neither / 18
```

Report exact gained and lost pose identities. Aggregate equality alone is insufficient because an
arm may detect a different set of GT poses.

The phrase **“substantially explains the H10→H5 Car improvement”** is allowed only when all three
preregistered conditions pass:

1. `G_car(X) >= 0.50`;
2. `R_gain(X) >= 0.50` (at least 16 exact E-only poses recovered);
3. `R_shared(X) >= 15/16` (no more than one shared positive lost).

These are descriptive engineering interpretation rules, not equivalence tests. A failing arm is
reported by its exact continuous metrics; no threshold may be selected afterward. Terms such as
“near-pass,” “practically equivalent,” “almost substantial,” “near equivalent,” “essentially
solved,” or new cutoffs are not authorized.

This three-part rule is deliberately conservative: an intervention must recover substantial
H5-only benefit without materially discarding Car positives on which native H10 and H5 already
agree. Because M7 is deterministic after the repeatability gate, loss of shared positives is part
of the intervention result and is not dismissed post hoc as threshold noise. An arm that narrowly
misses only `R_shared >= 15/16` is reported transparently with all exact continuous metrics, but
does not pass this interpretation gate.

## Pedestrian characterization

Frozen Pedestrian counts are A TP 219, E TP 268, for a gap of 49:

```text
G_ped(X) = (TP_X - 219) / 49
```

The paired sets are shared 204, E-only 64, A-only 15, and neither 113. Freeze their exact pose
identity lists/hashes from the same verified full M6b asset before inference. Report E-only
recovery, A-only retention, shared retention, and neither recovery. Pedestrian is not required to
pass the Car interpretation gate; class dependence is reported rather than used to change the
primary conclusion.

## Descriptive factorial contrasts

For any scalar outcome Y:

```text
L = ((Y_B - Y_A) + (Y_D - Y_C)) / 2
P = ((Y_C - Y_A) + (Y_D - Y_B)) / 2
I = Y_D - Y_B - Y_C + Y_A
```

L is the encoded-lag contrast, P is the total-point-count contrast, and I is the interaction on
that outcome scale. Report exact numerical values. Additive, synergistic, or antagonistic language
must be cautious and tied to the measured interaction; none is a population causal effect. F is
excluded from these equations.

## Residual history effects and outcome handling

If B/C/D/F do not reproduce most of E's benefit, residual candidates include physical historical
geometry, moving-object smear/displacement, occlusion, duplicated surfaces, exact sweep identity,
spatial-density distribution, pillar occupancy, interactions not captured by total count, and
other accumulated-history effects. The protocol need not explain the full difference and may not
add arms after seeing results.

If an arm fails scientifically, preserve and report it; do not tune, train, or add an arm. If an
arm succeeds, report it under the frozen rule; do not make it the default or begin production
changes. M7 contains no training, latency benchmark, ROS throughput campaign, GPU speed claim,
Jetson work, or deployment optimization.

## Planned evidence and repository discipline

The prospective final cycle may create:

- `docs/m7/M7_PROTOCOL.md` — owner-approved frozen protocol;
- `docs/m7/M7_RESULTS.md` — final result and limitations;
- `benchmarks/m7/preregistration/m7_protocol.json` — machine-readable protocol and paired-set hashes;
- `benchmarks/m7/inputs/m7_input_ledger.json` — compact input-only freeze;
- `benchmarks/m7/diagnostics/m7_input_characterization.json` — input-only summaries;
- `benchmarks/m7/diagnostics/m7_repeatability.json` — sentinel gate;
- `benchmarks/m7/results/m7_controlled_history_mechanism.json` — compact canonical result.

Full per-frame ledgers/checkpoints may remain external and hash-pinned. No generated artifact over
5 MiB may enter Git without explicit owner approval. This protocol-freeze task creates none of
these artifacts.

## Frozen protocol decisions

- B isolates encoded lag magnitude, not physical span or accumulated geometry.
- C matches exact E total point count, not general density or pillars.
- C quotas use exact integer largest-remainder allocation with lower-rank tie-break.
- C selection uses one specified SHA256/SplitMix64 rule and preserves A row order.
- D reuses C rows so D/C changes only lag.
- E is loaded from accepted M6b evidence and is not rerun.
- F keeps current plus ranks 2/4/6/8/10 and is a natural, unthinned, long-span comparator at
  matched history-sweep count; F/E does not isolate span.
- Car recall at score 0.25/IoU 0.50 is primary; precision/AP are annotation-conditioned secondary.
- `G_car`, exact paired sets, and the three-part interpretation rule are frozen before inference.
- Pedestrian remains secondary with its own continuous and paired characterization.
- Factorial contrasts exclude F and are descriptive.
- Residual history effects remain explicit; success or failure authorizes no tuning or training.
- The input-only ledger and owner review block outcome observation.
- Structural identity, repeatability, checkpoint, and fail-closed rules precede full inference.
- No symmetric H5 lag-expanded mirror arm or adaptive confirmation arm exists in M7.
- Protocol freeze authorizes neither implementation input construction nor detector inference.
