# M7 controlled history-mechanism study — frozen interpretation

**M7 SCIENTIFIC INTERPRETATION — OWNER APPROVED AND FROZEN.**

This document interprets the accepted preregistered M7 R2 measurement. It is separate from the
[raw evidence record](M7_MEASUREMENT_RAW.md), which remains unchanged. No detector rerun, input
regeneration, threshold change, or production-behavior change occurred during interpretation.

## Result in brief

Among the frozen M7 interventions, encoded lag magnitude is the dominant measured contributor to
the observed H10-to-H5 Car degradation. Changing only encoded lag from A to B recovered 23 of the
32 Cars detected by E but missed by A, retained all 16 Cars detected by both baselines, and passed
all three preregistered substantial-explanation conditions. Exact total-point-count matching alone
(C) failed those conditions. The combined lag-compressed and point-count-matched arm D also passed
and produced the highest primary Car recall, slightly above E at IoU 0.50.

The lag-only Car gain is also range-dependent in this corpus: B improves the 0–20 m and 20–35 m
slices but remains 0/6 at 35–50 m, equal to A in that six-instance far-range slice.

That conclusion is class- and criterion-dependent. E retained more Car matches than D at IoU 0.70,
had slightly higher Car AP, and had substantially higher Pedestrian recall than every M7
intervention. Lag compression reduced Pedestrian TP under both point-population settings. M7
therefore identifies a measured mechanism; it does not validate a universal deployment policy.

## Frozen design

| Arm | Frozen meaning |
| --- | --- |
| A — `H10_NATIVE` | Frozen M6b H10 baseline. |
| B — `H10_LAG_COMPRESSED` | Exact A points, rows, XYZ, sweep membership, and order; only encoded `time_lag` magnitude changes condition-wise. |
| C — `H10_POINT_COUNT_MATCHED` | All current A points plus a deterministic historical subset; exact total point count equals E; native A lag remains. |
| D — `H10_LAG_COMPRESSED_POINT_COUNT_MATCHED` | Exact C rows with the B lag mapping; D/C changes only encoded lag. |
| E — `H5_NATIVE` | Frozen M6b H5 comparator. |
| F — `H10_ALTERNATE_FULL_SPAN` | Current plus complete history ranks 2/4/6/8/10: a natural, unthinned, long-span comparator at matched history-sweep count. |

F is not a pure span intervention. F/E and F/B each differ in more than one physical or encoded
property and do not isolate temporal span.

## Primary Car result

The primary operating point is score `>= 0.25`, oriented BEV IoU `>= 0.50`, with 66 eligible Car
GT. The preregistered rule requires all of `G_car >= 0.50`, `R_gain >= 0.50`, and
`R_shared >= 15/16`.

| Arm | TP | Recall | G_car | R_gain | R_shared | R_novel | Gate conclusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A | 16 | 0.2424242424 | — | — | — | — | Baseline |
| B | 41 | 0.6212121212 | 0.78125 | 23/32 (0.71875) | 16/16 (1.0) | 2/18 (0.111111) | **PASS — all three** |
| C | 17 | 0.2575757576 | 0.03125 | 2/32 (0.0625) | 14/16 (0.875) | 1/18 (0.055556) | **FAIL** |
| D | 49 | 0.7424242424 | 1.03125 | 29/32 (0.90625) | 16/16 (1.0) | 4/18 (0.222222) | **PASS — all three** |
| E | 48 | 0.7272727273 | — | — | — | — | Comparator |
| F | 21 | 0.3181818182 | 0.15625 | 7/32 (0.21875) | 13/16 (0.8125) | 1/18 (0.055556) | **FAIL** |

Under the frozen rule, B and D **substantially explain the H10→H5 Car improvement**. C and F do
not. B's passing result shows that lag compression alone explains a large portion of the observed
gap while preserving the shared positives. Point-count matching alone does not.

`R_novel` also prevents an aggregate-only reading. B detects 2/18 and D 4/18 Car poses detected by
neither A nor E. D's 49 TP therefore are a partially different set, not simply E's 48 plus one;
D is not H5-equivalent.

## Descriptive Car factorial

| Contrast | Meaning | Value |
| --- | --- | ---: |
| L | Encoded-lag contrast | 0.4318181818 |
| P | Exact-total-point-count contrast | 0.0681818182 |
| I | Interaction | 0.1060606061 |

The encoded-lag contrast is much larger than the point-count contrast on the measured Car-recall
scale. These are descriptive mechanistic contrasts, not percentages of causal responsibility or a
population causal decomposition.

## IoU-threshold qualification

The full threshold progression is retained because the D/E ordering changes under the stricter
criterion. Cells are TP counts against the fixed class denominator.

| Car arm | IoU 0.30 | IoU 0.50 | IoU 0.70 |
| --- | ---: | ---: | ---: |
| A | 16 | 16 | 16 |
| B | 41 | 41 | 38 |
| C | 17 | 17 | 17 |
| D | 49 | 49 | 40 |
| E | 48 | 48 | 44 |
| F | 21 | 21 | 20 |

| Pedestrian arm | IoU 0.30 | IoU 0.50 | IoU 0.70 |
| --- | ---: | ---: | ---: |
| A | 280 | 219 | 54 |
| B | 276 | 199 | 42 |
| C | 276 | 224 | 59 |
| D | 279 | 212 | 48 |
| E | 316 | 268 | 76 |
| F | 288 | 236 | 46 |

D exceeds E at the primary Car operating point, 49 versus 48 TP, but E preserves more Car matches
at IoU 0.70, 44 versus 40. E also has slightly higher Car AP (`0.2423801048`) than D
(`0.2415680459`). D therefore slightly exceeds H5 only at the frozen primary Car recall operating
point; it does not dominate H5 across stricter localization or ranked AP. This comparison does not
by itself diagnose the localization-error mechanism.

## Car range qualification

These are primary-IoU matches, shown as `TP/eligible GT`.

| Arm | 0–20 m | 20–35 m | 35–50 m |
| --- | ---: | ---: | ---: |
| A | 12/27 | 4/30 | 0/6 |
| B | 27/27 | 14/30 | 0/6 |
| C | 10/27 | 6/30 | 1/6 |
| D | 27/27 | 19/30 | 3/6 |
| E | 27/27 | 17/30 | 4/6 |
| F | 11/27 | 10/30 | 0/6 |

B's observed gain over A is confined to the 0–20 m and 20–35 m bands in this corpus. Lag
compression alone recovered 0/6 eligible 35–50 m Cars, equal to A in that band. C, D, and E
recovered 1/6, 3/6, and 4/6 respectively. This six-instance far-range result is descriptive and
corpus-bound. It suggests that reduced point-population configurations recovered some far-range
instances that lag compression alone did not, but does not establish a unique density mechanism.

## Pedestrian class dependence

| Arm | TP/396 | Recall |
| --- | ---: | ---: |
| A | 219 | 0.5530303030 |
| B | 199 | 0.5025252525 |
| C | 224 | 0.5656565657 |
| D | 212 | 0.5353535354 |
| E | 268 | 0.6767676768 |
| F | 236 | 0.5959595960 |

Lag compression reduced Pedestrian TP by 20 from A to B and by 12 from C to D. This direction is
opposite the Car lag effect under both point-population settings. E remains materially stronger
for Pedestrian recall than every M7 intervention.

The descriptive Pedestrian-recall contrasts are `L = -0.04040404040404039`,
`P = +0.022727272727272763`, and `I = +0.02020202020202022`. No Pedestrian pass/fail gate exists,
and none is introduced here.

## Secondary F comparator

F produced 21 Car TP (`0.3181818182` recall) and 236 Pedestrian TP (`0.5959595960` recall). It is a
natural long-span, six-support comparator and recovers relatively little of the Car H10→H5 gap.
F/E does not isolate physical span, F/B is not a single-factor comparison, and F does not establish
that physical span is irrelevant or causal.

## Precision and AP limitation

KITTI Raw tracklets do not label the complete LiDAR world. Precision and AP are therefore
annotation-conditioned by incomplete tracklets. They are not official KITTI benchmark AP,
physical false-positive rates, or complete-world precision. Recall against eligible labelled GT is
the primary M7 outcome.

## What M7 establishes

1. The frozen H10/H5 difference was sufficiently mechanistically decomposable to test
   prospectively without retraining.
2. For Cars, changing only encoded lag magnitude from A to B recovers a large portion of the
   frozen H10→H5 recall gap and passes the preregistered substantial-explanation rule.
3. Exact point-count matching alone does not substantially explain the Car improvement.
4. Combining lag compression and point-count matching produces the highest primary Car recall in
   the study, slightly above H5.
5. D is not globally superior to E: E is stronger at IoU 0.70, slightly stronger in AP, and
   substantially stronger for Pedestrians.
6. The lag intervention is a measured class-dependent mechanism, not a universal deployment fix.

## What M7 does not establish

M7 does not establish that encoded lag is the sole cause, physical history span has no role, point
population has no role, or F isolates span. The factorial is not a population causal decomposition.
The intervention is not shown to generalize beyond this detector, dataset, and configuration. M7
evaluates a frozen pretrained detector and does not establish whether a model trained with broader
or different encoded-lag exposure would show the same sensitivity. C tests exact total point count
rather than natural H5 density, and F does not isolate sweep count; M7 therefore does not establish
that natural point-density effects or sweep-count effects are irrelevant. M7 does not authorize
lag compression as a product default, create a Pedestrian acceptance gate, constitute an official
KITTI benchmark, show that D universally outperforms H5, or validate a production policy.

## Future-work boundary

The evidence may motivate a separately preregistered study of deployment-time encoded-lag policy
or normalization across history depth and class behavior. A blanket policy is currently blocked
by two specific findings:

1. the Pedestrian reversal under both point-population settings (A→B and C→D); and
2. the absence of far-range Car recovery under lag compression alone (B remains 0/6, equal to A).

This frozen interpretation does not create an M8 protocol, freeze a follow-up design, implement
lag normalization, or change inference behavior.

## Evidence identity

- Frozen protocol commit: `fd4a143621ffc0692206c100279a9edfd5572d35`
- Raw measurement merge commit: `72f78f740b084776f6fe52994581760ade8844dd`
- Raw arm table SHA256: `2539286bc4ddf05e0526e0301aeb93e295afa1d549140d2ef341edc6cb725f44`
- Car factorial SHA256: `f6e6f7a25759948be894d3419055064775ae168081acfc2c9ae77422052bbb06`
- Secondary characterization SHA256: `1dcf152c000af820f008e5ccdc73549cebdb2297990ba8a960032a64c5c905c6`
- Measurement manifest SHA256: `21b1b93807c6e41a607f1c94bce182ca32fa995259298dc6e4c7392ee185598e`

**M7 SCIENTIFIC INTERPRETATION — OWNER APPROVED AND FROZEN.**
