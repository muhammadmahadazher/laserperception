# M8 P1-S1 scientific interpretation

Status: **INTERPRETATION COMPLETE**. This document interprets the already-published S1 primary
measurement. It adds no detector calls and does not modify the frozen protocol or raw evidence.

## Evidence basis

The accepted S1 campaign ran at execution commit
`6994d72c3e7691a86116d1417ac3ae08256d163f`. Three fresh processes each completed the frozen 856
A2/E2 conditions, for 2,568 accepted canonical calls. The authoritative inputs are the
[raw measurement narrative](M8_S1_MEASUREMENT_RAW.md), the compact
[primary](../../benchmarks/m8/results/m8_s1_primary_raw.json) and
[secondary](../../benchmarks/m8/results/m8_s1_secondary_raw.json) results, and the
[measurement manifest](../../benchmarks/m8/results/m8_s1_measurement_manifest.json). Their SHA256
identities are recorded in the compact
[interpretation record](../../benchmarks/m8/results/m8_s1_interpretation.json).

The earlier 779-condition incomplete attempt and the later interrupted 26-condition attempt each
contribute zero accepted canonical calls. Neither was combined with the three accepted processes.
All observations below use score at least 0.25 and primary oriented BEV IoU at least 0.50 unless
stated otherwise.

## Reproducibility across three runtime realizations

For every accepted process, the threshold-level TP, FP, FN, recall, annotation-conditioned
precision, and F1 values are identical. The within-process Car recall contrast is also identical:
`recall(E2) - recall(A2) = 0.3636363636363636`. This establishes high discrete evaluator-level
repeatability across the three observed runtime realizations on the same frozen corpus.

Score-ranked AP varies slightly. Car AP spans 0.11986748193919464–0.12001333009861485 for A2 and
0.2038383187081106–0.2038496433907682 for E2. Pedestrian AP spans
0.00029648142894901994–0.000296658658157745 for A2 and
0.004505068942876486–0.004506390197284403 for E2. The repeated processes are runtime
realizations of one fixed corpus, not independent dataset replicates. The result therefore does
not establish mathematical determinism, dataset-sampling uncertainty, population confidence
intervals, or universal hardware reproducibility. No p-values or confidence intervals are
reported.

## Car H10-to-H5 result

| Arm | History label | TP | FP | FN | Recall | Annotation-conditioned precision | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| A2 | H10 | 19 | 89 | 47 | 19/66 = 0.287879 | 0.175926 | 0.218391 |
| E2 | H5 | 43 | 188 | 23 | 43/66 = 0.651515 | 0.186147 | 0.289562 |

The prospectively frozen direction criterion is satisfied in every accepted process:
`TP_E2 = 43 > TP_A2 = 19`. The positive H10-to-H5 Car phenomenon therefore replicated
descriptively under S1, with a recall increase of 24/66, or 0.363636. H10 versus H5 is a compound
temporal/density intervention. This observation does not show that H5 is universally better,
isolate history length as the cause, or establish architecture causality.

## Range-localized observations

| Car range | A2/H10 TP/GT | E2/H5 TP/GT | TP change |
|---|---:|---:|---:|
| 0–20 m | 5/27 | 22/27 | +17 |
| 20–35 m | 13/30 | 20/30 | +7 |
| 35–50 m | 1/9 | 1/9 | 0 |

The observed 24-TP A2-to-E2 Car increase is concentrated in the bins at or below 35 m. The frozen
35–50 m count is unchanged. This is a localization of the observation, not a mechanism claim.

## Pedestrian cross-domain failure

Pedestrian transfer effectively fails under the frozen DSVT stack and protocol. A2/H10 records
0 TP, 1 FP, and 396 FN: recall 0/396 = 0. E2/H5 records 1 TP, 8 FP, and 395 FN: recall
1/396 = 0.00252525. This severe class-specific cross-domain failure is a central S1 result.

The evidence does not establish a unique cause. Plausible unresolved and confounded factors include
training and checkpoint differences, feature-contract differences, raw KITTI reflectance versus
the nuScenes-trained intensity channel, architecture and head differences, postprocessing and
score-calibration differences, spatial-discretization differences, and other stack-level
domain-shift interactions. These are hypotheses, not findings about causality.

## Comparison with historical PointPillars

This is a **frozen detector-stack comparison**, not an architecture benchmark. The two stacks
differ in architecture, training recipe, checkpoint, framework, feature contract, intensity
availability, spatial discretization, postprocessing, and deployment state.

| Class | Condition | DSVT recall | Historical PointPillars recall | DSVT minus PointPillars |
|---|---|---:|---:|---:|
| Car | H10 | 19/66 = 0.287879 | 16/66 = 0.242424 | +3/66 = +0.045455 |
| Car | H5 | 43/66 = 0.651515 | 48/66 = 0.727273 | -5/66 = -0.075758 |
| Pedestrian | H10 | 0/396 = 0 | 219/396 = 0.553030 | -219/396 = -0.553030 |
| Pedestrian | H5 | 1/396 = 0.002525 | 268/396 = 0.676768 | -267/396 = -0.674242 |

The comparison is mixed for Car: DSVT is descriptively higher at H10 and lower at H5. Historical
PointPillars is dramatically higher for Pedestrian in both conditions. These observations do not
select a universal model winner.

## Annotation and AP limitations

Precision and AP are annotation-conditioned inside the reference-camera field of view against
incomplete KITTI Raw tracklets. They are neither whole-world physical false-positive precision nor
official KITTI benchmark AP. Outside-FOV predictions are reported separately and are not ordinary
evaluated false positives. AP uses the preregistered score-ranked calculation and its small
between-process numerical spread must be retained alongside the identical threshold counts.

## What S1 establishes

- The frozen A2/E2 campaign completed three accepted processes and 2,568 canonical calls.
- Threshold-level results repeat exactly across those three observed runtime realizations.
- The preregistered positive Car direction criterion holds: E2/H5 has 24 more TP than A2/H10.
- The Car increase occurs in the two bins at or below 35 m; the 35–50 m TP count is unchanged.
- Pedestrian transfer under this frozen DSVT stack is severely unsuccessful.
- The historical PointPillars comparison is mixed by class and condition.

## What S1 does not establish

S1 does not establish mathematical determinism, sampling uncertainty, statistical significance,
population confidence intervals, cross-hardware reproducibility, architecture causality, intensity
causality, a universal model winner, a production history setting, production readiness, or an
explanation for the class-specific domain shift. It also does not make normalized S2 recovery a
valid reported quantity.

## Zero-intensity rationale

The frozen protocol preregistered `A2_zeroI` and `E2_zeroI`, using exact positive float32 zero
(`+0.0`) for intensity in three fresh processes. The severe Pedestrian failure, together with the
known difference between raw KITTI reflectance and the nuScenes-trained intensity feature,
provides a scientific reason to test intensity sensitivity. It is not evidence that intensity
caused the failure.

**ZERO-INTENSITY INTERVENTION RECOMMENDED FOR OWNER AUTHORIZATION**

This recommendation is scientific only. It neither authorizes nor executes the intervention.

## Requirements before S2

S1 supplies a positive Car denominator candidate, `TP_E2 - TP_A2 = 24`, and completes the first
measurement barrier. S2 is not ready now. Before any B2/C2/D2/F2 result exists, a prospective S2
protocol must:

1. define a numeric minimum-gap/denominator-stability criterion and the eligibility and formula for
   normalized recovery, without reusing the historical PointPillars gap of 32;
2. choose and freeze how V2 shared, E2-only, A2-only, and neither partitions are formed across the
   three numerically non-identical realizations, rather than automatically selecting pass 1,
   median predictions, a union, or an intersection;
3. prove each future V2 condition's `[x, y, z, time]` projection matches the frozen M7 XYZT hash;
4. freeze the partitions, S2 protocol, validation, identities, runtime binding, and owner
   authorization before execution.

Until those decisions are preregistered, raw deltas and paired counts are the valid descriptive
quantities; normalized recovery must not be headlined.

## Next experimental decision

The next scientifically motivated decision is whether the owner authorizes the already-frozen
zero-intensity intervention. S2 remains blocked on the prospective definitions above. No next
experiment was run as part of this interpretation.
