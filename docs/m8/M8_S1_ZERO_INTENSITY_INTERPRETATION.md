# M8 P1-S1 zero-intensity scientific interpretation

Status: **INTERPRETATION COMPLETE**. This is a CPU-only interpretation of the published compact
results. It adds no detector calls and does not change the frozen protocol or historical primary
interpretation. All threshold results below use score ≥0.25 and oriented BEV IoU ≥0.50 on the
fixed, annotation-conditioned KITTI Raw corpus.

## Scope and evidence identity

The primary raw-intensity campaign completed three accepted processes and 2,568 calls at execution
commit `6994d72c3e7691a86116d1417ac3ae08256d163f`. The separate zero-intensity campaign
completed three accepted processes and 2,568 calls at execution commit
`95fb66ac1f57c41f06f05bd9ef5dac27b1e3ea54`. The latter followed a fresh 10-process,
140-accepted-call Stage R on GPU `GPU-e43b9709-1c26-6b50-f52b-b32bb621048b`; all 2,568
condition input mappings were provenance-verified. The zero-intensity intervention set the
candidate-consumed intensity to positive float32 `+0.0` (`0x00000000`).

The authoritative inputs are the frozen [protocol](../../benchmarks/m8/preregistration/m8_s1_protocol.json),
primary [raw](../../benchmarks/m8/results/m8_s1_primary_raw.json),
[secondary](../../benchmarks/m8/results/m8_s1_secondary_raw.json), and
[manifest](../../benchmarks/m8/results/m8_s1_measurement_manifest.json) records, and the zero-intensity
[raw](../../benchmarks/m8/results/m8_s1_zero_intensity_raw.json),
[secondary](../../benchmarks/m8/results/m8_s1_zero_intensity_secondary.json),
[comparison](../../benchmarks/m8/results/m8_s1_zero_intensity_comparison.json),
[manifest](../../benchmarks/m8/results/m8_s1_zero_intensity_manifest.json), and
[raw narrative](M8_S1_ZERO_INTENSITY_RAW.md). Their tracked-blob SHA256 identities are bound in the
[compact interpretation](../../benchmarks/m8/results/m8_s1_zero_intensity_interpretation.json).
The incomplete 779-condition primary attempt and interrupted 26-condition attempt contribute no
accepted canonical calls to this interpretation.

Within each primary process, H10 and H5 share a runtime realization; the same is true within each
zero-intensity process. **Primary and zero-intensity process indices are not paired.** Differences
between campaigns compare separately repeated three-process distributions, not pass-1 versus
pass-1 (or corresponding pass-2/pass-3) effects.

## Pedestrian result

| Campaign | H10 TP/FP/FN | H10 recall | H5 TP/FP/FN | H5 recall |
|---|---:|---:|---:|---:|
| Primary | 0/1/396 | 0/396 | 1/8/395 | 1/396 |
| Zero intensity | 0/2/396 | 0/396 | 0/8/396 | 0/396 |

Removing the candidate-consumed raw intensity **did not recover Pedestrian detection** under this
frozen model, corpus, and evaluator. The severe cross-domain failure persists: H10 remains at zero
TP and H5 changes from one TP to zero. Zero-intensity Pedestrian range recall is zero in every
recorded range band. These observations do not support raw KITTI reflectance mismatch as a
*sufficient* explanation for the Pedestrian failure. They do not establish that intensity is
irrelevant or identify what caused the failure.

## Car result and H5-versus-H10 direction

| Campaign | H10 TP/FP/FN | H10 recall | H5 TP/FP/FN | H5 recall | Within-process H5 − H10 recall |
|---|---:|---:|---:|---:|---:|
| Primary | 19/89/47 | 19/66 = 0.287879 | 43/188/23 | 43/66 = 0.651515 | +24/66 = +0.363636 |
| Zero intensity | 17/81/49 | 17/66 = 0.257576 | 43/169/23 | 43/66 = 0.651515 | +26/66 = +0.393939 |

The positive H5-over-H10 Car recall direction survives zeroing intensity. H5 recall remains 43/66,
while H10 falls by two TPs, from 19/66 to 17/66. These are descriptive differences of campaign
medians: zero-intensity minus primary is −2/66 for H10 and zero for H5. The persistent positive
direction does not support an explanation depending *solely* on raw intensity values. H10-versus-H5
is a compound temporal/density/history intervention; this measurement does not isolate time lag,
density, cap pressure, or any other component as its mechanism.

## Range localization

| Car range | Primary H10 | ZeroI H10 | Primary H5 | ZeroI H5 |
|---|---:|---:|---:|---:|
| 0–20 m | 5/27 | 3/27 | 22/27 | 22/27 |
| 20–35 m | 13/30 | 13/30 | 20/30 | 20/30 |
| 35–50 m | 1/9 | 1/9 | 1/9 | 1/9 |

At the frozen score/IoU operating point, the two-TP H10 Car reduction is entirely in 0–20 m.
The other H10 range counts and all H5 range counts are unchanged. This is limited observed
operating-point sensitivity on the fixed corpus, not a significance or generalization claim.

## AP and prediction-population context

The following are the exact three-process annotation-conditioned AP values at IoU 0.50, in each
campaign's own process order. They are **not official KITTI AP** and are not paired across campaigns.

| Arm/class | Primary AP | Zero-intensity AP |
|---|---|---|
| H10 Car | 0.11987159627950453, 0.12001333009861485, 0.11986748193919464 | 0.11326007468832544, 0.11326137678004208, 0.11329556928526817 |
| H5 Car | 0.2038496433907682, 0.203840909119444, 0.2038383187081106 | 0.19984950622947859, 0.19985855707872371, 0.19976173874250422 |
| H10 Pedestrian | 0.00029648142894901994, 0.000296658658157745, 0.00029665032516698343 | 0.00015990202751554747, 0.00015991543762698692, 0.0001598870205625053 |
| H5 Pedestrian | 0.004505068942876486, 0.004506390197284403, 0.004505173438942965 | 0.0017920660721208106, 0.001817662852254153, 0.0017922196970378487 |

Pedestrian AP is lower under zero intensity in both histories. The score-ranked,
annotation-conditioned behavior also did not show a recovery. At score ≥0.25, inside-FOV
Pedestrian predictions are 1 versus 2 for primary versus zero-intensity H10 and 9 versus 8 for
H5; the corresponding TP counts are 0 versus 0 and 1 versus 0. AP can be positive even when
recall at this fixed score threshold is zero, because AP uses the ranked predictions. These
population and AP changes remain descriptive, without a statistical-worsening claim.

## What the intervention rules out — and what it does not

The zero-intensity intervention does not rescue severe Pedestrian transfer failure. It leaves the
strong positive H5-over-H10 Car recall direction intact, with H5 recall unchanged and a two-TP
H10 reduction localized to 0–20 m at the operating point. Raw KITTI intensity mismatch is thus
not supported as a sufficient explanation for either the Pedestrian collapse or the observed
positive Car direction. This does not establish that intensity cannot matter, identify the cause
of Pedestrian failure, or isolate a mechanism behind H5/H10. It does not support an architecture
winner or a result beyond this corpus, model, and evaluator.

## Reproducibility

Within each campaign, threshold-level TP/FP/FN outcomes were identical across all three accepted
processes; AP has small numerical spread. This is high discrete evaluator-level repeatability
across runtime realizations on the same fixed corpus. It is not independent dataset replication
or proof of deterministic floating-point inference. No significance test is reported.

## Implications for next science

Zero-intensity and S1 interpretation are complete; prospective S2 **design may begin**. S2 is
**not ready or authorized to execute**. Before any B2/C2/D2/F2 calls, the owner must establish
the numeric minimum-gap and denominator-stability rules, normalized-recovery eligibility and
formula, multi-realization V2 partition-selection rule, frozen V2 partitions, XYZT identity proof,
S2 protocol, validation, runtime binding, and authorization. B2/C2/D2/F2 calls remain zero.

## Claim boundaries

Primary-versus-zero-intensity processes are unpaired. No intensity causality, architecture
causality, H5/H10 component causality, statistical significance, universal model winner,
production readiness, or autonomous-driving safety is claimed.
