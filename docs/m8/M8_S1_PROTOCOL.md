# M8 P1-S1 frozen detector-stack comparison protocol

Status: **M8 P1-S1 SCIENTIFIC PROTOCOL — OWNER APPROVED AND FROZEN BEFORE ANY
GT-RELATIVE V2 MEASUREMENT.**

**NO STAGE R DETECTOR OUTPUT EXISTS. NO A2/E2 GT-RELATIVE V2 RESULT EXISTS. NO ZERO-INTENSITY
RESULT EXISTS. NO B2/C2/D2/F2 RESULT EXISTS.**

The source reviewed draft commit is `def09021dd033b8dfbc0a413c58cba84fcc7a863`.
Owner approval occurred before any scientific V2 detector measurement. This freeze does not
authorize inference: the implementation, resume behavior, artifact binding, owner implementation
approval, and explicit committed inference authorization remain mandatory before Stage R.

The machine-readable companion is
[`m8_s1_protocol.json`](../../benchmarks/m8/preregistration/m8_s1_protocol.json).

## Scientific question and interpretation boundary

The primary question is:

> How does the frozen pretrained DSVT detector stack behave on the same frozen KITTI Raw H10/H5
> cross-domain corpus and evaluation protocol previously used for the historical PointPillars
> stack?

This is a frozen detector-stack comparison under a common cross-domain physical corpus and
evaluation protocol. It is **not** a pure architecture-causal ablation. The stacks differ in
architecture, training recipe, checkpoint, framework, feature contract, postprocessing, intensity
availability, spatial discretization, point-order policy, supported range, and deployment
completeness. No Phase-1 pass/fail capability threshold is defined.

## Bound identities

### Accepted M8 P1-E engineering package

| Item | Frozen identity |
|---|---|
| P1-E canonical engineering commit | `77369c02e3486650cd06624cb796cf1efbc6e3d4` |
| P1-E normal merge commit | `8fcf71f527104e439a59bb8cc2376ec332fa5841` |
| Candidate | DSVT-Pillar with TransFusion head |
| DSVT upstream commit | `8cfc2a6f23eed0b10aabcdc4768c60b184357061` |
| Official upstream config SHA256 | `b0832e03ad11d4e0b61f0fb07d977e687763caae472a4f87ed750bdc2d13be0f` |
| Checkpoint | `DSVT_Nuscenes_val.pth`, 28,665,215 bytes |
| Checkpoint SHA256 | `a675149d095eef8ddc0c137ae46eeac075ccc504c7608162c71e7adf318793fb` |
| Candidate manifest | `configs/m8/dsvt_nuscenes_pillar.json`, 4,915 bytes |
| Candidate-manifest SHA256 | `aa456e0386e46e9d089a957b1f1a8a4f74ceae70435c7ad8e6ca5e67bb90f4e7` |
| Accepted M8 input ledger | `benchmarks/m8/diagnostics/m8_input_projection_ledger.json`, 669,345 bytes |
| Accepted input-ledger SHA256 | `474e87e34c64d669750d4b6f7a64ac46fc9c5c462693fad79ff7c9547a7f1f7c` |
| Final-head input replay | `benchmarks/m8/diagnostics/m8_input_projection_revalidation.json`, 966 bytes |
| Final-head replay SHA256 | `71ac9418c29da5efd64f9eaeb03e859f85d6b1c56dc2fe47cef6563a9f960341` |

Before any future inference, all 428 H10 and 428 H5 input identities must reproduce the accepted
ledger exactly. The final-head replay is evidence that the P1-E implementation did so at its
canonical engineering commit; it is not permission to skip the future pre-inference binding gate.

The selected runtime recorded by the candidate manifest is Python 3.10, PyTorch 2.1.0+cu118,
CUDA 11.8, spconv 2.3.8, torch-scatter 2.1.2+pt21cu118, NumPy 1.23.5, TensorRT 8.6.1, and `cuda:0`.
The future measurement manifest must record the actually verified versions and device rather than
assuming they match this draft.

### Accepted historical PointPillars baseline

PointPillars must not be rerun. The historical baseline is reused from the accepted M6b result:

| Item | Frozen identity |
|---|---|
| M6b recorded measurement/preregistration identity | `9159682fadfc069eeb70e07acb76dd0a929db98f` |
| Compact result | `benchmarks/m6b/results/kitti_raw_cross_domain_characterization.json`, 111,529 bytes |
| Compact-result SHA256 | `b9d47120aabb38733d79f987f16277ebab626d2b7c13159c40c30a68b76c1d26` |
| Compact input ledger SHA256 | `2413224808b0140856a0e00f884c53bc9b49ec6b545172f5e7fa57d00802dc15` |
| External full result | `kitti_raw_cross_domain_characterization_full.json`, 41,987,113 bytes |
| External full-result SHA256 | `87870b2aa0cc2a91d39331afc8154fdad0c8c796f1cabfb4f8530a3eb106de27` |
| Ordered 428-frame identity SHA256 | `76bd5f7adac3d892ad2fb00cb9cf5f4f73dd475682ee011b7ea9524060c46c95` |

The accepted primary operating-point baseline is:

| Condition | Class | TP | FP | FN | Recall |
|---|---|---:|---:|---:|---:|
| A — PointPillars H10 | Car | 16 | 144 | 50 | `16/66` |
| E — PointPillars H5 | Car | 48 | 261 | 18 | `48/66` |
| A — PointPillars H10 | Pedestrian | 219 | 3,831 | 177 | `219/396` |
| E — PointPillars H5 | Pedestrian | 268 | 3,868 | 128 | `268/396` |

The evaluator is the frozen M6b benchmark-inspired Raw-tracklet evaluator, not the official KITTI
benchmark. Current source identities to be reverified at implementation review are:

| Evaluator component | SHA256 |
|---|---|
| `configs/m6/kitti_m6b.yaml` | `c6d839f2b0d62bc57c958a99b25dbb4a897e3063f080d0a3d6ea24bdde307620` |
| `benchmarks/m6b/run_characterization.py` | `824b431bd8231db9afd7f7a2b943dd5b2877276113cd33a145997cea3b09cb91` |
| `src/laserperception/evaluation/kitti_m6b.py` | `07e731462a0a805474da7449f95e82a9e8c105204c0e87459cc1ff946065abb8` |
| `src/laserperception/evaluation/m6b_metrics.py` | `4ac80e80647aaf882798d91c20531c18c875b80253cc8d56f1c49ed67efa358b` |
| `src/laserperception/evaluation/m6b_pillars.py` | `bbde774ce44c5bf0d0e3bc45d86c25d0b2a4632cf0f113d61fae636e785e8577` |

## Corpus and V2 conditions

The corpus is the exact ordered 428-frame M6b set: indices 10–107 from
`2011_09_26_drive_0001` and 10–339 from `2011_09_26_drive_0091`. It may not be trimmed or
subsampled to fit a session.

- **A2:** frozen DSVT on H10, current acquisition plus ten historical acquisitions.
- **E2:** frozen DSVT on H5, the same current frame plus five historical acquisitions.

The primary feature contract is contiguous float32 `[x,y,z,intensity,time_lag]`, with raw KITTI
reflectance and the exact P1-E input contract. A2 and E2 each contain 428 conditions. H10 versus H5
is a compound temporal-and-density intervention; it does not isolate time lag, density, or cap
pressure.

## Evaluator and primary operating point

The protocol reuses the M6b definitions without tuning:

- score `>= 0.25`;
- oriented BEV IoU `>= 0.50`;
- 66 eligible Car and 396 eligible Pedestrian poses;
- DSVT `car -> car` and DSVT `pedestrian -> pedestrian`;
- no truck, bus, construction-vehicle, trailer, or other class maps to Car;
- Van is the Car neighbour-ignore class and Person (sitting) is the Pedestrian neighbour-ignore
  class;
- predictions outside the frozen reference-camera-0 annotation FOV are counted separately, not
  scored as false positives;
- matching uses the frozen descending-score, stable-detection-index, maximum-IoU, lower-track-ID
  tie rules.

Precision and AP are annotation-conditioned because KITTI Raw tracklets are incomplete. AP is the
same all-points, score-ranked monotonic PR-envelope area over the postprocessed prediction
population used by M6b; it is not official KITTI benchmark AP.

## Primary reporting

Execute exactly three complete primary corpus passes. For A2 and E2, report separately for Car and
Pedestrian for pass 1, pass 2, pass 3, their median, minimum, and maximum:

- TP, FP, FN;
- recall;
- annotation-conditioned precision;
- F1;
- annotation-conditioned AP.

TP, FN, recall, and the preregistered recall contrasts are the primary GT-linked quantities. FP is
also reported at the frozen operating point. Because Raw tracklets are incomplete,
annotation-conditioned precision, F1, and AP remain important descriptive quantities rather than
physical whole-world false-positive performance or official KITTI benchmark precision/AP.

Within a pass, A2 and E2 are produced by the same initialized runtime realization. Preserve this
pairing and compute the history contrast before summarizing passes. For class `c` and pass `i`:

```text
history_delta_i[c] = recall(E2_i, c) - recall(A2_i, c)

median_history_delta[c] = median(history_delta_1[c],
                                 history_delta_2[c],
                                 history_delta_3[c])
minimum_history_delta[c] = min(history_delta_1[c], history_delta_2[c], history_delta_3[c])
maximum_history_delta[c] = max(history_delta_1[c], history_delta_2[c], history_delta_3[c])

delta_H10_i[c] = recall(A2_i, c) - frozen_PointPillars_H10_recall(c)
delta_H5_i[c]  = recall(E2_i, c) - frozen_PointPillars_H5_recall(c)
```

Report all three history and historical-baseline deltas before median/minimum/maximum summaries.
Do not substitute `median(E2 recall) - median(A2 recall)` for the paired history contrast. Apply
the same within-pass construction to other mathematically meaningful H10-versus-H5 quantities,
including `TP(E2_i,c) - TP(A2_i,c)` if TP contrast is reported. PointPillars is one frozen
historical realization, not process-paired with DSVT; each V2 pass is compared with that fixed
baseline separately.

No numerical Phase-1 success threshold or model-selection rule is introduced.

## Secondary characterization

For A2 and E2, preregister the same score threshold with oriented BEV IoU thresholds 0.30, 0.50,
and 0.70. Also report, using the exact M6b definitions:

- recall at 0–20 m, 20–35 m, and 35–50 m;
- per-track detected-frame continuity;
- prediction population;
- outside-annotation-FOV counts;
- neighbour-ignore counts and behavior.

These outcomes are descriptive and inherit the Raw-tracklet annotation limitations. Every reported
aggregate retains pass 1, pass 2, pass 3, median, minimum, and maximum.

## Fixed three-pass replication

The complete-corpus pass count is frozen before Stage R results exist. Stage R does not select
between one and three passes: sentinel stability cannot establish stability over the remaining
corpus, the candidate already has non-byte-exact box/score/DetectionFrame engineering outputs,
and annotation-conditioned AP is globally ranked over a complete prediction population.

The three passes are repeated numerical/runtime realizations of the same frozen corpus. They
characterize detector/runtime numerical reproducibility and observed run-to-run outcome spread.
They are not independent dataset samples or population replicates, and three process executions do
not support conventional confidence intervals or p-values. Median, minimum, and maximum summarize
only the observed numerical-realization spread; no inferential statistics are introduced.

Execute exactly three complete primary passes. Each contains all 428 H10 and 428 H5 conditions,
for 856 accepted conditions per pass and 2,568 accepted primary calls. Aggregate passes
independently; do not average boxes, combine detections, form consensus predictions, or select a
favourable pass.

## Stage R — independent GT-relative repeatability characterization

No Stage R call may occur while this document is a draft. Stage R runs before the complete corpus
only after owner freeze, implementation/resume review, exact artifact binding, owner approval, and
a committed inference authorization. It characterizes local discrete repeatability and
cross-process variation, validates repeatability instrumentation, tests whether engineering
nondeterminism reaches evaluator outcomes, and preserves continuity with the M7 sentinel design.
It does not determine corpus-pass count, and its outputs are never reused in a corpus pass.

Observed Stage R variation does not by itself cancel S1. A contract failure, artifact mismatch,
corrupted evaluator, invalid detector output, or other structural defect remains fail-closed.

### Prospective GT-only sentinel audit

The five historical frames were audited using only the frozen KITTI Raw tracklet annotations,
reference-camera eligibility, neighbour-ignore roles, and model-frame range definitions. No
detector prediction was loaded and no DSVT or PointPillars inference occurred. Range entries below
are `0–20 / 20–35 / 35–50 m` eligible-target counts.

| Frame | Eligible Car | Eligible Pedestrian | Car ignore | Pedestrian ignore | Car range counts | Pedestrian range counts |
|---|---:|---:|---:|---:|---|---|
| `2011_09_26_drive_0001/0000000010` | 1 | 0 | 0 | 0 | `0 / 0 / 1` | `0 / 0 / 0` |
| `2011_09_26_drive_0001/0000000011` | 4 | 0 | 0 | 0 | `2 / 2 / 0` | `0 / 0 / 0` |
| `2011_09_26_drive_0001/0000000015` | 3 | 0 | 0 | 0 | `1 / 2 / 0` | `0 / 0 / 0` |
| `2011_09_26_drive_0001/0000000083` | 0 | 0 | 0 | 0 | `0 / 0 / 0` | `0 / 0 / 0` |
| `2011_09_26_drive_0091/0000000010` | 0 | 2 | 0 | 0 | `0 / 0 / 0` | `0 / 2 / 0` |

The historical set has three Car-bearing frames but only one Pedestrian-bearing frame. Applying the
prospective rule to the exact 428-frame order appends the first unused frame that improves a
deficient class, stopping as soon as both classes reach three GT-bearing frames:

1. corpus ordinal 100, `2011_09_26_drive_0091/0000000011`: one eligible Pedestrian at 20–35 m;
2. corpus ordinal 101, `2011_09_26_drive_0091/0000000012`: three eligible Pedestrians at 20–35 m.

Neither added frame has eligible Car or a primary-class neighbour-ignore box. The final frozen
sentinel order is therefore:

1. `2011_09_26_drive_0001/0000000010`
2. `2011_09_26_drive_0001/0000000011`
3. `2011_09_26_drive_0001/0000000015`
4. `2011_09_26_drive_0001/0000000083`
5. `2011_09_26_drive_0091/0000000010`
6. `2011_09_26_drive_0091/0000000011`
7. `2011_09_26_drive_0091/0000000012`

This gives exactly three distinct Car-bearing and three distinct Pedestrian-bearing frames. The
floor provides minimal scene diversity, not statistical representativeness. Audit inputs are bound
by tracklet SHAs `34f0672dee9dc94535893e653b4a66e6ddf534a09d2533bac4e62965935a91b8`
and `3d363ee40129e51aaf44764b9637bc7e946b6e3ec628784adcdedd395505feab`, camera-calibration SHA
`edc1eae281cb95e41798a98dcc545521449527145477ff852dbf3a4ec48c643c`, and Velodyne-calibration
SHA `9dc0c3e92dfceceb9500caa2c9488261a52640e43c3f7c3045ca1ed7927e7266`.

### Fresh-process design and recorded evidence

Run exactly ten fresh Stage R processes. Each process starts clean, verifies frozen runtime and
artifact identities, initializes DSVT once, executes the seven sentinels once each in final order
with H10 then H5, records evidence, and exits. This is 14 conditions per process and exactly 140
accepted Stage R calls. Do not run an eleventh process.

For every process/frame/history/class at score `>= 0.25` and IoU `>= 0.50`, record thresholded
prediction count, TP, FP, FN, ignored-prediction count, and exact matched-GT identity set. At IoU
0.30 and 0.70 also retain TP and matched-GT identities. Preserve sufficient score-ranked
postprocessed sentinel evidence to characterize mini-corpus ordering changes across processes. Do
not require exact raw-float equality or infer full-corpus AP stability from Stage R.

## Secondary intensity-zero intervention

- **A2_zeroI:** H10 with candidate-consumed intensity exactly positive float32 zero.
- **E2_zeroI:** H5 with candidate-consumed intensity exactly positive float32 zero.

Rows, XYZ, time lag, model, checkpoint, source-row order, and evaluator remain identical to A2/E2.
This secondary intervention characterizes sensitivity to the additional domain-shifted intensity
channel. It may not select the candidate, replace A2/E2, modify the model, or tune intensity.

Execute exactly three complete zero-intensity passes, each containing 428 H10 then 428 H5
conditions for 856 accepted calls and 2,568 accepted calls overall. Each pass is independent and
reported as pass 1, pass 2, pass 3, median, minimum, and maximum. Do not merge detections across
passes. These passes are separate processes from primary passes, so primary-versus-zero-intensity
process-index pairs are not statistically paired; compare the three-pass distributions
descriptively. The official five-feature A2/E2 contract remains primary.

## Canonical pass, order, failure, and accounting rules

One complete canonical corpus pass is one fresh Python process that verifies identities,
initializes the frozen candidate once, executes all 856 ordered conditions, and exits normally.
Primary passes 1–3 use three different fresh processes; zero-intensity passes 1–3 use another three.
No accepted pass may mix process outputs.

Every pass uses the frozen 428-frame order and executes `frame_1/H10`, `frame_1/H5`, then
`frame_2/H10`, `frame_2/H5`, through `frame_428/H10`, `frame_428/H5`. Stage R uses its final
seven-frame order with H10 then H5 in each fresh process. Results may not change this order.

If a canonical process dies before all 856 conditions, preserve its checkpoint/log as failure
evidence, mark the attempt incomplete, exclude its outputs from scientific aggregation, and restart
the entire logical pass from condition 1 in a new process. Never splice an incomplete attempt into
a replacement. Record attempt ID, logical pass ID, process identity, attempted calls, accepted
canonical calls, failed calls, and failure reason. A technical call failure with no accepted output
may be retried only under the later frozen implementation with full accounting.

Expected accepted detector calls are fixed:

| Experiment | Processes | Conditions per process | Accepted calls |
|---|---:|---:|---:|
| Primary A2/E2 | 3 | 856 | 2,568 |
| Secondary A2_zeroI/E2_zeroI | 3 | 856 | 2,568 |
| Stage R | 10 | 14 | 140 |
| **Total** | **16** | — | **5,276** |

Stage R outputs are not corpus outputs. Attempts may exceed 5,276 calls only when explicitly
recorded technical failures yielded no accepted canonical output. The corpus may not be trimmed.

## Claim boundaries

### Point order

The primary V2 policy preserves frozen source-row order. The official DSVT test data processor
enables random point shuffling; LaserPerception bypasses it prospectively. S1 includes no
shuffle-versus-preserved-order condition and makes no point-order-neutrality claim.

### Spatial discretization and range

Historical PointPillars uses 0.25 m XY voxels, a 400 x 400 grid, and
`[-50,-50,-5,50,50,3]`. DSVT uses 0.3 m XY pillars, a 360 x 360 grid, and supports
`[-54,-54,-5,54,54,3]`. The shared corpus remains restricted to
`[-50,-50,-5,50,50,3]`, so DSVT's outer supported XY ring is unpopulated. Occupied-pillar
differences are not architecture-only, and zero candidate-range drops do not imply source-training
distribution equivalence.

### Intensity

Historical PointPillars did not consume intensity. DSVT consumes raw KITTI reflectance through a
channel learned from nuScenes intensity. S1 does not assume score calibration equivalence or
intensity neutrality; the zero-intensity intervention is explicitly secondary.

### Deployment completeness

The historical PointPillars stack demonstrated the project's frozen end-to-end TensorRT path. M8
P1-E verified only a partial DSVT TensorRT boundary after DynPillarVFE plus DSVT InputLayer through
four transformer blocks. Raw point handling, DynPillarVFE, DSVT InputLayer, BEV scatter, the 2D
backbone, TransFusion head, and postprocess remain outside that partial engine as applicable. S1 is
an accuracy/cross-domain comparison and cannot establish end-to-end DSVT deployment parity.

## Measurement identity and authorization barrier

The future measurement manifest must bind at minimum:

- frozen S1 protocol commit;
- P1-E canonical engineering and merge commits;
- DSVT upstream commit, official config SHA, checkpoint SHA, and candidate-manifest SHA;
- input-ledger and final-head-revalidation SHAs;
- exact evaluator implementation identity;
- exact Python, PyTorch, CUDA runtime, NVIDIA driver, GPU name and available GPU UUID, spconv,
  torch-scatter, NumPy, and device identities;
- model eval/train state, inference/no-grad/inference-mode state, Python/NumPy/Torch CPU/Torch CUDA
  seeds, TF32 settings, cuDNN benchmark and deterministic settings, Torch deterministic-algorithm
  setting, and relevant CUDA/PyTorch environment variables;
- class mapping, score contract, point-order policy, and intensity policy.

The measurement must preserve accepted P1-E behavior; it must not enable a new deterministic
algorithm merely to make S1 cleaner. Any runtime-policy difference from P1-E requires owner review
before authorization.

Before explicit inference authorization, the future implementation must report expected
single-pass, Stage R, and full-campaign durations; measured model-initialization duration; and
pass/session operational feasibility. A GT-blind engineering preflight may size runtime only if it
loads no GT, calculates no accuracy, changes no frozen design, and discards semantic detector
outputs. Runtime inconvenience does not authorize reducing the corpus, pass count, or secondary
condition. If the campaign is operationally infeasible, stop before authorization and begin a new
prospective protocol revision; never modify this protocol mid-campaign.

There may be no caller-selectable model factory. Protocol freeze alone does not authorize
measurement. The mandatory chronology is:

1. this frozen protocol exists;
2. the scientific measurement implementation is created;
3. checkpoint/resume/process semantics are implemented;
4. all artifacts and runtime identities are bound fail-closed;
5. the implementation is reviewed;
6. the owner explicitly approves inference;
7. an inference-authorization act is committed **before** the first GT-relative V2 call;
8. only then may independent Stage R begin, followed by the already-fixed three primary and three
   zero-intensity complete-corpus processes.

## Output chronology

Measurement first produces raw results only, without scientific interpretation:

- `benchmarks/m8/results/m8_s1_stage_r.json`;
- `benchmarks/m8/results/m8_s1_primary_raw.json`;
- `benchmarks/m8/results/m8_s1_secondary_raw.json`;
- `benchmarks/m8/results/m8_s1_measurement_manifest.json`;
- `docs/m8/M8_S1_MEASUREMENT_RAW.md`.

The Stage R artifact retains the final sentinel set, GT-only audit identity, ten process identities,
every condition outcome, and repeatability summaries. Primary and secondary raw artifacts retain
three separate complete pass records; they are never collapsed to median-only evidence. Large
detector/checkpoint evidence remains external and is recorded by filename, byte count, and SHA256.
Owner review of the raw record precedes any separate interpretation act.

## S2 barrier and gap eligibility

S1 must finish and freeze the A2/E2 outputs before S2 can be designed. S1 authorizes no
B2/C2/D2/F2 inference. Because S1 produces three complete A2/E2 realizations, future S2 must
prospectively define how shared, E-only, A-only, and neither partitions are constructed for a
numerically nondeterministic detector. It may use a single frozen realization or a preregistered
multi-pass stability definition, but S1 selects neither. Pass 1, median predictions, union, and
intersection must not be adopted automatically. S2 makes that decision after raw S1 evidence is
frozen and before any B2/C2/D2/F2 output exists.

Prospectively, if Car `TP_E2 <= TP_A2`, the positive H10-to-H5 phenomenon did not replicate and
normalized recovery is undefined. If the gap is positive but too small under a future S2
preregistered stability rule, normalized recovery must not be headlined. S1 does not select that
future numerical cutoff.

## Claim audit

This protocol does **not** state or assume:

- pure architecture causality;
- that DSVT is universally better;
- end-to-end DSVT TensorRT parity;
- score-calibration equivalence;
- source-distribution equivalence;
- deterministic inference;
- intensity neutrality;
- point-order neutrality;
- that M7 replication has already been established.

At draft time there are zero Stage R calls, zero GT-relative V2 results, zero H10/H5 V2 accuracy
inferences, zero intensity-zero inferences, and zero B2/C2/D2/F2 inferences.
