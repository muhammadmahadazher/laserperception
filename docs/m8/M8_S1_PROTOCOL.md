# M8 P1-S1 frozen detector-stack comparison protocol — draft

Status: **M8 P1-S1 SCIENTIFIC PROTOCOL DRAFT — OWNER REVIEW REQUIRED — NO
GT-RELATIVE V2 RESULT EXISTS.**

This document is a preregistration draft. It is not frozen, does not authorize inference, and
contains no Stage R, A2, E2, or intensity-zero detector result. Owner approval and a committed
protocol freeze are necessary but still insufficient to begin measurement: the implementation,
resume behavior, artifact binding, and explicit inference authorization must be reviewed and
committed afterward.

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

For A2 and E2, report separately for Car and Pedestrian:

- TP, FP, FN;
- recall;
- annotation-conditioned precision;
- F1;
- annotation-conditioned AP.

Report the following frozen-baseline deltas without treating them as causal architecture effects:

```text
delta_H10_class       = recall(V2_H10_class) - recall(PointPillars_H10_class)
delta_H5_class        = recall(V2_H5_class)  - recall(PointPillars_H5_class)
delta_history_v2_class = recall(V2_H5_class) - recall(V2_H10_class)
```

No numerical Phase-1 success threshold or model-selection rule is introduced.

## Secondary characterization

For A2 and E2, preregister the same score threshold with oriented BEV IoU thresholds 0.30, 0.50,
and 0.70. Also report, using the exact M6b definitions:

- recall at 0–20 m, 20–35 m, and 35–50 m;
- per-track detected-frame continuity;
- prediction population;
- outside-annotation-FOV counts;
- neighbour-ignore counts and behavior.

These outcomes are descriptive and inherit the Raw-tracklet annotation limitations.

## Stage R — GT-relative repeatability before corpus measurement

No Stage R call may occur while this document is a draft. Stage R must run before the full A2/E2
corpus only after the protocol is owner-approved and frozen, the measurement implementation and
resume design are reviewed, exact artifacts are bound, and explicit inference authorization is
committed.

### Sentinel set and call count

Use exactly these five frozen frames:

1. `2011_09_26_drive_0001/0000000010`
2. `2011_09_26_drive_0001/0000000011`
3. `2011_09_26_drive_0001/0000000015`
4. `2011_09_26_drive_0001/0000000083`
5. `2011_09_26_drive_0091/0000000010`

Run both H10 and H5 for every frame: ten sentinel conditions. Repeat each condition exactly ten
times, for exactly 100 Stage R calls. Do not perform an eleventh repeat.

### Recorded evidence per repetition

For Car and Pedestrian at score `>= 0.25` and primary IoU `>= 0.50`, preserve enough evidence to
compare:

- identities of predictions surviving the score threshold;
- prediction count by primary class;
- TP, FP, FN;
- ignored-prediction count;
- exact matched-GT identity set.

At IoU 0.30 and 0.70, also record discrete TP and matched-GT identity information for secondary
characterization. Raw-tensor maximum differences do not establish these discrete quantities. The
engineering source-domain differences therefore neither pass nor fail Stage R in advance.

### Branch S — discrete stability

Branch S applies only if, for every one of the ten sentinel conditions and both primary classes,
all ten repeats have exactly identical TP, FP, FN, ignored-prediction count, and matched-GT
identity set at score `>= 0.25` and IoU `>= 0.50`.

Under Branch S, execute the full primary V2 corpus once. Stage R repeat #1 is the canonical A2/E2
result for each of the ten sentinel conditions and must not be rerun in the ordinary sweep. Reuse
is scientifically sound under Branch S because the branch requires the discrete scientific
outcomes to be identical across all ten repeats before reuse is selected.

### Branch R — discrete instability

If any required primary discrete quantity differs for either class on any sentinel, Branch R
applies automatically. The difference must not be reinterpreted as negligible.

Under Branch R, execute three complete independent primary V2 corpus passes. For each sentinel,
Stage R repeats #1, #2, and #3 become its pass 1, pass 2, and pass 3 results; those combinations
must not be rerun. Aggregate every pass independently and report pass 1, pass 2, pass 3, median,
minimum, and maximum for all primary aggregate outcomes. Do not merge detections across passes or
average boxes into synthetic predictions. The scientific report must identify discrete-level V2
nondeterminism.

## Secondary intensity-zero intervention

- **A2_zeroI:** H10 with candidate-consumed intensity exactly positive float32 zero.
- **E2_zeroI:** H5 with candidate-consumed intensity exactly positive float32 zero.

Rows, XYZ, time lag, model, checkpoint, source-row order, and evaluator remain identical to A2/E2.
This secondary intervention characterizes sensitivity to the additional domain-shifted intensity
channel. It may not select the candidate, replace A2/E2, modify the model, or tune intensity.

Use the branch selected by primary Stage R; do not conduct a second adaptive repeatability test. If
Branch S applies, run one complete zero-intensity corpus pass. If Branch R applies, run three
complete zero-intensity corpus passes. No zero-intensity result exists at protocol-draft time.

## Complete-call accounting and resume requirements

The future implementation must checkpoint atomically, resume without silently rerunning accepted
conditions, retain the complete corpus, and report exact attempted, accepted, reused, and failed
call counts.

| Branch | Stage R calls | Additional primary calls | Zero-intensity calls | Total detector calls | Canonical primary outputs |
|---|---:|---:|---:|---:|---:|
| S | 100 | 846 | 856 | 1,802 | 856 |
| R | 100 | 2,538 | 2,568 | 5,206 | 2,568 across three passes |

Branch S reuses ten Stage R repeat-#1 outputs. Branch R reuses 30 Stage R outputs: repeats #1–#3
for ten sentinel conditions. The remaining Stage R repetitions are repeatability evidence only.
The corpus may not be reduced because of runtime or session length.

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
- verified PyTorch, CUDA, spconv, torch-scatter, NumPy, and device identities;
- class mapping, score contract, point-order policy, and intensity policy.

There may be no caller-selectable model factory. Protocol freeze alone does not authorize
measurement. The mandatory chronology is:

1. owner reviews and freezes this protocol in a committed act;
2. scientific measurement implementation and checkpoint/resume design are created and reviewed;
3. all artifacts and runtime identities are bound fail-closed;
4. owner approves the implementation;
5. explicit inference authorization is committed **before** the first GT-relative V2 call;
6. Stage R runs, selects Branch S or Branch R mechanically, and only then does the corresponding
   complete-corpus execution proceed.

## Output chronology

Measurement first produces raw results only, without scientific interpretation:

- `benchmarks/m8/results/m8_s1_stage_r.json`;
- `benchmarks/m8/results/m8_s1_primary_raw.json`;
- `benchmarks/m8/results/m8_s1_secondary_raw.json`;
- `benchmarks/m8/results/m8_s1_measurement_manifest.json`;
- `docs/m8/M8_S1_MEASUREMENT_RAW.md`.

Large detector/checkpoint evidence remains external and is recorded by filename, byte count, and
SHA256. Owner review of the raw record precedes any separate interpretation act.

## S2 barrier and gap eligibility

S1 must finish and freeze the A2/E2 paired outputs before S2 can be designed. S1 authorizes no
B2/C2/D2/F2 inference. Only afterward may S2 define shared, E-only, A-only, and neither partitions
or a normalized recovery statistic.

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
