# M8 P1-S1 Stage R raw evidence

> **Status:** OWNER REVIEW REQUIRED. This record contains raw Stage R repeatability evidence only.
> Primary and zero-intensity corpus measurement remain unauthorized. No scientific interpretation
> is included.

The authorized Stage R run completed ten fresh sequential processes. Each process evaluated the
same seven frozen sentinel frames in H10-then-H5 order, producing 14 accepted calls per process and
140 accepted calls overall. There were no failed attempts, retries, primary calls, zero-intensity
calls, B2/C2/D2/F2 calls, PointPillars reruns, or training/fine-tuning calls.

## Frozen execution and authorization

| Identity | Value |
| --- | --- |
| Protocol freeze commit | `5061d5d2c6a6057fed1f3f537c5857d2d84f6b3f` |
| Protocol JSON SHA256 | `c132f60257c6a39debb548461c79bd59c98325484d233db6095b441c638d8e88` |
| Execution commit | `d8e5012312b6ee0b3c891e1c2d794424f8a35c36` |
| Authorization commit | `81eb156e1f384b18e16af8189ad546965e007b6e` |
| Authorization merge | `0f93c480acb6c98bc07781db8ed64b8433ec9238` |
| Authorization artifact SHA256 | `3c16a0c0ff9680b6418a53b20a5a51dd8f2a40d864a75c380397f2454ce06b9c` |
| Runtime-policy SHA256 | `703e453a8bca0e6e2e4b1c4b976deaa5bc4ed27b3a4847144204193baab77563` |
| Checkpoint SHA256 | `a675149d095eef8ddc0c137ae46eeac075ccc504c7608162c71e7adf318793fb` |
| Config SHA256 | `b0832e03ad11d4e0b61f0fb07d977e687763caae472a4f87ed750bdc2d13be0f` |
| Candidate manifest SHA256 | `aa456e0386e46e9d089a957b1f1a8a4f74ceae70435c7ad8e6ca5e67bb90f4e7` |
| Input ledger SHA256 | `474e87e34c64d669750d4b6f7a64ac46fc9c5c462693fad79ff7c9547a7f1f7c` |
| Evaluator | `m6b-r2-score-0.25-oriented-bev-iou-0.30-0.50-0.70` |

Before the first call, the detached worktree was clean at the exact execution commit. A dry,
pre-model check proved that its runtime recognizes authorization schema
`laserperception.m8.s1.authorization.v2` and runtime-policy schema
`laserperception.m8.s1.runtime-policy-binding.v1`. The live policy matched the bound policy exactly:
Python 3.10.12, PyTorch 2.1.0+cu118, CUDA 11.8, driver 610.88, RTX 4060 Laptop GPU UUID
`GPU-fb8f7552-21ae-1dbb-798c-e9fa3dca54fc`, spconv 2.3.8,
torch-scatter 2.1.2+pt21cu118, NumPy 1.23.5, `CUDA_MODULE_LOADING=LAZY`, and
`PYTORCH_CUDA_ALLOC_CONF` unset. No competing CUDA compute process was present.

## Attempt accounting

All rows are first attempts from one sequential orchestrator session. The exact per-process raw
file hashes, result hashes, directory sizes, timestamps, and deterministic tree identities are in
[`m8_s1_stage_r.json`](../../benchmarks/m8/results/m8_s1_stage_r.json).

| Logical pass | Process UUID | Calls | Status |
| --- | --- | ---: | --- |
| stage-r-1 | `ceae628a-7f0d-49c7-a9d1-a43cda679b92` | 14 | COMPLETE |
| stage-r-2 | `bade393f-b46e-475b-8216-dead3797acf7` | 14 | COMPLETE |
| stage-r-3 | `92101bbe-1f71-4eef-a24d-aec889f94b37` | 14 | COMPLETE |
| stage-r-4 | `607e4dcb-43da-4eb8-9f3d-8a15bea7b19a` | 14 | COMPLETE |
| stage-r-5 | `cb5d1b68-6b99-4164-8e34-94a2b5aba960` | 14 | COMPLETE |
| stage-r-6 | `57409512-ee0b-40c5-9d19-78646611beeb` | 14 | COMPLETE |
| stage-r-7 | `17fe6bc8-1b32-4ebf-a6a6-f817b637ae55` | 14 | COMPLETE |
| stage-r-8 | `c1b046f5-0183-4b0f-b250-587d1d61ed8e` | 14 | COMPLETE |
| stage-r-9 | `53cb9aae-e7b1-4982-9be0-eaca25a97fa7` | 14 | COMPLETE |
| stage-r-10 | `943c5ffe-ae29-4f53-ab54-23d6fda0d667` | 14 | COMPLETE |

Total attempted calls and total accepted canonical calls are both 140. No technical attempt failed.

## Mini-corpus GT denominators

These denominators are necessary for reading the discrete totals. H10 and H5 for one frame share
the same GT, so both unique-frame and evaluated-condition counts are shown.

| Class | Eligible GT in seven unique frames | Eligible GT across 14 evaluated conditions |
| --- | ---: | ---: |
| Car | 8 | 16 |
| Pedestrian | 6 | 12 |

The Stage R denominator is small and is not directly interchangeable with a full-corpus
denominator.

## Primary IoU 0.50 raw repeatability

Values are ordered `stage-r-1` through `stage-r-10`.

| Class | TP values | FP values | FN values | Ignored values | Thresholded-prediction values | TP min / median / max / range |
| --- | --- | --- | --- | --- | --- | --- |
| Car | `8, 8, 8, 8, 8, 8, 8, 8, 8, 8` | `4, 4, 4, 4, 4, 4, 4, 4, 4, 4` | `8, 8, 8, 8, 8, 8, 8, 8, 8, 8` | `0, 0, 0, 0, 0, 0, 0, 0, 0, 0` | `12, 12, 12, 12, 12, 12, 12, 12, 12, 12` | `8 / 8 / 8 / 0` |
| Pedestrian | `0, 0, 0, 0, 0, 0, 0, 0, 0, 0` | `2, 2, 2, 2, 2, 2, 2, 2, 2, 2` | `12, 12, 12, 12, 12, 12, 12, 12, 12, 12` | `0, 0, 0, 0, 0, 0, 0, 0, 0, 0` | `2, 2, 2, 2, 2, 2, 2, 2, 2, 2` | `0 / 0 / 0 / 0` |

Matched-GT identity stability is the primary repeatability signal: for both classes, every
condition's primary matched-GT identity set was exact across all ten processes. The primary
discrete outcome tuple was also exact for every condition and both classes. Thus equal TP counts
did not conceal churn between different matched objects in this run.

No wildly discrepant primary TP process was observed. Had one appeared, it would have remained in
the raw evidence and been flagged for owner review rather than retried.

## Per-condition primary outcomes

Each tuple is `predictions / TP / FP / FN / ignored` and was identical in all ten processes.

### Car

| Condition | Eligible GT | Primary tuple | TP at IoU 0.30 | TP at IoU 0.70 |
| --- | ---: | --- | ---: | ---: |
| drive_0001 frame 10 H10 | 1 | `1 / 0 / 1 / 1 / 0` | 0 | 0 |
| drive_0001 frame 10 H5 | 1 | `2 / 0 / 2 / 1 / 0` | 0 | 0 |
| drive_0001 frame 11 H10 | 4 | `1 / 1 / 0 / 3 / 0` | 1 | 0 |
| drive_0001 frame 11 H5 | 4 | `2 / 2 / 0 / 2 / 0` | 2 | 0 |
| drive_0001 frame 15 H10 | 3 | `2 / 2 / 0 / 1 / 0` | 2 | 0 |
| drive_0001 frame 15 H5 | 3 | `4 / 3 / 1 / 0 / 0` | 3 | 1 |
| drive_0001 frame 83 H10 | 0 | `0 / 0 / 0 / 0 / 0` | 0 | 0 |
| drive_0001 frame 83 H5 | 0 | `0 / 0 / 0 / 0 / 0` | 0 | 0 |
| drive_0091 frame 10 H10 | 0 | `0 / 0 / 0 / 0 / 0` | 0 | 0 |
| drive_0091 frame 10 H5 | 0 | `0 / 0 / 0 / 0 / 0` | 0 | 0 |
| drive_0091 frame 11 H10 | 0 | `0 / 0 / 0 / 0 / 0` | 0 | 0 |
| drive_0091 frame 11 H5 | 0 | `0 / 0 / 0 / 0 / 0` | 0 | 0 |
| drive_0091 frame 12 H10 | 0 | `0 / 0 / 0 / 0 / 0` | 0 | 0 |
| drive_0091 frame 12 H5 | 0 | `0 / 0 / 0 / 0 / 0` | 0 | 0 |

### Pedestrian

| Condition | Eligible GT | Primary tuple | TP at IoU 0.30 | TP at IoU 0.70 |
| --- | ---: | --- | ---: | ---: |
| drive_0001 frame 10 H10 | 0 | `0 / 0 / 0 / 0 / 0` | 0 | 0 |
| drive_0001 frame 10 H5 | 0 | `0 / 0 / 0 / 0 / 0` | 0 | 0 |
| drive_0001 frame 11 H10 | 0 | `0 / 0 / 0 / 0 / 0` | 0 | 0 |
| drive_0001 frame 11 H5 | 0 | `0 / 0 / 0 / 0 / 0` | 0 | 0 |
| drive_0001 frame 15 H10 | 0 | `0 / 0 / 0 / 0 / 0` | 0 | 0 |
| drive_0001 frame 15 H5 | 0 | `0 / 0 / 0 / 0 / 0` | 0 | 0 |
| drive_0001 frame 83 H10 | 0 | `0 / 0 / 0 / 0 / 0` | 0 | 0 |
| drive_0001 frame 83 H5 | 0 | `0 / 0 / 0 / 0 / 0` | 0 | 0 |
| drive_0091 frame 10 H10 | 2 | `1 / 0 / 1 / 2 / 0` | 0 | 0 |
| drive_0091 frame 10 H5 | 2 | `1 / 0 / 1 / 2 / 0` | 0 | 0 |
| drive_0091 frame 11 H10 | 1 | `0 / 0 / 0 / 1 / 0` | 0 | 0 |
| drive_0091 frame 11 H5 | 1 | `0 / 0 / 0 / 1 / 0` | 0 | 0 |
| drive_0091 frame 12 H10 | 3 | `0 / 0 / 0 / 3 / 0` | 0 | 0 |
| drive_0091 frame 12 H5 | 3 | `0 / 0 / 0 / 3 / 0` | 0 | 0 |

At IoU 0.30 and 0.70, every per-condition TP value and matched-GT identity set was also exact
across all ten processes. Aggregate Car TP values were 8 at IoU 0.30 and 1 at IoU 0.70 in every
process. Pedestrian TP values were 0 at both secondary thresholds in every process.

## Score-ranked mini-corpus evidence

The count of ranked entries was exact: 106 Car and 543 Pedestrian entries in every process. Exact
payload hashes differed in all ten processes because score values varied. Score-ranked ordering
identity had three Car signatures and eight Pedestrian signatures. Ordering variation occurred in
two Car conditions and three Pedestrian conditions, identified machine-readably in the compact
JSON. The largest corresponding score spread was `0.0003924369812011719` for Car and
`0.010171376168727875` for Pedestrian. These are raw observations; no cause is assigned here.

## External evidence identity and handling

The external attempt tree contains 190 files and 30,023,835 file bytes. Its deterministic
manifest-tree SHA256 is
`85fbffcb48f83d28ff74b7ab2bab77f99166090dc994a7436ac8b4fc6d3a3c7d`.
The physical path is intentionally not committed.

The WSL launcher initially materialized the untracked attempt directories beside the detached
execution worktree because the launch received an empty `--attempt-root` path. After all ten
processes completed, those directories were relocated to the external evidence root. A sorted
pre/post file-SHA manifest was byte-identical. No source file changed, no detector call was rerun,
and the detached execution worktree was clean after relocation. This was evidence-location
handling, not a failed scientific attempt.

## Prospective relationship to a future primary campaign

Before any primary result exists, the following reporting yardstick is fixed: the observed Stage R
spread, normalized by the applicable evaluated-condition class denominator, is the reference scale
for interpreting future S1 H10-versus-H5 and V2-versus-PointPillars deltas. Deltas within that scale
will be reported as within observed run-to-run variation.

This yardstick is not a pass/fail gate, does not make the Stage R mini-corpus denominator directly
interchangeable with a full-corpus denominator, and does not authorize a primary campaign.

Primary and zero-intensity measurement remain unauthorized pending a separate owner review and
authorization act.
