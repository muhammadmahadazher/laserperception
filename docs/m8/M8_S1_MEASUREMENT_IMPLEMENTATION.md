# M8 P1-S1 measurement runtime implementation

Status: **M8 P1-S1 MEASUREMENT IMPLEMENTATION CANDIDATE — OWNER REVIEW REQUIRED —
STAGE R AND GT-RELATIVE MEASUREMENT UNAUTHORIZED.**

This document describes the reviewable execution machinery for the already frozen P1-S1
protocol. It is not an inference authorization and records no S1 accuracy result. The frozen
protocol files remain byte-for-byte unchanged.

## Architecture and authorization barrier

`scripts/detection/run_m8_s1.py` exposes only `preflight`, `stage-r`, `primary-pass`,
`zero-intensity-pass`, and `aggregate`. The GT-blind preflight has a physically separate module
and imports no KITTI tracklet loader or evaluator. The three scientific inference modes first
verify the frozen protocol, candidate, checkpoint, upstream config, accepted input ledger, and
final input revalidation. They then require a separate, exact future owner authorization. Only
after that barrier passes can the CLI import the GT/evaluation worker or initialize DSVT.

No authorization artifact is created by this implementation. There is no force, unsafe,
skip-authorization, ignore-binding, or unfrozen bypass. A missing authorization is the expected,
normal fail-closed state.

The future authorization schema binds the protocol freeze commit and JSON hash, reviewed runtime
commit, runtime-binding identity, checkpoint, config, candidate manifest, input ledger, evaluator
identity, owner role and approval, and timestamp/provenance. The implementation manifest avoids a
self-referential commit field: the reviewed Git commit is the external canonical implementation
identity.

## Pass and Stage R process models

A canonical corpus pass is one fresh Python process. It revalidates all 428 H10 and all 428 H5
inputs before model initialization, captures runtime state, initializes DSVT once, and executes
the immutable 856-condition frame-major order (`H10`, then `H5`). The orchestrator launches three
separate primary processes and three separate zero-intensity processes. Process UUIDs must be
distinct across all six passes.

Stage R is implemented but cannot currently run. Each future repeat is one fresh process with the
seven frozen sentinels in protocol order, H10 then H5, for exactly 14 calls. Ten such processes
produce the future 140 accepted calls. Stage R outputs are independent and are never reused as
primary or zero-intensity corpus outputs.

The zero-intensity intervention is derived only from an already verified primary input. XYZ,
time-lag, row count, and row order remain byte-identical; every candidate-consumed intensity word
is IEEE-754 float32 positive zero (`0x00000000`).

## Failure, restart, and atomic evidence

Each attempt has a logical pass ID, attempt ID, process UUID/PID, timestamps, attempted/completed/
failed call counts, accepted canonical count, failure reason, and explicit status. Conditions can
be written only in the next frozen position. An interrupted attempt is `INCOMPLETE`, has zero
accepted canonical calls, is preserved externally, and cannot resume. Its replacement has a new
attempt directory and process identity and begins again at condition 1. Partial attempts cannot be
spliced.

Condition records, progress, the raw pass, and final manifests use sibling temporary files,
`flush`, `fsync`, and `os.replace`. A final pass manifest is possible only at exactly 856 conditions
(or 14 for one Stage R repeat). It records ordered condition-file hashes and a deterministic final
result hash. Partial evidence therefore cannot masquerade as canonical evidence.

## Evidence design and aggregation

Large per-condition predictions and session evidence remain external. Every external artifact is
identified compactly by logical name, role, bytes, SHA256, schema, and producer. The later tracked
results remain the protocol-listed compact Stage R, primary, secondary, measurement-manifest, and
raw-measurement documentation artifacts. No checkpoint, ONNX, engine, dataset, or large prediction
dump is added to Git.

The future per-condition schema retains frame/history/input identity, DetectionFrame identity,
stable prediction indices, score and box payload/hash, annotation-FOV classification, primary
match/ignore/FP disposition, matched GT/IoU, range band, track identity, all three IoU threshold
counts and matched sets, and score-ranked dispositions. This is sufficient to recompute the frozen
TP/FP/FN, recall, annotation-conditioned precision/F1/AP, range, continuity, outside-FOV, and
neighbour-ignore summaries without rerunning DSVT.

Offline aggregation initializes no detector. It keeps pass 1/2/3 distinct and reports only
minimum, median, and maximum. The history contrast is computed per pass first:

`history_delta_i[c] = recall(E2_i,c) - recall(A2_i,c)`

It explicitly does not subtract marginal medians. PointPillars comparisons use the fixed
historical baseline for each V2 pass. Annotation-conditioned AP delegates to the unchanged M6b
`all_points_average_precision` implementation. No boxes are averaged and no confidence interval,
p-value, standard error, or population-significance claim is produced.

## Runtime state and telemetry

Each future process records exact Python, PyTorch, CUDA, driver/device, GPU UUID where available,
spconv, torch-scatter, and NumPy identities; model eval state; inference/gradient state; actual
unseeded Python/NumPy/Torch policies and state hashes; TF32 and cuDNN flags; deterministic-
algorithm state; relevant environment variables; point-order policy; artifact identities; and CUDA
allocated/reserved/peak memory. The runner does not seed or enable deterministic algorithms. It
reuses `NvidiaSmiSampler` for clocks, temperature, P-state, power, GPU/memory utilization, and
driver/device context; telemetry is not a pass-selection tool.

## Authorized GT-blind sizing preflight

The engineering preflight is the only inference-capable mode authorized before a separate
scientific authorization. It uses the accepted H10 input-only pillar census, excludes all seven
Stage R sentinels, sorts by candidate pillar count with earliest frozen corpus order as the tie
break, and selects nearest ranks for quantiles 0.05 through 0.95. Selection is frozen before model
output exists. One earliest non-sentinel, non-selected frame supplies the excluded H10/H5 warmup
pair.

Exactly two fresh worker processes each verify bindings, reconstruct exact inputs, time DSVT
initialization, execute two warmups, synchronize CUDA, and time 20 fixed H10/H5 calls. Predictions
exist only transiently and are immediately discarded by a backend method that returns no semantic
value or prediction count. The artifact records 40 measured and four warmup engineering calls,
with GT/evaluator loaded and semantic output retained all explicitly false.

The preflight ran at runtime implementation commit
`f23de4d3bbf538166f25ba6fa5e96dccd3a5c7e3`. Exactly two fresh processes performed four
excluded warmup calls and 40 measured engineering calls. No ground truth or evaluator was loaded,
no prediction count was observed or stored, no semantic output was retained, and no scientific
campaign call was made. A later deterministic summary-only correction made the resource table use
the already captured post-sizing CUDA peaks instead of initialization memory; it caused no further
inference.

The selected input-only representatives were:

| Quantile | Frame | H10 candidate pillars |
| ---: | --- | ---: |
| 0.05 | `2011_09_26_drive_0091/0000000326` | 14,491 |
| 0.15 | `2011_09_26_drive_0091/0000000332` | 15,148 |
| 0.25 | `2011_09_26_drive_0091/0000000234` | 17,140 |
| 0.35 | `2011_09_26_drive_0091/0000000221` | 18,708 |
| 0.45 | `2011_09_26_drive_0091/0000000157` | 21,854 |
| 0.55 | `2011_09_26_drive_0091/0000000015` | 25,503 |
| 0.65 | `2011_09_26_drive_0091/0000000029` | 28,720 |
| 0.75 | `2011_09_26_drive_0001/0000000039` | 29,116 |
| 0.85 | `2011_09_26_drive_0091/0000000096` | 29,867 |
| 0.95 | `2011_09_26_drive_0091/0000000058` | 31,960 |

Initialization took 12.729 and 13.623 seconds (median 13.176 seconds). Observed synchronized
wall-clock inference summaries were:

| Input | Minimum (s) | Median (s) | Maximum (s) |
| --- | ---: | ---: | ---: |
| H10 | 0.299 | 0.336 | 0.367 |
| H5 | 0.214 | 0.233 | 0.260 |
| H10/H5 pair | 0.529 | 0.563 | 0.627 |

Both workers ran on an NVIDIA GeForce RTX 4060 Laptop GPU with driver 610.88. During measured
blocks, sampled SM clocks ranged from 2,280 to 2,655 MHz, memory clocks from 7,001 to 8,001 MHz,
temperature from 54 to 64 degrees Celsius, power from 17.12 to 80.72 W, and GPU utilization from
0% to 93%; observed P-states were P0 and P3. Power-limit telemetry was unavailable. The maximum
captured CUDA reserve was 7,073,693,696 of 8,585,216,000 bytes (82.39%), while peak process RSS
was 1,778,950,144 of 8,170,172,416 host bytes (21.77%).

## Operational estimate policy

The sizing artifact uses the observed median H10/H5 pair rate plus median initialization for a
central engineering estimate. Its conservative observed-rate envelope uses the observed minimum
and maximum pair and initialization times. It reports one 856-condition pass, Stage R (ten fresh
processes and 140 calls), three primary passes, three zero-intensity passes, and the full 5,276-call
campaign. These stratified observations are not confidence bounds and cannot automatically revise
the protocol.

The resulting central estimates (with conservative observed-rate envelopes) are:

| Scope | Central | Observed-rate envelope |
| --- | ---: | ---: |
| One 856-condition pass | 254.3 s | 239.0–282.0 s |
| Stage R: 10 processes / 140 calls | 171.2 s | 164.3–180.1 s |
| Three primary passes | 762.8 s | 716.9–846.0 s |
| Three zero-intensity passes | 762.8 s | 716.9–846.0 s |
| Full 5,276-call campaign | 1,696.8 s | 1,598.0–1,872.2 s |

Classification: **OPERATIONALLY PLAUSIBLE**. Both fresh processes completed the fixed sizing
workload and the recorded peak CUDA reserve and host RSS stayed below the preflight's documented
90% engineering capacity-review boundary. That boundary is operational only, not a scientific
acceptance threshold; the frozen scientific protocol is unchanged.
