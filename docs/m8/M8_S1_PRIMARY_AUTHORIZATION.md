# M8 P1-S1 primary measurement authorization

**Status: M8 P1-S1 PRIMARY A2/E2 — OWNER AUTHORIZED AFTER STAGE R RAW REVIEW —
ZERO-INTENSITY REMAINS UNAUTHORIZED.**

This is an authorization record only. No primary, zero-intensity, B2, C2, D2, or F2 detector
call occurred while preparing it.

## Frozen bindings

- Stage R raw merge commit: `eaecc404d2caed72e7c99e70242958fb387cea4b`
- Stage R compact raw result:
  `benchmarks/m8/results/m8_s1_stage_r.json`, SHA256
  `1549386c9bc9185c4240082398c5e0e5a64dc066ee5695c462bbfefde873952f`
- Stage R raw document: `docs/m8/M8_S1_STAGE_R_RAW.md`, SHA256
  `d6923df196de29471406a6fd88878cf5354ccb552ae3dd97517cbd0928707cd3`
- External Stage R evidence tree SHA256:
  `85fbffcb48f83d28ff74b7ab2bab77f99166090dc994a7436ac8b4fc6d3a3c7d`
- Scientific execution commit: `d8e5012312b6ee0b3c891e1c2d794424f8a35c36`
- Runtime-policy SHA256:
  `703e453a8bca0e6e2e4b1c4b976deaa5bc4ed27b3a4847144204193baab77563`
- Primary authorization: `benchmarks/m8/preregistration/m8_s1_primary_authorization.json`, SHA256
  `0f67b939dd57fd782ca55a525f609c58d0349b216f2ed89a697e328995cb4ddd`

The live runtime-binding capture at the scientific execution commit was byte-identical to the
frozen runtime-policy artifact before this authorization was issued. The policy describes the
execution environment and is reused unchanged; it does not carry Stage R scientific scope.

## Owner review finding

The primary discrete Stage R outcomes and matched-GT identities were exact across all ten frozen
sentinel processes. Numerical score payloads and ranked ordering were not globally exact. The
frozen design therefore remains three complete fresh-process primary passes; no Stage R output is
reused.

Stage R Pedestrian threshold sparsity is an observed property at the frozen operating point. It
does not establish whether the cause is score calibration, cross-domain capability, localization,
postprocessing, or another mechanism. The external score threshold remains `0.25`; no tuning, new
threshold, or new S1 metric is authorized. Annotation-conditioned AP remains the preregistered
descriptive view over the frozen retained score-ranked candidate population.

## Authorized scope

The only authorized future workload is:

- mode `primary-pass`;
- logical passes `primary-pass-1`, `primary-pass-2`, and `primary-pass-3`;
- three fresh sequential processes;
- 428 frozen frames per process, with H10 followed by H5 for every frame;
- 856 conditions per process and 2,568 accepted canonical detector calls in total;
- one S1 CUDA process at a time, no other user CUDA workload, and
  `PYTORCH_CUDA_ALLOC_CONF` unset.

`stage-r`, `zero-intensity-pass`, `primary-pass-4`, B2, C2, D2, F2, training, and tuning are not
authorized.

## Operational duration note

Actual Stage R process wall times included substantial fixed process and startup overhead, so the
earlier inference-only estimate understated end-to-end scientific-process duration. The primary
campaign remains operationally practical, but each pass is expected to include several minutes of
binding, revalidation, and model-startup cost plus inference time. This is not a timing acceptance
gate.
