# M8 S1 external-runtime status

This is the sanitized current operational record for M8 S1. Frozen protocol, Stage R, policy, and
authorization documents retain their historical meaning and are not superseded here.

## Frozen scientific state

- Selected candidate: official pretrained DSVT-Pillar + TransFusion.
- S1 protocol: unchanged.
- Historical retired-machine Stage R: preserved.
- Current future-execution commit:
  `6994d72c3e7691a86116d1417ac3ae08256d163f`.
- Current receipt file SHA256:
  `bef4c55575581aefe8f477e32d1b394f40823a0b0858c66c3ac5c1fae141ec4d`.
- Receipt: 856/856 conditions, mismatch count 0.

The prepared final execution capsule is private Drive state. No private path is published here.

## External-runtime chronology

1. The retired machine completed historical Stage R. Its policy and authorization remain
   machine-bound and non-portable.
2. Runtime migration used RunPod A40 workers. Fresh machine qualification was performed on the
   applicable A40 runtimes.
3. A fresh RunPod A40 Stage R campaign completed 10/10 accepted processes and 140 accepted detector
   calls. The [sanitized Stage R summary](../../benchmarks/m8/diagnostics/external_runtime_stage_r_summary.json)
   records the authorization, runtime, frozen identities, execution timestamps, and reviewed
   evidence-tree SHA256
   `f96eefc02760040df71da947e8e44fb573e1e61f85c4b9d64d0df112d81697a6`.
4. A later primary pass ended `INCOMPLETE` after 779 attempted detector conditions. It yielded
   zero accepted complete primary processes and zero accepted canonical primary calls. Primary
   passes 2 and 3 did not start. Partial output is not a scientific result and may not be spliced.
5. Engineering PRs #41–#45 decoupled Stage R from redundant replay, adopted frame-major primary
   replay, restored the complete live pre-inference revalidation, and made failures before source
   loading, gate completion, or backend construction preserve fail-closed `INCOMPLETE` evidence.
6. The corrected live input gate measured 479.365 s with one worker, 203.146 s with two, and
   121.057 s with four. Four workers were selected, a 3.96× speedup over measured serial. The
   [CPU benchmark record](../../benchmarks/m8/diagnostics/primary_input_revalidation_cpu_benchmark.json)
   preserves the input identity, environment, timestamps, timing boundary, single-observation
   policy, and dirty-tree limitation. These are CPU input-revalidation timings, not DSVT
   throughput or portable hardware guarantees.
7. Several 2026-09-22 A40/A6000 allocation attempts failed before Pod creation. No GPU billing or
   detector calls resulted, and no Pod remains active.

## Current boundary

The primary A2/E2 campaign is still pending. Capacity failures are infrastructure blockers rather
than scientific failures. Zero-intensity, S2, and training have not run. No DSVT accuracy,
PointPillars comparison, training, TensorRT parity, or production claim follows from this record.

The v0.4.0 release advances documentation and platform metadata but does not replace the frozen
execution commit or regenerate its receipt.

The frozen readiness planner is a pre-selection planning artifact: fields such as
`provider_selected=false` and its older “primary calls remain 0” missing-gate text are not the
current operational ledger. Current accounting distinguishes 779 attempted conditions from zero
accepted canonical primary calls. Correcting runtime-source wording is outside this documentation
release because the execution commit remains frozen.
