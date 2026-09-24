# M8 S1 external-runtime status

This is the sanitized supporting ledger for M8 S1 external-runtime work. Current execution state,
authorization boundaries, frozen campaign identities, and blockers are maintained only in
[PROJECT_STATUS.md](../PROJECT_STATUS.md). Frozen protocol, Stage R, policy, and authorization
documents retain their historical meaning and are not superseded here.

## Frozen scientific state

- Selected candidate: official pretrained DSVT-Pillar + TransFusion.
- S1 protocol: unchanged.
- Historical retired-machine Stage R: preserved.
- The live execution commit and receipt identity are recorded in
  [PROJECT_STATUS.md](../PROJECT_STATUS.md).

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
   The linked [repeatability record](../../benchmarks/m8/diagnostics/external_runtime_stage_r_repeatability.json)
   preserves the final 14-condition sentinel order, all ten process identities and artifact hashes,
   per-condition outcomes for both evaluated classes and all three IoU thresholds, primary discrete
   outcome distributions, and score-ranked repeatability statistics.
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
   policy, parent-process memory method, approximate combined working-set observations, and
   dirty-tree limitation. The combined-memory sampling method was not recorded and remains
   explicit null provenance. These are CPU input-revalidation timings, not DSVT throughput or
   portable hardware guarantees.
7. Several 2026-09-22 A40/A6000 allocation attempts failed before Pod creation. No GPU billing or
   detector calls resulted, and no Pod remains active.
8. A later authorized A40 campaign completed three accepted fresh primary processes with 856
   conditions each and 2,568 accepted canonical calls. The accepted raw measurement and exact
   evidence identities are published in [M8_S1_MEASUREMENT_RAW.md](M8_S1_MEASUREMENT_RAW.md).
   The earlier 779-condition attempt remains incomplete with zero accepted calls.

## Scope boundary

This supporting ledger does not grant scientific execution authority or supersede the canonical
status page. The accepted A2/E2 raw measurement is descriptive; no winner, architecture-causality,
training, TensorRT parity, statistical-significance, or production claim follows from this
external-runtime chronology. The frozen readiness planner remains a
pre-selection planning artifact rather than a live operational ledger.
