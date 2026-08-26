# M7 implementation candidate review

Status: **IMPLEMENTATION CANDIDATE FOR OWNER REVIEW.**
**NO REAL M7 INPUT OR DETECTOR OUTPUT GENERATED.**

This document describes a CPU-only implementation candidate for the frozen protocol. It records no
M7 scientific result and authorizes neither real B/C/D/F input construction nor detector inference.

## Identities and chronology

- Frozen scientific protocol commit:
  `fd4a143621ffc0692206c100279a9edfd5572d35`
- Source protocol draft commit:
  `7700216c234c0c4bf908dba6ab5a7106e730a627`
- Post-freeze editorial consistency commit:
  `ac86412e504fc2ddf5ce4b549565a969aba56898`
- Candidate implementation identity: the implementation commit containing this document, recorded
  as the final head of the implementation PR. It is not frozen before owner review and merge.

The editorial commit changed only the stale description of the already-approved C selection rule.
It did not change seed material, hashing, SplitMix64, quotas, arms, thresholds, or metrics.

## Code map and protocol mapping

| Frozen responsibility | Implementation |
|---|---|
| Protocol constants, A–F identities, corpus and B/C/D/F order | `benchmarks/m7/protocol.py` |
| Canonical float32/uint64 bytes, JSON hashes, external-asset checks, row provenance | `benchmarks/m7/provenance.py` |
| Pure B/C/D/F transformations, integer quotas, SHA256 seed and SplitMix64 | `benchmarks/m7/interventions.py` |
| Existing exact-fast pillar audit and B/A, C, D/C, F/A stop gates | `benchmarks/m7/structural_validation.py` |
| Frozen M6 paired-set parser and deterministic future input-ledger writer | `benchmarks/m7/evidence.py` |
| Authorization barrier, repeatability, checkpoints, ordering and metrics | `benchmarks/m7/execution.py` |
| Detector-free future input-only orchestration | `benchmarks/m7/prepare_inputs.py` |
| Authorization-first future measurement boundary | `benchmarks/m7/run_measurement.py` |

The primary matcher delegates to the unchanged `laserperception.evaluation.kitti_m6b.match_detections`
implementation with score 0.25 and oriented-BEV IoU 0.50. Pillar structure delegates to the existing
M6 `analyze_pillars` exact-fast audit. The implementation does not fork detector postprocessing,
GT conversion, FOV, neighbour-ignore, matching, or voxelization semantics.

## Canonical bytes and sweep provenance

- Model-ready arrays are normalized to C-contiguous little-endian IEEE-754 float32 with shape
  `(N, 4)` and `x, y, z, time_lag` order before hashing.
- Selected global A row identities are C-contiguous little-endian uint64.
- Seed text is encoded as UTF-8 without a terminator.
- Sweep provenance explicitly carries current/history rank, source sweep ID, zero-based ordinal in
  the range-filtered source sweep, and zero-based global A row index.
- Current lag must have the exact positive-zero `0x00000000` bits. Historical lag must be nonzero,
  and A must expose current plus ranks 1–10 in nearest-to-oldest order.
- Noncontiguous views and non-native-endian float32 inputs normalize to the same canonical bytes;
  malformed routes fail closed.

## Arm B numerical path

`derive_lag_scale` selects `T10_f32` and `T5_f32` from existing float32 lag values, converts them
exactly to binary64, and divides in binary64. `_apply_lag_scale` performs one binary64 multiply per
row and one final float32 cast, then writes current rows to positive zero. It rejects nonfinite or
zero extrema, malformed support, nonpositive scale, sign changes, and float32 support collapse.
XYZ, row order, membership, and provenance remain unchanged.

The synthetic known-answer fixture freezes scale `0x1.0000000000000p-1`, binary64 bits
`0x3fe0000000000000`, and hard-coded float32 output bits rather than computing expected values with
the production helper.

## Arm C integer and SplitMix64 path

`allocate_quotas` uses unbounded Python integer products, division, and remainders. Allocation order
is descending remainder then ascending rank, with exact sum/bounds checks. No floating-point quota
arithmetic exists.

`seed_text` freezes:

```text
laserperception-m7-c-v1|<drive>|<10-digit-frame>|<history-rank>
```

`seed_identity` takes the first eight SHA256 digest bytes as unsigned big-endian uint64.
`splitmix64_key` uses the frozen three constants and masks every stage to 64 bits.
`select_lowest_ordinals` sorts by `(key, ordinal)`, then `construct_c` restores global A row order.

The hard-coded known answer for drive `2011_09_26_drive_0001`, frame 10, rank 1 is:

- seed text bytes:
  `laserperception-m7-c-v1|2011_09_26_drive_0001|0000000010|1`
- SHA256: `66b9030d4f6a151ec1bed274fbd8ec3f85ec46a62a5e7d4353d8c5a063215ed7`
- seed: `0x66b9030d4f6a151e`
- keys for ordinals 0, 1, 2, and 9:
  `0x7ca05f4081ce1706`, `0x9a0bc31c44bd2c7b`, `0xe930618898348e1a`,
  `0x2f8939f2beee0dc0`
- lowest three ordinals over the hard-coded 0–9 key vector: `5, 9, 7`

Tests also cover zero/full targets, largest-remainder ties, zero quotas, a small sweep, forced equal
keys, duplicate/out-of-range rows, exact subset copies, and final global order.

## Arms D and F

`construct_d` accepts a `CResult`; it has no selection parameters and does not call selection logic.
It reuses C's selected-row identity and applies the exact B scale. D/C validators require identical
row vectors, selected-row SHA, XYZ, coordinate order, candidate order, and retained selection.

`construct_f` retains complete current and history ranks exactly `(2, 4, 6, 8, 10)` in A global
order. It does no thinning, lag remapping, or row mutation. Tests require rank 1 exclusion, ranks 2
and 10 inclusion, complete selected sweeps, and exact source bytes. No code or test claims F/E is a
physical-span-only comparison.

## Structural validation

`PillarStructure` derives candidate and retained identities through the existing M6 CPU
`analyze_pillars` helper. Validators fail closed for:

- B/A row, XYZ, provenance, coordinate, candidate-order, or retained-selection difference;
- C count, current retention, uniqueness, subset, native-lag, or global-order difference;
- D/C row, selected-row SHA, XYZ, coordinate, candidate-order, or retained-selection difference;
- F/A rank, complete-sweep, or byte-copy difference.

Only synthetic arrays and existing non-M7 CPU helpers were used during implementation review.

## Paired sets and frozen evaluator

The parser reads only the verified published v0.3.0 full M6b result and partitions the canonical
`(drive_id, frame_index, GT track ID)` keys in lexicographic order. Local parser validation recovered
the published M6 cardinalities exactly:

| Class | Shared | E-only | A-only | Neither |
|---|---:|---:|---:|---:|
| Car | 16 | 32 | 0 | 18 |
| Pedestrian | 204 | 64 | 15 | 113 |

This is validation of existing M6 evidence, not observation of an M7 outcome. No paired-set
preregistration artifact was written.

## Input-only ledger and barrier

The future compact ledger records frame/arm/generation identities, A/E commitments, point and byte
hashes, selected rows, lag bits/support/span, sweep identities/counts, pillar identities/counts,
B/D scale provenance, C/D quota/seed provenance, F ranks, and runtime versions. Serialization is
canonical and atomic. Duplicate, missing, malformed, noncanonical, shortened, or reordered
conditions are rejected.

`prepare_input_freeze` streams exactly 428 source frames and constructs exactly 1,712 B/C/D/F
conditions. It verifies the published full-asset hashes, corpus and ordered A/E commitments, every
per-frame source hash, and canonical order. The module imports neither TensorRT nor MMDeploy and has
no detector factory. It was implemented but not run on KITTI.

## Inference authorization and detector safety

The future runner requires an owner-created authorization object binding protocol commit,
implementation commit, input-ledger SHA256, engine, checkpoint, ONNX, evaluator, and an exact true
authorization boolean. It then hashes the actual external files and verifies the evaluator identity.
Only after both checks pass can the injected detector factory be called. Tests prove missing, false,
or mismatched authorization and mismatched artifact files stop before the factory invocation. No
authorization artifact was created here.

## Repeatability and checkpoint/resume

The repeatability implementation fixes the five sentinels, arms B/C/D/F, and ten calls. It compares
the three raw tensor identities and DetectionFrame identity exactly. Repeat 1 becomes canonical only
after all ten agree; there is no eleventh call. Mock tests cover exact pass and one changed raw or
DetectionFrame hash.

Atomic checkpoints bind protocol, implementation, input ledger, engine, checkpoint, ONNX,
evaluator, condition input, raw outputs, and DetectionFrame. Resume requires exact identity,
preserves canonical order, refuses malformed or duplicate completed conditions, and never deletes a
completed record. The canonical constructor refuses any corpus other than all 1,712 conditions;
synthetic unit tests use an explicitly private fixture-only path.

## Test scope and safety confirmation

The new tests use tiny synthetic arrays, synthetic sweep provenance, mock detector hashes, temporary
JSON files, and the already-published M6 full asset only for the permitted local parser validation.
They cover canonical bytes, B arithmetic, C quotas and SplitMix64, D reuse, F selection, structural
relations, paired parsing, ledger serialization, authorization/lazy initialization, repeatability,
checkpoint/resume, canonical order, and metric arithmetic.

No real B, C, D, or F model-ready input was generated. No M7 input ledger, paired-set
preregistration, repeatability record, detector output, or result file was created. TensorRT was not
imported or initialized, and no detector inference, training, tuning, GPU test, or ROS test occurred.
