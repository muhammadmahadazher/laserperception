# M7 measurement-runtime streaming amendment

Status: **MEASUREMENT-RUNTIME AMENDMENT CANDIDATE.**

**NO INFERENCE AUTHORIZATION EXISTS. NO M7 DETECTOR OUTPUT EXISTS.**

This prospective amendment addresses only the no-GPU ledger-loading resource failure recorded in
[M7 inference preflight failure](M7_INFERENCE_PREFLIGHT_FAILURE.md). The exact frozen scientific
protocol remains `fd4a143621ffc0692206c100279a9edfd5572d35`, and the scientific/input-generation
implementation remains `c989f7df5ca8c5ac8148c0ed3a2e91de48b754b2`.

## Frozen archival input

The amendment does not regenerate, reserialize, split, trim, or replace the full ledger:

- Logical filename: `m7_input_ledger_full.json`
- Bytes: `3,163,158,937`
- SHA256: `577a7ee3da5495611592ca3226a2adefd577fa54821bb859d25892d0cbcbb8ea`

The full-file SHA remains the authorization identity. The compact manifest is not a substitute.

## Failure and bounded-memory architecture

The original loader retained the complete UTF-8 source and a much larger Python object graph,
including all C/D `selected_ordinals`. It was OOM-killed before authorization on the first strict
preflight. The amended canonical path instead:

1. verifies the exact file byte count and streams its complete SHA256 before parsing;
2. parses with `ijson==3.5.1`, explicitly importing backend `yajl2_c`;
3. selects `use_float=False`, so YAJL emits exact `Decimal` values for non-integer numbers, then
   deterministically converts those values to Python binary64 floats to reproduce standard-library
   `json.loads` numeric behavior while preserving unsigned 64-bit seed integers;
4. validates every complete condition dictionary, one at a time, with the full frozen schema and
   canonical sequence;
5. converts each validated condition to a compact canonical runtime projection; and
6. discards the full condition dictionary and parser intermediates before advancing.

The small-fixture whole-file loader remains only as a test oracle. The canonical measurement path
uses the streaming loader and does not call `Path.read_text`, whole-file `json.load`, or whole-file
`json.loads`.

## Archival ledger versus runtime projection

The immutable runtime projection retains these top-level condition fields exactly:

`arm`, `condition_id`, `drive_id`, `f_history_ranks`, `frame_index`, `generation_commit`,
`lag_bit_patterns`, `lag_scale_provenance`, `lag_span_seconds`, `lag_support_count`,
`model_ready_sha256`, `per_sweep_point_counts`, `pillar_structure`, `point_count`,
`provenance_schema`, `quota_provenance`, `rank_source_identities`,
`rank_to_lag_bit_pattern`, `seed_provenance`, `selected_row_sha256`,
`source_a_sha256`, `source_e_sha256`, `sweep_ids`, and `xyz_sha256`.

`runtime_versions` remains a required, validated archival provenance field. After the first
authorized session exposed the metadata-binding regression, the scientific equality projection was
restored to the explicit historical `c989…` field set; see
[M7 binding-gate failure R1](M7_BINDING_GATE_FAILURE_R1.md). Different recorded environments are
permitted only when every frozen scientific binding and the exact model-ready bytes reproduce.

For each C/D seed, the retained `seed_provenance` keeps `history_rank`, `seed_text_utf8`, `sha256`,
`seed_uint64`, and `seed_uint64_hex`. The complete `selected_ordinals` lists remain in the frozen
archival ledger and are fully parsed and validated for schema, count, type, and source-sweep range.
They are omitted only from the post-validation resident projection.

This omission does not weaken exact row binding. The archival full-file SHA commits every ordinal.
Before inference, the unchanged canonical source adapter must independently regenerate the frozen
quota/seed intervention and match both `selected_row_sha256` and `model_ready_sha256`; the exact
same read-only array is then hashed again immediately before the detector call. The projection is a
runtime index derived from an already-authenticated full file, not an independently authorizable
scientific ledger.

The complete ordered projection is given a deterministic SHA256 over the canonical JSON array of
all 1,712 compact projections. It retains direct condition lookup without retaining source text,
full condition dictionaries, ordinal arrays, or parsed event history.

## Prospective runtime identity contracts

The scientific implementation identity is unchanged. A separate future
`m7_measurement_runtime_commit` identifies the exact parsing, binding, detector-construction, and
resume code. Inference authorization schema
`laserperception.m7.inference-authorization.v2` must bind both identities plus the protocol,
full-ledger SHA, checkpoint, ONNX, engine, evaluator, and explicit authorization boolean.

Future progress and condition checkpoints use schema v2 and include
`measurement_runtime_commit`. No M7 detector checkpoint currently exists, so no migration is
needed.

## Exact real-ledger preflights

The exact runtime candidate `a9c047e4cab37edee62ba577d392dc47b1588a20` was tested twice in
separate fresh Python processes against the unchanged frozen full ledger. Both runs used Python
3.10.12 under WSL2 Linux and returned the complete compact index:

| Run | Conditions | Wall clock (s) | Peak RSS (bytes) | Deep retained index (bytes) | Projection SHA256 |
|---:|---:|---:|---:|---:|---|
| 1 | 1,712 | 327.357896657 | 108,072,960 | 11,772,395 | `8ec32bcee88f2a624f5083e7f93cb06a216f73f1708492efc2730c0428a98533` |
| 2 | 1,712 | 321.320522443 | 108,568,576 | 11,772,395 | `8ec32bcee88f2a624f5083e7f93cb06a216f73f1708492efc2730c0428a98533` |

The first condition was
`2011_09_26_drive_0001/0000000010|H10_LAG_COMPRESSED`; the last was
`2011_09_26_drive_0091/0000000339|H10_ALTERNATE_FULL_SPAN`. Each process independently verified
the 3,163,158,937-byte file and SHA256 before parsing. Projection SHA equality passed. Peak RSS is
the exact Linux `resource.getrusage(RUSAGE_SELF).ru_maxrss` high-water value; retained index size is
an approximate recursive `sys.getsizeof` total with object-ID deduplication.

The compact tracked record is
[m7_streaming_ledger_preflight.json](../../benchmarks/m7/preregistration/m7_streaming_ledger_preflight.json).
The two complete 1,511-byte process records remain external with SHA256 values
`fabdb48a88cb147b336d49f0893c81caa4d116a328e2234053329dcd526dcbe2` and
`6f452f6ef4697465f3201a645bada28d3c03d16b5ec110ecd583985366c5b308`.

## Current execution barrier

The bounded-memory loader preflight passed, but inference remains unauthorized. This task did not
construct `CanonicalM7Detector` or `M2Backend`, initialize CUDA/TensorRT, call the detector, create a
checkpoint, or observe an M7 result. A later owner authorization must bind both the unchanged
scientific implementation identity and the reviewed measurement-runtime commit through
authorization schema v2.
