# M7 binding-gate failure R1

Status: **CASE A — RUNTIME-BINDING METADATA REGRESSION ONLY. INFERENCE BLOCKED.**

The first authorized M7 session used authorization commit
`dc4da349ef94a95ac87e84417fc1e10eb41588f1`, bound to reviewed runtime
`5d8cc81653d27ef513b1fd83f98e58793983506e`. It stopped on the first condition,
`2011_09_26_drive_0001/0000000010|H10_LAG_COMPRESSED`, with zero repeatability calls, zero detector
calls, and zero completed conditions. No M7 result existed. This was a measurement-runtime binding
failure, not a detector failure.

The preserved external `failed_progress_identity.json` is `247,229` bytes with SHA256
`5459ef97e3225d68962105bbed2384926fcd104ea5fe16b5483ac93e85358983`. It contains the exact
protocol, scientific implementation, reviewed measurement runtime, ledger, engine, checkpoint,
ONNX, and evaluator identities; all 1,712 condition states are `PENDING` and none is `COMPLETE`.

## Diagnosis

At scientific implementation `c989f7df5ca8c5ac8148c0ed3a2e91de48b754b2`, the explicit
`BOUND_RECORD_FIELDS` tuple contained 24 fields and did not include `runtime_versions`. PR #20
derived its runtime projection from every archival `REQUIRED_CONDITION_FIELDS` member, silently
adding `runtime_versions` to scientific equality. Regenerated records contain the current Python
and NumPy versions, so otherwise identical inputs from another environment could not bind.

A no-GPU diagnostic stream-hashed and validated the unchanged 3,163,158,937-byte ledger, then
regenerated only Arm B for the failed frame through `CanonicalM7SourceAdapter`. The complete
diagnostic JSON is external, `30,068` bytes, SHA256
`ccd34d40bc7e4e6cfdc73c04d79caae3d5d6606a0e62314b71159d0efcf7a0a5`. No torch, TensorRT,
MMDeploy, or MMDetection3D module was loaded.

The only differing field was `runtime_versions`:

| Provenance | OS | Python | NumPy |
|---|---|---|---|
| Authorized input freeze | `Windows-11-10.0.26200-SP0` | `3.12.10` | `2.2.1` |
| WSL diagnostic | `Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.35` | `3.10.12` | `1.26.4` |

Environment differences are provenance, not evidence of harmless arithmetic differences. They are
allowed during regeneration only when every frozen scientific binding and the authoritative
model-ready bytes reproduce exactly. Any scientific field or byte difference still stops the gate.

## Complete field comparison

Each digest is SHA256 over the field's canonical JSON representation.

| Field | Equal | Authorized digest | Regenerated digest |
|---|---:|---|---|
| `arm` | yes | `cf03143e507c853fed0abfbb88063a0feec2cb86813bd94c53864f1969b3bd48` | same |
| `condition_id` | yes | `d37a0f8fc3baf81a46c35627a5dfdee5822d053710b7aba9bee43a87fb31cfbe` | same |
| `drive_id` | yes | `a1ba221265e552f191c3a1371f77814327195882d66b06915909a800f20031ef` | same |
| `f_history_ranks` | yes | `38e0b9de817f645c4bec37c0d4a3e58baecccb040f5718dc069a72c7385a0bed` | same |
| `frame_index` | yes | `917df3320d778ddbaa5c5c7742bc4046bf803c36ed2b050f30844ed206783469` | same |
| `generation_commit` | yes | `ed7dd7d0af27018de9e373f84fde410f0bb7df71a2900f87e33cd57d24b2971b` | same |
| `lag_bit_patterns` | yes | `9413b9753d20dbe50774ec3051314bae2e7ed77e6c9d14cd7e5466cb0361fda4` | same |
| `lag_scale_provenance` | yes | `ee7614a8d08ac959c3f716e64d1d435348cf740a31007a1cdfb65a39b01bc9da` | same |
| `lag_span_seconds` | yes | `61ff92298f8169a0db3f4b68509965ac2ee7569c55b11173188fecb825764680` | same |
| `lag_support_count` | yes | `25d4f2a86deb5e2574bb3210b67bb24fcc4afb19f93a7b65a057daa874a9d18e` | same |
| `model_ready_sha256` | yes | `b72bbf6d8fbcda4f73126fd3dacd4541ea39015076f2f1f57640751105bc9d2b` | same |
| `per_sweep_point_counts` | yes | `617f5f6d353f3e5d52db24728f2d02f09aa7d6a7896a948cd2277bcf6bb005df` | same |
| `pillar_structure` | yes | `e31f1e262c853cf6f2c5c4a3a5351680fc6ec09bf014d48a9a1970d2a32045e7` | same |
| `point_count` | yes | `3930760b9e8bdb8b445273c014f19a11d757b6825752d4c78e0401839bac2a7b` | same |
| `provenance_schema` | yes | `c323f2c2b7e9cf39045ad2eefe7bb75fd2766dc98d02665f55575a84f2275050` | same |
| `quota_provenance` | yes | `38e0b9de817f645c4bec37c0d4a3e58baecccb040f5718dc069a72c7385a0bed` | same |
| `rank_source_identities` | yes | `5ee14dd7d67fc6df0738e42907fd3eac6c7c81e642c1b338d21453a13ec5fb12` | same |
| `rank_to_lag_bit_pattern` | yes | `d9c0a20c3acea155a707d2694cf6322cff546962682a8d04105c8cfe73822030` | same |
| `runtime_versions` | **no** | `0100b079556dd8e3cea4b170b57f0fe5f283d829e70d4cb8493c06e841ab1491` | `a91fd7e3a0bded147e8d83c535494b9e1371fdb483b554bdf81d136ef37fb5fe` |
| `seed_provenance` | yes | `38e0b9de817f645c4bec37c0d4a3e58baecccb040f5718dc069a72c7385a0bed` | same |
| `selected_row_sha256` | yes | `10109942b08e6cd5794690a19c9d881b0009c35dfb6a4ddf1957bf90dc1ead3f` | same |
| `source_a_sha256` | yes | `dbe13922a1263042c5946ab219ce3a4ac530a0fbd203feac9c19cf13796ee79f` | same |
| `source_e_sha256` | yes | `45ecb94f6127312862066f71c5d86eecd8e69e4e692cb0d0fb8fbcd2f99ebed4` | same |
| `sweep_ids` | yes | `c4ec3e727f5c4873d605e70d90d942ff97af447d8f9ac1a864be771f79db7fd4` | same |
| `xyz_sha256` | yes | `1d0b8835e7211732aabb3b47b167b821640ae9b00150ab35e8e3670947cd11ac` | same |

Important exact values were identical:

- Source A SHA256: `5ff825de4c351961f62b416c11042d50bf5d78f2d363f842ce4b5d182456b18a`
- Source E SHA256: `52d8a54a02bc04d85b22373934aae0051c6d1ebf64b44af219518e1d334a2f99`
- XYZ SHA256: `9b55fa90bd1bcfa432c89856538853e9d0c74da147c7e976dbe8049216eb7daa`
- Model-ready SHA256: `fac7242e920c93709a186757782d68b188747de2f672d83914a1346e231d40d9`
- Selected-row SHA256: `03dad10f891cff48365cf26972f08d52e87921af34117337309ac7d3feb8bc6a`
- Shape/dtype: `(1,312,220, 4)`, little-endian IEEE-754 float32
- Differing float32/XYZ/lag values: `0 / 0 / 0`
- Maximum absolute XYZ/lag difference: `0.0 / 0.0`
- First differing row, column, or float32 bits: none

All eleven rank lag patterns were exact, ranks 0 through 10 respectively:
`0x00000000`, `0x3d5321c0`, `0x3dd31bf0`, `0x3e1e5340`, `0x3e531718`,
`0x3e83eee8`, `0x3e9e5278`, `0x3eb8b4c8`, `0x3ed318dc`, `0x3eed7bf4`,
`0x3f03efb0`. `T5_f32` was `0x3f03efb0`, `T10_f32` was `0x3f83f078`, and the binary64
scale was `0x3fdfffcf7e2be031`; no arithmetic boundary differed.

## Corrective policy

`runtime_versions` remains required and strictly schema-validated in every archival condition. It
remains in the frozen ledger and input-freeze provenance. The scientific runtime projection is
restored to the explicit historical `c989…` 24-field tuple; it is no longer derived from the
archival required-field set.

The PR #20 selected-ordinal policy is unchanged: complete arrays remain present in the frozen
ledger, are parsed and validated after full-file SHA verification, and are discarded only from the
resident projection. Their exact row population remains bound by the archival ledger SHA,
`selected_row_sha256`, and `model_ready_sha256`.

The failed authorization remains permanently bound to runtime `5d8cc816…` and must not be reused.
A future measurement requires a new runtime identity and a new v2 authorization artifact.

## Corrected-candidate preflights

The corrected measurement-runtime candidate is
`23792950e1e00bbe7e128b29b73f77c13776cac1`. Two fresh-process, no-GPU preflights each
stream-hashed and validated the complete unchanged ledger, created a fresh
`CanonicalM7SourceAdapter`, and regenerated the five frozen sentinels across arms B/C/D/F (20
conditions per run). Both runs exactly matched the scientific projection plus each explicitly
reported `selected_row_sha256`, `xyz_sha256`, `model_ready_sha256`, `lag_bit_patterns`, and
`pillar_structure` value.

| Run | Conditions | Wall clock (s) | Peak RSS (bytes) | External record | SHA256 |
|---:|---:|---:|---:|---|---|
| 1 | 20/20 exact | `356.951063453` | `556,957,696` | `sentinel_binding_preflight_1.json` (`11,297` bytes) | `8162c2867878fc34e555e8eec68f9784a4939f8035f62a2340cabc8cab8ab867` |
| 2 | 20/20 exact | `355.473328778` | `548,306,944` | `sentinel_binding_preflight_2.json` (`11,292` bytes) | `3f8cdebe24c251b7c916807549779eedfc5a0e5e40809b73adf4f343288fecea` |

Both full-ledger projections produced SHA256
`efbde2858f0e9780b52cda3b6ae35d8e6d236a8b28b518af695de8a8343e1c92`, and both sets of
per-condition comparison records were identical. The frozen archive recorded Python `3.12.10`
and NumPy `2.2.1`; regeneration recorded Python `3.10.12` and NumPy `1.26.4`. These environment
versions were reported separately and did not enter scientific equality.

No detector was constructed, no detector call occurred, no CUDA or TensorRT initialization
occurred, no inference authorization was created, and the frozen ledger was not regenerated.
