# M7 input-only freeze

Status: **M7 INPUT-ONLY FREEZE CANDIDATE.**

**NO M7 DETECTOR OUTPUT EXISTS.**

**INFERENCE IS NOT AUTHORIZED.**

This record freezes the complete B/C/D/F input corpus for the controlled history-mechanism study.
It contains input structure only. It does not contain a detector observation, scientific outcome,
or permission to initialize the detector runtime. The scientific design remains the frozen
[M7 protocol](M7_PROTOCOL.md).

## Frozen identities

- Protocol freeze commit: `fd4a143621ffc0692206c100279a9edfd5572d35`
- Canonical implementation commit:
  `c989f7df5ca8c5ac8148c0ed3a2e91de48b754b2`
- Accepted PR #18 main merge commit:
  `cc938136e11796932ba25821d291428ac481bed1`
- Pre-generation implementation-freeze documentation commit:
  `9c96852e1b7e0225f36c886cbafce9606549b86c`
- v0.3.0 tag target, unchanged:
  `7e2a68dda394b12027d2c04864a4ae63cde2e338`

The implementation identity remains the exact reviewed implementation commit. The merge commit
records its accepted integration on `main`; it does not replace `implementation_commit`.

The two existing v0.3.0 M6 sources were verified before generation:

| Existing M6 asset | Bytes | SHA256 |
|---|---:|---|
| `pre_inference_input_ledger_full.json` | 5,837,452 | `e25b3d62113cc7e8c1fcf736caa68b1ab698f965f007c758ff91d3e498ca6caa` |
| `kitti_raw_cross_domain_characterization_full.json` | 41,987,113 | `87870b2aa0cc2a91d39331afc8154fdad0c8c796f1cabfb4f8530a3eb106de27` |

The canonical source adapter validated the same external KITTI Raw `2011_09_26` date root and the
frozen `2011_09_26_drive_0001_sync` and `2011_09_26_drive_0091_sync` sources. No dataset path or
point payload is present in tracked evidence.

## Corpus and source gate

The detector-free canonical `prepare_input_freeze` entrypoint reconstructed A/H10 and E/H5 from
the frozen M6 transforms and generated, in canonical frame-then-arm order:

| Input | Exact/source or structural gate | Conditions |
|---|---|---:|
| A/H10 source | model-ready SHA equals frozen M6 H10 | 428/428 |
| E/H5 source | model-ready SHA equals frozen M6 H5 | 428/428 |
| B | A rows, XYZ, provenance, sweep population, and pillar structure; lag only changed | 428/428 |
| C | E point count; current retained; historical A subset; native lag/order; deterministic quota/seed | 428/428 |
| D | C rows, selected-row SHA, XYZ, and pillar structure; lag only changed | 428/428 |
| F | complete ranks current + 2,4,6,8,10; native lag and A order | 428/428 |

There are 428 conditions in each of B, C, D, and F: 1,712 newly frozen inputs in total. C had no
condition with a zero historical quota, so no history rank reached zero. F retains the rank-10
endpoint but skips intervening ranks; it does not isolate physical temporal span.

The factorial structure passed for all 428 frames:

- A to B: lag only, with the native A/B point population identical;
- A to C: point population only;
- C to D: lag only on identical selected rows, with C/D point populations identical.

## Full ledger and public evidence

The full canonical authorization input is one indivisible external ledger:

- Logical filename: `m7_input_ledger_full.json`
- Bytes: `3,163,158,937`
- SHA256: `577a7ee3da5495611592ca3226a2adefd577fa54821bb859d25892d0cbcbb8ea`
- Storage: external under the ignored local evidence area; not tracked because it exceeds 5 MiB

The repository's `load_strict_input_ledger` accepted the exact file, enforcing its schema,
protocol/implementation identities, unique complete 1,712-condition corpus, B/C/D/F-only arm set,
and canonical order. A future owner authorization must bind this exact full-ledger SHA; the compact
manifest is not a substitute.

Tracked input-only evidence:

| Artifact | Bytes | SHA256 | Purpose |
|---|---:|---|---|
| [Compact input manifest](../../benchmarks/m7/preregistration/m7_input_manifest.json) | 1,151,365 | `8d4f74d783950d24956239f3a67a7a58fe10013e0e83a88d0f8b23e3139ffe90` | Public per-condition hashes/counts in canonical order |
| [Input characterization](../../benchmarks/m7/preregistration/m7_input_characterization.json) | 7,569 | `b30cb8f71ad3b060267f52f21fb3aad7db43735b4b9162c0394c59546c9dd30f` | Pre-inference A/B/C/D/E/F structure |
| [Paired GT sets](../../benchmarks/m7/preregistration/m7_paired_gt_sets.json) | 34,281 | `0f4ecf564bff30913a0cb35b2043a9a5cd0c8fdb26b220c4cb12072e186f8ba5` | Existing-M6 pose partitions for later preregistered evaluation |

No tracked artifact contains raw KITTI points, dataset binaries, private paths, model/deployment
assets, credentials, or detector values.

## Input-only characterization

These are structural properties observed before inference. They are not detector findings.

| Arm | Mean points | Median points | Lag supports | Mean candidate pillars | Overflow frames | Discarded pillars total |
|---|---:|---:|---:|---:|---:|---:|
| A | 1,330,654.35 | 1,334,420.0 | 11 | 31,192.08 | 68 | 124,696 |
| B | 1,330,654.35 | 1,334,420.0 | 11 | 31,192.08 | 68 | 124,696 |
| C | 726,560.59 | 728,279.0 | 11 | 28,348.96 | 0 | 0 |
| D | 726,560.59 | 728,279.0 | 11 | 28,348.96 | 0 | 0 |
| E | 726,560.59 | 728,279.0 | 6 | 25,319.65 | 0 | 0 |
| F | 725,750.47 | 727,854.5 | 6 | 26,744.39 | 0 | 0 |

- B's frozen lag scale ranged from `0.4995792092` to `0.5002258225`, with median
  `0.5000120752`; exact A XYZ identity passed 428/428.
- C matched E point count 428/428 and retained a median `0.5457946999` of A's points.
- D matched C selected-row SHA and XYZ 428/428.
- F's median point-count ratios were `0.5454415887` versus A and `0.9993124558` versus E. Its lag
  span equaled A for 428/428 frames and exceeded E for 428/428 frames because rank 10 is retained.

The characterization JSON records the complete requested min/median/mean/max distributions,
per-rank C retention, lag spans/supports, candidate/retained pillars, overflow, and discarded
pillar counts.

## Frozen paired GT partitions

The tracked preregistration was derived only from the already-published M6b result, using the
lexicographically ordered `(drive_id, frame_index, GT track ID)` key:

| Class | Shared | E-only | A-only | Neither |
|---|---:|---:|---:|---:|
| Car | 16 | 32 | 0 | 18 |
| Pedestrian | 204 | 64 | 15 | 113 |

It contains the actual pose keys and per-partition hashes. No M7 result was consulted.

## Fresh-adapter replay

After the full ledger was frozen, a fresh `CanonicalM7SourceAdapter` regenerated B/C/D/F for the
five frozen sentinels (`0001/10`, `0001/83`, `0001/11`, `0001/15`, and `0091/10`). Results:

- model-ready SHA256 identity: 20/20 exact;
- every bound input-ledger field: 20/20 exact;
- detector calls: zero.

This was input regeneration only, not detector repeatability.

## Operational generation record

- Canonical full-generation wall clock: `7,185.64733 s` (`01:59:45.647`)
- Full ledger bytes: `3,163,158,937`
- Python: `3.12.10`
- NumPy: `2.2.1`
- Runtime: `Windows-11-10.0.26200-SP0`
- Exact in-process peak RSS: not captured by the dependency-free wrapper
- OS-native monitoring observed a high-water mark of approximately `9,154 MiB`; this is an
  approximate operational observation, not a benchmark result
- Fresh-adapter five-sentinel replay wall clock: `184.9107445 s`

These resource figures describe one evidence-generation operation and are not performance claims.

## Owner barrier

No `inference_authorization.json` or equivalent authorization exists. During this task:

- `CanonicalM7Detector` was not constructed;
- `M2Backend` was not initialized for inference;
- CUDA, TensorRT, and MMDeploy detector execution were not initialized;
- no repeatability inference, raw network tensor, `DetectionFrame`, checkpoint, detector summary,
  or M7 result was produced;
- no training or tuning occurred;
- no protocol, accepted M7 implementation, M6 evidence, or v0.3.0 release identity changed.

The next step, if the owner approves these frozen inputs, is a separate explicit inference
authorization task. Until then: **INFERENCE IS NOT AUTHORIZED.**
