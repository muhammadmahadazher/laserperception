# M7 inference authorization R2

Status: **PROSPECTIVELY AUTHORIZED FOR THE CORRECTED FROZEN M7 R2 MEASUREMENT.**

**R1 AUTHORIZATION FAILED CLOSED WITH ZERO DETECTOR CALLS.**

**THIS R2 AUTHORIZATION IS A NEW PROSPECTIVE AUTHORIZATION.**

**NO M7 DETECTOR OUTPUT EXISTED BEFORE THE R2 AUTHORIZATION COMMIT.**

PR #21 was reviewed at head `5a8c02e8ba279ee44a8bb87eb2ec2984ca95e729` and accepted into `main` by
normal merge commit `742d8ebeb3a1ac8f248dbdd0a0b4e0ffcc4f990b`. The narrow code correction is
`23792950e1e00bbe7e128b29b73f77c13776cac1`. The reviewed PR head, not the code-only commit or
merge commit, is the R2 measurement-runtime identity.

## Frozen identities

- Scientific protocol: `fd4a143621ffc0692206c100279a9edfd5572d35`
- Scientific/input implementation: `c989f7df5ca8c5ac8148c0ed3a2e91de48b754b2`
- R2 measurement runtime: `5a8c02e8ba279ee44a8bb87eb2ec2984ca95e729`
- Full input ledger: `3,163,158,937` bytes, SHA256
  `577a7ee3da5495611592ca3226a2adefd577fa54821bb859d25892d0cbcbb8ea`
- Corrected full-ledger projection SHA256:
  `efbde2858f0e9780b52cda3b6ae35d8e6d236a8b28b518af695de8a8343e1c92`
- Paired GT SHA256: `0f4ecf564bff30913a0cb35b2043a9a5cd0c8fdb26b220c4cb12072e186f8ba5`
- PointPillars checkpoint SHA256:
  `f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0`
- ONNX SHA256: `61ce22a8ca31498675c32576bfb94f0093d31dc95d2762f7254bf915a59ecc16`
- TensorRT 40k engine SHA256:
  `2e790b1cdbdc1b88c2aafdc81b5921ebee152edd8408158f88437ae4dd1f3e7f`
- Evaluator: `m6b-r2-score-0.25-oriented-bev-iou-0.50`

The checkpoint, ONNX, and engine were rehashed before authorization. The pinned MMDetection3D and
MMDeploy checkout commits remained `fe25f7a51d36e3702f961e198894580d83c4387b` and
`bc75c9d6c8940aa03d0e1e5b5962bd930478ba77`; their required deployment config hashes also matched.
No engine was rebuilt.

## Merged-main no-GPU verification

One fresh process at exact merged `main` commit `742d8ebeb3a1ac8f248dbdd0a0b4e0ffcc4f990b`
stream-hashed and strictly validated all 1,712 ledger records, then used a fresh
`CanonicalM7SourceAdapter` to regenerate five frozen frames across arms B/C/D/F. All 20 conditions
exactly matched their scientific projections, selected-row SHA256, XYZ SHA256, model-ready SHA256,
lag bit patterns, and pillar structures.

- Wall clock: `358.665457872` seconds
- Process peak RSS: `557,400,064` bytes
- External record: `merged_main_no_gpu_preflight.json`, `11,364` bytes, SHA256
  `e56e87d1238dda2ab2fe71fdd61d3dc690ea06733700eda1f7417d977b68425c`
- Environment: Ubuntu 22.04 on WSL2, Linux
  `6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.35`, Python `3.10.12`, NumPy `1.26.4`
- Frozen archive environment recorded separately: Python `3.12.10`, NumPy `2.2.1`

No detector was constructed, and CUDA and TensorRT were not initialized during this verification.
The environment versions are archival provenance; authorization depends on the exact reproduced
scientific bindings and model-ready bytes.

## R1 separation

R1 remains permanently associated with authorization commit
`dc4da349ef94a95ac87e84417fc1e10eb41588f1` and runtime
`5d8cc81653d27ef513b1fd83f98e58793983506e`. It stopped with all 1,712 conditions pending, zero
repeatability calls, zero detector calls, and no M7 result. R2 uses a new branch, authorization,
checkpoint root, progress identity, and output root. No R1 progress is eligible for R2 reuse.

The exact authorization object is
[`m7_inference_authorization.json`](../../benchmarks/m7/preregistration/m7_inference_authorization.json).
It authorizes only the canonical public `run_measurement(...)` path under these frozen identities.
