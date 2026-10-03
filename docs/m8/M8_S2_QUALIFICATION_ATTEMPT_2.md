# M8 P1-S2 external qualification attempt 2 — accepted

On 2026-10-03, qualification-only work succeeded on a temporary, sole-GPU NVIDIA A40.
This is a historical structural qualification, not S2 detector execution or a scientific result.

| Identity | Value |
|---|---|
| Pod | `lv0hhtgcx3gbw6` / `laserperception-m8-s2-qual2-2536e5d` |
| Hostname | `77a4f74b7925` |
| GPU UUID | `GPU-d020b940-193f-f906-936c-2f3db7e16a01` |
| GPU memory | 46,068 MiB / 47,708,110,848 bytes |
| Driver | 570.211.01 |
| Execution commit | `2536e5d3eacefa738c6e2faf089262e442c3fab4` |
| DSVT upstream | `8cfc2a6f23eed0b10aabcdc4768c60b184357061` |
| Environment | Python 3.10.12; Torch 2.1.0+cu118; NumPy 1.23.5; spconv 2.3.8; torch-scatter 2.1.2+pt21cu118 |
| Development toolkit | CUDA 11.8; nvcc 11.8.89; GCC/G++ 11.4.0 |

The complete CUDA development pre-build gate passed. All **seven** required upstream extensions
built and imported; the native inventory contains seven in-place modules and seven build-directory
copies. The generated `pcdet/version.py` matched the exact pinned bytes and SHA256
`e5fcccc8123cb08c0a709b59d7ae2991a662ee9579e4fb9a04a953a857353cdd`.

The input gate passed **1,712/1,712**, 428 each for B2/C2/D2/F2. GT-blind qualification accepted
**28/28** frozen XYZIT inputs, **7/7** B2/A2 same-runtime CUDA coordinate identities, and
**7/7** D2/C2 identities. CPU/CUDA candidate-count equality was not required.

| Preserved file | SHA256 |
|---|---|
| Input-gate receipt | `58647610a4c9cd02e8209d244b86495afac3056c18060e62f0aa08c746d5e29b` |
| Runtime policy | `9b60109248b066cd1a38703cedf9f2964b8f0a5750bec831fc63a31369e78db6` |
| Qualification receipt | `d8fe3ed2e72c40a9dfa4e5584eb75435163f065db28f9fcebcb92658d6a5ce49` |
| Evidence archive | `ce0c309e2eaca7d35b9af83684a5c6cbe3731e78fd0e27a3031d6afd2e5ccd2b` |

GT and evaluator loaded: **false**. Detector/scientific calls, engineering sizing, repeatability,
full-corpus execution, and training: **zero / not executed**. Structural CUDA qualification
does not authorize inference.

Private evidence remains unchanged under the canonical Drive-backed project at
`_CLOUD_WORK/m8-s2-runtime-qualification-attempt-2-20261003-lv0hhtgcx3gbw6/`
(local project root `J:\My Drive\laserperception`). All 32 archived remote evidence files were
hash-verified off-Pod; the final local preservation manifest covers 57 files. Private authorization
contents and raw logs are not published.

Pod lifetime was approximately **12.405 minutes**. Estimated incremental GPU cost was $0.10131,
storage $0.00144, total **$0.10275**; this is an estimate, not a provider invoice.
Deletion was requested at `2026-10-03T14:13:04.844Z` and confirmed at
`2026-10-03T14:13:13.000Z`: DELETE returned 204, GET returned 404, and the provider listed no Pods.

The accepted policy and receipt are bound to this **terminated worker** and execution commit.
They are **not portable**. No active runtime exists. Any future sizing or scientific worker must
receive a fresh input gate, runtime binding, GT-blind qualification, and its own explicit owner
authorization. See the [sizing execution handoff](M8_S2_SIZING_EXECUTION.md).
