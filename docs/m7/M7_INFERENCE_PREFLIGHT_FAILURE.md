# M7 inference preflight failure

Status: **MEASUREMENT-RUNTIME RESOURCE FAILURE; INFERENCE NOT AUTHORIZED.**

On 2026-08-27, after the M7 input freeze was merged at
`227f1604466d2ee22ccdcb1ad0067fd8f374ab3f`, the first required no-GPU inference preflight
attempted to load and strictly parse the exact frozen full ledger.

## Frozen input identity

- Logical filename: `m7_input_ledger_full.json`
- Bytes: `3,163,158,937`
- SHA256: `577a7ee3da5495611592ca3226a2adefd577fa54821bb859d25892d0cbcbb8ea`
- Scientific/input-generation implementation:
  `c989f7df5ca8c5ac8148c0ed3a2e91de48b754b2`

The file identity was exact. The ledger was not malformed, regenerated, rewritten, split, or
substituted.

## Failure

The original `load_strict_input_ledger(...)` path read the complete approximately 3.16 GB UTF-8
JSON source and passed the resulting value to whole-file `json.loads(...)`. It then attempted to
retain all 1,712 complete condition dictionaries in a `StrictInputLedger`. The source includes
large C/D seed-provenance `selected_ordinals` lists, so the whole-file text and expanded Python
object graph substantially exceeded the on-disk size.

The fresh-process preflight environment and observation were:

- Python: `3.10.12`
- OS: WSL2 Linux
- WSL memory allocation: approximately `7.6 GiB` RAM and `2.0 GiB` swap
- Wall clock before termination: approximately `89.7 s`
- Kernel-observed anonymous RSS at termination: `7,461,628 KiB` (approximately `7.12 GiB`)
- Result: the operating system OOM-killed the Python process before strict parsing completed

No inference authorization artifact existed. The canonical detector was not constructed;
`M2Backend`, CUDA, TensorRT, and detector inference were not initialized. No repeatability call,
checkpoint, raw network output, `DetectionFrame`, or M7 result was produced.

This is a measurement-runtime resource failure, not an M7 detector result and not an
input-generation failure. No scientific protocol, intervention, input record, full-ledger byte,
compact manifest, paired GT set, detector, threshold, evaluator, M6 evidence, or v0.3.0 identity
changed.
