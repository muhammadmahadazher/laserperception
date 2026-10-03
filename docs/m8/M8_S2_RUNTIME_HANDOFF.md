# M8 P1-S2 prebilling runtime handoff

This is a CPU-only software and planning freeze for future B2/C2/D2/F2 execution. It creates no
runtime binding, qualification, inference authorization, detector output, or scientific result.
The [frozen S2 protocol](M8_S2_PROTOCOL.md), [CPU input freeze](M8_S2_INPUT_FREEZE.md), and
[machine-readable handoff](../../benchmarks/m8/preregistration/m8_s2_runtime_handoff.json) govern
the future work. No external worker was contacted for the original freeze. The later
[first external qualification attempt](M8_S2_QUALIFICATION_ATTEMPT_1.md) passed the input gate
but stopped before runtime-policy capture and qualification; it made zero scientific calls.
The prospective CPU-only compatibility repair below does not run an external worker.

## Software and input identities

The execution software and synthetic tests were committed before this handoff, beginning at
`aa38ae4d9ca291380bf44de2628f9a1a598b96bf`. Subsequent prospective evidence and review
corrections complete at `f62e81be2fa831c28408859e5cc9bf7c2708e891`
(`S2_EXECUTION_IMPLEMENTATION_COMMIT`). The future runtime must instead bind the **reviewed merged
execution commit** containing this code.
The first attempt used merge `b62f4dacc466790257031b3c15fcae4e39254e25`. After the compatibility
repair is reviewed and merged, future external execution and its fresh qualification-only grant
must bind that **new reviewed merge**, not the first attempt's commit or grant. The historical
machine-readable prebilling record is retained unchanged. The original S2 input implementation
commit remains `bf098b319744f1ec1df08207c1cd93853b1f31ae`.

| Frozen artifact | SHA256 |
|---|---|
| S2 protocol Markdown | `218ef2dcc03fa4ff75562e02f368f7624f16a4c768b1758147c1d46ac1d9c53d` |
| S2 protocol JSON | `341da352f2684d7eb85f3d43112cd88bd4652009463cb59163c67ebd79748720` |
| Frozen S1 GT provenance JSON | `c132f60257c6a39debb548461c79bd59c98325484d233db6095b441c638d8e88` |
| S2 paired partitions | `f52b0013d10d38c4fd91f52cfb42205f52b97ecf4433cd86654872dd70a59d6e` |
| S2 input-freeze record | `c08589bb9600633d5a8b675a16f8be805697a5302296ff52608e5845a3e371db` |
| Private full S2 input ledger | `a3ed54b276f77fb784035045b079573cd4e4ddfedc9d0f8eb774c1340a59396b` |
| Tracked compact input manifest | `239b563d5f850f2f20809950eca9d56f1a700677766afce8489099e940554ecc` |
| CPU input characterization | `c327dafe6287335c971fa8c5c5f2a15680c08aafcc367e4cdccda4e6e39f86db` |
| Frozen M7 compact manifest | `8d4f74d783950d24956239f3a67a7a58fe10013e0e83a88d0f8b23e3139ffe90` |

The private `m8_s2_input_ledger.jsonl` has 7,728,782 bytes and 1,712 ordered records, 428 per
arm. Its compact tracked counterpart is
[here](../../benchmarks/m8/inputs/m8_s2_input_manifest.json). A future worker also needs the
frozen M8 A2/E2 source ledger, M6/M7 provenance inputs, required KITTI data, official pinned DSVT
upstream checkout/config, and checkpoint. Hydrate only these required items from the canonical
private Drive and verify their recorded hashes; do not make the worker the sole copy of evidence.

## Future fail-closed sequence

### Setup before billed compilation

Capture the live hostname and sole GPU UUID only inside an owner-authorized external runtime,
then obtain its fresh qualification-only authorization. Clone the exact reviewed execution merge,
hydrate/hash the frozen assets, and prepare the pinned Python 3.10 / Torch 2.1.0+cu118 /
NumPy 1.23.5 / spconv 2.3.8 / torch-scatter 2.1.2+pt21cu118 environment.

Install a **complete CUDA 11.8 development toolkit**, with GCC and G++ 11, before compiling.
Use NVIDIA's [official CUDA 11.8 installation guidance](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-installation-guide-linux/index.html),
for example the full `cuda` package from the `nvidia/label/cuda-11.8.0` channel, or the
official `cuda-toolkit-11-8` package on a supported distribution. These are future installation
choices, not actions performed by this repair. Compiler-only `cuda-nvcc` plus `cuda-cudart-dev`
was insufficient. Do not use the base container's CUDA 12.8 headers or silently change Torch.

Set `CUDA_HOME` to that toolkit and `CC=gcc-11`, `CXX=g++-11`. Run the
[external-only pre-build checker](../../scripts/detection/check_m8_s2_development.py)
using the same isolated environment Python that will build the extensions:

```sh
python -I scripts/detection/check_m8_s2_development.py --external-worker \
  --cuda-home "$CUDA_HOME" --cc "$CC" --cxx "$CXX"
```

Require exit zero before `setup.py build_ext --inplace`. Preserve the checker JSON or failure
output. It checks Python/package metadata without importing Torch, nvcc release 11.8,
matching GCC/G++ 11, readable `cuda.h` with `CUDA_VERSION 11080`, `cuda_runtime.h`,
`cuda_runtime_api.h`, `cusparse.h`, `cublas_v2.h`, `cusolverDn.h`, and `curand.h`, plus
readable development-link names `libcudart.so`, `libcusparse.so`, `libcublas.so`,
`libcusolver.so`, and `libcurand.so` under that toolkit's `lib64` or `lib`.
Use the toolkit's include/library paths for compilation and loading. The fast static check
rejects file links that resolve outside the selected toolkit, preventing accidental mixing of
installation roots. It does not prove ABI compatibility or replace the full build/import gate.
It performs no
GPU discovery, model initialization, GT loading, or inference. Never invoke its external
probe mode on the local CPU workstation; CI uses supplied filesystem/version fixtures.

Build **all seven** pinned upstream extensions and verify their normal imports:
`iou3d_nms_cuda`, `ioubev_nms_cuda`, `roiaware_pool3d_cuda`, `roipoint_pool3d_cuda`,
`pointnet2_stack_cuda`, `pointnet2_batch_cuda`, and `ingroup_inds_cuda` in their existing
`pcdet.ops` locations. Incomplete or failed build products cannot qualify a runtime.
Disable bytecode writes during setup/import checks, and keep generated build bookkeeping
outside importable source where appropriate; all arbitrary untracked Python and caches
remain rejected.

The only permitted generated Python is the regular, non-symlink `pcdet/version.py` in the
exact pinned DSVT checkout `8cfc2a6f23eed0b10aabcdc4768c60b184357061`. Require exact bytes
`__version__ = "0.6.0+8cfc2a6"` plus one LF and SHA256
`e5fcccc8123cb08c0a709b59d7ae2991a662ee9579e4fb9a04a953a857353cdd`.
The source verifier checks the fixed commit and frozen hash; no moving Git value or arbitrary
generated Python exception is accepted. Hash every permitted native extension into the live
policy and reverify that policy before qualification. Native source-shadowing checks remain.

The next-worker order is: live identity → fresh qualification-only grant → exact reviewed
checkout → hydrate/hash frozen assets → complete development toolkit → pre-build gate →
complete extension build/import checks → exact generated-version check → native inventory
hashes → input receipt → fresh runtime policy → GT-blind 28-condition qualification →
off-worker receipt/hash persistence → **STOP for owner review**. No repeatability, engineering
sizing, or full pass is included in that qualification-only sequence.

### Qualification and separately authorized later phases

The [S2 runner](../../scripts/detection/run_m8_s2.py) exposes `input-gate`, `runtime-binding`,
`qualification-plan`, `qualification`, `repeatability`, `full-pass`, and `aggregate` modes.
Invoke it with isolated Python (`python -I scripts/detection/run_m8_s2.py ...`); the runner
checks this before shadowable imports, performs its source inventory, and only then adds the
reviewed repository and `src` checkout to the import path.
`qualification-plan` is static and CPU-only. Worker modes require `--external-worker`;
runtime-binding, qualification, and scientific modes additionally require the exact applicable
owner authorization. No authorization file is created by this handoff.
The `seal-interrupted` mode is a CPU-only, external-worker recovery operation; it never imports
the detector or loads GT.

1. Obtain a fresh **qualification-only** owner authorization for the selected worker hostname,
   GPU UUID, and reviewed merged commit. The external worker must compare the grant to its live
   hostname and the sole NVIDIA-SMI-visible GPU UUID before Torch import. Reject multi-GPU
   visibility; policy capture also requires exactly one Torch CUDA device and the same GPU UUID.
   Verify Git HEAD, a clean tracked execution tree, no untracked importable Python source in the
   repository or pinned upstream checkout except the exact generated-version contract above,
   and no project bytecode caches; the runner disables
   new bytecode writes before importing project modules. Verify all frozen identities and the
   streamed full ledger. The
   input-gate receipt is evidence, not inference permission.
2. Capture a fresh runtime policy for that worker. Bind its hostname, GPU name, UUID, VRAM, driver, CUDA,
   Python, PyTorch, spconv, torch-scatter, NumPy, DSVT upstream commit, candidate config, and
   checkpoint, plus SHA256 hashes of in-checkout upstream native extensions and the S2
   protocol/input hashes. A historical S1 or zero-intensity policy cannot
   authorize S2.
   Check that the pinned DSVT upstream checkout has no tracked edits outside the hashed config.
   Reject local native extensions and upstream extensions that shadow Python source; rehash each
   permitted upstream build product against the fresh live policy before qualification or science.
   The backend's environment-selected upstream and checkpoint paths must resolve to those same
   verified CLI paths.
3. Under the same qualification scope, run the GT-blind seven-sentinel structural preflight on
   that bound runtime. Check 28/28 frozen full XYZIT inputs, then compute candidate coordinates
   with the bound DSVT CUDA arithmetic. Require B2/A2 and D2/C2 exact same-runtime coordinate
   identity in all seven frames. Record CUDA candidate counts and coordinate hashes.
   The receipt records the qualification-only grant ID and SHA256; preserve that grant and
   supply it as `--qualification-authorization` when later verifying the receipt for science.
   CPU-versus-CUDA count equality is
   **not** a gate: the historical H10 example was 30,623 CPU analytic versus 30,624 CUDA.
4. Present the fresh qualification receipt to the owner. Only a new **repeatability-only**
   authorization must bind one durable `--campaign-root` and can permit ten independent processes,
   each in seven-sentinel frame-major
   B2/C2/D2/F2 order (28 calls, 280 accepted calls total). Each process has a fresh UUID and model
   initialization. Each attempt uses a fresh `--attempt-root` directly under that campaign root.
   Before importing the scoring path, verify both frozen KITTI tracklet XML files and both camera
   calibration files against the recorded S1 protocol SHA256 values. No GT is loaded for the
   qualification step.
   The serialized pass-claim ledger there permits a retry only after the prior attempt is preserved
   and sealed `INCOMPLETE`; a complete pass consumes its logical ID. Preserve the ledger and
   persist/hash-verify every complete or failed attempt off-worker. Offline review verifies each
   incomplete attempt's sealed file inventory and its original authorization;
   a retry may bind a separately issued fresh authorization. The seal includes any partial
   conditions, runtime state, telemetry, and interruption recovery files that were produced.
   The offline
   review requires exact discrete agreement for the frozen per-class/threshold gates. Its
   `ACCEPTED` result alone does not authorize the corpus. Keep the machine-generated review
   unchanged. Record the owner's decision in a separate attestation bound to that review's
   SHA256, and supply it as `--repeatability-owner-attestation` for full-pass preflight.
   Full-pass preflight also requires that review to name the currently verified qualification
   receipt, preserving the same qualification chain across both phases.
   If a process stops before sealing its attempt, keep the lock and partial files. On the same
   worker, after verifying its process is dead, `seal-interrupted` preserves the original manifest,
   writes a recovery receipt and zero-accepted-call incomplete manifest, then releases the lock.
   If a final pass manifest was already written, validate its complete evidence before restoring
   terminal bookkeeping and releasing a stale lock; never reclassify it as incomplete.
   A stale lock after a normally failed, already sealed incomplete attempt is released only after
   verifying that attempt's complete sealed file inventory and zero accepted calls.
   A lock left before the pass claim was written can be released on the original worker only after
   verifying the process is dead, the lock identity matches, and no attempt directory exists.
   A live or different worker cannot use that recovery path. Persist the recovery evidence before
   seeking any fresh retry authorization.
5. After owner review of repeatability and a separate **full-pass-only** authorization that binds
   both the review and owner-attestation hashes, execute
   three fresh complete processes. Each consumes 428 frames in frame-major B2/C2/D2/F2 order:
   1,712 conditions per pass and 5,136 accepted calls in three passes. Each input is reconstructed
   once, verified against the frozen ledger immediately, and that same verified array is passed
   to the detector. An incomplete attempt has zero accepted canonical calls, remains preserved,
   and cannot be spliced or resumed as a canonical pass. The same durable claim rule prevents
   successful reruns or selection among multiple complete attempts. Persist/hash-verify each
   process and its claim ledger before continuing. The completed manifest binds the campaign
   claim, runtime state, and telemetry by SHA256; CPU aggregation verifies the complete claim
   history, including prior incomplete attempts. Stop the worker when no unique state remains there.
6. Aggregate on CPU only. Report pass 1/2/3 and min/median/max by arm/class/IoU for TP, FP, FN,
   recall, annotation-conditioned precision/F1/AP, range, track continuity, population, FOV, and
   ignore behavior. Frozen Car denominators are A_REF=19, E_REF=43, D_REF=24 and partitions
   shared/E2-only/A2-only/neither=19/24/0/23. Report unclamped G, paired R metrics, frozen
   Car TP/recall factorials (excluding F2), and the all-three descriptive gate. The hashed
   aggregate retains the common runtime/input/qualification bindings and each ordered
   attempt's owner authorization ID and document SHA256 alongside its result identity.
   Aggregation requires the clean reviewed execution commit, which is recorded in its hashed output.
   No p-values,
   confidence intervals, AP factorials, or causal prose are generated.

The future accepted scientific plan totals **5,416 calls**: 280 repeatability plus 5,136 full.
No such calls have occurred. The runner checks external context, commit, frozen ledger/receipt,
policy, qualification, exact scope and logical pass **before** importing detector, Torch through
the science module, or real GT/evaluator code. The qualification path loads no GT or detector.

## Bounded GT-blind sizing proposal

The [fixed input-only selection](../../benchmarks/m8/preregistration/m8_s2_qualification_plan.json)
was generated after the implementation commit and before any detector output. Its SHA256 is
`f3bb7ca94785ccaed71c821ecfa1e9494c9015b7fa87ded51c33c96c51adda5e`. It selects four
different conditions per arm: low, median, and high CPU-analytic candidate-pillar regions, plus a
high point-count case. The proposed future sizing uses two fresh processes, each with two
excluded warmups and 16 measured GT-blind calls: **4 warmup + 32 measured = 36 engineering
detector calls**. These are not S2 scientific calls. Discard semantic outputs immediately; keep
only timing, memory, input identity, candidate pillar count, and runtime telemetry. Future owner
review is required before this separate engineering sizing execution.

The deterministic [cost estimator](../../src/laserperception/detection/m8_s2_planning.py) requires
all 32 measured call durations, two fresh-initialization durations, and an actual hourly price.
It projects the 280-call repeatability, one 1,712-call full pass, three 5,136-call full passes,
and combined 5,416-call accepted workload with initialization overhead. It reports median and
observed minimum/maximum envelopes, not a confidence interval. No future duration or cost is
measured yet.

Current state: runtime bound **false**; S2 inference authorized **false**; first qualification
attempt **blocked**, with no accepted qualification; sizing, repeatability, full-corpus inference,
and training **unexecuted**. Current S2 detector calls **0**.
