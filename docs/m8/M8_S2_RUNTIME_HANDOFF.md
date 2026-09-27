# M8 P1-S2 prebilling runtime handoff

This is a CPU-only software and planning freeze for future B2/C2/D2/F2 execution. It creates no
runtime binding, qualification, inference authorization, detector output, or scientific result.
The [frozen S2 protocol](M8_S2_PROTOCOL.md), [CPU input freeze](M8_S2_INPUT_FREEZE.md), and
[machine-readable handoff](../../benchmarks/m8/preregistration/m8_s2_runtime_handoff.json) govern
the future work. No external worker was contacted for this freeze.

## Software and input identities

The execution software and synthetic tests were committed before this handoff, beginning at
`aa38ae4d9ca291380bf44de2628f9a1a598b96bf`. Subsequent prospective evidence and review
corrections complete at `66925ad2da0d634c1a4fd976ba49c1410dda237c`
(`S2_EXECUTION_IMPLEMENTATION_COMMIT`). The future runtime must instead bind the **reviewed merged
execution commit** containing this code.
That merged identity is not known in this prebilling record. The original S2 input implementation
commit remains `bf098b319744f1ec1df08207c1cd93853b1f31ae`.

| Frozen artifact | SHA256 |
|---|---|
| S2 protocol Markdown | `218ef2dcc03fa4ff75562e02f368f7624f16a4c768b1758147c1d46ac1d9c53d` |
| S2 protocol JSON | `341da352f2684d7eb85f3d43112cd88bd4652009463cb59163c67ebd79748720` |
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

The [S2 runner](../../scripts/detection/run_m8_s2.py) exposes `input-gate`, `runtime-binding`,
`qualification-plan`, `qualification`, `repeatability`, `full-pass`, and `aggregate` modes.
`qualification-plan` is static and CPU-only. External modes require `--external-worker` and the
exact applicable owner authorization. No authorization file is created by this handoff.

1. Obtain a fresh **qualification-only** owner authorization for the selected worker hostname,
   GPU UUID, and reviewed merged commit. The external worker must compare the grant to its live
   hostname and the sole NVIDIA-SMI-visible GPU UUID before Torch import. Reject multi-GPU
   visibility; policy capture also requires exactly one Torch CUDA device and the same GPU UUID.
   Verify Git HEAD, a clean tracked execution tree, all frozen identities, and the streamed full
   ledger. The
   input-gate receipt is evidence, not inference permission.
2. Capture a fresh runtime policy for that worker. Bind its hostname, GPU name, UUID, VRAM, driver, CUDA,
   Python, PyTorch, spconv, torch-scatter, NumPy, DSVT upstream commit, candidate config, and
   checkpoint, plus the S2 protocol/input hashes. A historical S1 or zero-intensity policy cannot
   authorize S2.
3. Under the same qualification scope, run the GT-blind seven-sentinel structural preflight on
   that bound runtime. Check 28/28 frozen full XYZIT inputs, then compute candidate coordinates
   with the bound DSVT CUDA arithmetic. Require B2/A2 and D2/C2 exact same-runtime coordinate
   identity in all seven frames. Record CUDA candidate counts and coordinate hashes.
   CPU-versus-CUDA count equality is
   **not** a gate: the historical H10 example was 30,623 CPU analytic versus 30,624 CUDA.
4. Present the fresh qualification receipt to the owner. Only a new **repeatability-only**
   authorization must bind one durable `--campaign-root` and can permit ten independent processes,
   each in seven-sentinel frame-major
   B2/C2/D2/F2 order (28 calls, 280 accepted calls total). Each process has a fresh UUID and model
   initialization. Each attempt uses a fresh `--attempt-root` directly under that campaign root.
   The serialized pass-claim ledger there permits a retry only after the prior attempt is preserved
   and sealed `INCOMPLETE`; a complete pass consumes its logical ID. Preserve the ledger and
   persist/hash-verify every complete or failed attempt off-worker. The offline
   review requires exact discrete agreement for the frozen per-class/threshold gates. Its
   `ACCEPTED` result alone does not authorize the corpus.
5. After owner review of repeatability and a separate **full-pass-only** authorization, execute
   three fresh complete processes. Each consumes 428 frames in frame-major B2/C2/D2/F2 order:
   1,712 conditions per pass and 5,136 accepted calls in three passes. Each input is reconstructed
   once, verified against the frozen ledger immediately, and that same verified array is passed
   to the detector. An incomplete attempt has zero accepted canonical calls, remains preserved,
   and cannot be spliced or resumed as a canonical pass. The same durable claim rule prevents
   successful reruns or selection among multiple complete attempts. Persist/hash-verify each
   process and its claim ledger before
   continuing. Stop the worker when no unique state remains there.
6. Aggregate on CPU only. Report pass 1/2/3 and min/median/max by arm/class/IoU for TP, FP, FN,
   recall, annotation-conditioned precision/F1/AP, range, track continuity, population, FOV, and
   ignore behavior. Frozen Car denominators are A_REF=19, E_REF=43, D_REF=24 and partitions
   shared/E2-only/A2-only/neither=19/24/0/23. Report unclamped G, paired R metrics, frozen
   Car TP/recall factorials (excluding F2), and the all-three descriptive gate. The hashed
   aggregate retains the common runtime/input bindings and ordered attempt/result identities.
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

Current state: runtime bound **false**; S2 inference authorized **false**; qualification, sizing,
repeatability, full-corpus inference, and training **unexecuted**. Current S2 detector calls **0**.
