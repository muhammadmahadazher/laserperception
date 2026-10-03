# M8 P1-S2 GT-blind engineering sizing execution freeze

This CPU-only implementation creates no runtime, real authorization, measurement, or scientific
result. [Qualification attempt 2](M8_S2_QUALIFICATION_ATTEMPT_2.md) was accepted historically;
its A40 worker was deleted. Its policy cannot authorize a future worker.

The unchanged [sizing plan](../../benchmarks/m8/preregistration/m8_s2_qualification_plan.json)
has SHA256 `f3bb7ca94785ccaed71c821ecfa1e9494c9015b7fa87ded51c33c96c51adda5e`.
Selection is read from these frozen bytes, never regenerated from detector output.
Two separate OS processes, `s2-sizing-01` and `s2-sizing-02`, each initialize a fresh model and UUID,
run the two prescribed warmups, then the sixteen prescribed measured conditions in artifact order:
**4 excluded warmups + 32 measured = 36 engineering detector calls**. These are not S2 scientific,
repeatability, or full-corpus calls.

## Authorization and import boundary

`sizing-only` is independent of qualification, repeatability and full-pass scopes. The authorization
uses the existing S2 schema with an additional mandatory `sizing_plan_sha256` field **only for this
scope**. Bind the exact reviewed execution commit, frozen protocol/partitions/input/full-ledger/
compact-manifest identities, input-receipt, runtime-policy and accepted qualification-receipt SHA256s,
canonical absolute `authorized_campaign_root`, and allowed logical IDs from the two sizing IDs.
Worker hostname/GPU fields are null because the verified policy supplies that identity. Both review/
attestation fields are null. The qualification grant is separately required to verify the qualification
receipt and compare the live hostname/sole NVIDIA-SMI GPU to the bound policy before Torch import.
No real grant is created by this PR.

The [runner](../../scripts/detection/run_m8_s2.py) verifies external context, checkout, all input and
receipt bindings, exact scope, source/checkpoint/native identities, frozen sizing plan, and live
worker qualification identity before accelerator/backend import. It compares the freshly captured
live policy to the supplied policy before executing the sizing module. It never enters
`m8_s2_science`, the tracklet loader, evaluator, or scoring path.

For each authorized **separate process**, invoke isolated Python with mode `sizing` and:

```text
--external-worker --execution-commit <reviewed merge>
--full-ledger <verified full S2 ledger>
--input-gate-receipt <fresh receipt> --runtime-policy-binding <fresh policy>
--qualification-receipt <fresh accepted receipt>
--qualification-authorization <fresh qualification grant>
--authorization <fresh sizing-only grant>
--upstream-root <pinned DSVT> --checkpoint <verified checkpoint>
--date-root <KITTI date root> --m6-ledger <frozen transform ledger>
--campaign-root <bound sizing root> --attempt-root <new direct child>
--logical-pass-id s2-sizing-01 --attempt-id <unique attempt identifier>
```

Use `s2-sizing-02` in a new OS process for the second logical process. Do not run concurrently.
Attempt paths are new direct children of the bound campaign; existing attempts cannot resume.

## Consumption, measurement and discard

The existing reconstruction path verifies full XYZIT, M7 XYZT projection, selected rows and point
count against the frozen compact identities (bound to the streamed full ledger). That same verified
array is passed to the existing `DsvtBackend.run_gt_blind_timing_call`. This boundary discards GPU
predictions directly: no boxes, scores, labels, DetectionFrame or detection counts are exposed,
inspected, printed, serialized, or scored by sizing.

Fresh model initialization is timed separately with `perf_counter` and CUDA synchronization.
Each measured call synchronizes before timing and after exactly one backend call. Reconstruction,
hash validation, peak-stat reset and evidence serialization are outside the interval. Input-to-backend
preparation remains inside the existing backend timing boundary. Warmup durations are retained only
as engineering audit records; they are excluded from the 32-observation estimator distribution.

Each call records its fixed condition/arm, input hashes, point count, CPU-analytic candidate pillar
characterization, monotonic duration, process UUID and Torch allocated-before/peak-allocated/
reserved-after byte metrics. Peak statistics reset per call. These are **Torch allocator** metrics,
not device/system memory measurements or scientific outcomes. Schemas reject semantic fields.

## Attempt accounting and CPU aggregation

An exclusive campaign lock and atomic claim ledger prevent concurrency, repeated successful logical
processes, or two accepted processes sharing an OS PID. Each attempt seals its copied authorization,
ordered call records and exact bindings. COMPLETE requires all eighteen calls. Exceptions seal
INCOMPLETE with zero accepted calls, preserving partial evidence; accepted timings never splice
across attempts. Persist/hash-verify each successful **or failed** attempt and updated claims before
proceeding. A retry is a fresh process/attempt under a valid explicit sizing grant, preserving prior
sealed attempts; no fastest-run selection is permitted.

A hard-killed process leaves a lock and RUNNING claim. Do not delete the lock manually. On its
original worker, `sizing-seal-interrupted --external-worker --execution-commit <merge>
--campaign-root <root> --attempt-root <attempt>` verifies hostname and process death before sealing
INCOMPLETE. A manifest already COMPLETE remains COMPLETE. A live or different worker cannot recover
that claim. Exact atomic-write temporaries belonging to the verified dead PID are retained and
hash-accounted without promoting uncommitted bytes into canonical evidence; foreign temporaries
remain rejected. Persist the recovery before retry. An unrecoverable pre-claim/setup failure must fail
closed and return for owner review.

Mode `sizing-aggregate` is CPU-only. Supply the same reviewed `--execution-commit`, exactly two
`--pass-input` attempt directories, actual future `--usd-per-hour`, and `--output`. Aggregate on the
worker's CPU at the original authorized campaign path before the final off-worker archive. It checks
the complete claim history including sealed failures, matching bindings/hashes, distinct PIDs/UUIDs,
fixed call order, strict nonsemantic schemas and exactly two complete logical processes. Input order
does not affect deterministic output. Failed attempts contribute no timings.

The `laserperception.m8.s2.sizing-result.v1` result retains two initialization durations, all 32
measured observations, observed min/median/max, allocator summaries, and estimator results.
It reuses `estimate_future_cost()` for 280 calls/10 initializations, 1,712/1, 5,136/3 and 5,416/13.
The minimum/median/maximum projections are descriptive engineering scenarios, **not confidence
intervals**. The budget decision fields expose repeatability, three-pass and total median/observed-max
USD at the supplied actual GPU rate. Provider storage/transfer/setup overhead is outside this estimate.
The result also reports the GPU-cost equivalent of observed sizing call and initialization durations
(including warmups); this excludes reconstruction and provider overhead and is not a billing receipt.
Cost does not authorize science.

## Future hands-off lifecycle — requires a new owner task

1. Prepare CPU assets/commands before billing; provision **one** temporary A40/A6000.
2. Capture fresh live identity and issue qualification-only authorization for the reviewed merge.
3. Hydrate/hash assets, bootstrap the frozen environment, pass the development gate, build/import
   all seven extensions and verify generated source/native identities.
4. Pass fresh input gate, capture fresh live policy and pass fresh GT-blind qualification; persist
   and hash-verify each receipt off-worker.
5. Obtain fresh sizing-only authorization bound to those receipts, plan and campaign path.
6. Run the two separate complete sizing processes, preserving each attempt/claim before continuing.
7. Aggregate on CPU and generate descriptive cost estimates. Persist/hash-verify all evidence off-Pod.
8. **DELETE** the Pod after success or any terminal failure, confirm deletion, and return to the owner.

Never leave a Pod running for owner review, substitute a second worker automatically after failure,
or proceed into repeatability. After future accepted sizing, `S2_RUNTIME_SIZED=true`, but
`S2_REPEATABILITY_AUTHORIZED=false` and `S2_REPEATABILITY_EXECUTED=false`. Frozen scientific
definitions, denominators, recovery/factorial formulas and repeatability/full-pass rules are unchanged.
