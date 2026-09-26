# M8 external-runtime readiness and qualification runbook

M8-R is static engineering readiness. No provider is selected, no worker is provisioned, no hardware
is probed and no scientific detector call is authorized by this bundle. Historical M8 P1-E/S1/Stage R
and the retired primary authorization retain their original identities and status. The retired runtime
executed 140 Stage R calls and zero primary calls. Its policy and authorizations are non-portable.

## One-command CPU dry plan

From the exact authoritative checkout:

```console
laserperception worker plan --task m8-qualification --execution-commit FULL_CURRENT_MAIN_SHA --task-id m8-runtime-qualification --repository-root .
```

Use the actual full `git rev-parse HEAD` value. The command compares it with the selected checkout,
then emits a deterministic `laserperception.m8.readiness-plan.v1` record. It binds repository SHA,
selected candidate/config/checkpoint/upstream/input identities, verified-artifact descriptions,
Drive capsule `lpdrive:_CLOUD_WORK/m8-runtime-qualification`, raw-source requirements, environment,
resource guidance, static availability snapshot and missing fresh gates. `executes`, `hardware_probed`,
`provider_selected` and `source_payload_verified` are false; scientific_calls is 0. It launches only a
CPU Git HEAD read, never a provider request, download, transfer, bootstrap or GPU process.

This preliminary plan deliberately does not manufacture a TaskManifest or placeholder authorization.
After owner runtime selection and fresh qualification-only scope, create the existing TaskManifest
with its matching AuthorizationReference/runtime/execution binding; `worker plan MANIFEST --model
dsvt-pillar-transfusion-m8` then uses the normal provider-neutral planner. `worker bootstrap` remains
external-only and independently verifies task, Git, owner scope and artifacts before collection.
Basic bootstrap records environment facts with detector-call counts 0. Any GT-blind detector sizing,
load/forward or capacity operation needs a distinct explicit runtime-specific owner scope and honest
attempt/call accounting. It cannot be concealed inside zero-call environment discovery.

## Frozen identities

| Binding | Exact identity |
|---|---|
| Candidate | DSVT-Pillar with TransFusion, M8 P1-E |
| Candidate JSON | configs/m8/dsvt_nuscenes_pillar.json; 4915 B; SHA256 aa456e0386e46e9d089a957b1f1a8a4f74ceae70435c7ad8e6ca5e67bb90f4e7 |
| DSVT upstream | Haiyang-W/DSVT commit 8cfc2a6f23eed0b10aabcdc4768c60b184357061 |
| OpenPCDet reference | Audit-only commit 233f849829b6ac19afb8af8837a0246890908755; not the installed DSVT runtime |
| Official YAML | tools/cfgs/dsvt_models/dsvt_plain_1f_onestage_nusences.yaml; 5048 B; SHA256 b0832e03ad11d4e0b61f0fb07d977e687763caae472a4f87ed750bdc2d13be0f |
| Checkpoint | DSVT_Nuscenes_val.pth; 28665215 B; SHA256 a675149d095eef8ddc0c137ae46eeac075ccc504c7608162c71e7adf318793fb |
| Checkpoint source | Recorded official upstream Drive file 10d7c-uJxg5w4GN-JmRBQi4gQDwHiOHxP; private verified copy available; never redistribute |
| S1 protocol JSON | 15956B; SHA256 c132f60257c6a39debb548461c79bd59c98325484d233db6095b441c638d8e88 |
| S1 Markdown LF | 23802B; SHA256 1ad58ebbdd04897558ef9802fee6288b806c5e633d393f1bed957ecc6d6f6b10 |
| Protocol freeze | 5061d5d2c6a6057fed1f3f537c5857d2d84f6b3f |
| Projected input ledger | 669345B; SHA256 474e87e34c64d669750d4b6f7a64ac46fc9c5c462693fad79ff7c9547a7f1f7c |
| Input revalidation | 966B; SHA256 71ac9418c29da5efd64f9eaeb03e859f85d6b1c56dc2fe47cef6563a9f960341 |
| Full M6b transform ledger | 5837452B; SHA256 e25b3d62113cc7e8c1fcf736caa68b1ab698f965f007c758ff91d3e498ca6caa |
| Ordered 428-frame corpus | SHA256 76bd5f7adac3d892ad2fb00cb9cf5f4f73dd475682ee011b7ea9524060c46c95 |

Feature contract: contiguous float32 `[x,y,z,intensity,time_lag]`, raw KITTI reflectance without
normalization/synthesis, preserved frozen source-row order without test-time shuffle. Time lag is
current seconds minus history seconds, positive zero on current rows and positive for older history.
Physical corpus range is [-50,-50,-5,50,50,3]; candidate range is [-54,-54,-5,54,54,3]. All original
score/filter/evaluator/call-accounting conventions remain governed by the frozen protocol.

The Windows S1 Markdown working tree is 24232 B with 430 CRLF pairs; its Git index is LF. In-memory
canonical LF matches the frozen hash. A future clean external checkout must use canonical LF
(e.g. configure core.autocrlf=false before checkout). Do not alter frozen content or relax byte checks.

## Bounded artifact availability snapshot — 2026-09-17

| Artifact | Status | Verification and remaining work |
|---|---|---|
| Candidate JSON/protocol JSON/projected ledger/revalidation | READY | Exact bounded file size/SHA256 matched; Git-authoritative; reverify worker bytes |
| Protocol Markdown | READY | Canonical Git LF identity matched; materialize LF externally |
| Official YAML and checkpoint | READY | Exact private retirement copies independently hashed; no model import |
| Full M6b transform ledger | READY | Exact 5.8 MB copy at .local/m6b-r2/evidence/pre_inference_input_ledger_full.json hashed |
| KITTI drive0001/0091/calibration/timestamps | READY structurally | Exact 108/340 contiguous Velodyne and OXTS paths and boundaries exist; source payload hashes/transforms remain unverified |
| Clean pinned DSVT upstream checkout | RECONSTRUCTIBLE | Reconstruct from official pinned Git; verify exact checkout/config before use |
| Isolated scientific environment | RECONSTRUCTIBLE | Frozen candidate/retirement environment records; live versions and capacity require fresh qualification |
| Optional nuScenes source smoke data | REDOWNLOAD_REQUIRED | Not required by frozen KITTI source; only if owner separately scopes a source-domain smoke and approves hydration |
| Fresh runtime/provider choice, qualification authorization | MISSING | Owner selection and fresh runtime-specific scope required |
| Fresh machine policy and owner review | MISSING | Bind new reported runtime/inputs and fresh capacity evidence |
| Fresh Stage R authorization/run/raw review | MISSING | Historical Stage R cannot substitute |
| Fresh primary authorization | MISSING | Only after fresh Stage R raw owner review/freeze |
| Retired Stage R/policy/primary authorization | HISTORICAL_ONLY | Non-portable; preserved; zero historical primary calls |
| PointPillars/ONNX/TensorRT historical artifacts | HISTORICAL_ONLY | Not M8 primary qualification or end-to-end DSVT parity |

The required exact source root is the Drive-backed 2011_09_26 date containing synchronized drives 0001
and 0091. Drive 0001 frames 0..107 and drive 0091 frames 0..339 provide complete reconstruction history;
accepted evaluated indices are 10..107 and 10..339. Require all three date calibrations, Velodyne/OXTS
files and acquisition timestamp files. FrozenInputSource reconstructs 428 ordered frames and 856
H10/H5 conditions. Qualification must stay GT-blind: do not give it tracklet/GT loader access.
Filename/count checks are not payload verification. The unrelated 3.16 GB M7 ledger was untouched and
is not the required full transform ledger. Never mirror/hydrate private Drive recursively.

## Resource and environment envelope

Historical accepted capacity recorded device_total = 8585216000 B, peak_allocated = 2922354688 B and
peak_reserved = 8443133952 B (about 98.35% of device capacity). Reserved memory is not the active working
set. These facts only support planning guidance: 16 GB VRAM practical lower bound; 24 GB preferred.
A larger worker still needs fresh identity/capacity/runtime qualification; these are no portable
memory/latency guarantees and do not select a provider.

The frozen scientific environment records Python 3.10, Torch 2.1.0+cu118, CUDA 11.8, spconv 2.3.8,
torch-scatter 2.1.2+pt21cu118, NumPy 1.23.5, cuda:0 and batch 1. The historical partial TensorRT route
records 8.6.1 but is not the eager primary scientific path. Keep it optional. The lightweight core
wheel declares NumPy>=1.24: do not install its development dependency set into the frozen scientific
environment and silently upgrade NumPy. Use an isolated source runtime and externally verify every
live version; package metadata is not proof of runtime equivalence.

## Chronological future worker stages

Before starting billed GPU time, run the CPU-only
`scripts/detection/revalidate_m8_input_projection.py` command at the exact prospective execution
commit with `--receipt-output`. Persist its complete revalidation record and
`laserperception.m8.s1.input-gate-receipt.v1` receipt. The command must reconstruct all 856 frozen
H10/H5 conditions with zero mismatches. Verify the receipt against the same checkout and the
authoritative full transform/source ledger; a receipt from another commit or source fails closed.
Stage R and primary processes require this receipt; omission or invalidity fails before backend
construction. Each primary process additionally performs its own live 856-condition pre-inference
revalidation. The receipt is a provenance binding and does not replace that live gate.

1. Owner selects/provisions an external worker and explicitly scopes qualification. No provider/API
   operation is part of M8-R. Preserve fresh task/runtime identity and qualification authorization.
2. Hydrate exact authoritative Git SHA with a clean tracked checkout and canonical LF. Verify origin,
   full HEAD, upstream identity and no tracked modifications; never substitute a moving branch.
3. Hydrate only the declared verified artifacts and exact required source files. Verify size/SHA256
   plus source/transformation identities. Transport success does not authorize inference.
4. Construct the isolated frozen runtime. Record externally reported OS/Python/GPU/VRAM/driver/CUDA/
   Torch/framework versions; no collection occurs on the development workstation.
5. Perform owner-scoped GT-blind qualification. Basic bootstrap collects environment only. Separately
   scoped sizing/capacity operations require their own explicit permission and call accounting.
6. Verify candidate/config/checkpoint/upstream, ordered input identities and capacity/runtime preflight.
   Persist failed attempts too; a memory size or available model alone does not qualify a worker.
7. Create a new machine-specific runtime-policy binding. Bind exact new runtime, execution/config/
   checkpoint/input/environment identities. Retired policy cannot be reused or cosmetically relabeled.
8. Owner reviews fresh qualification/capacity/policy artifacts and binds the prospective policy.
9. Obtain fresh Stage-R-only authorization for that exact runtime/policy/execution identity. This does
   not authorize primary or zero-intensity calls.
10. Run fresh Stage R under the existing verifier/accounting path: 10 fresh processes with 14 canonical
    calls each. Every process verifies the global 856-condition receipt before backend construction,
    then freshly reconstructs and verifies the exact 14 sentinel inputs it consumes. No historical
    process/evidence substitution or spliced partial passes.
11. Persist raw Stage R evidence and every failed attempt to its Drive capsule before further work.
12. Owner reviews/freezes the fresh raw Stage R result. Engineering receipts alone do not satisfy this.
13. Obtain fresh primary authorization bound to that reviewed Stage R and machine policy.
14. Run three uninterrupted primary processes, each completing 856 H10/H5 conditions. Every fresh
    process verifies the complete global receipt, then completes a live 856-condition reconstruction
    and identity check before GT loading, backend/model construction, CUDA initialization, or any
    detector call. `--input-revalidation-workers` bounds only this CPU gate to 1 through 8 workers,
    with a default of 4. Spawned worker processes own isolated lazy sequence and native math runtime
    state; contiguous frame partitions reduce overlapping source reads, and canonical aggregation
    makes evidence independent of completion order. After the gate succeeds, scientific execution
    reconstructs each of the 428 frame pairs once and consumes H10 followed by H5, producing 856
    serial detector calls and deterministic consumed-input evidence. Pair data is not reused between
    the gate and execution, across frames, or across processes. No splicing; preserve incomplete
    attempts and restart a whole canonical process when required by protocol.
15. Persist and verify each canonical pass/process immediately before proceeding. Expected accepted
    primary total is 2568 only after all three complete; never claim a result from this plan.
16. Owner performs final evidence review. Keep raw/failed evidence distinct from compact final Git
    evidence; preserve all original provenance and scientific interpretation boundaries.
17. Verify GitHub/Drive persistence and retire the disposable worker only when no unique state remains.

## Record and execution boundaries

`worker.m8_readiness` exposes M8ReadinessRequest/Plan, M8ArtifactAvailability, M8CapacityRecord,
M8QualificationRecord, M8GateEvidence and M8QualificationProgress. They accept externally supplied
facts and synthetic CPU fixtures, perform strict identity/shape/status checks and serialize through
the existing JsonRecord system. Progress indexes artifacts→environment→passed capacity→new policy→
owner review→fresh Stage R authorization→fresh Stage R raw review→fresh primary authorization.
Changed stable GPU/software/Python/OS identities between environment and capacity reports, mixed
runtime/repository/policy identities, skipped gates, missing authorization references and
retired references fail. These records neither verify purported authorization-file contents nor
unlock scientific execution. The frozen runner must independently verify the real authorization,
live policy, artifact bytes, pass scope and AtomicAttempt accounting.

The nine M8 GPU CLIs require explicit --external-worker before GPU execution/child launch; sizing
children validate it independently. CPU help/dry plans and aggregate remain local-safe. This flag
protects against accidental local invocation; it is not owner permission or proof of an external
machine. Standalone historical engineering scripts still require separately scoped owner authorization
in their selected external runtime. No new execution command bypasses the existing worker/bootstrap
or scientific verifier. Historical evidence and algorithms are unchanged.

The completed zero-intensity raw measurement had a separate authorization bound to its A40 runtime;
it grants no permission for new execution. See [Project status](../PROJECT_STATUS.md) for current
scientific state. B2/C2/D2/F2 are unauthorized. S2 and training have not started. This engineering
runbook grants no execution permission by itself; each
external qualification or scientific stage still requires the applicable direct owner scope.
