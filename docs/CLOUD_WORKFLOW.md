# Compute workflow — local CPU development and external GPUs on demand

This document is the repository-side authoritative policy for persistent LaserPerception work.
The canonical private Google Drive binding is `lpdrive:` at folder ID
`18Q73IkiVcFT0EXAowIPlOiISNk0mkhHJ`; the repository is
`muhammadmahadazher/laserperception`.

Normal development takes place on a CPU-capable local workstation. Codex Cloud is not required
and is no longer the primary development/runtime assumption. External GPU compute remains useful
and is selected explicitly per authorized task; no provider is chosen by this policy.

```text
local CPU workstation -> GitHub source/history + Drive private/large state
                     -> explicitly selected ephemeral external GPU worker
                     -> verified GitHub/Drive persistence -> worker retirement
```

The historical filename is retained for link compatibility. This operational policy grants no
scientific authorization and changes no frozen protocol, threshold, artifact identity, or result.

## Local workstation boundary

Core editing, documentation, CPU tests, lint, type checking, and packaging must work without CUDA,
PyTorch, TensorRT, OpenPCDet/DSVT, or ROS. Install only core/dev dependencies for normal development.
Do not probe optional GPUs, import optional GPU frameworks to discover devices, or execute a
GPU-worker bootstrap locally without explicit owner/runtime authorization. GPU discovery and
execution occur only inside explicitly authorized GPU runtimes. Static worker-code review, shell
syntax checks, and CPU tests with mocked commands are permitted without running a GPU path.

Before a local test run, inspect collection hooks, plugins, and optional integration entry points.
Use the CPU suite (`python -m pytest -m "not gpu"`) only after verifying it does not discover
hardware during imports or collection; deselecting a marker alone does not prevent import effects.
Skip GPU/ROS integrations before discovery when their authorized environment is absent.

Verify latest main and protect existing user work before creating a feature branch. Use the owner's
configured human Git identity, push to the feature branch, open a PR, and fix the same PR until
required CI is green. Leave it open for owner review; never push implementation directly to main,
merge, or enable auto-merge. On Drive-backed workspaces, preserve private untracked state and avoid
recursive hydration, metadata optimization, or cleanup. Diagnose unexpected changes rather than
resetting files or suppressing Git status with configuration/index flags.

## Persistent systems and disposable workers

### GitHub

GitHub is authoritative for source code, tracked documentation, tests, configs, pull requests,
release history, and compact or final evidence suitable for Git review. Changes reach that source
of truth through the normal branch, commit, push, and GitHub PR flow.

### Google Drive

Google Drive is authoritative for private, large, raw, and non-Git state: datasets; checkpoints;
ONNX and TensorRT binaries; large scientific raw evidence (including failed attempts); logs;
temporary project files worth preserving; untracked project artifacts; cloud task capsules;
environment snapshots and manifests; large upstream/local state; and migration backups.

The durable roots are:

- `lpdrive:_CLOUD_STATE/` for current state and storage/artifact/dataset/task indexes;
- `lpdrive:_CLOUD_WORK/` for immutable-by-convention task capsules;
- `lpdrive:_MACHINE_RETIREMENT_2026-09-06/` for the retired-machine handoff.

Never commit or upload credentials, OAuth tokens, rclone configuration, or secret environment
values. Do not use destructive `rclone sync` or delete Drive data by default.

### External worker

An external compute worker is disposable. No unique LaserPerception state may remain only
on a worker when a task finishes. Git-suitable changes must exist in Git/PR; valuable non-Git state
must be Drive-backed. Preserve failed scientific attempts and valuable debug or temporary files.

Require the explicit expected repository identity and execution commit for each worker; never
silently default to a historical SHA. An execution checkout may be detached at its authorized
commit. A branch name is not an artifact identity. Provision credentials externally and never
snapshot credential stores or an unfiltered environment.

## External task lifecycle

### Phase 1 — Hydrate

1. Verify the repository commit (and clean starting tree when required).
2. Read `_CLOUD_STATE/CURRENT_STATE.json`.
3. Read the storage, artifact, dataset, and cloud-work indexes.
4. Download only the Drive inputs required for the task—never mirror the project root.
5. Verify known hashes before use.
6. Record input paths, identities, and provenance in the task manifest.

### Phase 2 — Work

Perform only the authorized engineering or scientific task. Frozen protocols and authorization
barriers continue to apply after migration.

### Phase 3 — Checkpoint

After any expensive or important operation, finalize the current artifact, hash it, upload it to
Drive, verify the uploaded copy, and only then continue. Scientific/GPU results must be
checkpointed after every canonical process or pass. Incomplete and failed attempts are evidence and
must also be preserved rather than silently discarded.

### Phase 4 — Finalize

Before declaring completion:

1. commit and hand Git-suitable work to GitHub through a feature-branch PR;
2. upload all non-Git outputs, valuable temporary/untracked files, and logs to Drive;
3. update the Drive state and indexes without blindly overwriting useful state;
4. create and verify a task manifest and environment record;
5. inventory untracked, ignored, outside-repository, and project-relevant `/tmp` files; and
6. confirm `unique_worker_state_remaining = false`.

## Drive layout and task capsules

Every external task receives a unique directory under the existing root. The `_CLOUD_*` names
remain for compatibility; they do not require Codex Cloud:

```text
_CLOUD_WORK/YYYY-MM-DD_<task_slug>_<short_id>/
├── 00_MANIFEST/
├── 01_FINAL_OUTPUTS/
├── 02_RAW_EVIDENCE/
├── 03_LOGS/
├── 04_TEMP_PRESERVED/
├── 05_ENVIRONMENT/
└── 06_EXTERNAL_ARTIFACTS/
```

Create only the applicable subdirectories. A full repository copy is unnecessary when tracked
source is in GitHub. Record Git HEAD and preserve a patch when useful, plus untracked files, ignored
task artifacts, logs, temporary work, external artifacts, and safe environment metadata. Preserve
substantially more for expensive or scientific jobs. Use `scripts/cloud/persist_task.sh` for a
non-destructive checked copy. The helper refuses an existing capsule destination rather than
merging new files into old state. Its SHA256 manifest records canonical local identities; the
small manifest is downloaded and compared byte-for-byte, while ordinary `rclone check` verifies
the uploaded files through hashes supported by the remote (or size when no common hash exists).
It does not download every remote byte by default. A scientific protocol or task may additionally
require an explicit full round-trip of individual critical artifacts.

## GPU worker policy

A future GPU worker is an ephemeral external runtime. S1 remains paused, with zero retired-runtime
primary calls. The required order for a new runtime is:

1. obtain owner selection of an external runtime and explicit scope for its engineering
   qualification; provisioning or owning a GPU alone is not authorization;
2. hydrate the verified GitHub execution commit and only required Drive artifacts; verify exact
   candidate, config, checkpoint, input, and evaluator identities;
3. construct a clean optional GPU environment and perform authorized GT-blind qualification,
   including capacity/runtime preflight; record software and policy differences for owner review;
4. capture a new machine-specific runtime-policy binding and obtain owner review;
5. obtain a committed, runtime-specific **Stage-R-only authorization** before any GT-relative call;
6. repeat Stage R on that runtime by default, preserving and verifying each process's raw evidence;
7. obtain owner review and freeze the new Stage R raw evidence;
8. obtain a committed, **new primary authorization** for that exact runtime and execution identity;
9. run only the newly authorized primary workload, sequentially, preserving incomplete/failed
   attempts and uploading/verifying evidence after every canonical pass before proceeding;
10. compact Git-suitable evidence through a PR and finalize/verify the Drive task capsule; and
11. destroy the worker only after confirming no unique worker-only state remains.

The retired-machine primary authorization is not portable and must never be reused. No provider is
selected by this policy.

Zero-intensity remains separately unauthorized. S2 and training have not started. The frozen S1
corpus, pass counts, feature order, evaluator, and failure accounting remain unchanged. Historical
Stage R is preserved and cannot stand in for new-runtime Stage R. A bootstrap, successful transfer,
or capacity check must never emit or manufacture scientific authorization.

## Provider-neutral worker tooling — implemented engineering layer

The `laserperception.worker` package provides strict task/artifact records, portable paths,
verified individual-object transfers, CPU-only planning, and guarded external qualification
bootstrap. See [External workers](EXTERNAL_WORKERS.md) for APIs, schemas and failure behavior.
No provider or scientific execution is selected or authorized by these tools. The review record
below remains historical; bounded whole-workspace snapshots are not implemented.

## Closed PR #32 — historical selective review and implementation plan

[PR #32](https://github.com/muhammadmahadazher/laserperception/pull/32) is closed and unmerged.
Its capsule remains historical; the Codex Cloud migration plan is inactive. This workflow retains
its useful concepts: verified individual-object transfer, explicit task manifests, bounded workspace
snapshots, and fail-closed worker qualification. No PR #32 executable helper is imported here.
The existing `scripts/cloud/persist_task.sh` remains compatible and unchanged.

Implementing all transfer and bootstrap utilities with adequate failure tests would obscure this
operational/architecture change. A separate, owner-scoped engineering follow-up should deliver:

| Component | Required behavior and acceptance tests |
|---|---|
| Object pull/push | Require expected SHA256 and byte size; stream full remote bytes for critical-object verification. Use fresh destinations, refuse overwrites, preserve failed uploads for accounting, and report transfer errors. Mock success, truncation, wrong digest, network failure, and existing destinations. No destructive sync/move/delete default. |
| Path validation | Reject absolute, drive-qualified, UNC, backslash, empty, dot and parent components, control characters, and paths escaping the selected root, including symlink escapes. Validate before filesystem/network actions. Test POSIX and Windows inputs. |
| Task manifest | Versioned task/repository/commit identity, safe environment description, input/output paths with sizes/hashes, provenance, authorization references, attempted/accepted/failed call accounting, and completion state. Do not force all future scientific call counters to zero; distinguish engineering-only tasks from separately authorized science. |
| Workspace snapshot | Explicit allowlist of task-value files plus reviewed patch and Git commit identity; no automatic whole-repository `bundle --all`, recursive private-tree capture, credentials, or environment dump. Refuse existing output; test secret exclusions and nested output paths. |
| Remote-only bootstrap | Require explicit expected repository identity and SHA, reviewed artifact manifest, and scoped external-runtime authorization before GPU inspection. Separate CPU validation from GPU execution. No silent SHA/provider defaults, arbitrary requirements execution, detector call, or authorization creation. Unit-test commands with mocks; syntax-check without executing GPU paths locally. |

Integration acceptance should use only tiny synthetic transfer objects in a separately authorized
test destination, prove no overwritten historical object, and record exact round-trip identities.
Provider choice and any real GPU qualification are separate owner decisions. The existing helper's
normal hash/size check is not a substitute for required exact critical-object verification.

## Scope and safety

Do not download whole datasets, checkpoint collections, retirement archives, or `.local`, and do
not recursively hash Drive merely to hydrate or index a task. Use existing manifests, file
metadata, and narrowly selected inputs. Scientific execution always requires current explicit
scope and runtime-specific authorization; cloud migration itself grants none.
