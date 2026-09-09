# Cloud-first workflow

This document is the repository-side authoritative policy for persistent LaserPerception work.
The canonical private Google Drive binding is `lpdrive:` at folder ID
`18Q73IkiVcFT0EXAowIPlOiISNk0mkhHJ`; the repository is
`muhammadmahadazher/laserperception`.

## Persistent systems and disposable workers

### GitHub

GitHub is authoritative for source code, tracked documentation, tests, configs, pull requests,
release history, and compact or final evidence suitable for Git review. Changes reach that source
of truth through the normal branch, commit, and Codex/GitHub PR flow.

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

### Cloud worker

A Codex Cloud or rented GPU worker is disposable. No unique LaserPerception state may remain only
on a worker when a task finishes. Git-suitable changes must exist in Git/PR; valuable non-Git state
must be Drive-backed. Preserve failed scientific attempts and valuable debug or temporary files.

An internal worker branch may be named `work`, and a worker may have no normal Git remote. That is
not repository corruption. Verify the required base commit rather than requiring the literal branch
name `main`; source identity is the starting Git commit plus the Codex/GitHub PR handoff.

## Cloud task lifecycle

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

1. commit and hand Git-suitable work to GitHub through the normal Codex/GitHub PR flow;
2. upload all non-Git outputs, valuable temporary/untracked files, and logs to Drive;
3. update the Drive state and indexes without blindly overwriting useful state;
4. create and verify a task manifest and environment record;
5. inventory untracked, ignored, outside-repository, and project-relevant `/tmp` files; and
6. confirm `unique_worker_state_remaining = false`.

## Drive layout and task capsules

Every cloud task receives a unique directory:

```text
_CLOUD_WORK/YYYY-MM-DD_<task_slug>_<short_id>/
├── 00_MANIFEST/
├── 01_FINAL_OUTPUTS/
├── 02_RAW_EVIDENCE/
├── 03_LOGS/
├── 04_TEMP_PRESERVED/
├── 05_ENVIRONMENT/
└── 06_WORKSPACE_SNAPSHOT/
```

Create only the applicable subdirectories. A full repository copy is unnecessary when tracked
source is in GitHub. Record Git HEAD and preserve a patch when useful, plus untracked files, ignored
task artifacts, logs, temporary work, external artifacts, safe environment metadata, and a workspace snapshot or patch bundle. Preserve
substantially more for expensive or scientific jobs. Use `scripts/cloud/persist_task.sh` for a
non-destructive checked copy. The helper refuses an existing capsule destination rather than
merging new files into old state. Its SHA256 manifest records canonical local identities; the
small manifest is downloaded and compared byte-for-byte, while ordinary `rclone check` verifies
the uploaded files through hashes supported by the remote (or size when no common hash exists).
It does not download every remote byte by default. A scientific protocol or task may additionally
require an explicit full round-trip of individual critical artifacts.

## GPU worker policy

A future GPU worker is an ephemeral external runtime. The required order is:

1. rent a provider-neutral GPU worker;
2. hydrate the verified GitHub commit and only required Drive artifacts;
3. conduct GT-blind runtime qualification;
4. bind a machine-specific runtime policy;
5. repeat Stage R on that runtime;
6. obtain owner review and a **new** primary authorization;
7. run only the newly authorized primary workload;
8. upload and verify raw evidence after every pass;
9. compact Git-suitable evidence through GitHub;
10. finalize and verify the Drive task capsule; and
11. destroy the worker only after confirming no unique worker-only state.

The retired-machine primary authorization is not portable and must never be reused. No provider is
selected by this policy.

## Scope and safety

Do not download whole datasets, checkpoint collections, retirement archives, or `.local`, and do
not recursively hash Drive merely to hydrate or index a task. Use existing manifests, file
metadata, and narrowly selected inputs. Scientific execution always requires current explicit
scope and runtime-specific authorization; cloud migration itself grants none.
