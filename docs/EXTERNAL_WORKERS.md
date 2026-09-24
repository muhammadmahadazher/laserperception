# External worker engineering tools

`laserperception.worker` separates transport integrity, environment qualification and scientific
authorization. Importing it and generating plans do not select a provider, contact a service,
inspect hardware, download a model or execute a detector.

Provider neutrality describes the software boundary, not project history. RunPod A40 workers have
been used for M8 qualification and Stage R, while bounded later allocation attempts were blocked
before Pod creation. No provider is required by the package. Current operational state is recorded
in [PROJECT_STATUS.md](PROJECT_STATUS.md) and
[M8_S1_EXTERNAL_RUNTIME_STATUS.md](m8/M8_S1_EXTERNAL_RUNTIME_STATUS.md).

## CPU-safe commands

```text
laserperception worker manifest validate task.json
laserperception worker plan task.json --model dsvt-pillar-transfusion-m8
laserperception worker verify-artifact artifact.json --root <selected-input-root>
```

The plan contains the explicit execution commit, declared input/output identities, reviewed model
artifact identities, runtime requirements, capsule target and prerequisite review steps. Commands
are argument arrays, not executable shell strings. Only qualification tasks receive a proposed
bootstrap command. A reference in a plan is not proof that authorization has been verified.

Construct a `TaskManifest` through its typed Python constructor or strict `from_json()` loader.
All fields are required, including explicit nulls/empty arrays where appropriate. This minimal
engineering-only example has no artifact transfers or scientific execution:

```json
{
  "schema_version": "1.0",
  "task_id": "engineering-example",
  "created_utc": "2026-09-13T00:00:00Z",
  "repository": "muhammadmahadazher/laserperception",
  "execution_commit": "21b5afb4765ef0a8109dbbe191e003f6a9bccbc6",
  "branch": null,
  "task_type": "engineering_only",
  "runtime": {
    "runtime_id": "unselected",
    "provider": null,
    "external": false,
    "description": "CPU planning example"
  },
  "environment": [],
  "inputs": [],
  "outputs": [],
  "steps": ["inspect-model"],
  "authorizations": [],
  "attempted_calls": 0,
  "accepted_calls": 0,
  "failed_calls": 0,
  "checkpoint_state": "pending",
  "completion_state": "planned",
  "unique_worker_state_remaining": false
}
```

The example commit identifies the P0 merge, not a runtime default. Select and verify the execution
commit for each actual task. Task kinds are `engineering_only`, `qualification`, `stage_r`,
`primary`, and `training`; representing a kind does not authorize it. Training plans are rejected.
Qualification and scientific kinds require matching role/runtime/commit authorization references.
Engineering/qualification tasks cannot record detector calls. A complete task requires verified
persistence and no unique worker-only state. Failed or incomplete scientific accounting can retain
attempted calls exceeding accepted plus failed calls; it must not invent accepted results.

Environment metadata has an explicit key allowlist. Do not supply credentials as values or dump an
environment. Artifact identities require name, relative_path, byte_size, sha256, role and provenance.

## Individual-object transfers

`VerifiedArtifact`, `RemoteObject`, and `ArtifactTransfer` are public Python APIs. Supply explicit
source/destination roots and a Drive-backed attempt-log directory or a worker capsule that will be
persisted before retirement. Rclone configuration/credentials are provisioned separately.

```python
from pathlib import Path
from laserperception.worker import ArtifactTransfer, RemoteObject, VerifiedArtifact

artifact = VerifiedArtifact.from_json(Path("artifact.json").read_text())
remote = RemoteObject("lpdrive", "_CLOUD_WORK/selected-task/input.bin")
transfer = ArtifactTransfer(Path("selected-task/transfer-attempts"))
# Explicit transfer actions, only when their source/destination have been selected:
# transfer.pull(artifact, remote, Path("selected-input-root"))
# transfer.push(artifact, Path("selected-output-root"), remote)
```

Pulls require a fresh destination and verify streamed byte count and SHA256 before installation.
Pushes refuse an existing remote object and verify a full downloaded round trip for every object.
`--ignore-existing` prevents intentional overwrites; if another writer races to create a differing
object, verification fails. This is not an atomic distributed transaction. Allocate unique capsule
paths and coordinate concurrent writers. No destructive sync, remote move, delete or cleanup runs.
Failed attempts retain staged/round-trip bytes and append-only sanitized event journals. Raw rclone
stderr is excluded because it can expose credentials. Missing rclone fails with an actionable error.

Portable paths reject absolute/drive/UNC forms, dot/parent/empty components, backslashes, control
characters, Windows device names and resolved symlink escapes. Keep selected roots stable during
operations; this API is not an isolation boundary against hostile concurrent filesystem mutation.
No whole-project traversal, recursive hydration or automatic snapshot is performed.

## External-only bootstrap

`worker bootstrap` is for a separately authorized external runtime. It requires all of:
an explicit external-worker flag, qualification mode, exact expected commit, runtime ID, a
qualification task manifest, a clean matching repository, verified input artifacts, and a hashed
owner qualification authorization file. Missing context fails before GPU collection.

The authorization file has exactly `schema_version` equal to
`laserperception.worker.qualification-authorization.v1`, `owner_approved` equal to true, `task_id`,
`runtime_id`, `execution_commit`, and `mode` equal to `qualification`. The owner supplies this file;
the tooling neither creates authorization nor converts historical authorization into new permission.
Its SHA256 and role are recorded in the task's `authorizations` array.

Only after every guard passes does bootstrap collect GPU model/VRAM/free memory/driver and optional
framework versions. A fresh JSON output and failure journal record the result for owner review.
It installs nothing, runs no detector, and never grants Stage R/primary/training authorization.
Tests use injected collectors and repository readers; no GPU path runs on the local workstation.

New-runtime policy binding, owner review, fresh Stage-R-only authorization, repeated Stage R,
preserved raw evidence, and a new primary authorization remain mandatory under
[the compute workflow](CLOUD_WORKFLOW.md). Qualification metadata alone is not a qualified scientific
runtime. Preserve each valuable output and failed attempt to Drive before retiring an external worker.

`scripts/cloud/persist_task.sh` and historical `_CLOUD_STATE`/`_CLOUD_WORK` names remain compatible.
The new tools do not reinterpret that helper's historical checksum checks as full-byte verification.

## M8 preliminary readiness without a selected runtime

`worker plan --task m8-qualification --execution-commit FULL_CURRENT_MAIN_SHA --repository-root .`
creates a CPU-only plan while future permissions are missing. It never fabricates a qualification
TaskManifest. After runtime selection and fresh owner qualification scope, use the existing manifest/
plan/bootstrap sequence. Bootstrap remains environment-only with detector calls 0; any GT-blind
DSVT sizing/capacity requires a distinct explicit runtime scope and honest attempt accounting.
See [M8 readiness/runbook](m8/M8_EXTERNAL_RUNTIME_RUNBOOK.md).
