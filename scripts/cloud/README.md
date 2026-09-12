# External task persistence helper

The `scripts/cloud/` path is retained for compatibility. Normal development is local and
CPU-capable; Codex Cloud is not required. This helper only transfers a deliberately prepared task
capsule. It does not provision a GPU or authorize scientific execution. See the
[compute workflow](../../docs/CLOUD_WORKFLOW.md) for the full runtime gates and the scoped
follow-up plan from closed, unmerged PR #32. That PR's other helpers are not installed here.

`persist_task.sh` copies a prepared task directory into the canonical `_CLOUD_WORK` root and checks
its contents. It refuses an existing capsule destination, uses non-destructive `rclone copy`,
writes a SHA256 manifest, downloads that small manifest for an exact comparison, and runs ordinary
`rclone check` without `--download`. It contains no credentials or scientific logic and refuses a
remote root other than `lpdrive:`.

```bash
LP_DRIVE_ROOT=lpdrive: scripts/cloud/persist_task.sh \
  /path/to/2026-09-08_example_ab12cd
```

Optionally pass a second argument to set the destination capsule name. Prepare the task manifest,
applicable capsule subdirectories, safe environment record, and final outputs before invoking it.
Never put rclone configs, tokens, secret environment values, or credentials in a capsule.

The SHA256 manifest records the canonical local identities. The ordinary remote check compares
uploaded files using hashes supported by both sides, falling back to size when no common hash is
available; it does not force a full byte download of a potentially large capsule. When a scientific
protocol or task requires stronger verification for a critical artifact, perform and record an
explicit full round-trip of that artifact separately.
