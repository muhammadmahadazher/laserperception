# Canonical Drive helpers

These fail-closed helpers address only the configured `lpdrive:` root. They never read or store
credentials; provision rclone externally. Paths are relative to that root.

- `drive_push.sh LOCAL DRIVE_PATH [SHA256]` stages to a temporary object, verifies it by a full
  round trip, moves it to the final name, and verifies again.
- `drive_pull.sh DRIVE_PATH LOCAL SHA256` downloads to a temporary local file and publishes it only
  after verification.
- `verify_drive_object.py DRIVE_PATH SHA256` streams and verifies one remote object.
- `task_snapshot.sh TASK_ID OUTPUT.tar.gz` captures a Git bundle, binary working-tree patch, status,
  and provenance. Upload its output into the task capsule.
- `persist_task.sh TASK_DIRECTORY [CAPSULE_NAME]` non-destructively uploads a prepared capsule and
  refuses an existing destination. Its normal remote check may use a common hash or file size, so
  use the full-round-trip helpers above for critical scientific objects.

A capsule uses `_CLOUD_WORK/<TASK_ID>/{00_MANIFEST,01_FINAL_OUTPUTS,02_RAW_EVIDENCE,03_LOGS,04_TEMP_PRESERVED,05_ENVIRONMENT,06_WORKSPACE_SNAPSHOT}/` and validates its manifest against
`TASK_MANIFEST.schema.json`. Transfers do not grant scientific authorization. Never put rclone
configs, tokens, secret environment values, or credentials in a capsule.
