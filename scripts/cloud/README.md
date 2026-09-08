# Cloud task persistence helper

`persist_task.sh` copies a prepared task directory into the canonical `_CLOUD_WORK` root and checks
its contents. It uses non-destructive `rclone copy`, writes a SHA256 manifest, downloads that
manifest for an exact comparison, and runs `rclone check`. It contains no credentials or scientific
logic and refuses a remote root other than `lpdrive:`.

```bash
LP_DRIVE_ROOT=lpdrive: scripts/cloud/persist_task.sh \
  /path/to/2026-09-08_example_ab12cd
```

Optionally pass a second argument to set the destination capsule name. Prepare the task manifest,
applicable capsule subdirectories, safe environment record, and final outputs before invoking it.
Never put rclone configs, tokens, secret environment values, or credentials in a capsule.
