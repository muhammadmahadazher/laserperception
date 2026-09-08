#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: LP_DRIVE_ROOT=lpdrive: $0 TASK_DIRECTORY [CAPSULE_NAME]" >&2
  exit 2
}

[[ $# -ge 1 && $# -le 2 ]] || usage
command -v rclone >/dev/null || { echo "rclone is required" >&2; exit 127; }
command -v sha256sum >/dev/null || { echo "sha256sum is required" >&2; exit 127; }

: "${LP_DRIVE_ROOT:?Set LP_DRIVE_ROOT to the canonical Drive remote root (normally lpdrive:)}"
[[ "$LP_DRIVE_ROOT" == "lpdrive:" ]] || {
  echo "Refusing non-canonical LP_DRIVE_ROOT: $LP_DRIVE_ROOT" >&2
  exit 2
}

source_dir=${1%/}
[[ -d "$source_dir" ]] || { echo "Task directory not found: $source_dir" >&2; exit 2; }
capsule_name=${2:-$(basename "$source_dir")}
[[ "$capsule_name" != */* && -n "$capsule_name" ]] || usage
destination="${LP_DRIVE_ROOT}_CLOUD_WORK/${capsule_name}/"

manifest=$(mktemp)
remote_manifest=$(mktemp)
trap 'rm -f "$manifest" "$remote_manifest"' EXIT
(
  cd "$source_dir"
  find . -type f -print0 | sort -z | xargs -0 -r sha256sum
) >"$manifest"

# Copy is intentionally non-destructive: this helper never deletes remote files.
rclone copy "$source_dir" "$destination" --checksum
rclone copyto "$manifest" "${destination}00_MANIFEST/FILE_SHA256.txt"
rclone copyto "${destination}00_MANIFEST/FILE_SHA256.txt" "$remote_manifest"
cmp --silent "$manifest" "$remote_manifest" || {
  echo "Remote checksum manifest did not round-trip exactly" >&2
  exit 1
}
rclone check "$source_dir" "$destination" --one-way --download
printf 'Persisted and verified %s -> %s\n' "$source_dir" "$destination"
