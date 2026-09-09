#!/usr/bin/env bash
set -euo pipefail

usage() { echo "Usage: $0 TASK_ID OUTPUT_TAR_GZ" >&2; exit 2; }
[[ $# -eq 2 && "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || usage
repo=$(git rev-parse --show-toplevel)
out=$(realpath -m "$2")
mkdir -p "$(dirname "$out")"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git -C "$repo" bundle create "$tmp/repository.bundle" --all
git -C "$repo" diff --binary HEAD > "$tmp/working-tree.patch"
git -C "$repo" status --porcelain=v1 > "$tmp/status.txt"
printf 'task_id=%s\nhead=%s\nbranch=%s\ncreated_at_utc=%s\n' \
  "$1" "$(git -C "$repo" rev-parse HEAD)" "$(git -C "$repo" branch --show-current)" "$(date -u +%FT%TZ)" > "$tmp/snapshot.txt"
tar -C "$tmp" -czf "${out}.tmp" .
mv -f "${out}.tmp" "$out"
sha256sum "$out"
