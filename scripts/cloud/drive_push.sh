#!/usr/bin/env bash
set -euo pipefail

usage() { echo "Usage: $0 LOCAL_FILE DRIVE_RELATIVE_PATH [EXPECTED_SHA256]" >&2; exit 2; }
[[ $# -ge 2 && $# -le 3 ]] || usage
command -v rclone >/dev/null || { echo "rclone is required" >&2; exit 1; }
command -v sha256sum >/dev/null || { echo "sha256sum is required" >&2; exit 1; }
source_file=$1
destination=${2#/}
[[ -f "$source_file" && -n "$destination" && "$destination" != */ ]] || usage
[[ "$destination" != *..* ]] || { echo "Drive path must not contain '..'" >&2; exit 2; }
local_sha=$(sha256sum "$source_file" | awk '{print $1}')
[[ $# -lt 3 || "$local_sha" == "$3" ]] || { echo "local SHA256 mismatch" >&2; exit 1; }
tmp="${destination}.uploading.$(date -u +%Y%m%dT%H%M%SZ).$$"
cleanup() { rclone deletefile "lpdrive:$tmp" >/dev/null 2>&1 || true; }
trap cleanup EXIT
rclone copyto --no-traverse "$source_file" "lpdrive:$tmp"
verify_copy=$(mktemp)
trap 'rm -f "$verify_copy"; cleanup' EXIT
rclone copyto --no-traverse "lpdrive:$tmp" "$verify_copy"
remote_sha=$(sha256sum "$verify_copy" | awk '{print $1}')
[[ "$remote_sha" == "$local_sha" ]] || { echo "temporary upload SHA256 mismatch" >&2; exit 1; }
rclone moveto "lpdrive:$tmp" "lpdrive:$destination"
rclone copyto --no-traverse "lpdrive:$destination" "$verify_copy"
final_sha=$(sha256sum "$verify_copy" | awk '{print $1}')
[[ "$final_sha" == "$local_sha" ]] || { echo "final upload SHA256 mismatch" >&2; exit 1; }
rm -f "$verify_copy"
trap - EXIT
printf '%s  lpdrive:%s\n' "$final_sha" "$destination"
