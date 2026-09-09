#!/usr/bin/env bash
set -euo pipefail

usage() { echo "Usage: $0 DRIVE_RELATIVE_PATH LOCAL_FILE EXPECTED_SHA256" >&2; exit 2; }
[[ $# -eq 3 ]] || usage
command -v rclone >/dev/null || { echo "rclone is required" >&2; exit 1; }
source_object=${1#/}
destination=$2
expected=$3
[[ -n "$source_object" && "$source_object" != *..* && "$expected" =~ ^[0-9a-fA-F]{64}$ ]] || usage
mkdir -p "$(dirname "$destination")"
tmp="${destination}.downloading.$$"
trap 'rm -f "$tmp"' EXIT
rclone copyto --no-traverse "lpdrive:$source_object" "$tmp"
actual=$(sha256sum "$tmp" | awk '{print $1}')
[[ "${actual,,}" == "${expected,,}" ]] || { echo "download SHA256 mismatch" >&2; exit 1; }
mv -f "$tmp" "$destination"
trap - EXIT
printf '%s  %s\n' "$actual" "$destination"
