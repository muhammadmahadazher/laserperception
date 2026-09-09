#!/usr/bin/env bash
set -euo pipefail

# Engineering bootstrap only. It deliberately contains no detector entry point.
EXPECTED_REPO_SHA=${EXPECTED_REPO_SHA:-56ee126bd188de3753134e7ab4ba4496392c60e4}
ARTIFACT_MANIFEST=${ARTIFACT_MANIFEST:?Set ARTIFACT_MANIFEST to an owner-reviewed JSON manifest}
WORK_ROOT=${WORK_ROOT:-"$HOME/laserperception-cloud"}
RCLONE_CONFIG_SOURCE=${RCLONE_CONFIG_SOURCE:-}
APPROVED_REQUIREMENTS=${APPROVED_REQUIREMENTS:-}
DSVT_SHA=8cfc2a6f23eed0b10aabcdc4768c60b184357061
OPENPCDET_SHA=233f849829b6ac19afb8af8837a0246890908755

[[ "$(uname -s)" == Linux ]] || { echo "Linux is required" >&2; exit 1; }
for command in git python3 rclone nvidia-smi rg; do command -v "$command" >/dev/null || { echo "$command is required" >&2; exit 1; }; done
nvidia-smi -L | rg '^GPU [0-9]+:' >/dev/null || { echo "No NVIDIA GPU detected" >&2; exit 1; }
repo=$(git rev-parse --show-toplevel)
actual_sha=$(git -C "$repo" rev-parse HEAD)
[[ "$actual_sha" == "$EXPECTED_REPO_SHA" ]] || { echo "Expected repo $EXPECTED_REPO_SHA, got $actual_sha" >&2; exit 1; }
python3 -m json.tool "$ARTIFACT_MANIFEST" >/dev/null

if [[ -n "$RCLONE_CONFIG_SOURCE" ]]; then
  [[ -f "$RCLONE_CONFIG_SOURCE" ]] || { echo "RCLONE_CONFIG_SOURCE is not a file" >&2; exit 1; }
  export RCLONE_CONFIG="$RCLONE_CONFIG_SOURCE"
fi
rclone lsf lpdrive: --max-depth 1 >/dev/null
mkdir -p "$WORK_ROOT/artifacts" "$WORK_ROOT/upstream"
python3 - "$ARTIFACT_MANIFEST" "$repo" "$WORK_ROOT/artifacts" <<'PY'
import json, pathlib, subprocess, sys
manifest, repo, output = sys.argv[1:]
for item in json.load(open(manifest, encoding="utf-8"))["artifacts"]:
    subprocess.run([f"{repo}/scripts/cloud/drive_pull.sh", item["drive_path"], str(pathlib.Path(output) / item["local_name"]), item["sha256"]], check=True)
PY

clone_exact() {
  local url=$1 destination=$2 sha=$3
  if [[ ! -d "$destination/.git" ]]; then git clone --filter=blob:none --no-checkout "$url" "$destination"; fi
  git -C "$destination" fetch --depth 1 origin "$sha"
  git -C "$destination" checkout --detach "$sha"
  [[ "$(git -C "$destination" rev-parse HEAD)" == "$sha" ]]
}
clone_exact https://github.com/Haiyang-W/DSVT.git "$WORK_ROOT/upstream/DSVT" "$DSVT_SHA"
clone_exact https://github.com/open-mmlab/OpenPCDet.git "$WORK_ROOT/upstream/OpenPCDet" "$OPENPCDET_SHA"

if [[ -n "$APPROVED_REQUIREMENTS" ]]; then
  [[ -f "$APPROVED_REQUIREMENTS" ]] || { echo "APPROVED_REQUIREMENTS is not a file" >&2; exit 1; }
  python3 -m pip install --require-hashes -r "$APPROVED_REQUIREMENTS"
fi
cat <<'STOP'
Bootstrap integrity checks completed.
STOP: NO SCIENTIFIC INFERENCE IS AUTHORIZED OR STARTED.
A new runtime policy, owner review, and runtime-specific Stage-R-only authorization are required.
STOP
