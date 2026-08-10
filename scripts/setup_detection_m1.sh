#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${LASERPERCEPTION_M1_PYTHON:-python3.10}"
VENV_PATH="${LASERPERCEPTION_M1_VENV:-$HOME/.venvs/laserperception-m1}"
CACHE_ROOT="${LASERPERCEPTION_M1_CACHE:-$HOME/.cache/laserperception}"
MMDET3D_ROOT="$CACHE_ROOT/mmdetection3d-v1.4.0"
CHECKPOINT_DIR="$CACHE_ROOT/checkpoints"
MMDET3D_COMMIT="fe25f7a51d36e3702f961e198894580d83c4387b"
CHECKPOINT_NAME="hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth"
CHECKPOINT_SHA256="f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0"
CHECKPOINT_URL="https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d/$CHECKPOINT_NAME"
MMCV_WHEEL="https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -r /proc/sys/kernel/osrelease ]] || ! grep -qi microsoft /proc/sys/kernel/osrelease; then
  echo "error: M1 setup must run inside Ubuntu 22.04 on WSL2" >&2
  exit 2
fi

for command_name in "$PYTHON_BIN" git curl gcc g++ nvidia-smi; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "error: required command is missing: $command_name" >&2
    echo "install the prerequisites documented in docs/DETECTION_ENVIRONMENT.md" >&2
    exit 2
  fi
done

if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 10))'; then
  echo "error: M1 requires Python 3.10" >&2
  exit 2
fi

mkdir -p "$(dirname "$VENV_PATH")" "$CACHE_ROOT" "$CHECKPOINT_DIR"
if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_PATH"
fi
PYTHON="$VENV_PATH/bin/python"

"$PYTHON" -m pip install --upgrade \
  'pip==24.3.1' 'setuptools==69.5.1' 'wheel==0.45.1'
"$PYTHON" -m pip install \
  'torch==2.1.0' 'torchvision==0.16.0' \
  --index-url https://download.pytorch.org/whl/cu118
"$PYTHON" -m pip install 'numpy==1.26.4' 'opencv-python==4.10.0.84'
"$PYTHON" -m pip install 'mmengine==0.10.7'
"$PYTHON" -m pip install "$MMCV_WHEEL"
"$PYTHON" -m pip install 'mmdet==3.2.0'

if [[ ! -d "$MMDET3D_ROOT/.git" ]]; then
  git clone --branch v1.4.0 --depth 1 \
    https://github.com/open-mmlab/mmdetection3d.git "$MMDET3D_ROOT"
fi
ACTUAL_COMMIT="$(git -C "$MMDET3D_ROOT" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$MMDET3D_COMMIT" ]]; then
  echo "error: MMDetection3D checkout is $ACTUAL_COMMIT, expected $MMDET3D_COMMIT" >&2
  exit 3
fi

"$PYTHON" -m pip install -e "$MMDET3D_ROOT"
"$PYTHON" -m pip install 'numpy==1.26.4' 'opencv-python==4.10.0.84'
"$PYTHON" -m pip install -e "$REPO_ROOT"
"$PYTHON" -m pip check

CHECKPOINT_PATH="$CHECKPOINT_DIR/$CHECKPOINT_NAME"
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  curl -L --fail --progress-bar -o "$CHECKPOINT_PATH.part" "$CHECKPOINT_URL"
  mv "$CHECKPOINT_PATH.part" "$CHECKPOINT_PATH"
fi
ACTUAL_CHECKPOINT_SHA256="$(sha256sum "$CHECKPOINT_PATH" | cut -d ' ' -f 1)"
if [[ "$ACTUAL_CHECKPOINT_SHA256" != "$CHECKPOINT_SHA256" ]]; then
  echo "error: checkpoint SHA256 mismatch; remove the cached file and rerun setup" >&2
  exit 4
fi

"$PYTHON" - <<'PY'
import mmcv
import mmdet
import mmdet3d
import mmengine
import numpy
import torch
from mmcv.ops import Voxelization, nms

versions = {
    "torch": torch.__version__,
    "mmcv": mmcv.__version__,
    "mmengine": mmengine.__version__,
    "mmdet": mmdet.__version__,
    "mmdet3d": mmdet3d.__version__,
    "numpy": numpy.__version__,
}
expected = {
    "torch": "2.1.0+cu118",
    "mmcv": "2.1.0",
    "mmengine": "0.10.7",
    "mmdet": "3.2.0",
    "mmdet3d": "1.4.0",
    "numpy": "1.26.4",
}
if versions != expected:
    raise SystemExit(f"version mismatch: expected {expected}, found {versions}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA")
if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4060 Laptop GPU":
    raise SystemExit(f"unexpected M1 GPU: {torch.cuda.get_device_name(0)}")
probe = torch.ones((32, 32), device="cuda")
_ = probe @ probe
torch.cuda.synchronize()
print("M1 environment verified")
print(versions)
print({"cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})
PY

echo "activate with: source $VENV_PATH/bin/activate"
echo "checkpoint verified: $CHECKPOINT_NAME"
