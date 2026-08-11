#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${LASERPERCEPTION_M2_PYTHON:-python3.10}"
VENV_PATH="${LASERPERCEPTION_M2_VENV:-$HOME/.venvs/laserperception-m2}"
M1_CACHE_ROOT="${LASERPERCEPTION_M1_CACHE:-$HOME/.cache/laserperception}"
M2_CACHE_ROOT="${LASERPERCEPTION_M2_CACHE:-$HOME/.cache/laserperception}"
MMDET3D_ROOT="$M1_CACHE_ROOT/mmdetection3d-v1.4.0"
MMDEPLOY_ROOT="$M2_CACHE_ROOT/mmdeploy-v1.3.1"
CHECKPOINT_DIR="$M1_CACHE_ROOT/checkpoints"
MMDET3D_COMMIT="fe25f7a51d36e3702f961e198894580d83c4387b"
MMDEPLOY_COMMIT="bc75c9d6c8940aa03d0e1e5b5962bd930478ba77"
CHECKPOINT_NAME="hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth"
CHECKPOINT_SHA256="f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0"
CHECKPOINT_URL="https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth"
MMCV_WHEEL="https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl"
CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
CUDA_KEYRING_SHA256="d93190d50b98ad4699ff40f4f7af50f16a76dac3bb8da1eaaf366d47898ff8df"
TENSORRT_PACKAGE_VERSION="8.6.1.6-1+cuda11.8"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -r /proc/sys/kernel/osrelease ]] || ! grep -qi microsoft /proc/sys/kernel/osrelease; then
  echo "error: M2 setup must run inside Ubuntu 22.04 on WSL2" >&2
  exit 2
fi

for command_name in "$PYTHON_BIN" git curl gcc g++ nvidia-smi sha256sum dpkg apt-get; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "error: required command is missing: $command_name" >&2
    exit 2
  fi
done

if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 10))'; then
  echo "error: M2 requires Python 3.10" >&2
  exit 2
fi

if [[ "$EUID" -eq 0 ]]; then
  AS_ROOT=()
elif command -v sudo >/dev/null 2>&1; then
  AS_ROOT=(sudo)
else
  echo "error: install sudo or run as root for the pinned NVIDIA system packages" >&2
  exit 2
fi

KEYRING_PACKAGE="$(mktemp --suffix=.deb)"
trap 'rm -f "$KEYRING_PACKAGE"' EXIT
curl -L --fail --progress-bar -o "$KEYRING_PACKAGE" "$CUDA_KEYRING_URL"
ACTUAL_KEYRING_SHA256="$(sha256sum "$KEYRING_PACKAGE" | cut -d ' ' -f 1)"
if [[ "$ACTUAL_KEYRING_SHA256" != "$CUDA_KEYRING_SHA256" ]]; then
  echo "error: NVIDIA CUDA keyring SHA256 mismatch" >&2
  exit 3
fi
"${AS_ROOT[@]}" dpkg -i "$KEYRING_PACKAGE"
"${AS_ROOT[@]}" apt-get update
"${AS_ROOT[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  'cuda-toolkit-config-common=11.8.89-1' \
  'cuda-toolkit-11-config-common=11.8.89-1' \
  'cuda-toolkit-11-8-config-common=11.8.89-1' \
  'libcublas-11-8=11.11.3.6-1' \
  'libcublas-dev-11-8=11.11.3.6-1' \
  'libcudnn8=8.9.7.29-1+cuda11.8' \
  'libcudnn8-dev=8.9.7.29-1+cuda11.8' \
  "libnvinfer8=$TENSORRT_PACKAGE_VERSION" \
  "libnvinfer-plugin8=$TENSORRT_PACKAGE_VERSION" \
  "libnvinfer-vc-plugin8=$TENSORRT_PACKAGE_VERSION" \
  "libnvonnxparsers8=$TENSORRT_PACKAGE_VERSION" \
  "libnvparsers8=$TENSORRT_PACKAGE_VERSION" \
  "libnvinfer-headers-dev=$TENSORRT_PACKAGE_VERSION" \
  "libnvinfer-headers-plugin-dev=$TENSORRT_PACKAGE_VERSION" \
  "libnvinfer-dev=$TENSORRT_PACKAGE_VERSION" \
  "libnvinfer-plugin-dev=$TENSORRT_PACKAGE_VERSION" \
  "libnvonnxparsers-dev=$TENSORRT_PACKAGE_VERSION" \
  "python3-libnvinfer=$TENSORRT_PACKAGE_VERSION"

mkdir -p "$(dirname "$VENV_PATH")" "$M1_CACHE_ROOT" "$M2_CACHE_ROOT" "$CHECKPOINT_DIR"
if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_PATH"
fi
PYTHON="$VENV_PATH/bin/python"
SITE_PACKAGES="$VENV_PATH/lib/python3.10/site-packages"
if [[ ! -e "$SITE_PACKAGES/tensorrt" ]]; then
  ln -s /usr/lib/python3.10/dist-packages/tensorrt "$SITE_PACKAGES/tensorrt"
fi
if [[ ! -e "$SITE_PACKAGES/tensorrt-8.6.1.dist-info" ]]; then
  ln -s /usr/lib/python3.10/dist-packages/tensorrt-8.6.1.dist-info \
    "$SITE_PACKAGES/tensorrt-8.6.1.dist-info"
fi

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
if [[ "$(git -C "$MMDET3D_ROOT" rev-parse HEAD)" != "$MMDET3D_COMMIT" ]]; then
  echo "error: MMDetection3D checkout differs from the pinned v1.4.0 commit" >&2
  exit 4
fi
"$PYTHON" -m pip install -e "$MMDET3D_ROOT"

if [[ ! -d "$MMDEPLOY_ROOT/.git" ]]; then
  git clone --branch v1.3.1 --depth 1 --recurse-submodules --shallow-submodules \
    https://github.com/open-mmlab/mmdeploy.git "$MMDEPLOY_ROOT"
fi
if [[ "$(git -C "$MMDEPLOY_ROOT" rev-parse HEAD)" != "$MMDEPLOY_COMMIT" ]]; then
  echo "error: MMDeploy checkout differs from the pinned v1.3.1 commit" >&2
  exit 4
fi
"$PYTHON" -m pip install -e "$MMDEPLOY_ROOT" --no-deps
"$PYTHON" -m pip install \
  'aenum==3.1.17' \
  'grpcio==1.83.0' \
  'multiprocess==0.70.19' \
  'onnx==1.14.1' \
  'prettytable==3.18.0' \
  'protobuf==3.20.2' \
  'six==1.17.0' \
  'tensorboard==2.14.1' \
  'terminaltables==3.1.10'
"$PYTHON" -m pip install 'numpy==1.26.4' 'opencv-python==4.10.0.84'
"$PYTHON" -m pip install -e "$REPO_ROOT"
"$PYTHON" -m pip check

CHECKPOINT_PATH="$CHECKPOINT_DIR/$CHECKPOINT_NAME"
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  curl -L --fail --progress-bar -o "$CHECKPOINT_PATH.part" "$CHECKPOINT_URL"
  mv "$CHECKPOINT_PATH.part" "$CHECKPOINT_PATH"
fi
if [[ "$(sha256sum "$CHECKPOINT_PATH" | cut -d ' ' -f 1)" != "$CHECKPOINT_SHA256" ]]; then
  echo "error: checkpoint SHA256 mismatch; remove the cached file and rerun setup" >&2
  exit 5
fi

"$PYTHON" "$REPO_ROOT/scripts/detection/check_m2_tensorrt.py"
echo "activate with: source $VENV_PATH/bin/activate"
echo "M2 environment verified; M1 virtual environment was not modified"
