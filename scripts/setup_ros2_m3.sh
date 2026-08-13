#!/usr/bin/env bash
set -euo pipefail

M2_VENV="${LASERPERCEPTION_M2_VENV:-$HOME/.venvs/laserperception-m2}"
PYTHON="$M2_VENV/bin/python"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_APT_SOURCE_VERSION="1.2.0"
ROS_APT_SOURCE_SHA256="767884cf4ed03116b9d64438930a832ed854147ae435279a7924dfdf60f94433"
ROS_APT_SOURCE_URL="https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.jammy_all.deb"

if [[ "$(. /etc/os-release && printf '%s' "$VERSION_ID")" != "22.04" ]]; then
  echo "error: M3 requires Ubuntu 22.04 Jammy" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "error: existing M2 environment is missing; run scripts/setup_detection_m2.sh first" >&2
  exit 2
fi
if ! "$PYTHON" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 10))'; then
  echo "error: the existing M2 environment must use Python 3.10" >&2
  exit 2
fi

if [[ "$EUID" -eq 0 ]]; then
  AS_ROOT=()
elif command -v sudo >/dev/null 2>&1; then
  AS_ROOT=(sudo)
else
  echo "error: install sudo or run as root to install official ROS packages" >&2
  exit 2
fi

if [[ ! -f /etc/apt/sources.list.d/ros2.sources ]]; then
  package="$(mktemp --suffix=.deb)"
  trap 'rm -f "$package"' EXIT
  curl -L --fail --progress-bar -o "$package" "$ROS_APT_SOURCE_URL"
  if [[ "$(sha256sum "$package" | cut -d ' ' -f 1)" != "$ROS_APT_SOURCE_SHA256" ]]; then
    echo "error: official ROS apt-source package SHA256 mismatch" >&2
    exit 3
  fi
  "${AS_ROOT[@]}" dpkg -i "$package"
fi
"${AS_ROOT[@]}" apt-get update
"${AS_ROOT[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  ros-humble-ros-base \
  ros-humble-rviz2 \
  ros-humble-vision-msgs \
  ros-humble-sensor-msgs-py \
  ros-humble-visualization-msgs \
  ros-dev-tools

source /opt/ros/humble/setup.bash
"$PYTHON" -m pip install -e "$REPO_ROOT"
"$PYTHON" - <<'PY'
import rclpy
import sensor_msgs
import sensor_msgs_py
import vision_msgs
import visualization_msgs
import torch
import tensorrt
import mmdeploy
import mmdet3d

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("CUDA device unavailable in combined ROS/M2 environment")
probe = torch.ones((16, 16), device="cuda:0") @ torch.ones((16, 16), device="cuda:0")
if not torch.isfinite(probe).all().item():
    raise SystemExit("CUDA verification operation failed")
torch.cuda.synchronize(0)
print({
    "ros_distro": "humble",
    "pytorch": torch.__version__,
    "tensorrt": tensorrt.__version__,
    "mmdeploy": mmdeploy.__version__,
    "mmdet3d": mmdet3d.__version__,
    "gpu": torch.cuda.get_device_name(0),
})
PY

BUILD_ROOT="${LASERPERCEPTION_M3_BUILD_ROOT:-$HOME/.cache/laserperception/m3/colcon}"
mkdir -p "$BUILD_ROOT"
colcon --log-base "$BUILD_ROOT/log" build \
  --base-paths "$REPO_ROOT/ros2/laserperception_ros" \
  --build-base "$BUILD_ROOT/build" \
  --install-base "$BUILD_ROOT/install" \
  --symlink-install

echo "M3 ROS environment verified"
echo "source /opt/ros/humble/setup.bash"
echo "source $M2_VENV/bin/activate"
echo "source $BUILD_ROOT/install/setup.bash"
