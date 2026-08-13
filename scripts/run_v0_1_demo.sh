#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "error: $*" >&2
  exit 2
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
M2_VENV="${LASERPERCEPTION_M2_VENV:-$HOME/.venvs/laserperception-m2}"
M3_BUILD_ROOT="${LASERPERCEPTION_M3_BUILD_ROOT:-$HOME/.cache/laserperception/m3/colcon}"
DATA_ROOT="${LASERPERCEPTION_NUSCENES_ROOT:-}"
RUN_RVIZ="${LASERPERCEPTION_DEMO_RVIZ:-true}"
PYTHON="$M2_VENV/bin/python"

[[ -r /etc/os-release ]] || fail "run this demo inside the documented Ubuntu 22.04 WSL2 environment"
# shellcheck disable=SC1091
source /etc/os-release
[[ "$VERSION_ID" == "22.04" ]] || fail "v0.1 GPU/ROS demo requires Ubuntu 22.04; found ${VERSION_ID:-unknown}"
[[ -r /proc/sys/kernel/osrelease ]] && grep -qi microsoft /proc/sys/kernel/osrelease \
  || fail "v0.1 GPU/ROS demo is validated on WSL2; use docs/QUICKSTART_V0_1.md"
[[ -n "$DATA_ROOT" ]] \
  || fail "set LASERPERCEPTION_NUSCENES_ROOT to an externally obtained nuScenes v1.0-mini root"
[[ -x "$PYTHON" ]] \
  || fail "missing M2 Python environment at $M2_VENV; run scripts/setup_detection_m2.sh"
[[ -r /opt/ros/humble/setup.bash ]] \
  || fail "ROS 2 Humble is missing; run scripts/setup_ros2_m3.sh"
[[ -r "$M3_BUILD_ROOT/install/setup.bash" ]] \
  || fail "built ROS workspace is missing at $M3_BUILD_ROOT; run scripts/setup_ros2_m3.sh"
[[ "$RUN_RVIZ" == "true" || "$RUN_RVIZ" == "false" ]] \
  || fail "LASERPERCEPTION_DEMO_RVIZ must be true or false"

for command_name in nvidia-smi sha256sum ros2; do
  command -v "$command_name" >/dev/null 2>&1 \
    || fail "required command is missing: $command_name"
done

export LASERPERCEPTION_M1_CACHE="${LASERPERCEPTION_M1_CACHE:-$HOME/.cache/laserperception}"
export LASERPERCEPTION_M2_CACHE="${LASERPERCEPTION_M2_CACHE:-$HOME/.cache/laserperception}"
export LASERPERCEPTION_NUSCENES_ROOT="$DATA_ROOT"

# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1090
source "$M2_VENV/bin/activate"
# shellcheck disable=SC1090
source "$M3_BUILD_ROOT/install/setup.bash"

ROS_SHARE="$($PYTHON -c 'from ament_index_python.packages import get_package_share_directory; print(get_package_share_directory("laserperception_ros"))')"
"$PYTHON" "$REPO_ROOT/scripts/detection/check_v0_1_assets.py" \
  --repo-root "$REPO_ROOT" \
  --data-root "$DATA_ROOT" \
  --ros-share "$ROS_SHARE"

"$PYTHON" - <<'PY'
import mmdeploy
import mmdet3d
import rclpy
import tensorrt
import torch

if torch.__version__ != "2.1.0+cu118":
    raise SystemExit(f"error: expected PyTorch 2.1.0+cu118, found {torch.__version__}")
if tensorrt.__version__ != "8.6.1":
    raise SystemExit(f"error: expected TensorRT 8.6.1, found {tensorrt.__version__}")
if mmdeploy.__version__ != "1.3.1" or mmdet3d.__version__ != "1.4.0":
    raise SystemExit("error: pinned OpenMMLab versions differ; rerun setup_detection_m2.sh")
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("error: CUDA device 0 is unavailable")
probe = torch.ones((16, 16), device="cuda:0") @ torch.ones((16, 16), device="cuda:0")
if not torch.isfinite(probe).all().item():
    raise SystemExit("error: CUDA verification operation returned a non-finite value")
torch.cuda.synchronize(0)
print({
    "gpu": torch.cuda.get_device_name(0),
    "pytorch": torch.__version__,
    "tensorrt": tensorrt.__version__,
    "mmdeploy": mmdeploy.__version__,
    "mmdet3d": mmdet3d.__version__,
    "ros": "humble",
})
PY

echo "Launching real W1 replay (mini_val index 42) through exact_fast/live production config."
echo "Close RViz or press Ctrl-C to stop. Set LASERPERCEPTION_DEMO_RVIZ=false for headless launch."
exec ros2 launch laserperception_ros m3_demo.launch.py \
  run_replay:=true \
  run_rviz:="$RUN_RVIZ"