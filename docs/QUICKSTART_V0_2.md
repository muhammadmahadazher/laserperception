# LaserPerception v0.2.0 quickstart

This guide presents the two supported ROS 2 paths in v0.2.0: the existing v0.1-compatible
model-ready replay and the new raw XYZ plus time-aware TF boundary. It does not download or
redistribute nuScenes, the checkpoint, ONNX, or TensorRT engine, and it does not imply that the
lightweight Python wheel installs the GPU/ROS stack.

## Prerequisites and external assets

The core package supports Python 3.10–3.13 on CPU. The validated detector/ROS stack is Ubuntu 22.04
under WSL2, Python 3.10, CUDA 11.8, TensorRT 8.6.1, and ROS 2 Humble. Timings from the measured RTX
4060 Laptop environment are not portable hardware guarantees.

Clone the repository and keep heavy caches on WSL ext4:

```bash
git clone https://github.com/muhammadmahadazher/laserperception.git
cd laserperception

export LASERPERCEPTION_M1_CACHE="$HOME/.cache/laserperception"
export LASERPERCEPTION_M2_CACHE="$HOME/.cache/laserperception"
export LASERPERCEPTION_M2_VENV="$HOME/.venvs/laserperception-m2"
export LASERPERCEPTION_M3_BUILD_ROOT="$HOME/.cache/laserperception/m3/colcon"
export LASERPERCEPTION_NUSCENES_ROOT="$HOME/datasets/nuscenes"
```

Obtain nuScenes v1.0-mini from the [official site](https://www.nuscenes.org/nuscenes), accept its
terms, and extract it outside the repository. Then install and prepare the pinned external stack:

```bash
bash scripts/setup_detection_m2.sh
source "$LASERPERCEPTION_M2_VENV/bin/activate"
python scripts/detection/prepare_nuscenes_mini.py \
  --data-root "$LASERPERCEPTION_NUSCENES_ROOT"
```

Produce the external ONNX and engine only if they are absent; they are not distributed:

```bash
python scripts/detection/check_m2_tensorrt.py
python scripts/detection/export_m2_onnx.py \
  --data-root "$LASERPERCEPTION_NUSCENES_ROOT"
python scripts/detection/build_m2_tensorrt.py
bash scripts/setup_ros2_m3.sh
```

See the historical [v0.1.0 quickstart](QUICKSTART_V0_1.md) for the frozen asset locations, SHA256
values, and detailed setup troubleshooting.

## A. Model-ready replay (v0.1-compatible)

This path publishes an already accumulated `PointCloud2` with scalar float32 `x`, `y`, `z`, and
`time_lag` directly to the detector:

```text
/laserperception/points_model_ready
    -> laserperception_detector
    -> /laserperception/detections
```

Launch the real W1 replay and RViz visualization:

```bash
bash scripts/run_v0_1_demo.sh
```

The historical script name is retained for compatibility. It validates the pinned assets and runs
`m3_demo.launch.py`; it does not exercise the new raw builder. The screenshot in the README is real
output from this W1 model-ready replay.

## B. Raw XYZ plus time-aware TF (new in v0.2)

The raw path requires each non-empty `sensor_msgs/PointCloud2` message to contain scalar float32
fields `x`, `y`, and `z`. Field order may vary. Extra fields such as intensity, ring, or per-point
timestamp are ignored by the frozen detector path. Non-finite XYZ rows are removed without
reordering; an all-invalid message is rejected.

The message header stamp is the acquisition time. A valid time-aware TF tree must connect every
historical source acquisition to the current acquisition through the configured fixed frame. The
same frame name at different timestamps is not assumed to be identity. Missing TF rejects the
current output; there is no latest-TF substitution or silent sweep skip.

Default topics and executables:

| Stage | Topic or executable |
|---|---|
| Raw input | `/laserperception/points_raw` |
| Builder | `laserperception_multisweep_builder` |
| Model-ready output | `/laserperception/points_model_ready` |
| Detector | `laserperception_detector` |
| Detections | `/laserperception/detections` |

The accepted launch and configuration are:

- `ros2/laserperception_ros/launch/m45b_raw_multisweep.launch.py`;
- `ros2/laserperception_ros/config/m45b_multisweep.yaml`.

Source the installed environment, then run the real nuScenes raw replay:

```bash
source /opt/ros/humble/setup.bash
source "$LASERPERCEPTION_M2_VENV/bin/activate"
source "$LASERPERCEPTION_M3_BUILD_ROOT/install/setup.bash"

ros2 launch laserperception_ros m45b_raw_multisweep.launch.py
```

The default launch publishes actual nuScenes raw acquisitions and the matching time-indexed TF,
runs the builder and unchanged detector, and opens RViz. It is a reproducibility example, not a
physical-sensor validation or vendor driver.

For an external compatible raw topic and existing TF source, disable the bundled replay:

```bash
ros2 launch laserperception_ros m45b_raw_multisweep.launch.py \
  run_raw_replay:=false run_rviz:=false
```

The packaged replay fixed frame is `nuscenes_map`; a live system must configure an appropriate fixed
frame and supply complete time-indexed TF history. LaserPerception does not provide localization,
odometry, sensor calibration, or per-point intra-scan deskew.

## What to expect

The builder starts with the current sweep, grows to current plus at most ten prior acquisitions,
and publishes float32 XYZT in current-then-nearest-history order. The downstream `exact_fast`,
TensorRT FP16 PointPillars network, MMDeploy postprocess, and Detection3DArray conversion are the
same path used by model-ready input.

M4.5/M4.5b was correctness and integration work, not a throughput campaign. Historical model-ready
M3 measurements do not establish a raw-ingestion Hz capability. See
[`RAW_LIDAR_ROS2.md`](RAW_LIDAR_ROS2.md) for the complete failure contract and
[`MULTISWEEP.md`](MULTISWEEP.md) for exactness evidence and transform conventions.

## Core-only smoke

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
python - <<'PY'
import numpy as np
from laserperception import PointCloud, __version__

cloud = PointCloud(xyz=np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
print(__version__, len(cloud), cloud.xyz.dtype)
PY
```
