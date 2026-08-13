# LaserPerception v0.1.0 quickstart

This guide takes a fresh clone to the real ROS 2 W1 replay. It does not download nuScenes silently,
ship a checkpoint/ONNX/engine, or imply that installing the lightweight Python wheel installs the
GPU/ROS stack.

## Choose the path you need

| Path | Supported environment | What it installs |
|---|---|---|
| Core Python package | Python 3.10–3.13, CPU | `laserperception`, NumPy, laspy |
| GPU detector and ROS demo | Ubuntu 22.04 under WSL2, Python 3.10, CUDA 11.8, TensorRT 8.6.1, ROS 2 Humble | Isolated external detector environment and ROS workspace |

The pinned GPU path was measured on one NVIDIA GeForce RTX 4060 Laptop GPU. Other CUDA GPUs may
work, but the published timings are not portable capability guarantees.

## 1. Clone and choose external roots

Run the GPU/ROS commands inside Ubuntu 22.04 on WSL2:

```bash
git clone https://github.com/muhammadmahadazher/laserperception.git
cd laserperception

export LASERPERCEPTION_M1_CACHE="$HOME/.cache/laserperception"
export LASERPERCEPTION_M2_CACHE="$HOME/.cache/laserperception"
export LASERPERCEPTION_M2_VENV="$HOME/.venvs/laserperception-m2"
export LASERPERCEPTION_M3_BUILD_ROOT="$HOME/.cache/laserperception/m3/colcon"
export LASERPERCEPTION_NUSCENES_ROOT="$HOME/datasets/nuscenes"
```

Keep these roots on WSL ext4, not inside the repository or a synchronized drive. The repository
never redistributes the dataset or deployment binaries.

## 2. Obtain nuScenes v1.0-mini

Register with nuScenes, review its terms, and download v1.0-mini from the
[official nuScenes site](https://www.nuscenes.org/nuscenes). Extract it so the root contains at
least `v1.0-mini/`, `samples/`, and `sweeps/`.

LaserPerception cannot accept or download the dataset for you. See [DATASETS.md](DATASETS.md) for
layout and licensing details.

## 3. Install the pinned detector environment

The setup installs pinned CUDA/TensorRT system packages and a Python 3.10 environment, clones
pinned MMDetection3D/MMDeploy sources into the external caches, downloads the official checkpoint,
and verifies CUDA. It may require `sudo` for apt packages.

```bash
bash scripts/setup_detection_m2.sh
source "$LASERPERCEPTION_M2_VENV/bin/activate"
```

Prepare the official nuScenes ten-sweep metadata:

```bash
python scripts/detection/prepare_nuscenes_mini.py \
  --data-root "$LASERPERCEPTION_NUSCENES_ROOT"
```

## 4. Produce or verify the external deployment assets

The v0.1 demo expects these external artifacts:

| Artifact | Expected location under the configured cache | SHA256 |
|---|---|---|
| Pretrained checkpoint | `$LASERPERCEPTION_M1_CACHE/checkpoints/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth` | `f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0` |
| ONNX | `$LASERPERCEPTION_M2_CACHE/m2/pointpillars.onnx` | `61ce22a8ca31498675c32576bfb94f0093d31dc95d2762f7254bf915a59ecc16` |
| TensorRT engine | `$LASERPERCEPTION_M2_CACHE/m2/engines/pointpillars_fp16.engine` | `a005f75852097cd9b193750560b214cc3d5237ae9b6c106c7fca3d4fc348714b` |

The checkpoint setup is automatic. ONNX and the TensorRT engine are generated locally because the
serialized engine is environment-specific and is not distributed. If they are absent, run the
existing frozen M2 path:

```bash
python scripts/detection/check_m2_tensorrt.py
python scripts/detection/export_m2_onnx.py \
  --data-root "$LASERPERCEPTION_NUSCENES_ROOT"
python scripts/detection/build_m2_tensorrt.py
```

These commands use the pinned manifests and existing profile; they are not optimization steps. See
[TENSORRT.md](TENSORRT.md) for the exact deployment boundary and historical parity evidence.

## 5. Install and build ROS 2 Humble

```bash
bash scripts/setup_ros2_m3.sh
```

This installs external ROS packages, installs the lightweight core editable into the existing M2
environment, and builds `ros2/laserperception_ros` under the external M3 build root.

## 6. Launch the real W1 demo

```bash
bash scripts/run_v0_1_demo.sh
```

The wrapper:

- validates the configured cache and dataset roots;
- verifies the frozen checkpoint, ONNX, and engine hashes;
- verifies the installed W1 `exact_fast` / `live` ROS config;
- verifies pinned packages and executes a small CUDA operation; and
- launches the existing `m3_demo.launch.py` replay and RViz configuration.

It does not download nuScenes, rebuild the engine, or hide a missing prerequisite. The source
workload is `mini_val` index 42: 10 historical sweeps plus the current keyframe, 354,182 points in
the measured evidence. The launch config offers replay at 20 Hz as the preserved M3 stress cadence;
the measured system did **not** sustain 20 Hz. The highest tested clean sustained rate was 10 Hz.

RViz displays:

- `/laserperception/points_model_ready` (`sensor_msgs/PointCloud2`); and
- `/laserperception/markers` (actual predicted 3D box markers).

The detector also publishes `/laserperception/detections` (`vision_msgs/Detection3DArray`). Set
`LASERPERCEPTION_DEMO_RVIZ=false` to launch without RViz.

The demo input is already model-ready and contains `x`, `y`, `z`, and `time_lag`. v0.1.0 does not
build sweep history from a raw physical LiDAR topic.

## Core-only quickstart

The wheel is intentionally independent of ROS, CUDA, PyTorch, OpenMMLab, and TensorRT:

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

## Troubleshooting

- `nuScenes ... missing or unprepared`: confirm the external dataset layout, then rerun
  `prepare_nuscenes_mini.py`.
- `SHA256 mismatch`: do not bypass it. Remove only the named external artifact and reproduce it from
  the pinned setup/export path.
- `built ROS workspace is missing`: rerun `scripts/setup_ros2_m3.sh` with the same
  `LASERPERCEPTION_M3_BUILD_ROOT`.
- no RViz window under WSL2: confirm WSLg is enabled; run with `LASERPERCEPTION_DEMO_RVIZ=false` to
  verify the headless ROS path separately.
- performance differs from the release notes: warm the GPU and record clocks, power, temperature,
  and host state where available. One timing session is not portable across systems or sessions.