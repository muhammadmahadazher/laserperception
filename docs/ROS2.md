# ROS 2 Humble interface (M3A)

M3A wraps the frozen M2 TensorRT FP16 detector with a ROS 2 interface. It consumes a
**model-ready multi-sweep** point cloud and publishes canonical `vision_msgs/Detection3DArray`
messages. It is not a raw physical-LiDAR adapter. The implementation passed the 20-sample
transport-fidelity gate, but its first 20 Hz rate diagnostic failed the preregistered M3A target.
M3B performance work requires separate owner/reviewer authorization.

## Frozen environment and artifacts

M3A targets Ubuntu 22.04 Jammy, ROS 2 Humble, Python 3.10, CUDA 11.8, TensorRT 8.6.1, and the
existing M2 virtual environment. It does not upgrade the system Python, NVIDIA driver, CUDA,
TensorRT, MMDeploy, or MMDetection3D.

The runtime retains the M2 assets outside the repository and fails during startup if the engine is
missing or its checksum differs:

| Artifact | Frozen SHA256 |
|---|---|
| PointPillars checkpoint | `f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0` |
| ONNX | `61ce22a8ca31498675c32576bfb94f0093d31dc95d2762f7254bf915a59ecc16` |
| TensorRT FP16 engine | `a005f75852097cd9b193750560b214cc3d5237ae9b6c106c7fca3d4fc348714b` |

Install the official Humble packages and build the isolated ROS workspace while keeping build
outputs on WSL ext4:

```bash
export LASERPERCEPTION_M1_CACHE="$HOME/.cache/laserperception"
export LASERPERCEPTION_M2_CACHE="$HOME/.cache/laserperception"
export LASERPERCEPTION_NUSCENES_ROOT="$HOME/datasets/nuscenes"
bash scripts/setup_ros2_m3.sh

source /opt/ros/humble/setup.bash
source "$HOME/.venvs/laserperception-m2/bin/activate"
source "$HOME/.cache/laserperception/m3/colcon/install/setup.bash"
```

The script uses the official ROS apt-source package and apt packages; it does not install an
unofficial `rclpy` wheel. The core `laserperception` wheel remains ROS-free and CPU-testable.

## Exact input contract

The default input is `/laserperception/points_model_ready` with type
`sensor_msgs/msg/PointCloud2`. Every message must be non-empty and contain these named fields:

| Field | Type | Meaning |
|---|---|---|
| `x` | `float32` | forward coordinate in the current LiDAR frame |
| `y` | `float32` | left coordinate in the current LiDAR frame |
| `z` | `float32` | up coordinate in the current LiDAR frame |
| `time_lag` | `float32` | current-sweep timestamp minus source-sweep timestamp |

Field order is arbitrary and extra fields are ignored. Organized and unorganized layouts are
accepted when `point_step`, `row_step`, field offsets, and data length are valid. Required values
must be finite. A missing `time_lag` is rejected with a throttled error; it is never treated as
intensity.

The points must already combine the current sweep and up to ten prior sweeps using the pinned
MMDetection3D semantics. Historical XYZ values must already be transformed into
`msg.header.frame_id` for the current sweep; current points use `time_lag = 0`. The M3A detector
performs no TF lookup, ego-motion reconstruction, history buffering, or sweep aggregation.

The adapter constructs only the minimum official MMDetection3D tensor/data-sample batch. The
existing official data preprocessor still owns voxelization, and M3 does not change the range,
voxel size, maximum points, maximum voxels, engine profile, network, classes, or postprocessing.

## Output contract

The default machine output is `/laserperception/detections` with type
`vision_msgs/msg/Detection3DArray`. For every accepted input:

- the array header and every contained `Detection3D.header` copy the exact input stamp and
  `frame_id`;
- `bbox.center.position` copies the LaserPerception geometric center directly, with **no**
  `height / 2` Z shift;
- `(length, width, height)` maps to `bbox.size.(x, y, z)`;
- x-forward/y-left/z-up yaw maps to `(0, 0, sin(yaw/2), cos(yaw/2))`;
- the sole hypothesis retains the upstream class name and detector score and uses the same box
  pose;
- `Detection3D.id` is empty because M3 does not track objects; and
- detector velocity is not transported because `Detection3DArray` has no canonical velocity
  field. No unrelated field is overloaded.

Optional `/laserperception/markers` output uses `visualization_msgs/MarkerArray`. Markers are
cleared and recreated on every frame, share the input header, and do not imply persistent object
identity.

## QoS

The checked-in defaults are explicit and bounded:

- input: volatile, keep-last depth 1, best effort;
- detections: volatile, keep-last depth 5, reliable; and
- markers: the same bounded output profile.

Depth and reliability parameters are configurable in
`ros2/laserperception_ros/config/m3_ros2.yaml`. No queue is unbounded.

## Replay and visualization

The replay node calls the existing verified nuScenes preparation path, extracts the exact `Nx4`
model-ready array, and serializes it to PointCloud2. It never independently reconstructs sweeps.

```bash
# Detector, repeating mini_val index 0 replay, and RViz2
ros2 launch laserperception_ros m3_demo.launch.py

# One-shot example without the launch file
ros2 run laserperception_ros laserperception_replay --ros-args \
  -p one_shot:=true -p loop:=false -p start_index:=0
```

`start_index`, `sample_count`, `one_shot`, `loop`, and `publish_rate_hz` support one sample or a
sequence. The default 20 Hz mode is a synthetic throughput stress cadence, **not** native nuScenes
annotated-keyframe timing.

The supplied RViz2 config displays `/laserperception/points_model_ready` and
`/laserperception/markers` with `nuscenes_lidar_top` as the replay fixed frame. Change the fixed
frame to the actual input `frame_id` for another model-ready source. No map/odom transform is
fabricated. Foxglove can subscribe to the same three topics. Capture screenshots or recordings
locally; do not commit bags or large video files.

## Correctness and latency gates

Run the frozen 20-sample transport gate after sourcing the environment:

```bash
python scripts/ros2/validate_m3_roundtrip.py
```

For each parity-v2 sample it compares exact point hashes, official voxel hashes, TensorRT raw
output hashes/statistics, and final detections between the original dataset path and the real M3
PointCloud2 round trip. A mismatch stops benchmarking. The frozen suite contains 19 samples with
10 historical sweeps plus the current keyframe and one scene-start sample, index 0, with zero
historical sweeps.

The M3 benchmark records two distinct boundaries:

1. **Callback processing:** callback entry through PointCloud2 validation/conversion, official
   voxelization, TensorRT, shared postprocessing, Detection3DArray construction, and return from
   `publisher.publish()`.
2. **Same-host ROS loopback:** replay publication stamp through same-host input transport,
   scheduling, detector processing, output transport, and sink reception. It is not
   sensor-to-actuator latency.

The protocol repeats mini_val index 0—a scene-start keyframe with zero accumulated historical
sweeps—performs 20 warmups and 200 measured messages at a synthetic 20 Hz, and reports full
distribution, deadline, count, effective-rate, and queue evidence. At exact
implementation commit `d54da837602de2924825d3045cb4a17b72c5b7b0`, replay held 19.945 Hz but
callback median/P95 were 238.255/274.637 ms and same-host loopback median/P95 were
303.283/352.550 ms. All 200 observations in both boundaries exceeded 50 ms. The detector produced
221 outputs from 1,096 published inputs (875 bounded-QoS input drops), for 3.990 Hz effective output;
there was no rejected message, detector-to-sink loss, or final processing backlog.

The M3A gate therefore failed. See `benchmarks/m3/README.md` and its diagnostic-only JSON. M3B is
indicated for review, but no optimization, postprocess change, profiling claim, or bottleneck
assumption is part of M3A.

## Live-sensor limitation

M3A must not be described as a drop-in single-sweep sensor driver interface. A future raw-sensor
adapter would need timestamped transforms, scan history, motion compensation into the current
LiDAR frame, and exact `time_lag` construction. That adapter, tracking, C++, custom CUDA, INT8,
training, Jetson, camera fusion, and M4 are outside this milestone.
