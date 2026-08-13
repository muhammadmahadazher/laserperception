# ROS 2 Humble interface (M3 complete)

M3 wraps the frozen M2 TensorRT FP16 detector with a ROS 2 Humble interface. It consumes a
**model-ready multi-sweep** `sensor_msgs/PointCloud2` and publishes canonical
`vision_msgs/Detection3DArray` messages. It is not a raw physical-LiDAR adapter.

The final production policy is explicit:

- deployed ROS path: `voxelization_mode: exact_fast`, `provenance_mode: live`;
- historical M2/evidence default: `voxelization_mode: official`, `provenance_mode: full`; and
- no fallback from `exact_fast` to `deterministic=False` or another semantics-changing path.

If exact-fast initialization fails, startup fails loudly. Optional `official` mode remains available
for debugging and historical-reference use.

## Frozen environment and artifacts

M3 targets Ubuntu 22.04 under WSL2, ROS 2 Humble, Python 3.10, CUDA runtime 11.8,
TensorRT 8.6.1, MMDeploy 1.3.1, and the existing M2 environment. The final measurement used
`rmw_fastrtps_cpp` on the NVIDIA GeForce RTX 4060 Laptop GPU.

| Artifact | Frozen SHA256 |
|---|---|
| PointPillars checkpoint | `f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0` |
| ONNX | `61ce22a8ca31498675c32576bfb94f0093d31dc95d2762f7254bf915a59ecc16` |
| TensorRT FP16 engine | `a005f75852097cd9b193750560b214cc3d5237ae9b6c106c7fca3d4fc348714b` |

No checkpoint, ONNX, or engine was changed, re-exported, or rebuilt in M3.

Build the isolated ROS workspace on WSL ext4:

```bash
export LASERPERCEPTION_M1_CACHE="$HOME/.cache/laserperception"
export LASERPERCEPTION_M2_CACHE="$HOME/.cache/laserperception"
export LASERPERCEPTION_NUSCENES_ROOT="$HOME/datasets/nuscenes"
bash scripts/setup_ros2_m3.sh

source /opt/ros/humble/setup.bash
source "$HOME/.venvs/laserperception-m2/bin/activate"
source "$HOME/.cache/laserperception/m3/colcon/install/setup.bash"
```

The setup uses official ROS packages; it does not add ROS to the lightweight core wheel.

## Production voxelization and provenance

`exact_fast` reproduces the pinned MMCV deterministic hard-voxelization outputs exactly while
using the pinned dynamic coordinate CUDA operation plus PyTorch tensor grouping. It is a supported
LaserPerception deployment optimization preserving upstream semantics; it is not described as an
upstream MMDetection3D implementation. It adds no custom CUDA, C++, or TensorRT plugin.

The historical `full` provenance mode retains per-frame voxel SHA256 metadata. The ROS `live`
mode records lightweight semantic provenance without hashing complete voxel tensors. The policy
does not change voxel values, TensorRT outputs, postprocessing, or detections.

## Input contract

The default input topic is `/laserperception/points_model_ready`. Every non-empty message must
contain finite `float32` fields:

| Field | Meaning |
|---|---|
| `x` | forward coordinate in the current LiDAR frame |
| `y` | left coordinate in the current LiDAR frame |
| `z` | up coordinate in the current LiDAR frame |
| `time_lag` | current-sweep timestamp minus source-sweep timestamp |

Field order is arbitrary and extra fields are ignored. Missing `time_lag` is rejected; it is never
treated as intensity. Historical XYZ values must already be transformed into the current LiDAR
frame, current points use `time_lag = 0`, and the input must already combine the current keyframe
with up to ten historical sweeps under the pinned MMDetection3D semantics. M3 performs no TF
lookup, history buffering, motion compensation, or sweep aggregation.

M3 does not change point-cloud range, voxel size, maximum points, maximum voxels, engine profile,
network, class names, thresholds, or postprocessing.

## Output contract

The default output topic is `/laserperception/detections`. For every accepted input:

- the array and each detection preserve the exact source stamp and frame;
- geometric center is copied directly, with no height/2 Z shift;
- length/width/height map to `bbox.size.x/y/z`;
- x-forward/y-left/z-up yaw maps to `(0, 0, sin(yaw/2), cos(yaw/2))`;
- class names and scores remain the upstream detector values;
- IDs remain empty because M3 does not add tracking; and
- velocity is not overloaded into an unrelated message field.

Optional `/laserperception/markers` output recreates visualization markers per frame and does not
imply persistent object identity.

## QoS, replay, and visualization

Checked-in QoS remains bounded:

- input: volatile, keep-last depth 1, best effort;
- detections: volatile, keep-last depth 5, reliable; and
- markers: the same bounded output profile.

The deployment YAML explicitly selects exact-fast/live and replays representative index 42 by
default. The replay node uses the existing nuScenes preparation path and serializes its exact
model-ready `Nx4` array; it does not independently rebuild sweep history.

```bash
ros2 launch laserperception_ros m3_demo.launch.py
```

A configured replay rate is a synthetic throughput stress cadence, not native nuScenes annotated
keyframe timing. The supplied RViz2 configuration displays model-ready points and markers in
`nuscenes_lidar_top`. Foxglove can subscribe to the same topics.

## Final production correctness gates

Correctness was rerun through the actual production integration at exact measurement commit
`a129b3507597b25f44ab1a833562f68883ebe8ce`:

| Gate | Result |
|---|---:|
| Official vs production exact-fast `voxels`/`num_points`/`coors` | **81/81 bit-exact** |
| Frozen raw TensorRT outputs | **20/20 exact** |
| Frozen final DetectionFrames | **20/20 exact** |
| PointCloud2 point values/hashes and Detection3DArray semantics | **20/20 exact** |
| Low-rate W1 PointCloud2 → exact-fast runtime → Detection3DArray smoke | **1/1 pass** |

The external correctness record SHA256 is
`000ba4bd15bc4349a0df29a2252819e00326c406e5b1dc0e787c0c060359d388`.
A difference would have stopped performance measurement.

## Final representative ROS performance

The canonical record is
[`benchmarks/m3/results/rtx4060_ros2_humble_exact_tensorrt_fp16.json`](../benchmarks/m3/results/rtx4060_ros2_humble_exact_tensorrt_fp16.json).
It measures W1, `mini_val` index 42: 10 historical sweeps plus the current keyframe,
354,182 points, exact-fast voxelization, live provenance, and the unchanged FP16 engine.

The eligible session used AC power, the existing host Ultimate Performance plan, a 30-second
sustained GPU warmup, 20 message warmups, and 200 measured accepted/output opportunities.
The immutable PointCloud2 replay payload was built once before timing; only its source timestamp
was refreshed immediately before each publish.

Timing boundaries:

1. **Callback processing:** PointCloud2 callback entry through publish-return after validation,
   conversion, exact-fast voxelization, TensorRT, unchanged MMDeploy postprocessing,
   DetectionFrame/Detection3DArray construction, and publication.
2. **Same-host loopback:** source publisher ROS timestamp through Detection3DArray sink reception.
   This is not sensor-to-actuator latency and excludes one-time replay-payload construction.

### Offered 20 Hz — not sustained

| Boundary | Count | Mean | Median | P90 | P95 | Min | Max | Population std | >50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Callback processing | 200 | 77.391 ms | 75.701 ms | 85.496 ms | 89.197 ms | 63.475 ms | 258.069 ms | 14.799 ms | 200 (100%) |
| Same-host loopback | 200 | 138.457 ms | 134.250 ms | 158.999 ms | 165.446 ms | 93.869 ms | 458.677 ms | 42.475 ms | 200 (100%) |

| Rate/count | Measured |
|---|---:|
| Requested offered rate | 20.000 Hz |
| Effective offered rate | 19.509 Hz |
| Replay published total | 398 |
| Detector received/accepted/published total | 221 / 221 / 221 |
| Sink received total | 221 |
| Measured offered inputs | 359 |
| Measured input drops | 159 |
| Detector-to-sink drops | 0 |
| Final processing backlog | 0 |
| Effective detector output rate | 10.825 Hz |

Backlog behavior was measured, not inferred from the latency median:

| Half | Callback entry interval median | Mean | P95 | Input drops |
|---|---:|---:|---:|---:|
| First | 87.243 ms | 90.605 ms | 100.266 ms | 77 |
| Second | 90.463 ms | 94.027 ms | 106.794 ms | 82 |

Entry intervals and drop counts both grew between halves. The run therefore fell behind and did
not demonstrate sustained 20 Hz. The telemetry session was eligible: both halves were P0, memory
clock was 8001 MHz, median SM clocks were 2595/2640 MHz, temperatures were 65.0/65.5 °C, and power
draw medians were 37.69/39.145 W. WSL did not expose a power-limit value. These observations do not
establish clock causality.

### Bounded sustainable-rate characterization

Only the authorized two follow-ups were run:

| Offered rate | Effective output | Measured drops | Half deterioration | Result |
|---|---:|---:|---|---|
| 10 Hz | 9.949 Hz | 0/200 | none | **sustained** |
| 15 Hz | 13.336 Hz | 21/221 | no growth, but loss remained | **not sustained** |

At 10 Hz, callback and loopback medians were 65.483 and 81.400 ms. At 15 Hz they were 60.215 and
121.219 ms. No 5 Hz run or further search was performed. The highest tested clean rate was 10 Hz.

M3 is complete despite the honest 20 Hz failure; the milestone completion criterion required exact
integration, preserved ROS semantics, representative measurement, and accurate disclosure—not a
forced sensor-rate pass.

## Preserved chronology

- **M3A:** scene-start index 0 20 Hz stress failed at 238.255 ms callback median, 303.283 ms
  loopback median, 3.990 Hz output, and 875 bounded input drops.
- **M3B-V1:** `deterministic=False` was fast but rejected because saturated retained-point subsets
  and observable repeatability/fidelity changed.
- **M3B-V2:** the exact tensor candidate passed 81/81 voxel, W1/W2 repeatability, frozen raw output,
  and final-frame gates. In that direct diagnostic, W1 live median fell from 333.137 to 43.168 ms.
- **Final M3:** exact-fast/live was integrated fail-closed and measured through ROS on W1; 20 Hz
  failed, 10 Hz sustained, and 15 Hz failed.

Historical diagnostics remain under [`benchmarks/m3/`](../benchmarks/m3/); the final result does
not overwrite them.

## Deferred post-v0.1 work

The following are documented backlog only:

- unchanged MMDeploy postprocessing optimization (about 21 ms in the V2 component ledger);
- ROS/DDS/executor investigation if later required;
- further exact-fast tuning;
- custom CUDA only if future evidence justifies it;
- INT8; and
- other detector architectures.

No postprocess, DDS, executor, custom CUDA, tracking, raw-sensor adapter, INT8, training, Jetson,
camera-fusion, or M4 work was implemented in this cycle.
