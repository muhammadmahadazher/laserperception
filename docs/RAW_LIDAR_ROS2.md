# Raw LiDAR PointCloud2 ingestion (v0.2.0)

v0.2.0 releases the accepted M4.5b ROS boundary that turns compatible raw single-sweep PointCloud2
messages plus time-aware TF into the existing model-ready PointCloud2 contract:

```text
/laserperception/points_raw
          +
source-time/current-time tf2 through a fixed frame
          |
          v
laserperception_multisweep_builder
          |
          v
/laserperception/points_model_ready
          |
          v
unchanged exact_fast + TensorRT detector
          |
          v
/laserperception/detections
```

The node is not a localization system, calibration service, vendor SDK driver, or physical-sensor
accuracy claim. It accepts compatible ROS 2 PointCloud2 streams containing float32 XYZ when a
valid time-aware TF tree is supplied.

## Input contract

The default raw topic is `/laserperception/points_raw` with best-effort, volatile, keep-last depth
5 QoS. Each non-empty message must contain scalar fields:

| Field | ROS datatype | Meaning |
|---|---|---|
| `x` | float32 | raw point X in `header.frame_id` |
| `y` | float32 | raw point Y in `header.frame_id` |
| `z` | float32 | raw point Z in `header.frame_id` |

Field order, row padding, endianness, organized layout, and additional fields are handled by the
shared PointCloud2 decoder. Extra vendor fields such as intensity, ring, or per-point timestamp are
allowed but ignored by the frozen detector path. They are not silently substituted for XYZ or
`time_lag`.

### Non-finite rows

Rows with non-finite X, Y, or Z are removed deterministically. Retained rows keep their original
PointCloud2 traversal order. The node records the number filtered. A cloud with no valid XYZ rows
is rejected and is not published as model-ready input.

### Timestamp policy

The PointCloud2 header stamp is the acquisition time. Integer nanoseconds identify the live
acquisition; the M4.5a boundary quantizes down to integer microseconds, matching the pinned nuScenes
source contract. The current sweep stamp is the transform target time. Every historical sweep keeps
its own source stamp, and `time_lag` is computed from current minus historical acquisition time.
There is no per-point firing-time deskew.

The model-ready output preserves the current input header and uses the configured target frame (or
the current message frame when `target_frame` is empty).

## Time-aware TF contract

The packaged configuration requires `fixed_frame: nuscenes_map`. This is a replay default; a live
system must provide an appropriate fixed frame and complete time-indexed TF history. `target_frame`
is empty by default, meaning the current raw message frame. A non-empty target is accepted only
when it equals that current frame; cross-frame conversion of the current sweep is not implemented.

For each historical acquisition the node calls:

```python
buffer.lookup_transform_full(
    target_frame,  # current raw frame
    current_stamp,  # target time
    historical.frame_id,  # historical raw frame
    historical_stamp,  # source time
    fixed_frame,
    timeout=Duration(seconds=transform_timeout_sec),
)
```

The important rule is that the same frame name at two timestamps does **not** imply identity:

```text
lidar@t0 -> map/odom -> lidar@t1
```

can be non-identity because the platform moved between `t0` and `t1`. Latest-TF substitution would
change the accumulated geometry and is not permitted.

### Buffer, listener, executor, and timeout

Production constructs `tf2_ros.Buffer(cache_time=Duration(seconds=10.0))` and
`TransformListener(buffer, None, spin_thread=True)`. The builder node's main process uses normal
`rclpy.spin(node)` for the raw callback, while the listener owns a dedicated node, executor, and
thread. Therefore a bounded lookup wait inside the raw callback does not prevent the listener from
receiving TF updates. Shutdown explicitly stops and joins that listener thread.

The packaged transform timeout is 0.2 seconds. The manual final measurement harness used 0.5
seconds and a `MultiThreadedExecutor(num_threads=3)` for replay/builder orchestration; the listener
still ran on its dedicated thread. These are bounded waits, not availability guarantees beyond the
tested setup.

## Transform storage convention

ROS supplies a conventional column-vector source-to-target transform:

```text
p_target = R @ p_source + t
```

The pinned M4.5a `SweepTransform` builder operation requires:

```text
rotation_storage = R.T
translation_storage = -R.T @ t
```

The preserved first implementation stored `-t`, which is wrong under non-identity rotation. A
fail-first synthetic rotation-plus-translation regression caught that defect. The synthetic test is
a regression guard; authoritative agreement comes from actual raw nuScenes sweeps traversing the
complete ROS/tf2 path and matching the accepted M4.5a oracle byte-for-byte. See
[`docs/MULTISWEEP.md`](MULTISWEEP.md) for the evidence distinction and chronology.

## Bounded history and failure behavior

- Maximum depth is ten historical acquisitions; each output contains current plus up to ten prior.
- Startup naturally emits current-only, then grows history. No padding or duplicated sweep is made.
- Selected history order is nearest-to-farthest; the M4.5a builder emits current first.
- An equal or regressing timestamp clears history before the new current acquisition.
- A positive `history_reset_gap_sec` can reset after a large gap; the packaged value `0.0` disables
  gap-based reset.
- Missing TF for any selected historical sweep rejects the current model-ready output. The node does
  not silently skip that sweep and does not use latest TF.
- Malformed PointCloud2, invalid timestamps/frames, builder contract failures, and an all-invalid
  cloud fail closed and increment rejection counters.

After a build attempt, a valid decoded current acquisition becomes available to later callbacks.
This keeps acquisition history chronological while preventing a partially built current output from
being published.

## Output and detector contract

The builder publishes `/laserperception/points_model_ready` as best-effort, volatile, keep-last
depth 1 with scalar float32 `x`, `y`, `z`, `time_lag`. The existing detector then publishes reliable,
volatile, keep-last depth 5 `/laserperception/detections` messages. The detector, checkpoint, ONNX,
TensorRT engine, exact-fast voxelizer, thresholds, class mapping, voxel geometry, and MMDeploy
postprocessing are unchanged.

The older v0.1/M3 path remains supported: publishers may still provide model-ready PointCloud2
directly and bypass the raw builder.

## Launch and replay

After installing the ROS package and providing the documented external assets and nuScenes root:

```bash
ros2 launch laserperception_ros m45b_raw_multisweep.launch.py
```

The checked-in launch starts the builder, unchanged detector, an optional nuScenes raw replay, and
optional RViz. Disable replay or visualization for a live compatible source:

```bash
ros2 launch laserperception_ros m45b_raw_multisweep.launch.py \
  run_raw_replay:=false run_rviz:=false
```

The replay publishes actual nuScenes raw acquisitions and TF for validation; it is not a vendor
sensor driver or localization substitute.

## Correctness evidence

At measurement commit `9e0f4dfacbfc997945825d86a85a3609594a059e`, the frozen 20 samples
passed exact raw ROS model-ready bytes, voxel tensors, TensorRT raw tensors, `DetectionFrame`, and
`Detection3DArray` semantic/geometric gates. W1 index 42 reconstructed 354,182 points with SHA256
`5c1f5590ce7925d9312fe9ef755d0d849f579ff3d7ae842af75d41a0252cb29a`; scene start and
low/median/high-rotation sentinels also matched exactly. The old model-ready M3 smoke passed on the
same commit.

Canonical evidence:
[`benchmarks/m45b/results/raw_ros_multisweep_correctness.json`](../benchmarks/m45b/results/raw_ros_multisweep_correctness.json),
SHA256 `09ec61bee8b005b7f006a3cb56186cdb08e4da7f8d822174a34e3185267f7224`.
No raw points or private paths are stored in that record.

## Limitations

M4.5b proves ingestion correctness for compatible PointCloud2 plus valid TF in the pinned evidence
environment. It does not prove detection accuracy on arbitrary physical sensors. PointPillars
remains pretrained on nuScenes. Not implemented or proven: arbitrary sensor calibration
automation, localization, odometry, intra-scan deskew, tracking, camera fusion, INT8, Jetson
deployment, physical-LiDAR accuracy, or safety certification. Do not describe this boundary as
“works with any LiDAR.”
