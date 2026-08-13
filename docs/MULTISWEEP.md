# Offline multi-sweep reconstruction

LaserPerception M4.5a adds a ROS-independent library that reconstructs the pinned PointPillars
model input from raw individual nuScenes LIDAR_TOP acquisitions and known calibration/ego-pose
metadata:

```text
current raw sweep + historical raw sweeps + timestamps + poses
                              |
                              v
                 MultiSweepBuilder (NumPy)
                              |
                              v
            ModelReadyPointCloud: float32 N x 4 XYZT
```

This is an **offline reconstruction boundary**. It does not accept ROS `PointCloud2`, query TF,
buffer live sensor history, or chain into the ROS detector node. Those capabilities remain planned
for a separately reviewed M4.5b. A physical LiDAR cannot be plugged into this library without a
caller that supplies correctly ordered raw sweeps and exact acquisition-time poses.

## Contracts

The production implementation is
[`src/laserperception/detection/multisweep.py`](../src/laserperception/detection/multisweep.py).
It depends on NumPy and the existing `ModelReadyPointCloud`; it does not import MMDetection3D,
MMCV, the nuScenes devkit, ROS, PyTorch, or TensorRT.

- `RawSweep` owns a contiguous float32 `N x 5` matrix in raw nuScenes
  `x, y, z, intensity, ring_index` order, an integer-microsecond timestamp, and a source ID.
- `LidarPose` holds float64 sensor calibration and ego-pose matrices/translations.
- `SweepTransform` reproduces the pinned converter's row-vector operation sequence and float32
  transform-storage cast while retaining source and current-target IDs.
- `HistoricalSweep` binds a raw acquisition to its exact transform.
- `MultiSweepBuilderConfig` records only pinned reference behavior: ten historical sweeps,
  `remove_close=False`, radius `1.0`, and `pad_empty_sweeps=False`.
- `MultiSweepBuilder` selects the caller-provided history in nearest-to-farthest order, applies
  exact pinned arithmetic, concatenates current first, keeps XYZT, applies strict spatial bounds,
  and returns the existing `ModelReadyPointCloud`.

The builder fails closed on malformed shapes/dtypes, non-finite values, invalid homogeneous
matrices, and transform source/target mismatches. Point order is never geometrically sorted.

## Pinned semantics

The source-level discovery record is
[`docs/m45/UPSTREAM_MULTISWEEP_CONTRACT.md`](m45/UPSTREAM_MULTISWEEP_CONTRACT.md). In summary:

| Property | Frozen behavior |
|---|---|
| Raw load | five native float32 values per point |
| History | up to ten acquisitions plus the current keyframe |
| Test selection | first N prepared sweeps, nearest-to-farthest |
| Scene start | current keyframe only; no padding |
| `remove_close` | disabled; dormant rule is strict `abs(x) < 1 && abs(y) < 1` before history transform |
| Transform | float32 points multiplied by float64 reload of a float32-quantized matrix; float32 write-back after rotation and again after translation |
| Lag | `(current_us / 1e6) - (historical_us / 1e6)`, assigned to float32; current is zero |
| Concatenation | current, then history nearest-to-farthest; source row order retained |
| Spatial filter | strict `-50 < x,y < 50`, `-5 < z < 3` |
| Output | contiguous float32 `x, y, z, time_lag` |

The builder deliberately matches operations rather than replacing them with an algebraically
equivalent SE(3) expression; the intermediate rounding points are part of the evidence contract.
The pinned upstream pipeline does not deskew individual points.

## Exact parity evidence

The manual validator is
[`scripts/detection/validate_m45a_multisweep.py`](../scripts/detection/validate_m45a_multisweep.py).
For the candidate side it independently follows raw nuScenes sample-data history, reads raw files,
constructs poses from calibration/ego tables, and invokes `MultiSweepBuilder`. Only the reference
side invokes the pinned official PointPillars test pipeline. Production code never calls
`LoadPointsFromMultiSweeps`.

At implementation commit `cc0f20b16412d98939c9544002d02029b35a5971`:

- **81/81** mini-val samples had identical shape, row order, float32 bytes, and SHA256;
- the reverified distribution was **2/2** current-only scene starts and **79/79** samples with ten
  historical sweeps;
- the frozen 20-sample detector check produced exact `voxels`, `num_points`, and `coors` tensors,
  exact raw TensorRT outputs, and exact final `DetectionFrame` values;
- Tier B was not used.

The sanitized per-sample evidence is
[`benchmarks/m45a/results/nuscenes_mini_multisweep_parity.json`](../benchmarks/m45a/results/nuscenes_mini_multisweep_parity.json).
It contains tokens, counts, shapes, hashes, artifact identities, and exact booleans—not dataset
points or private filesystem paths. nuScenes data and frozen model/deployment artifacts remain
external.

## Manual reproduction

Use the already documented pinned detection environment and external assets. From the repository
root in that environment:

```bash
export LASERPERCEPTION_NUSCENES_ROOT=/external/path/to/nuscenes
python scripts/detection/validate_m45a_multisweep.py
```

The command refuses a dirty implementation tree, verifies checkpoint/ONNX/engine hashes, requires
exactly 81 mini-val samples, stops on the first Tier A mismatch, and runs the frozen detector check
only after Tier A succeeds. It is a manual integration gate; ordinary CPU CI does not require
nuScenes, MMDetection3D, CUDA, TensorRT, or ROS.

## Scope boundary

M4.5a provides:

```text
raw sweep + known pose/calibration metadata -> model-ready temporal cloud
```

M4.5b remains planned:

```text
raw PointCloud2 + time-travel TF + live history buffering -> detector-ready stream
```

No ROS node, tf2 integration, live buffer, detector runtime, voxelizer, model, engine, threshold,
or benchmark timing path changed in M4.5a.
