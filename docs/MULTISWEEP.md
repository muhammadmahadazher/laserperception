# Multi-sweep reconstruction (v0.2.0)

LaserPerception v0.2.0 supports two accepted reconstruction boundaries that produce the unchanged
PointPillars model-ready `float32 N x 4` XYZT contract:

```text
M4.5a offline: raw sweeps + integer timestamps + known poses
M4.5b live ROS: raw PointCloud2 + acquisition stamps + time-indexed tf2
                                  |
                                  v
                       MultiSweepBuilder (NumPy)
                                  |
                                  v
                   ModelReadyPointCloud: float32 XYZT
                                  |
                                  v
                          unchanged detector
```

M4.5a is the accepted ROS-independent reconstruction core. M4.5b is a narrow ROS boundary around
that core: it decodes raw XYZ, retains bounded chronological history, resolves historical-to-current
transforms, serializes model-ready PointCloud2, and feeds the existing detector. Neither path
changes the model, voxel geometry, TensorRT engine, thresholds, classes, or exact-fast voxelizer.

## Shared M4.5a reconstruction core

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
- `MultiSweepBuilderConfig` records ten historical sweeps, `remove_close=False`, radius `1.0`, and
  `pad_empty_sweeps=False`.
- `MultiSweepBuilder` selects history nearest-to-farthest, applies pinned arithmetic, concatenates
  current first, keeps XYZT, applies strict spatial bounds, and preserves source-row order.

The builder fails closed on malformed shapes or dtypes, non-finite values, invalid homogeneous
matrices, and transform source/target mismatches. Point order is never geometrically sorted.

## Pinned accumulation semantics

The source-level discovery record is
[`docs/m45/UPSTREAM_MULTISWEEP_CONTRACT.md`](m45/UPSTREAM_MULTISWEEP_CONTRACT.md).

| Property | Frozen behavior |
|---|---|
| Raw load | five native float32 values per point |
| History | up to ten acquisitions plus the current keyframe |
| Selection | previous acquisitions nearest-to-farthest |
| Scene start | current keyframe only; no fabricated padding |
| `remove_close` | disabled; dormant rule is strict `abs(x) < 1 && abs(y) < 1` before transform |
| Transform | float32 points use float64 reload of a float32-quantized matrix; float32 write-back after rotation and translation |
| Lag | `(current_us / 1e6) - (historical_us / 1e6)`, assigned to float32; current is zero |
| Concatenation | current, then history nearest-to-farthest; source-row order retained |
| Spatial filter | strict `-50 < x,y < 50`, `-5 < z < 3` |
| Output | contiguous float32 `x, y, z, time_lag` |

The implementation matches the pinned operation sequence instead of substituting a merely
algebraically equivalent SE(3) expression. Intermediate rounding points are part of the evidence
contract. The pinned pipeline does not deskew individual points.

## Live ROS history boundary

The `laserperception_multisweep_builder` executable subscribes to
`/laserperception/points_raw` and publishes `/laserperception/points_model_ready`. Its complete
contract is in [`docs/RAW_LIDAR_ROS2.md`](RAW_LIDAR_ROS2.md).

- Required input fields are scalar float32 `x`, `y`, and `z`; other fields are ignored.
- The acquisition time is the PointCloud2 header stamp. Historical source stamps are retained and
  the current stamp is the target time. There is no per-point firing-time deskew.
- Non-finite XYZ rows are removed deterministically without reordering retained rows; a cloud with
  no valid rows is rejected.
- History starts current-only and grows to current plus ten prior acquisitions. It is selected
  nearest-to-farthest with no padding.
- Equal or regressing timestamps reset history. An optional positive configured gap also resets it;
  the packaged configuration leaves gap reset disabled (`0.0`).
- Every selected historical acquisition requires a valid bounded cross-time TF. A missing transform
  rejects the current output; the node does not skip a sweep or substitute latest TF.

## Transform convention and regression evidence

For a conventional ROS column-vector transform,

```text
p_target = R @ p_source + t
```

The pinned builder applies `p_source_row @ A - b`. The adapter must therefore store:

```text
rotation_storage = R.T
translation_storage = -R.T @ t
```

The first M4.5b adapter incorrectly stored `-t`, which is wrong whenever `R != I`. A fail-first
synthetic rotation-plus-translation fixture demonstrated that the regression test detects this
specific implementation bug; pure translation alone could not expose it.

That synthetic fixture is a **regression guard**, not proof of agreement with MMDetection3D. The
authoritative correctness evidence uses actual raw nuScenes sweeps and poses through ROS
PointCloud2, tf2 time travel, the repaired adapter, `MultiSweepBuilder`, and model-ready
PointCloud2, then compares the result byte-for-byte with the accepted M4.5a oracle.

| Authoritative raw ROS sample | History | Exact result |
|---|---:|---|
| Scene start index 0 | current only | 33,587 points, exact |
| W1 index 42 | current + 10 | 354,182 points, SHA256 `5c1f5590ce7925d9312fe9ef755d0d849f579ff3d7ae842af75d41a0252cb29a` |
| Low-rotation index 21 | current + 10 | exact |
| Median-rotation index 39 | current + 10 | exact |
| High-rotation index 58 | current + 10 | exact |

The chronology is intentionally preserved under `benchmarks/m45b`: the original W1 failure, TF
transform ledger, fail-first repair evidence, and final passing record remain separate artifacts.

## Exact correctness evidence

M4.5a commit `cc0f20b16412d98939c9544002d02029b35a5971` passed 81/81 mini-val
model-ready matrices and the frozen 20 detector samples. The M4.5b final measurement commit
`9e0f4dfacbfc997945825d86a85a3609594a059e` then ran the same frozen detector set through the
actual raw ROS integration path:

- **20/20** model-ready inputs: exact shape, point count, float32 bytes, and SHA256;
- **20/20** `voxels`, `num_points`, and `coors`: exact;
- **20/20** raw TensorRT `cls_score`, `bbox_pred`, and `dir_cls_pred`: exact;
- **20/20** full `DetectionFrame` values: exact; and
- **20/20** `Detection3DArray` semantic/geometric content: exact, with the raw acquisition header
  preserved. The older model-ready replay's wall-clock header is documented separately and is not
  treated as a semantic detector difference.

The canonical sanitized record is
[`benchmarks/m45b/results/raw_ros_multisweep_correctness.json`](../benchmarks/m45b/results/raw_ros_multisweep_correctness.json),
SHA256 `09ec61bee8b005b7f006a3cb56186cdb08e4da7f8d822174a34e3185267f7224`.
It contains tokens, counts, shapes, hashes, first-difference fields, frozen artifact identities, and
scope guards—not dataset points or private filesystem paths.

## Reproduction boundaries

The offline M4.5a validator is:

```bash
export LASERPERCEPTION_NUSCENES_ROOT=/external/path/to/nuscenes
python scripts/detection/validate_m45a_multisweep.py
```

M4.5b ROS-native tests and validation require the documented pinned ROS/GPU environment, external
nuScenes mini data, and frozen model/deployment assets. Start with:

```bash
python scripts/ros2/validate_m45b_raw_ros.py --help
python scripts/ros2/validate_m45b_detector_chain.py --help
```

Both evidence paths fail closed. Ordinary CPU CI does not require nuScenes, MMDetection3D, CUDA,
TensorRT, or ROS.

## Scope and limitations

M4.5 proves reconstruction and ingestion correctness for the pinned nuScenes inputs and for
compatible ROS 2 PointCloud2 streams containing float32 XYZ when a valid time-aware TF tree is
supplied. It does **not** prove detection accuracy on arbitrary physical sensors or provide
calibration automation, localization, odometry, intra-scan deskew, tracking, camera fusion, INT8,
Jetson deployment, physical-LiDAR accuracy, or safety certification. Vendor-specific extra fields
are ignored by the frozen detector path. LaserPerception does not claim that it “works with any
LiDAR.”
