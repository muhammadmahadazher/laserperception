# Pinned upstream multi-sweep contract

This document freezes the source-level contract that M4.5a must reproduce. It is a discovery
record, not a description of a newer OpenMMLab release. The production implementation must remain
independent of MMDetection3D; MMDetection3D is the parity oracle only.

## Exact environment and sources

| Component | Pinned identity | Relevance |
|---|---|---|
| LaserPerception base | `320e146d4cb0d272e8e569a914fbc6fdb450875b` | Post-v0.1 screenshot `main` used to start M4.5a |
| MMDetection3D | v1.4.0, commit `fe25f7a51d36e3702f961e198894580d83c4387b` | PointPillars config, nuScenes converter, loaders, point structures |
| MMCV | 2.1.0, tag commit `57c4e25e06e2d4f8a9357c84bcd24089a284dc88` | Installed CUDA 11.8 / Torch 2.1 wheel and transform base classes |
| MMCV wheel | `mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl`, SHA256 `82a3eb54106f643dace1f9bd4d96e96899a0bb2360a5bb8e2f252a36d95051d9` | Exact installed binary distribution |
| PyTorch | 2.1.0+cu118 | `BasePoints` storage and concatenation |
| NumPy | 1.26.4 | File, matrix, timestamp, and filter arithmetic |
| nuScenes devkit | 1.2.0, tag commit `eff381829dc86fa75caf7dbbbe862d2091dacf64` | Raw `.pcd.bin` format and raw calibration/pose records |
| pyquaternion | 0.9.9, tag commit `2ccfdd5ec6b214092efcbebacd74eabba5c072e1` | Float64 quaternion-to-matrix conversion used by the info converter |

The tracked detector manifest pins the MMDetection3D commit, dependency versions, PointPillars
config, checkpoint, and nuScenes preparation settings in
[`configs/detection/m1_pointpillars_nuscenes.yaml`](../../configs/detection/m1_pointpillars_nuscenes.yaml).

Primary upstream references at the pinned commits are:

- [PointPillars nuScenes test pipeline](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/configs/_base_/datasets/nus-3d.py#L74-L98)
- [`LoadPointsFromMultiSweeps`](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/mmdet3d/datasets/transforms/loading.py#L316-L453)
- [`LoadPointsFromFile`](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/mmdet3d/datasets/transforms/loading.py#L554-L683)
- [`BasePoints` construction, slicing, concatenation, and `new_point`](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/mmdet3d/structures/points/base_points.py#L33-L52)
- [nuScenes old-format info generation and sweep traversal](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/tools/dataset_converters/nuscenes_converter.py#L168-L226)
- [`obtain_sensor2top` transform construction](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/tools/dataset_converters/nuscenes_converter.py#L283-L341)
- [nuScenes v2 info conversion](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/tools/dataset_converters/update_infos_to_v2.py#L249-L316)
- [Strict `PointsRangeFilter` bounds](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/mmdet3d/structures/points/base_points.py#L230-L254)
- [nuScenes raw LiDAR reader](https://github.com/nutonomy/nuscenes-devkit/blob/eff381829dc86fa75caf7dbbbe862d2091dacf64/python-sdk/nuscenes/utils/data_classes.py#L236-L258)
- [pyquaternion rotation-matrix property](https://github.com/KieranWynn/pyquaternion/blob/2ccfdd5ec6b214092efcbebacd74eabba5c072e1/pyquaternion/quaternion.py#L980-L993)

## A. Current sweep load

nuScenes LIDAR_TOP files are headerless `.pcd.bin` files containing a flat sequence of native
float32 values. Each point has five columns in this order:

```text
x, y, z, intensity, ring_index
```

The pinned test pipeline constructs `LoadPointsFromFile` with `load_dim=5` and `use_dim=5`, which
expands to `[0, 1, 2, 3, 4]`. It therefore initially retains all five columns. `np.frombuffer` or
`np.fromfile` returns float32, and `BasePoints` calls `torch.as_tensor(..., dtype=torch.float32)` and
clones it. The initial and stored point dtype is consequently float32.

The subsequent multi-sweep transform sets the current sweep's fifth column to float32 zero. At the
end of that transform it selects its own default `use_dim=[0, 1, 2, 4]`. Intensity is discarded and
the original ring-index column has already been replaced by time lag.

## B. Historical sweep selection

`sweeps_num=10` means **up to ten historical sweeps in addition to the current keyframe**, so the
model can receive at most eleven acquisitions.

The nuScenes converter starts at the current LIDAR_TOP `sample_data` record and follows `prev`.
Every previous acquisition is appended immediately, so the prepared `lidar_sweeps` list is nearest
to farthest in time. Conversion stops at ten sweeps or an empty `prev` token.

In test mode:

- if `len(lidar_sweeps) <= 10`, the loader selects `np.arange(len(lidar_sweeps))`;
- otherwise it selects `np.arange(10)`;
- it never calls the random training selection path.

The selection is deterministic and preserves prepared-list order. The actual v1.0-mini validation
info contains 81 samples: **2 have zero historical sweeps and 79 have ten**. There are no partial
1–9-sweep records in this prepared split. This distribution was re-read from
`nuscenes_infos_val.pkl`, not assumed from earlier evidence.

## C. `remove_close`

The PointPillars test config does not pass `remove_close`, so the loader default is
`remove_close=False`. No current or historical point is removed by this option in the pinned path.

The dormant upstream rule was nevertheless source-verified. Its default radius is `1.0`, and it
computes:

```python
x_filt = np.abs(points[:, 0]) < radius
y_filt = np.abs(points[:, 1]) < radius
not_close = np.logical_not(np.logical_and(x_filt, y_filt))
points = points[not_close]
```

This is a strict, axis-aligned square test, not Euclidean radius. A point with `abs(x) == 1.0` or
`abs(y) == 1.0` is retained. For real historical sweeps the rule, if enabled, runs on raw sweep
coordinates **before** transformation. Boolean indexing preserves the relative order of retained
rows. The normal non-empty-sweep path does not apply it to the current keyframe. In the separate
padding branch it can filter each appended current-sweep duplicate, but the original current sweep
at list position zero remains unfiltered. None of these filtering branches runs in this config.

## D. Empty and padding behavior

`pad_empty_sweeps=False` by default and is not overridden. The v2 infos contain a
`lidar_sweeps` key even at scene start, where its value is an empty list. The loader selects an empty
index range and emits the current keyframe only. It does not duplicate points.

If the key were missing entirely, the actual configuration would also leave the list as current
only because padding is disabled. The upstream keyframe-duplication loop is therefore inactive.

## E. Timestamps and time lag

The old-format converter records the current nuScenes `sample['timestamp']` and each historical
LIDAR_TOP `sample_data['timestamp']`, both integer microseconds. Across all 81 prepared validation
samples, the sample timestamp equals the current LIDAR_TOP sample-data timestamp.

The v2 info converter divides both values by `1e6` before serializing them. They are stored and
loaded as Python `float` values (binary64 seconds). The loader computes:

```python
lag_seconds_binary64 = current_timestamp_seconds - sweep_timestamp_seconds
points_sweep[:, 4] = lag_seconds_binary64
```

Assignment to the raw float32 point matrix rounds the lag to float32. Every row from one historical
sweep receives the same lag. Historical lag is positive because the equation is current minus
past. The current keyframe's fifth column is set directly to float32 `0.0`.

For representative sample index 42, the 11 final unique lags run from `0.0` to the float32 value
`0.5004169940948486` seconds.

## F. Transform construction and application

### Frames and source metadata

Each historical point starts in its acquisition's LIDAR_TOP sensor frame. It is transformed through
that acquisition's calibrated sensor-to-ego pose and ego-to-global pose, then through the inverse
current ego-to-global and current LIDAR_TOP-to-ego poses, ending in the **current keyframe's
LIDAR_TOP frame**. No per-point deskew is performed.

nuScenes calibration and ego-pose rotations are quaternion records in `w, x, y, z` order.
`pyquaternion.Quaternion(...).rotation_matrix` produces float64 3×3 matrices. Translation records
participate in NumPy operations as float64.

### Old-format converter arithmetic

The pinned converter uses row-vector conventions. With suffix `_s` for the historical sweep and no
suffix for the current keyframe, its exact sequence is:

```python
l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
    np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
)
T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
    np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
)
T -= e2g_t @ (
    np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
) + l2e_t @ np.linalg.inv(l2e_r_mat).T

sensor2lidar_rotation = R.T
sensor2lidar_translation = T
```

`R` and `T` are float64. This sequence intentionally repeats the `np.linalg.inv` expressions as
shown; a parity implementation must not algebraically simplify it before proving identical bytes.

### V2 info conversion and cast point

The v2 converter turns that transform into the field named `lidar2sensor`:

```python
lidar2sensor = np.eye(4)                         # float64
rot = sensor2lidar_rotation                     # float64
trans = sensor2lidar_translation                # float64
lidar2sensor[:3, :3] = rot.T
lidar2sensor[:3, 3:4] = -1 * np.matmul(rot.T, trans.reshape(3, 1))
stored = lidar2sensor.astype(np.float32).tolist()
```

This is the first transform-matrix cast to float32. JSON-like list serialization converts the
elements to Python floats, but every value is already quantized to float32. All 790 prepared
historical transforms were independently rebuilt from raw calibration/ego-pose tables and matched
the stored float32 matrices byte-for-byte.

### Loader arithmetic and intermediate dtypes

The loader reconstructs the stored list with `np.array(...)`, giving a float64 matrix whose numeric
values remain float32-quantized. Historical raw points are float32. It then executes, in order:

```python
points_sweep[:, :3] = points_sweep[:, :3] @ lidar2sensor[:3, :3]
points_sweep[:, :3] -= lidar2sensor[:3, 3]
```

The matrix multiplication expression is float64 and is assigned into the float32 point slice,
rounding after rotation. The in-place subtraction uses the float64 translation and writes back to
the same float32 slice, rounding again after translation. Combining these expressions into one
affine operation changes the rounding contract and is not allowed for Tier A parity.

The current keyframe is not transformed.

## G. Concatenation and final test filter

The multi-sweep loader builds this list:

1. current keyframe;
2. selected history in nearest-to-farthest order.

It never sorts points geometrically. Raw point order is retained within each sweep; `remove_close`
does not run. Each historical NumPy matrix becomes a float32 `BasePoints` tensor. `BasePoints.cat`
uses `torch.cat(..., dim=0)`, preserving list and row order, then `points[:, [0, 1, 2, 4]]` selects
the final feature columns.

The pinned test augmentation has rotation `0`, scale `1`, translation standard deviation `0`, and
no flip. Those identity operations are byte-preserving in the measured environment. The remaining
`PointsRangeFilter` uses strict bounds:

```text
-50 < x < 50
-50 < y < 50
 -5 < z < 3
```

Its boolean mask preserves relative row order. `Pack3DDetInputs` returns the resulting contiguous
float32 tensor. The candidate builder must apply the same strict range mask and return an explicit
contiguous array.

## H. Final PointPillars input

The final frozen model input is exactly:

```text
x, y, z, time_lag
```

It is a contiguous `N×4` float32 matrix. Intensity is not retained. Ring index is overwritten by
time lag before the four-column selection. This is compatible with LaserPerception's existing
`ModelReadyPointCloud`; no contract review stop is required.

## Discovery validation checkpoint

A temporary independent reconstruction used only raw point files plus the prepared transform and
timestamp metadata, without calling `LoadPointsFromMultiSweeps`. It matched the pinned official
test-pipeline output for **81/81** mini-val samples, including both scene starts and all 79
full-history samples, with identical shapes, row order, float32 bytes, and SHA256 values. This
diagnostic establishes that the operation contract above is sufficient; it is not the final M4.5a
evidence artifact or production implementation.

Concise frozen summary:

| Item | Discovered contract |
|---|---|
| `load_dim` | 5 for current and historical raw files |
| Current `use_dim` | Initial `[0,1,2,3,4]` |
| Multi-sweep `use_dim` | Final `[0,1,2,4]` |
| `sweeps_num` | Up to 10 historical plus current |
| Test selection | First N prepared sweeps, nearest-to-farthest; deterministic |
| `remove_close` | Disabled; dormant strict 1 m axis-aligned square rule before historical transform |
| Padding | Disabled; scene start is current only |
| Transform | Float32 points × float64 stored matrix; float32 assignment after rotation; float32 in-place assignment after translation |
| Transform metadata | Raw float64 quaternion/pose composition → float32 `lidar2sensor` storage → float64 reload of quantized values |
| Lag | `(current_us / 1e6) - (historical_us / 1e6)`, then float32 assignment |
| Concatenation | Current, then history nearest-to-farthest; source point order retained |
| Final range | Strict `(-50,-50,-5) < xyz < (50,50,3)` |
| Final features | `x, y, z, time_lag`, contiguous float32 |
