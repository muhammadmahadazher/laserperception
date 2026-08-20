# M6a preregistered offline protocol

Status: frozen before implementation measurement and before any KITTI detector execution.

## Question and scope

M6a asks only whether LaserPerception can decode official KITTI Raw data, recover verified poses,
align axes, and reconstruct the existing model-ready multi-sweep contract offline. It does not
initialize PointPillars, TensorRT, postprocessing, or ROS.

The official drives, timestamp rule, calibration directions, OXTS equations, frame alignment, and
selection algorithm are recorded in `configs/m6/kitti_raw.yaml` and the companion contract docs.

## Tier-A pose oracle

Production OXTS poses from `2011_09_30_drive_0016` are compared with official KITTI odometry
training sequence 04, frames 0–270. The official odometry 3x4 rows map rectified camera-0
coordinates at frame i into rectified camera-0 at frame 0. Production OXTS matrices are converted
to that same frame with the raw calibration and only the dataset-defined first-frame origin is
used; no fitted SE(3), scale, or trajectory alignment is permitted.

### Tolerances frozen before comparison

Both routes use binary64 matrix arithmetic. Official odometry text is decimal serialization, and
the candidate route adds calibration conjugation. The pass gate is all of:

- maximum rotation-matrix element absolute difference ≤ `1e-9`;
- relative rotation angle ≤ `1e-8` radians;
- translation-vector norm difference ≤ `1e-5` metres.

For production relative-transform inverse/composition closure:

- maximum rotation-matrix element error ≤ `1e-10`;
- translation norm error ≤ `1e-9` metres.

These are preregistered numerical-representation tolerances, not tuned scientific thresholds. Any
failure stops M6a evidence promotion.

## Frozen reconstruction-frame selection

The canonical drive has 108 chronological frames. Exactly 24 frames are selected without detector
information:

1. structural frames `0, 1, 2, 5, 10` exercise current-only and shallow history;
2. 16 full-history indices are selected by inclusive, integer-floor systematic spacing from frame
   11 through the final frame;
3. three additional full-history indices represent low, median, and high previous-frame motion,
   ranked by `translation_norm_m + relative_rotation_angle_rad`; if an index is already selected,
   take the next closest ranked unused candidate, breaking ties by lower frame index.

The exact resulting list is committed before the clean measurement commit. At least ten frames
must have all ten historical sweeps.

History is current first, then up to ten preceding acquisitions nearest-to-farthest. No padding or
synthetic sweep is permitted. `MultiSweepBuilder` remains unchanged.

## Hard reconstruction gates

For each selected frame:

- source `.bin` decodes to exact little-endian float32 XYZR bytes, row count, and ordering;
- axis alignment changes no row count and creates finite float32 five-column `RawSweep` data;
- output is finite, C-contiguous float32 `N x 4` XYZT;
- current lag is exact float32 zero; each sweep has one constant lag; older lags increase;
- pre-builder rows equal the sum of selected acquisitions;
- because the existing builder applies the frozen strict detector-range crop, final rows must equal
  an independently counted strict `(-50,-50,-5) < XYZ < (50,50,3)` mask after transforms;
- surviving rows retain current-then-history and within-file order;
- hashes are stable; selected sentinel frames run ten times with identical SHA256.

The point-count exception is only the already-existing builder crop. The KITTI reader, basis
rotation, and pose transform may not drop rows.

## Input-shift diagnostics

For full-history frames, measure only source/accumulated counts, in-range counts, time span, unique
candidate pillar coordinates under the frozen 0.25 m XY grid, and the amount above
`max_voxels=40000`. These are non-detector diagnostics. They do not authorize a voxel, model, or
threshold change and do not assert which spatial region would be retained.

## Evidence discipline

Implementation and CPU tests are committed cleanly. Evidence is then generated from that exact
commit and stored, sanitized, in
`benchmarks/m6a/results/kitti_raw_offline_reconstruction.json`. Dataset bytes, private paths,
credentials, and raw labels are excluded. Any accidental KITTI detector run contaminates the
preregistration boundary and stops M6a.
