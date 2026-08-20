# KITTI Velodyne to frozen model-frame alignment

Status: frozen for the M6a Protocol R2 offline oracle before KITTI detector initialization.

## Physical bases

Official KITTI Raw defines the Velodyne sensor basis as:

- +X: vehicle forward;
- +Y: vehicle left;
- +Z: up.

The official nuScenes devkit's KITTI conversion code documents the complementary lidar axes:
KITTI +X is forward while the nuScenes lidar +X axis is right, and applies a +90-degree yaw about
+Z when converting KITTI lidar data to nuScenes lidar convention. That convention is corroborated
by all three parts of the actual frozen input path rather than inferred from one quaternion:

1. The tracked M4.5b W1 ledger contains the real current `LIDAR_TOP` calibrated-sensor record.
   Its sensor-to-ego quaternion `(w,x,y,z)` is
   `[0.7077955119163518, -0.006492242056004365, 0.010646214713995808,
   -0.7063073142877817]`. Its sensor axes expressed in vehicle ego `(forward,left,up)` are
   `+X=[0.002033,-0.999981,-0.005900]`, `+Y=[0.999704,0.002176,-0.024229]`, and
   `+Z=[0.024242,-0.005849,0.999689]`. Thus raw sensor +X is physically right, +Y is
   forward, and +Z is up, with the small measured mounting tilt retained.
2. The pinned MMDetection3D `LoadPointsFromFile` keeps native `LIDAR_TOP` columns, and
   `LoadPointsFromMultiSweeps` transforms history into the current `LIDAR_TOP` sensor frame. Its
   final `[0,1,2,4]` selection applies no later sensor-axis rotation.
3. The accepted M4.5a contract ends accumulation in the current acquisition's raw
   `LIDAR_TOP` frame.

The calibration record is tracked at
`benchmarks/m45b/diagnostics/w1_tf_transform_ledger.json` (SHA256
`0363fd23ff426aca7a9d88518203062a8e7440b0155a49879f639b3c96c18f2d`). This section describes
the raw frozen model-input basis; it does not redefine the public `DetectionFrame` contract.

Therefore the frozen model-input physical basis is:

- model +X: vehicle right;
- model +Y: vehicle forward;
- model +Z: up.

## Frozen alignment

For column vectors:

```text
p_model = A @ p_kitti

A = [[ 0, -1, 0],
     [ 1,  0, 0],
     [ 0,  0, 1]]
```

`det(A) = +1`; this is a proper +90-degree rotation about +Z, not a reflection. Examples:

- KITTI forward `[1, 0, 0]` becomes model forward `[0, 1, 0]`;
- KITTI left `[0, 1, 0]` becomes model left `[-1, 0, 0]` because model +X is right;
- up is unchanged.

For the repository's row-vector point matrices, the implementation uses `points @ A.T`.

The alignment is non-identity and changes axes only. It adds no translation and does not imitate
nuScenes mounting height, range, beam pattern, density, vertical field of view, or motion.

## Pose/calibration consistency

Let `T_ego_from_velo` be the native KITTI Velodyne-to-IMU calibration. Since
`p_model = A @ p_velo`, then `p_velo = A.T @ p_model`; the virtual sensor calibration is:

```text
R_ego_from_model = R_ego_from_velo @ A.T
t_ego_from_model = t_ego_from_velo
```

Equivalently, a native historical-to-current rotation expressed in KITTI Velodyne axes is
conjugated as `A @ R @ A.T` in model axes. Translations rotate by `A`. Production code constructs
virtual-sensor `LidarPose` values and delegates relative-transform arithmetic to the unchanged
`SweepTransform.from_poses`; tests cover identity, rotations, translations, inverses, and combined
motion so a sign or transpose error fails.

## Evidence boundary

The basis was derived from official coordinate documentation, the actual tracked calibration, and
the pinned input path. No KITTI prediction, score, or rendered box was generated or inspected while
choosing it.
