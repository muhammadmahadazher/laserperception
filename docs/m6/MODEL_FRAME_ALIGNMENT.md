# KITTI Velodyne to frozen model-frame alignment

Status: frozen for the M6a candidate before KITTI detector initialization.

## Physical bases

Official KITTI Raw defines the Velodyne sensor basis as:

- +X: vehicle forward;
- +Y: vehicle left;
- +Z: up.

The official nuScenes devkit's KITTI conversion code documents the complementary lidar axes:
KITTI +X is forward while the nuScenes lidar +X axis is right, and applies a +90-degree yaw about
+Z when converting KITTI lidar data to nuScenes lidar convention. The pinned MMDetection3D
nuScenes loader does not apply a later sensor-axis rotation. LaserPerception M4.5a also preserves
the current nuScenes `LIDAR_TOP` sensor frame as the final current-sweep frame.

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

The basis was derived from official coordinate documentation and the pinned input path. No KITTI
prediction, score, or rendered box was generated or inspected while choosing it.
