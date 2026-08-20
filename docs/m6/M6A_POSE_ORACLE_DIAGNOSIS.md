# M6a Tier-A pose-oracle failure diagnosis

Status: post-failure diagnostic design, not part of the original preregistration.

This diagnosis was designed after observing the immutable Tier-A failure in
`benchmarks/m6a/diagnostics/pose_oracle_failure_ec9e341.json` (SHA256
`894960c34b96aac02db2a8a10f13c66e39e3ac99b3ef34ee76c0ef3e88eabaf3`). It does
not change the original tolerances, result, implementation, or evidence status. M6a remains failed.

## Authoritative source products

The official KITTI Raw devkit and IJRR data paper define synchronized Raw data as approximately
10 Hz, referenced to `image_00`. Cameras are triggered by the Velodyne forward-facing instant.
For GPS/IMU, KITTI selects the closest packet from the native 100 Hz OXTS stream, with a stated
worst-case difference of 5 ms. The synchronized OXTS timestamp remains the selected packet's own
timestamp; it is not necessarily identical to the image or Velodyne timestamp.

The official Odometry devkit defines each row of `poses/04.txt` as a left-rectified-camera pose:
the matrix maps a point from camera frame i into camera frame 0. Its mapping table assigns sequence
04 to `2011_09_30_drive_0016`, raw frames 000000 through 000270 inclusive. The KITTI website states
that on 31 October 2013 the odometry pose files were replaced with a properly interpolated
(subsampled) product. The distributed devkit does not document the exact interpolation algorithm.

## Frame and transform ledger

All equations below use homogeneous column vectors. `T^b_a` maps coordinates from source frame
`a` into target frame `b` by `p_b = T^b_a p_a`.

| Symbol | Source -> target | Official source | Operation |
|---|---|---|---|
| `T^V_I` | IMU -> Velodyne | `calib_imu_to_velo.txt` | direct `R, T` |
| `T^Craw_V` | Velodyne -> raw camera 0 | `calib_velo_to_cam.txt` | direct `R, T` |
| `T^Crect_Craw` | raw camera 0 -> rectified camera 0 | `R_rect_00` in `calib_cam_to_cam.txt` | rotation embedded in 4x4 |
| `T^Crect_I` | IMU -> rectified camera 0 | raw calibration chain | `T^Crect_Craw T^Craw_V T^V_I` |
| `G_i` | IMU i -> Mercator navigation frame | synchronized OXTS and Raw devkit | `Rz(yaw) Ry(pitch) Rx(roll), t(lat, lon, alt)` |
| `P_i` | IMU i -> IMU 0 | Raw devkit normalization | `G_0^-1 G_i` |
| `O_i` | rectified camera 0 at i -> rectified camera 0 at 0 | `poses/04.txt` | direct odometry row |
| `Tr` | Velodyne -> left rectified camera | sequence-04 `calib.txt` | direct odometry calibration row |

The production Tier-A candidate is exactly:

`C_i = T^Crect_I P_i (T^Crect_I)^-1`.

The independent raw-derived Velodyne calibration is:

`Tr_raw = T^Crect_Craw T^Craw_V`.

No fitted alignment, index search, time shift, scale fit, or residual-minimizing frame change is
permitted.

## Diagnostic gates frozen before new oracle comparisons

These gates are diagnostic arithmetic checks and do not replace or relax the original Tier-A
tolerances.

- Raw-derived `Tr_raw` versus sequence-04 `Tr`: rotation-matrix max absolute difference <= `1e-9`,
  rotation angle <= `1e-8 rad`, and translation norm <= `1e-6 m`.
- Candidate and odometry frame 0 versus identity: matrix max absolute difference <= `1e-12` and
  translation norm <= `1e-12 m`.
- Production OXTS pose versus a separate direct transcription of official
  `convertOxtsToPose.m`: matrix max absolute difference <= `1e-12`, rotation angle <= `1e-10 rad`,
  and translation norm <= `1e-9 m` across all 271 mapped frames.
- Production composed camera chain versus the direct-devkit pose plus independently parsed raw
  calibration: the same `1e-12`, `1e-10 rad`, and `1e-9 m` arithmetic limits.

## Controlled diagnostic comparisons

The diagnosis will report absolute errors and relative-pose errors for spacings 1, 2, 5, and 10.
It will preserve integer nanoseconds for `image_00`, synchronized OXTS, Velodyne start, end, and
forward-facing timestamps, and will compare them with sequence-04 `times.txt`.

The official unsynced archive is inspected only for its original OXTS timestamps and packets. The
following timing variants are frozen before comparison and remain non-authoritative because the
Odometry interpolation implementation is not published in the distributed devkit:

1. synchronized OXTS packet (the failed production input);
2. nearest previous unsynced OXTS packet to the `image_00` timestamp;
3. nearest unsynced OXTS packet to the `image_00` timestamp;
4. linear translation plus quaternion SLERP between bracketing unsynced OXTS poses at the
   `image_00` timestamp.

All variants will be reported. None may be selected post hoc and promoted as authoritative. The
original Tier-A failure remains unchanged regardless of the outcome.
