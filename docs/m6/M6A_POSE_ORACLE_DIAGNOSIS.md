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

## Diagnostic result

Classification: **DATA-PRODUCT / TIMING**.

The diagnosis establishes that LaserPerception implements the official KITTI Raw OXTS pose
semantics and the raw calibration chain correctly. The failed comparison used a different pose
product as a numerical-equality oracle: nearest-packet synchronized Raw OXTS versus the current,
post-2013 interpolated KITTI Odometry ground truth. The products share frames and nominal times,
but not the same temporal sampling/provenance.

The complete non-canonical record is
[`pose_oracle_diagnosis_ec9e341.json`](../../benchmarks/m6a/diagnostics/pose_oracle_diagnosis_ec9e341.json).
It was generated at `47b81b78a1469ba98ccea126243447df444b5499` from the diagnostic protocol
commit `be463437d1f873f03265e6cabd8f0cd680ee29bb`; its SHA256 is
`44509f4c28fafbdd848c2627c99cde4615bd8e6011520c2a371b1ee3ce6853d8`.

## Provenance findings

### Product A: synchronized KITTI Raw OXTS

- KITTI records OXTS at 100 Hz. The synchronization procedure uses `image_00` as its reference and
  selects the closest OXTS packet rather than interpolating or averaging it. The IJRR paper's data
  synchronization discussion states that the worst-case camera-to-OXTS difference is 5 ms.
- `image_00`, synchronized OXTS, and Velodyne retain separate timestamp files. The synchronized
  OXTS timestamp is the selected native packet timestamp, not a rewritten image timestamp.
- For all 271 mapped frames, the synchronized OXTS timestamps exactly matched entries in the
  unsynchronized 100 Hz stream and exactly matched the packets nearest to `image_00`.
- Sources: the official [Raw data page](https://www.cvlibs.net/datasets/kitti/raw_data.php), Raw
  devkit `readme.txt` and `convertOxtsToPose.m`, and the official
  [KITTI data paper](https://www.cvlibs.net/publications/Geiger2013IJRR.pdf), section 2.3.

### Product B: KITTI Odometry ground truth

- Each `poses/04.txt` row maps a point in the left rectified camera at frame i into that camera at
  frame 0. The trajectory is GPS/IMU ground truth, expressed in the camera coordinate system.
- The official mapping is odometry sequence 04 to `2011_09_30_drive_0016`, frames 000000 through
  000270 inclusive. This is 271 frames at index offset zero.
- The synchronized raw drive contains 279 records; the odometry mapping table deliberately ends at
  raw frame 270, leaving raw tail frames 271 through 278 outside sequence 04.
- KITTI's official [dataset changelog](https://www.cvlibs.net/datasets/kitti/) states that the pose
  files were replaced on 31 October 2013 by a properly interpolated, subsampled version. The
  currently distributed pose archive has that corrected provenance. Neither the Odometry devkit
  nor the public history recovered here specifies the exact interpolation arithmetic.
- Sources: the official [Odometry page](https://www.cvlibs.net/datasets/kitti/eval_odometry.php),
  Odometry devkit `readme.txt`, and the official dataset changelog.

## Calibration and frame checks

The independent raw calibration result
`T^Crect_Craw T^Craw_V` and sequence-04 `Tr` both mean Velodyne to left rectified camera. Their
comparison passed every frozen diagnostic limit:

| Metric | Result | Limit |
|---|---:|---:|
| Rotation-matrix max absolute difference | `4.451994328746878e-14` | `1e-9` |
| Rotation-angle difference | `0 rad` | `1e-8 rad` |
| Translation-vector norm difference | `5.868706941768216e-15 m` | `1e-6 m` |

The raw-derived and odometry rotation determinants were `1.000000069040493` and
`1.000000069040554`; their orthonormality max errors were `8.076707569415476e-8` and
`8.076712121329876e-8`. The complete matrices are retained in the JSON.

The frame-zero diagnostic is effectively the expected normalized identity relation. The candidate
was within `4.653966101386686e-10` matrix max and `4.665698802017416e-10 m` translation of identity;
the serialized odometry pose was within `3.562503e-10` matrix max and
`2.288783348495058e-16 m` translation. Both technically miss the predeclared `1e-12` identity
check because that check was tighter than the serialized/numerically inverted products. The
rotation angle was zero, the discrepancy is ten orders below the observed 0.088 m failure, and
there is no meaningful frame-zero offset. The frozen diagnostic check is reported as failed rather
than loosened after observation.

## Official Raw devkit gates

Across all 271 frames, production Raw OXTS poses and a separate direct transcription of official
`convertOxtsToPose.m` were identical:

| Gate | Matrix max | Rotation max | Translation max | Result |
|---|---:|---:|---:|---|
| Raw OXTS pose | `0` | `0 rad` | `0 m` | PASS |
| Raw-devkit pose + independent camera calibration | `0` | `0 rad` | `0 m` | PASS |

The composed check evaluates the exact production chain
`T^Crect_I P_i inverse(T^Crect_I)` against independently parsed calibration plus direct-devkit
poses. These results exclude a production OXTS-conversion, calibration-direction, rectification,
or normalization implementation error at the recorded arithmetic precision.

## Absolute error shape

The immutable original gate maxima remain `0.00029970412004469253` matrix difference at frame 108,
`0.0004166289537925316 rad` trace-derived relative angle at frame 147, and
`0.08847669331706698 m` translation at frame 220.

For distributional analysis, the diagnostic uses a stable quaternion geodesic angle. This does not
replace the original metric:

| Metric | Min | Median | p90 | p95 | Max |
|---|---:|---:|---:|---:|---:|
| Translation norm (m) | `4.6657e-10` | `0.0345813` | `0.0670795` | `0.0741020` | `0.0884767` |
| Rotation angle (rad) | `0` | `3.78494e-5` | `1.18158e-4` | `1.45397e-4` | `3.06132e-4` |

The signed translation error is dominated by camera z (forward): range `-0.0884578` to
`+0.0652047 m`, with 218 sign changes. Camera x stays within about `-0.000818` to
`+0.001309 m`; camera y stays within about `-0.005883` to `+0.000978 m`. Error does not accumulate:
translation-norm correlation is `0.1110` with frame and `0.1118` with odometry distance; rotation
correlation with frame is `-0.1417`. The oscillation and sign changes are inconsistent with a
fixed calibration bias or monotonic integration drift.

## Relative-pose diagnostics

| Spacing | Translation median / p95 / max (m) | Rotation median / p95 / max (rad) |
|---:|---:|---:|
| 1 | `0.0624233 / 0.0912770 / 0.0973667` | `7.47855e-5 / 2.12084e-4 / 5.01298e-4` |
| 2 | `0.0254479 / 0.1216831 / 0.1340926` | `5.36997e-5 / 2.21954e-4 / 4.40414e-4` |
| 5 | `0.00786645 / 0.0953031 / 0.1537174` | `5.29221e-5 / 2.37052e-4 / 3.91490e-4` |
| 10 | `0.0161879 / 0.1153231 / 0.1531478` | `5.92016e-5 / 2.14750e-4 / 4.19854e-4` |

The non-monotonic spacing pattern and alternating signed errors resemble independently sampled
poses, not a single wrong rigid transform or steadily accumulating scale error.

## Timestamp ledger and unsynchronized OXTS

All timestamp arithmetic used integer nanoseconds. For the 271 mapped frames:

| Offset | Min | Median | p90 | p95 | Max |
|---|---:|---:|---:|---:|---:|
| OXTS sync - `image_00` (ms) | `-4.971505` | `+0.059572` | `+3.923774` | `+4.413290` | `+4.993358` |
| Velodyne forward - `image_00` (ms) | `-7.581053` | `-4.453231` | `-4.390394` | `-4.375484` | `-4.311226` |
| Velodyne start - `image_00` (ms) | `-59.635772` | `-56.518286` | `-56.454408` | `-56.439107` | `-56.366920` |
| Velodyne end - `image_00` (ms) | `+44.473666` | `+47.607271` | `+47.677283` | `+47.697118` | `+47.744468` |
| Odometry `times.txt` - exact image elapsed (ms) | `-0.004832` | `+0.000168` | `+0.003776` | `+0.004512` | `+0.005008` |

The last row reflects decimal serialization of `times.txt`, not a 5 ms sensor offset: its units are
milliseconds after converting the original nanosecond differences.

The official unsynchronized drive exposes 2,967 native OXTS packets spanning and bracketing every
mapped `image_00` timestamp. Synchronized frames map to unsynchronized packet indices 45 through
2856. The nearest previous and following packets are a median `5.054542 ms` before and
`4.934178 ms` after the image timestamp. Only OXTS timestamps and text packets were range-extracted
from the official 1,724,245,728-byte archive for this diagnosis; no raw images or point clouds were
added to the repository.

## Timing-hypothesis tests

The preregistered variants were all retained:

| Variant | Translation median / p95 / max (m) | Rotation median / p95 / max (rad) |
|---|---:|---:|
| Synchronized OXTS production | `0.0345813 / 0.0741020 / 0.0884767` | `3.78494e-5 / 1.45397e-4 / 3.06132e-4` |
| Nearest previous raw OXTS | `0.0640279 / 0.1119349 / 0.1283112` | `2.03577e-4 / 6.27394e-4 / 1.38981e-3` |
| Nearest raw OXTS | `0.0345121 / 0.0738934 / 0.0881610` | `3.78494e-5 / 1.45397e-4 / 3.06132e-4` |
| Linear translation + quaternion SLERP | `0.0009430 / 0.0019810 / 0.0029335` | `4.38274e-5 / 1.67572e-4 / 4.39612e-4` |

The nearest variant reproduces the synchronization policy. The generic interpolation candidate is
not authoritative and does not pass the original tolerance; it reduces translation median by
about 36.7x and the maximum by about 30.2x, while not improving the orientation distribution. The
exact KITTI Odometry interpolation arithmetic was not recovered, so no candidate variant is
promoted as official.

The physical timing signature independently supports the cause:

- `speed * abs(OXTS-image offset)` has median `0.0342856 m`, close to the observed
  `0.0345813 m`, and Pearson correlation `0.862406` with translation error;
- `angular rate * abs(offset)` has median `3.28187e-5 rad`, close to the observed
  `3.78494e-5 rad`, and Pearson correlation `0.824961` with rotation error;
- the production-to-interpolated translation displacement correlates `0.999803` with the observed
  translation error, and the analogous rotation signature correlates `0.726196`.

This analysis applies no time shift, pose fit, correction, or change to production data.

## Conclusion and prospective protocol recommendation

The original numerical-equality oracle conflated two distinct official products. Production is
exact against official Raw devkit semantics, sequence mapping and camera calibration are correct,
and the residual's scale and shape follow the measured Raw synchronization offsets. The cause is
therefore classified **DATA-PRODUCT / TIMING**, not IMPLEMENTATION or MIXED.

A prospective M6a Protocol Revision 2 is recommended for owner review:

1. **Correctness oracle:** direct official KITTI Raw devkit semantics, with a preregistered strict
   arithmetic tolerance for OXTS conversion, calibration, normalization, and reconstruction.
2. **External trajectory consistency check:** KITTI Odometry ground truth, with all discrepancy
   statistics reported independently and no expectation of byte or numerical equality.

This task does not implement or promote that revision. The original Tier-A artifact, tolerances,
maxima, and **FAIL** status remain unchanged; M6a remains failed pending an explicitly authorized
protocol revision.
