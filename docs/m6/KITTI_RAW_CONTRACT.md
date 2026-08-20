# KITTI Raw source and dataset contract

Status: M6a complete under Protocol R2. This contract was frozen before any KITTI detector
initialization or prediction. M6a validates ingestion and reconstruction only.

## Authoritative sources

- [KITTI Raw](https://www.cvlibs.net/datasets/kitti/raw_data.php) describes synchronized data at
  10 Hz, binary Velodyne point clouds, OXTS, calibration, and human-validated tracklets.
- [KITTI IJRR paper](https://www.cvlibs.net/publications/Geiger2013IJRR.pdf) defines the sensor,
  synchronized archive layout, timestamp files, coordinate frames, and tracklet conventions.
- The official KITTI Raw devkit is the source for the calibration and OXTS equations transcribed
  here. It remains validation material, not a runtime dependency.
- The official odometry devkit maps odometry sequence 04 to
  `2011_09_30_drive_0016`, frames 0 through 270. Under prospective Protocol R2, its public
  ground-truth poses provide separately labelled external trajectory context, not the KITTI Raw
  correctness equality oracle.
- [KITTI dataset terms](https://www.cvlibs.net/datasets/kitti/) apply. KITTI data are not
  redistributed by LaserPerception; the official page identifies the data as CC BY-NC-SA 3.0.

Archive presence and byte sizes in `configs/m6/kitti_raw.yaml` are metadata from the official
`avg-kitti` object store. Protocol R2 identities and hashes are frozen in
`configs/m6/kitti_raw_r2.yaml`.

## Frozen drive selection

Selection used official metadata and did not use detector results.

| Role | Raw drive | Category | Frames | Tracklets | Reason |
|---|---|---:|---:|---:|---|
| Canonical reconstruction | `2011_09_26_drive_0001` | City | 108 | yes | Small sequential urban drive with OXTS, calibration, timestamps, more than ten history frames, and future-M6b labels |
| Adapter pose-oracle drive | `2011_09_30_drive_0016` | Road | 279 raw / 271 mapped | no | Its first 271 synchronized OXTS records exercise the production adapter against a direct official Raw-devkit transcription; odometry sequence 04 is external context only |
| Future M6b candidate | `2011_09_26_drive_0017` | City | 114 | yes | Compact independent tracklet drive |
| Future M6b candidate | `2011_09_26_drive_0019` | City | 481 | yes | Longer independent tracklet drive |

M6a downloads the canonical drive and adapter pose-oracle drive only. These roles are intentionally
separate: passing all 271 mapped frames on `drive_0016` validates the adapter implementation, while
a second all-108-frame transfer check validates the same official Raw semantics on reconstruction
`drive_0001`. Neither result claims that odometry sequence 04 supplies poses for `drive_0001`.
The other two candidates are frozen metadata, not detector workloads.

## Velodyne point records

The sensor is a Velodyne HDL-64E. Each `velodyne_points/data/##########.bin` file is a headerless,
row-aligned sequence of four little-endian IEEE-754 float32 values:

1. `x`: forward, metres;
2. `y`: left, metres;
3. `z`: up, metres;
4. reflectance: the source intensity/reflectivity channel.

The native Velodyne frame is right-handed. The official devkit reads the file directly into four
rows and transposes it; it does not sort points. LaserPerception therefore preserves the file's
measurement order byte-for-byte. The source material does not define a repair rule for NaN or
infinite rows. The production adapter fails closed on non-finite data before constructing
`RawSweep`; it never silently removes or rewrites rows. Reflectance is retained in the ignored
feature slot for provenance but the frozen PointPillars input remains `x, y, z, time_lag`.

KITTI notes that the rotating scan is not ego-motion untwisted. M6a does not deskew it or infer
per-point time.

## Timestamp files and selected acquisition stamp

The Velodyne directory contains three official nanosecond-text clocks:

- `timestamps_start.txt`: start of the physical revolution;
- `timestamps_end.txt`: end of the physical revolution;
- `timestamps.txt`: the forward-facing/reference instant synchronized to the camera trigger.

M6a freezes `velodyne_points/timestamps.txt`. It is defined for every synchronized frame and is the
best documented single rigid acquisition reference for later ROS headers. The decision precedes
all detector execution.

Timestamp text has no timezone designator. LaserPerception parses the Gregorian fields and exactly
nine fractional digits using integer arithmetic, then maps the timezone-naive calendar to a UTC
Unix-epoch number solely as a deterministic representation. This is not a claim that KITTI's text
declares UTC. The exact nanoseconds remain provenance. `RawSweep` receives
`timestamp_nanoseconds // 1000`, matching v0.2 live ingestion; the remainder
`timestamp_nanoseconds % 1000` is recorded. The existing binary64 microseconds-to-seconds lag
conversion is unchanged.

## Calibration frame graph

Official rigid calibration text stores a 3x3 `R` and three-vector `T` for the column-vector rule
`p_target = R @ p_source + T`, in metres:

- `calib_imu_to_velo.txt`: IMU/OXTS to Velodyne, `T_velo_from_imu`;
- `calib_velo_to_cam.txt`: Velodyne to unrectified camera 0, `T_cam0_from_velo`;
- `calib_cam_to_cam.txt`: camera intrinsics/extrinsics and `R_rect_00` rectification.

LaserPerception uses the IMU as ego. Native Velodyne-to-ego therefore requires inversion of
`T_velo_from_imu`. The virtual model-frame sensor has the same physical origin; only its basis is
rotated. The production parser validates finite float64 values, orthonormal rotations,
determinant +1, and homogeneous inverse closure.

For the odometry external check:

`T_cam0rect_from_imu = R_rect_00 @ T_cam0_from_velo @ T_velo_from_imu`.

The OXTS-derived pose is conjugated by this calibration before being reported beside official
camera-0 odometry poses. This comparison no longer defines the Raw adapter's correctness gate.

## OXTS pose semantics

The first six OXTS fields are latitude, longitude, altitude, roll, pitch, and yaw. The production
adapter directly transcribes the official `convertOxtsToPose.m` path in float64:

- `scale = cos(latitude_0 * pi / 180)`;
- Mercator `x = scale * longitude * pi * 6378137 / 180`;
- Mercator `y = scale * 6378137 * log(tan((90 + latitude) * pi / 360))`;
- `z = altitude`;
- `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`;
- normalize every pose by the inverse of the first valid pose.

The resulting homogeneous matrix maps coordinates in the current OXTS/IMU frame into the first
OXTS/IMU frame. Angles enter sine/cosine directly, so no explicit yaw-unwrapping step is required.
Protocol R2 compares these direct float64 4x4 matrices to an independent official Raw-devkit
transcription and requires exact equality of all 16 scalars. If any non-zero difference appears,
measurement stops; no tolerance is adopted after the fact.

The Raw devkit overview informally describes vehicle-aligned forward/left/up sensor frames, while
the OXTS conversion source documents its RT3000 base convention. The production implementation
does not reconcile these comments by guesswork: it uses the official matrices and equations
exactly. The later odometry comparison remains a non-blocking common-camera-frame consistency
check because R1 established that synced Raw OXTS and odometry ground truth are distinct official
timing products.

## Tracklets for later M6b

Tracklets are XML and are not detector-evaluation inputs in M6a. The official devkit defines:

- native Velodyne coordinates;
- class strings including `Car`, `Van`, `Truck`, `Pedestrian`, `Person (sitting)`, `Cyclist`,
  `Tram`, and `Misc`;
- dimensions `h, w, l`;
- translation `tx, ty, tz` at the bottom/contact-centre convention;
- rotation `rx, ry, rz`, with validated labels using yaw about +Z and roll/pitch effectively zero;
- frame span, occlusion, truncation, and state fields.

Coverage is not uniform: KITTI publishes tracklets only for sequences passing its third human
validation stage, and not every object category is exhaustively interchangeable with nuScenes.
The canonical `2011_09_26_drive_0001` archive is present and verified: its tracklet archive SHA256
is `fe1a9a054f0cf24459d6637b54800b6d0c1d632fa0c6a42a1b1ae81efe4168f7`, and its extracted XML
SHA256 is `34f0672dee9dc94535893e653b4a66e6ddf534a09d2533bac4e62965935a91b8`.
It contains 15 tracklets (`Car`: 12, `Cyclist`: 2, `Tram`: 1), 572 temporal poses, and coverage
from frame 0 through frame 107. Thus M6b can remain on the same canonical drive, subject to the
still-unresolved taxonomy and evaluation policy below.

### Proposed M6b GT mapping — NOT YET ACTIVE

`Car` and `Pedestrian` are possible overlapping concepts. `Van`, `Truck`, `Tram`, seated persons,
and `Misc` require an explicit review. `Cyclist` must not be silently equated with nuScenes
`bicycle` or `motorcycle`. No mapping, matching, scoring, or detector output is active in M6a.
