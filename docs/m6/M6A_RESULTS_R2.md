# M6a Protocol R2 result — KITTI Raw offline oracle

Status: canonical PASS, awaiting review. M6b has not started.

## Scientific chronology

The original protocol remains historical evidence. Protocol v1 was frozen at
`4d6bc3704f5404fbb761cc758c60f7958e17b872`; its measurement at
`ec9e341056807d5549353c8ef362fd109b25f2f2` failed the preregistered odometry-equality gate. The
unchanged failure artifact retains SHA256
`894960c34b96aac02db2a8a10f13c66e39e3ac99b3ef34ee76c0ef3e88eabaf3`.

R1 then diagnosed the discrepancy as **DATA-PRODUCT / TIMING** at
`a4fb2625db5f95b4eb81e0a70051037285c0be61`. The accepted diagnostic artifact SHA256 is
`44509f4c28fafbdd848c2627c99cde4615bd8e6011520c2a371b1ee3ce6853d8`. Only after that failure and
diagnosis was prospective Protocol R2 committed at
`17924559ca852d23e661e0451bf1a22fc3af9bf6`.

The new canonical run used clean measurement commit
`1ab832df89109546abedc9f4e7f21c16c4cd0dca`. No original observation, tolerance, status, filename,
or artifact hash was rewritten.

## Revised pose roles

The correctness oracle now compares LaserPerception's adapter directly with an independent
transcription of official KITTI Raw devkit arithmetic at the in-memory float64 4x4 matrix boundary.
All 16 scalars must be exactly equal; the allowed matrix, rotation, and translation maxima are
zero. Any non-zero value would stop measurement rather than trigger a new tolerance.

- Adapter pose-oracle drive `2011_09_30_drive_0016`: 271/271 exact; matrix, rotation, and
  translation maxima all zero.
- Canonical reconstruction drive `2011_09_26_drive_0001`: 108/108 exact transfer check; all maxima
  zero.

These are different drives with different roles. Passing `drive_0016` alone would validate the
adapter, not the poses used for reconstruction; the separate 108-frame check closes that gap.

The individual frame-zero matrices differ from ideal identity by
`4.656612873077393e-10`, so the earlier `1e-12` ideal-identity diagnostic remains recorded as a
historical failure. Production and direct-reference frame-zero matrices are nevertheless exactly
equal. Protocol R2 treats distance from ideal identity as a known non-blocking serialized-inverse
diagnostic and does not invent a replacement limit.

KITTI Odometry sequence 04 is reported separately as external trajectory context, not an equality
oracle. Its Raw-derived absolute translation discrepancy has median `0.0345813 m` and maximum
`0.0884767 m`; absolute rotation has median `3.78494e-05 rad` and maximum `0.000306132 rad` in this
rerun. Absolute distributions, relative Δ1/2/5/10 distributions, calibration comparison, and Raw
OXTS/image timing provenance remain in the canonical record. No interpolation method was promoted
into production.

## Model frame and offline reconstruction

The frozen raw model input uses +X vehicle-right, +Y vehicle-forward, and +Z up. Native KITTI uses
+X forward, +Y left, and +Z up. The proper alignment is:

```text
A = [[0, -1, 0],
     [1,  0, 0],
     [0,  0, 1]]
```

Its determinant is +1 and its inverse is `A.T`. The derivation uses official coordinate sources,
the actual tracked `LIDAR_TOP` calibration, the pinned MMDetection3D preparation path, and M4.5a's
final current-sensor frame. It was frozen before any KITTI detector result.

All 108 canonical `.bin` frames decoded byte-for-byte as little-endian float32 XYZR with exact row
count and order. Reflectance was recognized but not promoted to the four-column XYZT model input.

The frozen 24-frame set is:

```text
0, 1, 2, 5, 10, 11, 14, 17, 23, 30, 36, 43,
49, 55, 62, 65, 68, 75, 81, 87, 94, 100, 106, 107
```

It contains one current-only startup frame, three shallow-history frames, and 20 full-history
frames. The unchanged `MultiSweepBuilder` produced contiguous finite float32 `N×4` XYZT matrices,
current first, then history nearest-to-farthest, preserving source order within the existing strict
range mask. Current lag was exact positive zero; historical lag was constant per acquisition,
positive, distinct, and increasing with age. Every selected frame reproduced one exact SHA256 over
ten independent builds (24/24 sentinels, 240 builds total).

Source acquisitions contained 98,322 to 123,259 points. Selected accumulated rows ranged from
121,015 to 1,354,461, and final in-range rows ranged from 118,545 to 1,327,389. Full-history span
ranged from 1.030776 to 1.032076 seconds.

## Input-shift and tracklet findings

Full-history frames contained 38,091 to 41,437 candidate occupied 0.25 m XY pillars. Five of 20
exceeded the frozen `max_voxels=40000`; the maximum candidate overflow fraction was 3.4679%. This
is an input-only capacity observation, not a proven spatial-drop or detector-quality cause. No
optional retained-pillar spatial characterization and no voxelizer change were performed.

The canonical drive's tracklet archive is present. Its XML contains 15 tracklets (`Car`: 12,
`Cyclist`: 2, `Tram`: 1), 572 poses, and coverage from frame 0 through 107. The official contract is
Velodyne-frame `h,w,l`, bottom/contact-centre translation, and yaw `rz` about +Z. Taxonomy mapping,
especially `Cyclist` versus nuScenes `bicycle`/`motorcycle`, remains unresolved for owner review.

## Canonical evidence and scope

- Artifact: `benchmarks/m6a/results/kitti_raw_offline_reconstruction.json`
- SHA256: `a62da9808079994d971c4d47bbdf04f2c50d44ddcfd0c7958f7534512552155b`
- Measurement commit: `1ab832df89109546abedc9f4e7f21c16c4cd0dca`

No KITTI detector inference, TensorRT KITTI run, ROS KITTI run, threshold tuning, model change,
training, performance campaign, or M6b work occurred. The artifact freezes the offline model-ready
hash targets that a future separately authorized M6b ROS path must reproduce.
