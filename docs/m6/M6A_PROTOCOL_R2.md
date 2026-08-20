# M6a Protocol Revision 2 — prospective KITTI Raw offline oracle

Status: preregistered prospectively after the original Tier-A failure and after the M6a-R1
root-cause diagnosis, but before any new canonical M6a measurement.

## Chronology and revision boundary

The scientific chronology is immutable:

1. Protocol v1 was frozen at `4d6bc3704f5404fbb761cc758c60f7958e17b872`.
2. The clean v1 measurement implementation was
   `ec9e341056807d5549353c8ef362fd109b25f2f2`.
3. The preregistered Tier-A comparison failed. Its artifact is
   `benchmarks/m6a/diagnostics/pose_oracle_failure_ec9e341.json`, SHA256
   `894960c34b96aac02db2a8a10f13c66e39e3ac99b3ef34ee76c0ef3e88eabaf3`.
4. M6a-R1 diagnosed the cause as **DATA-PRODUCT / TIMING**. Its artifact is
   `benchmarks/m6a/diagnostics/pose_oracle_diagnosis_ec9e341.json`, SHA256
   `44509f4c28fafbdd848c2627c99cde4615bd8e6011520c2a371b1ee3ce6853d8`, committed
   in the diagnosis chronology ending at `a4fb2625db5f95b4eb81e0a70051037285c0be61`.
5. This document creates Protocol R2. Only measurements from a later clean measurement commit may
   be considered under R2.

The original Tier-A result remains **FAIL**. R2 does not change its observation, tolerance,
status, filename, or hash and does not reinterpret it as a pass.

## Scientific reason for the revision

Protocol v1 incorrectly assigned one numerical-equality role to two official products with
different timing provenance. R1 established all of the following:

- LaserPerception and a direct transcription of official KITTI Raw `convertOxtsToPose.m` produced
  identical matrices for all 271 mapped frames;
- the independently derived raw Velodyne-to-rectified-camera calibration and odometry sequence-04
  `Tr` differed by only `4.451994328746878e-14` in matrix elements and
  `5.868706941768216e-15 m` in translation;
- the complete production and independent Raw-devkit camera-frame chains were identical;
- synchronized Raw OXTS selects the closest packet from the native 100 Hz stream, whereas the
  currently distributed KITTI Odometry poses are a separately corrected, interpolated/subsampled
  trajectory product; and
- measured displacement and angular signatures followed the actual OXTS-to-image time offsets.

The revision separates software correctness from external trajectory context. It does not alter
the production pose adapter.

## Two frozen oracle roles

### A. KITTI Raw pose correctness oracle — blocking

Purpose: verify that LaserPerception implements official KITTI Raw OXTS and calibration semantics.

Reference: a diagnostic-only direct transcription of the official Raw devkit equations, using the
same official text inputs and binary64 matrix representation as the production adapter, plus an
independently parsed raw calibration chain.

Candidate: the unchanged production `laserperception.datasets.kitti_raw` pose and calibration
adapter.

Two explicitly distinct drives are covered:

1. **Adapter pose-oracle drive:** all 271 mapped frames of
   `2011_09_30_drive_0016`, the drive mapped to odometry sequence 04. This validates the adapter
   and supports the external odometry comparison.
2. **Canonical reconstruction drive transfer check:** all 108 frames of
   `2011_09_26_drive_0001`. This validates the same Raw-devkit semantics on the actual drive whose
   poses feed the offline reconstruction. It has official tracklets; the official tracklet archive
   contains `tracklet_labels.xml`. It has no odometry numerical-equality oracle.

Passing the 271-frame drive does not, by itself, claim that the poses used for reconstruction were
checked on that same drive. The two results must remain separately labelled.

#### Exact comparison boundary

Exact equality is required only at the direct comparison boundary where both routes consume the
same parsed float64 OXTS/calibration scalars, use the same 4x4 homogeneous column-vector layout,
apply the documented Raw-devkit arithmetic, normalize to their own first frame, and return the
matrices before JSON serialization or any additional reporting composition.

For every frame, require:

- `numpy.array_equal` for the complete float64 4x4 production and reference matrices;
- all 16 scalar elements numerically equal, including identical signed finite values;
- derived matrix maximum difference exactly `0.0`;
- derived rotation maximum exactly `0.0 rad`;
- derived translation maximum exactly `0.0 m`; and
- exact-equality count equal to frame count.

If any non-zero difference appears, **STOP and report it as the finding**. Do not introduce or
adopt a tolerance after observing it. If another route adds serialization, a different matrix
composition, or a different arithmetic order, its differences must be reported separately and
must not be substituted for this direct gate.

The independently parsed raw-calibration relationship remains separately reported with the
predeclared R1 diagnostic limits: rotation-matrix maximum `1e-9`, rotation angle `1e-8 rad`, and
translation norm `1e-6 m`. This comparison involves official decimal products and different
composition routes, so it is not relabelled as an exact-equality gate.

### B. KITTI Odometry external consistency check — non-blocking context

Purpose: report independent trajectory context for sequence 04. KITTI Odometry ground truth is
not the KITTI Raw pose correctness oracle.

The check must preserve:

- absolute translation and stable rotation distributions;
- relative-pose distributions for deltas 1, 2, 5, and 10;
- raw OXTS, `image_00`, Velodyne, and odometry-time relationships;
- raw-versus-odometry calibration comparison; and
- the corrected/interpolated/subsampled odometry provenance.

No numerical-equality pass criterion applies. No fitted alignment, time shift, inferred
interpolation procedure, or production pose replacement is allowed. The original Tier-A failure
is the permanent evidence that the roles must remain separate.

## Frame-zero disposition

The R1 frame-zero distance-to-ideal-identity check used a predeclared `1e-12` threshold and
technically failed: the candidate matrix/translation residual was approximately `4.65e-10`, and
the serialized odometry matrix residual was approximately `3.56e-10`. Both rotations were zero;
the values arise at the serialized/inverse-composition reporting boundary and are about ten orders
below the original `0.088 m` trajectory discrepancy.

R2 resolves this openly:

- direct production-versus-Raw-devkit equality at frame zero remains part of the blocking exact
  gate and must be exact;
- distance of either numerically normalized/serialized product from mathematical identity is a
  known, non-blocking diagnostic with no pass threshold;
- the previous `1e-12` result remains recorded as failed and is not silently ignored or relaxed;
  and
- candidate and reference frame-zero matrices, rotation, and translation residuals must be
  reported in canonical evidence.

## Timestamp contract

The selected acquisition stamp remains `velodyne_points/timestamps.txt`. Preserve the source
integer nanoseconds and record:

```text
timestamp_microseconds = timestamp_nanoseconds // 1000
submicrosecond_remainder = timestamp_nanoseconds % 1000
```

The existing `RawSweep`/`MultiSweepBuilder` lag arithmetic is unchanged. Odometry interpolation
timestamps may not replace KITTI Raw acquisition semantics.

## Gate ledger at the R2 boundary

| Original M6a item | R2 status before new measurement | Reason |
|---|---|---|
| Dataset/point/timestamp/calibration adapter CPU tests | PASS — re-run | Already implemented without external measurements |
| Original Tier-A odometry equality | Historical FAIL | Permanently preserved, not revalidated as a gate |
| Raw-devkit pose correctness | REVALIDATE | New blocking role on both explicit drives |
| Odometry trajectory comparison | REVALIDATE | External context only |
| Model-frame derivation and analytic basis tests | REVALIDATE | Must be documented against official nuScenes conversion, actual LIDAR_TOP calibration, and pinned preparation path |
| Raw decode on official canonical data | NOT YET RUN | Blocked by the v1 Tier-A stop |
| 24-frame offline reconstruction invariants | NOT YET RUN | Blocked by the v1 Tier-A stop |
| Determinism and ROS-future oracle hashes | NOT YET RUN | Blocked by the v1 Tier-A stop |
| Input-shift/pillar diagnostics | NOT YET RUN | Blocked by the v1 Tier-A stop |
| Tracklet contract/selected-drive availability | REVALIDATE | Contract drafted; actual canonical archive must be recorded |
| M6b draft | REVALIDATE | Draft only; remains inactive |

## Frozen model-frame and reconstruction rules

The model-frame alignment must be established without detector output from:

- official KITTI Velodyne basis;
- the official nuScenes KITTI conversion relationship;
- the actual pinned `LIDAR_TOP` calibrated-sensor record;
- the pinned MMDetection3D test preparation path; and
- the M4.5a current-sensor-frame contract.

The existing frozen candidate remains subject to revalidation, not post-result selection:

```text
A = [[0, -1, 0],
     [1,  0, 0],
     [0,  0, 1]]
```

No translation, scaling, beam modification, deskew, or detector-guided rotation is allowed.

The canonical reconstruction drive is `2011_09_26_drive_0001`, not the sequence-04 pose-oracle
drive. Its 108 frames and frozen 24 indices remain:

`[0, 1, 2, 5, 10, 11, 14, 17, 23, 30, 36, 43, 49, 55, 62, 65, 68, 75, 81, 87, 94, 100, 106, 107]`.

`MultiSweepBuilder` remains unchanged. Each current frame is followed by at most ten earlier
acquisitions, nearest to farthest, without padding. The only permitted row-count change is the
builder's existing strict detector-range crop. Source and surviving row order remain fixed.

Every selected frame is a determinism sentinel: rebuild each one ten times and require identical
output bytes and SHA256 across all repetitions.

## Remaining canonical gates

After the Raw pose correctness gate passes, the clean measurement must verify:

- exact little-endian float32 XYZR decode, bytes, row count, XYZ values, and source order for all
  source files used by the 24 selected reconstructions;
- finite C-contiguous float32 `N x 4` XYZT output;
- current lag exactly positive float32 zero, one constant lag per acquisition, positive strictly
  increasing historical lags, and distinct lag count equal to acquisition count;
- current-first, history-nearest-to-farthest, within-file row order;
- expected rows from an independent strict `(-50,-50,-5) < XYZ < (50,50,3)` count after the
  unchanged transforms;
- canonical timestamp ns/us/remainder, history identities, counts, lags, and output hashes for all
  24 frames; and
- input-only accumulated/in-range counts, temporal span, unique candidate 0.25 m XY pillars,
  `max_voxels=40000` engagement, overflow count, and overflow fraction for full-history frames.

If the cap engages, an optional input-only spatial characterization may report candidate versus
retained coordinates only if the existing exact ordering can be evaluated without changing the
voxelizer. It may be called a candidate mechanism for M6b, never a detector-quality cause.

No detector, TensorRT, ROS, threshold, model, voxel geometry, or performance path is part of M6a.

## Measurement and promotion discipline

Implementation, CPU tests, finalized contracts, and the inactive M6b draft must be committed in a
clean measurement commit after this protocol commit. Canonical evidence may be generated only
from that exact clean commit. Any blocking failure stops promotion.

The first passing artifact, if all gates pass, is
`benchmarks/m6a/results/kitti_raw_offline_reconstruction.json`. It must preserve the v1 failure and
R1 diagnosis chronology, remain sanitized, and identify both drive roles explicitly. M6b remains
not started until separate owner authorization.
