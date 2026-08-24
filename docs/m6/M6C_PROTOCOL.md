# M6c protocol — KITTI Raw ROS replay exactness

Status: prospective Revision R2, frozen before accepted KITTI ROS output

Protocol date: 2026-08-24

Frozen base `main`: `ebbbc0bbc4423e3be476abcd1165f75a136fa54c`

Original implementation commit: `0b74d048423e78ad349c35a55cdc8a9cc082eb8b`

Revision R1 implementation commit: `d74aca083f708ee98f50e08b5a5cf8171ace1397`

Revision R2 implementation commit: `3c4c76d9fbf9ff8787cee9fb8fa0e7dc1e72de18`

Protocol commit: the commit that freezes this latest revision; measurement tools require that exact
commit as `HEAD`.

## Preserved pre-output Revision R1 chronology

The first harness invocation at original protocol commit
`ddfd9f0eb6b8f82d66078bd6bff15d9f9769dbc8` stopped during ROS node construction. The generated
topic token began with the numeric drive suffix (`0001_h10`), which ROS correctly rejected as an
invalid topic name. No raw PointCloud2, model-ready output, or detector output was published, and no
Gate A/B/C/D condition was evaluated or recorded.

Revision R1 prospectively prefixes that orchestration-only topic token with `drive_`. It does not
alter the replay bytes, timestamps, TF, builder configuration, corpus, source identities, detector,
engine, or exactness criteria below. This failure remains recorded rather than being treated as a
scientific gate result.

The Revision R1 invocation then stopped while constructing frame 0, again before publishing any
PointCloud2 or evaluating a gate. The pinned KITTI calibration text produces a rotation whose
maximum orthonormality residual is approximately `8.491793e-8` and whose determinant is
approximately `1.0000000287`. The replay helper had imposed `1e-10`, stricter than the existing
accepted KITTI transform validator's `1e-6`, and rejected that already accepted calibration.

Revision R2 uses the existing `1e-6` transform-validity bound solely when accepting the serialized
official rotation for conversion to the unit quaternion required by ROS. Quaternion normalization
does not relax any output gate: every Gate A/B model-ready byte and every Gate D tensor/detection
identity below still requires exact equality. Any representational difference introduced by ROS
quaternion encoding is therefore retained as the measured finding rather than tuned away.

## Objective and boundary

M6c tests integration exactness only. It asks whether official KITTI Raw acquisitions replayed
through the accepted live ROS 2 boundary reproduce the frozen M6a model-ready inputs and M6b
detector outputs exactly. It does not create a new accuracy or performance result.

The tested chain is:

```text
KITTI Raw acquisition
  -> virtual model-axis float32 XYZ PointCloud2
  -> OXTS-derived time-aware tf2
  -> LaserPerceptionMultiSweepNode
  -> model-ready float32 XYZT PointCloud2
  -> exact_fast
  -> frozen structural-40k TensorRT FP16 engine
  -> unchanged MMDeploy postprocess
  -> DetectionFrame
  -> Detection3DArray
```

No model, checkpoint, ONNX, engine, precision, voxel geometry, maximum voxel count, threshold,
postprocessing, H10/H5 policy, or v0.2 default is changed.

## Independent and shared components

The replay adapter uses `KittiRawSequence` only to decode official files, apply the frozen KITTI
native-to-model-axis rotation, and obtain the accepted OXTS/calibration pose. The offline oracle
uses the same geometry source and the same `MultiSweepBuilder` arithmetic.

The ROS path under test is independent at these boundaries:

- float32 raw PointCloud2 serialization and decoding;
- TF publication, transport, buffering, and `lookup_transform_full` across acquisition times;
- `LiveSweepHistory` selection;
- model-ready PointCloud2 serialization and decoding.

`LaserPerceptionMultiSweepNode` consumes only published PointCloud2 bytes and tf2. It does not
import or call `KittiRawSequence`. Thus the gate is not merely a second call to the offline dataset
adapter inside the builder node.

## Frozen frame, timestamp, TF, and history contracts

Native KITTI Velodyne axes are +X forward, +Y left, +Z up. Replay publishes the accepted virtual
model-axis frame `kitti_model_aligned_lidar`, where +X is right, +Y is forward, and +Z is up, using:

```text
A = [[0, -1, 0],
     [1,  0, 0],
     [0,  0, 1]]
det(A) = +1
```

Point row order is unchanged after this rotation; reflectance is ignored. The fixed TF frame is
`kitti_world`. Every PointCloud2 header carries its official KITTI nanosecond acquisition timestamp.
The existing live boundary then floors that stamp to integer microseconds exactly once for the
builder. Historical time lag remains current acquisition time minus historical acquisition time.

TF publishes the accepted virtual model-lidar pose derived from official Raw OXTS and date-level
calibration. Same frame names at different acquisition times are not identity. The builder must use
`lookup_transform_full` through `kitti_world`; missing required TF rejects the frame and fails the
gate.

H10 is current plus up to ten preceding acquisitions. H5 is current plus up to five. Historical
order is nearest-to-farthest and builder output remains current first.

## Frozen source evidence and detector artifacts

| Artifact | SHA256 |
|---|---|
| `benchmarks/m6a/results/kitti_raw_offline_reconstruction.json` | `a62da9808079994d971c4d47bbdf04f2c50d44ddcfd0c7958f7534512552155b` |
| `benchmarks/m6b/results/kitti_raw_cross_domain_characterization.json` | `b9d47120aabb38733d79f987f16277ebab626d2b7c13159c40c30a68b76c1d26` |
| `benchmarks/m6b/diagnostics/pre_inference_input_ledger.json` | `2413224808b0140856a0e00f884c53bc9b49ec6b545172f5e7fa57d00802dc15` |
| local full M6b result (41,987,113 bytes) | `87870b2aa0cc2a91d39331afc8154fdad0c8c796f1cabfb4f8530a3eb106de27` |
| local full M6b input ledger (5,837,452 bytes) | `e25b3d62113cc7e8c1fcf736caa68b1ab698f965f007c758ff91d3e498ca6caa` |
| M6c detector sentinel preregistration | `e80e803fe8923a52ced1bdecd17841a4cf915352a56ea5427607746409cf3be3` |
| checkpoint | `f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0` |
| ONNX | `61ce22a8ca31498675c32576bfb94f0093d31dc95d2762f7254bf915a59ecc16` |
| structural-40k TensorRT engine | `2e790b1cdbdc1b88c2aafdc81b5921ebee152edd8408158f88437ae4dd1f3e7f` |

The engine profile remains min/opt/max `4352 / 18207 / 40000`. The sentinel preregistration is
`benchmarks/m6c/preregistration/detector_sentinels.json` (1,492,097 bytes). It was generated without
M6c inference after verifying both full external M6b artifacts. The ten accepted checkpoint payloads
were cross-checked against the verified full result; their expected model-ready, voxel, raw tensor,
DetectionFrame, and semantic detection payloads are embedded in the tracked preregistration.

## Gate A — M6a ROS input exactness

Drive: `2011_09_26_drive_0001`, H10. Frozen current frames:

```text
0, 1, 2, 5, 10, 11, 14, 17, 23, 30, 36, 43, 49, 55, 62, 65, 68,
75, 81, 87, 94, 100, 106, 107
```

For each output require exact timestamp, history depth, point count, float32 `(N, 4)` shape, row
order, bytes, and SHA256. Acceptance is 24/24. Any difference stops M6c before detector execution.

## Gate B — complete M6b ROS input exactness

Frozen population:

- `2011_09_26_drive_0001`, frames 10–107: 98 frames;
- `2011_09_26_drive_0091`, frames 10–339: 330 frames;
- both H10 and H5: 428 + 428 = 856 conditions.

For every condition require exact drive/frame/condition identity, official timestamp, history depth,
point count, float32 `(N, 4)` shape, and model-ready SHA256. No detector inference is run for the
complete corpus. Acceptance is 856/856.

The run is restart-safe under `.local/m6c`. Atomic progress records bind every PASS to the protocol
commit, implementation commit, drive, frame, condition, and source-evidence hashes. A resumed pass
may replay preceding acquisitions only to warm the required history; it skips comparison for already
verified conditions. It may span multiple wall-clock sessions. Progress and session wall-clock time
must be recorded. The corpus must never be reduced to fit a session.

## Gates C/D — frozen detector sentinels and ROS output exactness

The sentinel current frames were selected before M6c inference:

```text
2011_09_26_drive_0001/0000000010
2011_09_26_drive_0001/0000000083
2011_09_26_drive_0001/0000000011
2011_09_26_drive_0001/0000000015
2011_09_26_drive_0091/0000000010
```

Both H10 and H5 are required, for ten detector conditions. Detector execution is forbidden until
Gates A and B pass.

For every sentinel require exact:

- model-ready input SHA256;
- voxel count and retained voxel tensor hashes;
- raw TensorRT `cls_score`, `bbox_pred`, and `dir_cls_pred` hashes;
- complete DetectionFrame payload and SHA256;
- Detection3DArray header, per-detection headers, count, class identity, score, center, L/W/H, and
  quaternion orientation.

The current LaserPerception `Detection3DArray` contract does not expose `velocity_xy` and does not
overload it into another field. Velocity is therefore frozen as absent from this ROS equality gate,
not treated conditionally. Acceptance is 10/10.

## Equality and failure policy

All gates use exact equality. No numeric tolerance may be introduced after seeing a difference.
Unexpected TF failure, rejected target, history reset, missing artifact, artifact hash mismatch,
engine mismatch, or identity mismatch is a gate failure. The first expected/observed boundary must
be preserved, downstream execution stops, and remediation requires an explicit prospective protocol
revision.

Wall-clock values are orchestration-progress records only. M6c reports no latency, throughput, FPS,
playback-rate, domain metric, or real-time claim.

## Evidence and scope freeze

Local resumable state and detailed captures remain under `.local/m6c`. On full success, compact
tracked evidence will be written to:

- `benchmarks/m6c/results/kitti_raw_ros_exactness.json`;
- `docs/m6/M6C_RESULTS.md`.

No raw points, PointCloud2 payloads, dataset paths, private machine paths, engine bytes, or raw
network arrays may enter tracked evidence. M6c does not authorize training, tuning, localization,
odometry, deskew, tracking, camera fusion, performance work, Jetson work, a release, or another
milestone.
