# M6c post-failure diagnostic D1 plan

Status: **preregistered diagnostic only**. Protocol R2 remains failed and unchanged. This is not
M6c R3, a success-protocol revision, a new M6b measurement, or permission to run Gate B.

## Immutable starting state

- Base main: `ebbbc0bbc4423e3be476abcd1165f75a136fa54c`.
- Diagnostic branch starting HEAD: `f2c8fda3eeda444be8caa8799767d9a3e2f39d04`.
- Frozen R2 protocol/measurement commit: `0a8419978d265571b51f943ffc797b5fcc78c4ca`.
- Frozen R2 result: frame 0 PASS, frame 1 FAIL, Gate A 1 PASS / 1 FAIL / 22 pending;
  Gate B and all detector sentinels were not started.
- Preserved failure record:
  [`gate_a_failure_frame_0000000001.json`](../../benchmarks/m6c/diagnostics/gate_a_failure_frame_0000000001.json),
  SHA256 `fd319dfcb13da570c62d83b4475531bbbd735845808225ebffee4afe03f1cea4`.

The diagnostic must not change M6a/M6b artifacts, R2 hashes or equality rules, the checkpoint,
ONNX, 40k engine, exact-fast behavior, score threshold, geometry, H10, or H5.

## Accepted prior numerical evidence

M6b already isolated a pre-ROS portability boundary. The accepted
[`M6B_PROTOCOL.md`](M6B_PROTOCOL.md#prospective-pre-inference-portability-correction) and
[`pre_inference_platform_reproduction.json`](../../benchmarks/m6b/diagnostics/pre_inference_platform_reproduction.json)
record that Windows reproduced the M6a frame-10 H10 input, while direct WSL2 reconstruction from
byte-identical files differed in one float32 transform translation by one ULP. M6b then
prospectively froze canonical float32 sweep transforms and passed them through the unchanged
`MultiSweepBuilder`. The production rationale and reconstruction contract remain implemented in
[`m6b_input_oracle.py`](../../src/laserperception/evaluation/m6b_input_oracle.py).

This D1 diagnostic cannot use that portability correction as proof of ROS equivalence: deriving
OXTS/calibration poses, representing them as ROS quaternions, and composing them through real tf2
are the boundaries under test.

## Frozen artifacts and identities

| Artifact | SHA256 |
|---|---|
| M6a compact reconstruction evidence | `a62da9808079994d971c4d47bbdf04f2c50d44ddcfd0c7958f7534512552155b` |
| M6b compact result | `b9d47120aabb38733d79f987f16277ebab626d2b7c13159c40c30a68b76c1d26` |
| M6b compact input ledger | `2413224808b0140856a0e00f884c53bc9b49ec6b545172f5e7fa57d00802dc15` |
| M6b full result, external and verified | `87870b2aa0cc2a91d39331afc8154fdad0c8c796f1cabfb4f8530a3eb106de27` |
| M6b full input ledger, external and verified | `e25b3d62113cc7e8c1fcf736caa68b1ab698f965f007c758ff91d3e498ca6caa` |
| M6c preregistered sentinels | `e80e803fe8923a52ced1bdecd17841a4cf915352a56ea5427607746409cf3be3` |
| Structural 40k TensorRT engine | `2e790b1cdbdc1b88c2aafdc81b5921ebee152edd8408158f88437ae4dd1f3e7f` |

The compact and external evidence identities were verified before the R2 campaign and again before
freezing the sentinel manifest. Expected detector identities below come only from those accepted
artifacts, never from a D1 ROS result.

## Diagnostic questions and hypotheses

The diagnostic tests four independently reportable mechanisms rather than presuming one cause:

1. **Platform arithmetic:** T0 differs from T1 before quaternion or ROS work.
2. **Unit-quaternion projection:** T1 differs from T2 when slightly non-orthonormal serialized
   rotations are normalized into a proper unit-quaternion representation.
3. **tf2:** T2 differs from T3 after real publication, buffering, `lookup_transform_full`, and
   relative composition.
4. **Final storage:** T3 differs from T4 at the builder's rotation/translation mapping or float32
   cast boundary.

Any combination may be present. Dominance is determined only from affected float32 transform
values, maximum absolute/ULP deltas, and propagated point differences. If the complete ladder does
not explain the preserved ROS transform, classification is `UNRESOLVED_TRANSFORM_BOUNDARY` and no
GPU/network diagnosis is allowed.

## Primary frame and transform ladder

The CPU/ROS forensic condition is fixed to
`2011_09_26_drive_0001/0000000001`, H10 with actual history depth 1.

- Frozen model-ready SHA: `4088c7ca546aa4b9a00f485153d4a00fd7ed92cde1e7c70f3a24bb6ab883bf7e`.
- Preserved ROS SHA: `5bd1d66a1cfe553ae91493b7eb48f36233afe0947f8ab096576f40d2557f16f7`.
- Frozen transform SHA: `c0c66df4237968a1c0ced2f3bc260d01158e97ef5a5e4bb359efaace9369e733`.
- Preserved ROS transform SHA: `a57ade3532cca9ff0e6a3eb8998a5ba57c882482f10f97d0fc6112abd5336f9e`.

Run and retain the stages in this order:

- **T0:** load the accepted frozen float32 historical transform; do not recompute it.
- **T1:** recompute the same historical-to-current transform in WSL using accepted
  `KittiRawSequence`, OXTS/calibration, and matrix composition only.
- **T2:** convert the same WSL absolute-pose rotations through the production
  matrix-to-normalized-quaternion helper, independently reconstruct matrices from each quaternion,
  then compose historical-to-current using matrix arithmetic without ROS. Verify quaternion norms
  and q/-q equivalence.
- **T3:** publish the same absolute poses through real ROS/tf2 and obtain the relative transform
  through `lookup_transform_full` with the frozen timestamps and fixed frame.
- **T4:** apply the accepted ROS-column-vector to builder-storage mapping and final dtype/cast
  points used by the live builder.

For each stage record float64 where applicable, canonical float32 bytes/hash, rotation determinant,
and maximum orthonormality residual. For each adjacent pair record exact equality, differing
float32 elements, maximum absolute delta, per-element ULP distances, and separate rotation and
translation summaries. The T0/T1 comparison is mandatory even if later stages fail. The
quaternion-to-matrix calculation used for checking is independent of the production helper; q and
-q must reconstruct the same proper rotation.

## Frame-1 propagation and voxel comparisons

Compare historical rows only while separately confirming the already exact current rows and time
lags. Record counts of differing rows and X/Y/Z/time values, maxima, median/p95/p99 non-zero
deltas, and ULP distributions. These are descriptive diagnostics, not tolerances.

Using the frozen PointPillars geometry, measure rather than estimate:

- range-mask membership and points changing membership;
- per-point discrete voxel coordinates and changed-coordinate count;
- expected/observed candidate pillar counts and added/removed key sets;
- retained pillar count, key set, and order under the 40k cap;
- retained point membership and order where deterministic provenance permits;
- `coors`, `num_points`, pillar identity/order, point membership/order, and voxel feature values as
  separate results.

No TensorRT initialization is required for frame 1.

## One authorized downstream condition

Only after the transform ladder explains the preserved ROS transform may D1 execute the detector
for `2011_09_26_drive_0001/0000000010`, H10. No other detector condition is authorized.

Frozen frame-10 references from the verified sentinel manifest are:

- model-ready SHA `5ff825de4c351961f62b416c11042d50bf5d78f2d363f842ce4b5d182456b18a`,
  point count `1,312,220`;
- retained voxel count `40,000`;
- `coors` SHA `3cbdf69da7037d35dcce207c1bb948a21a19957928fa6a5a782c20b1105e81b9`;
- `num_points` SHA `8dd660a9092b3bd075a1c930a42b5a8c1e96000a7e343a8692679fbd1b7df74a`;
- voxel-feature SHA `ec282516ade006647d8b39b6aea63766d3cb2eeac3c587616eb7126deb41978b`;
- raw `cls_score` SHA `cd065f7381305c14ffb1353d9c170a8be4a65d0227e0db8ce867731b006a7242`;
- raw `bbox_pred` SHA `f19ba02133b698bdac6a56408eb0a3378b199517d965e43ac4899e7c925ba678`;
- raw `dir_cls_pred` SHA `dd7a096abcc5591adeb59444304aa1430a66b23507c72b3c329fe2fa03dcd5ac`;
- `DetectionFrame` SHA `565cdd71123b78f2fd5b23456702c7ef0e6410d6f1195a36a0a02c1b4f47b132`,
  with the frozen 301-detection payload embedded in the sentinel manifest.

The current `Detection3DArray` contract exposes header/frame, class, score, pose/orientation, and
size; it does not expose or overload `velocity_xy`. Normalized ROS semantics therefore freeze only
the exposed fields and use the existing M4.5b normalization method.

First run the frozen expected frame-10 payload through `exact_fast` and the verified existing 40k
engine in the same session. If any accepted coors, num-points, voxel, raw-output, or DetectionFrame
identity fails, stop with `M6c DIAGNOSTIC INCONCLUSIVE — CANONICAL DOWNSTREAM CONTROL FAILED` and
do not run the ROS variant. If the control passes, run only the ROS frame-10/H10 variant and
continue through model-ready bytes, voxel structure, voxel values, all three raw tensors,
DetectionFrame, and normalized Detection3DArray semantics even after an intermediate byte
difference. Non-exact deltas are descriptive only; no new gate or tolerance may be introduced.

## Evidence and stop conditions

Tracked outputs are limited to:

- `benchmarks/m6c/diagnostics/post_failure_tf_representation.json`;
- `benchmarks/m6c/diagnostics/post_failure_downstream_frame10_h10.json` only if the detector
  diagnostic is reached;
- `docs/m6/M6C_POST_FAILURE_DIAGNOSIS.md`.

Each JSON must remain compact, contain no full point/tensor arrays or private paths, prefer under
1 MB, and hard-stop before 5 MB. Local arrays, captures, and runtime artifacts remain untracked.

Stop before GPU execution if the transform ladder cannot explain the observed ROS transform. Stop
before the ROS detector variant if the same-session frozen control fails. At no point may D1 revise
R2, rerun Gate A as a success attempt, start Gate B, run the remaining nine detector sentinels,
change a success gate, create R3, update completion governance, open a PR, or begin a performance,
accuracy, release, or unrelated milestone campaign.

The final diagnosis may document future options—same-platform derived references, canonical
transforms as replay pose data, or another data-supported design—but must not select or implement
one.
