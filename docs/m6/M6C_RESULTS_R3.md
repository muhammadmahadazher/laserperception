# M6c final R3 projected-reference ROS validation

Status: **PASS — M6c complete with a positive projected-reference ROS validation result**

M6c's final authorized cycle passed all three frozen boundaries: 24/24 Gate 1A conditions, 856/856
Gate 1B conditions, and 860/860 unique live conditions matched the committed projected references
byte-for-byte; the ten frozen detector sentinels passed unchanged parity-v2 Stage 1; and all 10
published `Detection3DArray` messages represented their candidate `DetectionFrame` exactly under
the accepted ROS semantic comparison. Stage 2 was not triggered. M6c and M6 are complete.

This result does not erase the original R2 failure. R2 compared ROS output to the original M6a
serialized matrices and stopped at frame 1. R3 instead prospectively froze a same-platform reference
after mapping the accepted KITTI poses through the unit-quaternion/SO(3) representation that ROS TF
can carry. The R3 reference is not described as the M6a oracle.

## Frozen identities

| Boundary | Identity |
|---|---|
| Starting reviewed branch | `4e996b81df9481ff3a5b253db58c6580bbff91d7` |
| Projected-reference boundary | `03ce7729bea0d76028783234dee559fe32cf21db` |
| Measurement implementation | `28d81f3f9d4a5ce92d2dde7b3a6635c5079d1f4b` |
| Frozen protocol | `07c4ba293c3d0efbf01c7efb18d389a67828c3fc` |
| Projected manifest | `c06cddc6884fef87de99d1c68ec2b5c1f1945f7f9e5ecae6fcb3e4275dd952a2` |
| Gate 1 evidence | `a84a501fd7c5a48fc5421c8c507102b2b8a02aea2266b8fe0cbf18a3f3874549` |
| Gate 2 evidence | `e415b3067b12a0c501ef4854e2fa7df9cbf809456ffd3dbd7979d2ce7b50177b` |
| Checkpoint | `f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0` |
| ONNX | `61ce22a8ca31498675c32576bfb94f0093d31dc95d2762f7254bf915a59ecc16` |
| Structural 40k engine | `2e790b1cdbdc1b88c2aafdc81b5921ebee152edd8408158f88437ae4dd1f3e7f` |

The projected population was generated before live output and without ROS, tf2, the detector, or
GPU initialization. It contains 860 unique identities: 24 Gate 1A memberships, 856 Gate 1B
memberships, and 20 shared memberships stored once. Generation and measurement used Linux/WSL2
x86_64, Python 3.10.12, and NumPy 1.26.4; the live run used ROS 2 Humble with
`rmw_fastrtps_cpp`. The hashes are qualified to that platform and pinned arithmetic path.

## Chronology

### R2: preserved failure

R2 required the live ROS path to reproduce the original M6a bytes. Frame 0 passed. Frame 1—the
first condition with history—retained timestamp, history depth, point count, shape, current-sweep
rows, and time lag, but its historical XYZ bytes differed. R2 stopped at 1 PASS / 1 FAIL / 22
pending. Gate B and detector execution did not start. No tolerance was adopted.

### D1: post-failure diagnosis

D1 separated four numerical boundaries through the T0/T1/T2/T3/T4 ladder:

- direct platform arithmetic contributed a small difference;
- converting the slightly non-orthonormal serialized rotation to a unit quaternion and back to an
  SO(3) rotation produced the dominant float32 difference;
- real tf2 differed from the projected matrix only below float32 resolution and was float32-faithful;
- builder storage changed no serialized float32 transform or model-ready output after projection.

Frame 1 retained identical range membership, voxel coordinates, pillar keys/order, `coors`,
`num_points`, and retained-point membership/order, although coordinate-bearing feature bytes were
not exact. The one authorized downstream frame 10 accumulated enough representation-level change
for six points to cross discrete voxel-coordinate boundaries. Retained membership changed,
`num_points` and raw TensorRT outputs diverged, and the result contained 302 rather than 301
detections. D1 explained R2; it did not rescue it or define a tolerance.

### Feasibility and prospective R3

The preregistered feasibility study established 30/30 H10 transform repeatability, 3/3 H10
model-ready byte exactness, and optional H5 feasibility against the projected representation. R3
then generated the full reference population offline, committed it, implemented the final gates,
and froze the protocol before any canonical live R3 output was observed.

## Original versus projected input characterization

The original and projected model-ready hashes differed for all 856 M6b conditions. Point counts
were nevertheless identical for all 856; signed and absolute point-count deltas were zero. This was
descriptive evidence, not an R3 gate.

| Population | Conditions | SHA same / different | Point count same / different | Max count delta | Original / projected points |
|---|---:|---:|---:|---:|---:|
| H10 | 428 | 0 / 428 | 428 / 0 | 0 | 569,520,061 / 569,520,061 |
| H5 | 428 | 0 / 428 | 428 / 0 | 0 | 310,967,933 / 310,967,933 |
| Total | 856 | 0 / 856 | 856 / 0 | 0 | 880,487,994 / 880,487,994 |

## Gate 1: live ROS input exactness

| Gate | Exact | Required | Failed | Result |
|---|---:|---:|---:|---|
| 1A: M6a H10 targets | 24 | 24 | 0 | PASS |
| 1B: full M6b H10/H5 population | 856 | 856 | 0 | PASS |
| Unique live identities | 860 | 860 | 0 | PASS |

The 20 shared Gate 1A/1B conditions were not replayed redundantly. Across chronological sessions,
the ROS nodes received 886 raw frames, accepted all 886 as valid, and published 886 model-ready
outputs. Rejected frames, TF failures, history resets, and filtered invalid points were all zero.
Every target matched timestamp, requested and actual history depth, point count, shape, float32
dtype, row order, complete XYZT bytes, and SHA256. Frame 1 had no special case.

Gate 1 validates PointCloud2 transport/decoding, timestamps, TF publication and transport,
time-aware fixed-frame composition, live history selection, ROS orchestration, and model-ready
PointCloud2 serialization. It does not independently revalidate official KITTI pose derivation or
the internal mathematics of `MultiSweepBuilder`; those are shared, earlier-validated components.

## Gate 2: unchanged detector semantic envelope

All ten frozen conditions reverified their projected model-ready SHA before inference. The frozen
checkpoint, ONNX, and structural 40k engine hashes matched. Parity-v2 Stage 1 passed every frozen
check, so Stage 2 was not run.

| Stage 1 aggregate | Result |
|---|---:|
| Exported detections, reference / candidate | 113 / 113 |
| Per-condition count disagreements | 0 |
| High-confidence coverage, both directions | 81/81 (1.000) |
| High-confidence class mismatches | 0 |
| Continuous outliers | 0/81 |
| Heading-direction agreement | 81/81 (1.000) |
| Threshold-edge disagreements | 0 |

Each continuous metric passed 81/81 comparisons. Maximum differences were 0.06535 m XY,
0.01766 m Z, 2.184% relative dimension error, 0.001926 score, and 0.62947 degrees box-axis yaw.
All are reported under the unchanged preregistered gates; none became a new threshold.

Parity-v2 was originally accepted for rewritten/native FP32 versus TensorRT FP16 on identical
model-ready inputs. R3 reused that unchanged envelope as LaserPerception's existing semantic
deployment standard for a different source of variation. Passing means representation-induced
detector changes remained inside that inherited envelope. It does **not** establish that
quaternion-projection noise is experimentally equivalent to FP16 deployment arithmetic.

## Final ROS output contract

All 10 published `Detection3DArray` messages exactly represented their own candidate
`DetectionFrame` under the accepted M4.5b semantic comparison: timestamp, frame, count, class,
score, center, length/width/height, and orientation. The current message contract did not expose
velocity. Detector counters were 10 received, 10 accepted, 10 published, and zero rejected.

This was a conversion-contract check, not a demand for byte identity with frozen M6b ROS messages.

## Byte-identity implementation note

The protocol commit is the canonical preregistration identity. Gate 1 recorded the LF-materialized
protocol file SHA frozen in its evidence. The sentinel and pre-existing parity evaluator identities
retain their preregistered Windows CRLF byte forms. A clean execution worktree was materialized with
the required line endings for those frozen byte checks. This altered no commit, program semantics,
protocol threshold, input, or output. The redundant measurement-script worktree byte identity can
differ between LF and CRLF materializations even though the exact measurement implementation commit
is fixed. No source or protocol was changed after output observation.

## Claim boundary and closure

The positive claim is narrow: on the recorded WSL2/ROS environment, the live ROS chain reproduced
all committed projected model-ready references byte-for-byte, and downstream detector changes stayed
inside the inherited parity-v2 semantic envelope while ROS output conversion remained exact.

R3 does not claim original-M6a byte identity, portability of projected hashes across platforms,
physical-LiDAR validation, performance, latency, throughput, or real-time operation. It does not
revalidate the KITTI pose oracle or make ROS/tf2 “more exact” than a unit-quaternion representation
permits. No performance campaign or tuning occurred.

The compact canonical result is
[`kitti_raw_ros_projected_validation_r3.json`](../../benchmarks/m6c/results/kitti_raw_ros_projected_validation_r3.json).
Detailed Gate 1 and Gate 2 ledgers remain separately tracked for auditability. M6a is complete,
M6b is complete, M6c is complete with a positive projected-reference ROS validation result, and M6
is complete. M5 remains conditional/inactive. No technical submilestone is active, and no R4 or
next milestone is authorized.
