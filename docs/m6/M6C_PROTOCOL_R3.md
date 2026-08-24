# M6c Protocol R3 — projected-reference ROS validation

STATUS:
**FROZEN — FINAL M6c EXECUTION CYCLE**

Protocol date: 2026-08-24

Base `main`: `ebbbc0bbc4423e3be476abcd1165f75a136fa54c`

Branch: `feat/m6c-kitti-ros-exactness`

Reviewed draft commit: `4e996b81df9481ff3a5b253db58c6580bbff91d7`

R2 failed protocol/measurement: `0a8419978d265571b51f943ffc797b5fcc78c4ca`

D1 final diagnosis: `e64d80ff46bc735b7bec4ad568fa015731ada9eb`

R3 feasibility implementation: `cafc67f41e9abc12fa0e9a9e76a2ef6add197bf1`

R3 feasibility evidence SHA256:
`b3d2503ed513d258fe2526c162e8e53a51df509a38c6a258a248fcbe29be6b4b`

Projected-reference generator implementation: `3b39e2b3d47d39d57f21c75e87fb8122e31cd058`

Projected-reference preregistration boundary: `03ce7729bea0d76028783234dee559fe32cf21db`

R3 measurement implementation: `28d81f3f9d4a5ce92d2dde7b3a6635c5079d1f4b`

Protocol commit: the commit adding this file. Both execution tools require that exact commit as
`HEAD` and verify that this file was last changed by it.

No canonical live R3 output was observed before this protocol freeze.

## Preserved chronology and final-cycle rule

R2 remains a real failure: frame 0 passed original-M6a byte exactness, frame 1 failed, Gate A
stopped at 1 PASS / 1 FAIL / 22 pending, and no downstream gate started. D1 remains post-failure
diagnostic evidence. It separated small platform arithmetic effects from the dominant
unit-quaternion/SO(3) projection and showed that tf2 and builder storage are float32-faithful to
the projected representation. R3 does not rewrite either result.

R3 is the final authorized M6c cycle. Gate 1 failure, Gate 2 failure, or final ROS conversion
failure closes M6c and M6 with a negative result. Passing all three closes M6c and M6 positively.
There is no automatic R4, tolerance revision, population change, or remediation cycle.

## Frozen artifacts and identities

| Artifact | Path or role | SHA256 |
|---|---|---|
| M6a compact oracle | `benchmarks/m6a/results/kitti_raw_offline_reconstruction.json` | `a62da9808079994d971c4d47bbdf04f2c50d44ddcfd0c7958f7534512552155b` |
| M6b characterization | `benchmarks/m6b/results/kitti_raw_cross_domain_characterization.json` | `b9d47120aabb38733d79f987f16277ebab626d2b7c13159c40c30a68b76c1d26` |
| M6b input ledger | `benchmarks/m6b/diagnostics/pre_inference_input_ledger.json` | `2413224808b0140856a0e00f884c53bc9b49ec6b545172f5e7fa57d00802dc15` |
| R2 failure | `benchmarks/m6c/diagnostics/gate_a_failure_frame_0000000001.json` | `fd319dfcb13da570c62d83b4475531bbbd735845808225ebffee4afe03f1cea4` |
| D1 transform evidence | `benchmarks/m6c/diagnostics/post_failure_tf_representation.json` | `07ea0434fb5833c96d8e6c619a8459cb43c30bbde97d5cfdba96ac8288f3db5d` |
| D1 downstream evidence | `benchmarks/m6c/diagnostics/post_failure_downstream_frame10_h10.json` | `6346a9d0f9916ea4c6e2abb4e7f9c58587a49a5f3b4cbe7ac9d2a6b4b2c3cd3c` |
| Detector sentinels | `benchmarks/m6c/preregistration/detector_sentinels.json` | `e80e803fe8923a52ced1bdecd17841a4cf915352a56ea5427607746409cf3be3` |
| Projected manifest | `benchmarks/m6c/preregistration/projected_reference_manifest.json` | `c06cddc6884fef87de99d1c68ec2b5c1f1945f7f9e5ecae6fcb3e4275dd952a2` |
| Original/projected characterization | `benchmarks/m6c/diagnostics/r3_projected_vs_original_characterization.json` | `3b4c04e4347af6d2d3640147b17be5152f90317c7943ca5a6132826e044d14f8` |
| PointPillars checkpoint | external, frozen | `f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0` |
| ONNX | external, frozen | `61ce22a8ca31498675c32576bfb94f0093d31dc95d2762f7254bf915a59ecc16` |
| Structural 40k TensorRT engine | external, frozen | `2e790b1cdbdc1b88c2aafdc81b5921ebee152edd8408158f88437ae4dd1f3e7f` |

The projected manifest contains 860 unique identities: 24 Gate 1A memberships, 856 Gate 1B
memberships, and 20 overlapping memberships represented once. It was generated on
Linux/WSL2 x86_64, Python 3.10.12, NumPy 1.26.4. The later live run uses ROS 2 Humble and
`rmw_fastrtps_cpp`. These projected hashes are platform-qualified M6c references, not portable
hash guarantees for arbitrary NumPy/platform combinations.

## Reference lineage and claim boundary

M6a independently validates official KITTI Raw OXTS/calibration pose and reconstruction semantics.
M6b freezes detector behavior on the accepted offline inputs. R3 begins after mapping the accepted
poses into the unit-quaternion representation ROS TF can carry:

```text
accepted KITTI absolute poses
  -> frozen matrix-to-unit-quaternion conversion
  -> independent quaternion-to-matrix reconstruction
  -> offline historical-to-current composition
  -> existing MultiSweepBuilder arithmetic
  -> projected float32 XYZT reference
```

The live lineage is:

```text
KITTI Raw -> raw PointCloud2 -> unit-quaternion TF -> lookup_transform_full
  -> LiveSweepHistory -> LaserPerceptionMultiSweepNode -> model-ready PointCloud2
```

R3 does not claim that ROS reproduces the original M6a serialized matrices or bytes.

| Component | Projected offline reference | Live ROS path | Relationship |
|---|---|---|---|
| KITTI Raw decoding | accepted `KittiRawSequence` semantics | same accepted source decoding/replay | shared source semantics |
| Absolute KITTI poses | accepted OXTS/calibration poses | same accepted poses | shared |
| Matrix → unit quaternion definition | frozen conversion | frozen conversion used for TF publication | shared representation boundary |
| Historical relative composition | direct offline matrices after projection | `lookup_transform_full` through fixed frame | independent computation |
| Raw point transport | direct `RawSweep` input | PointCloud2 serialization/transport/decoding | independent live boundary |
| History selection | offline requested set | `LiveSweepHistory` in ROS node | independent |
| Builder mathematics | existing `MultiSweepBuilder` contract | existing builder in live node | shared arithmetic |
| Model-ready transport | direct in-memory reference | PointCloud2 publication/decoding | independent live boundary |

Gate 1 validates PointCloud2 transport, timestamps, TF publication and transport, time-aware
fixed-frame composition, live history selection, ROS builder orchestration, and model-ready
PointCloud2 serialization. It does not independently revalidate official KITTI pose derivation or
the internal mathematics of `MultiSweepBuilder`; earlier milestones cover those claims.

## Preregistered original-versus-projected characterization

This is descriptive and is not a gate:

| Population | Compared | SHA identical / different | Point count identical / different | Signed point delta min / max | Max absolute delta | Original / projected points |
|---|---:|---:|---:|---:|---:|---:|
| H10 | 428 | 0 / 428 | 428 / 0 | 0 / 0 | 0 | 569,520,061 / 569,520,061 |
| H5 | 428 | 0 / 428 | 428 / 0 | 0 / 0 | 0 | 310,967,933 / 310,967,933 |
| Total | 856 | 0 / 856 | 856 / 0 | 0 / 0 | 0 | 880,487,994 / 880,487,994 |

All original/projected SHAs differ, while every point count is identical. Gate 1 compares live ROS
against the projected reference, never against the original M6b SHA.

## Gate 1 — projected ROS input byte exactness

Every condition requires exact official timestamp, requested and actual history depth, point
count, shape, float32 dtype, row order, complete XYZT bytes, SHA256, and model-ready frame.

- Gate 1A: drive 0001 frames 0, 1, 2, 5, 10, 11, 14, 17, 23, 30, 36, 43, 49, 55, 62, 65,
  68, 75, 81, 87, 94, 100, 106, and 107 at requested H10. Acceptance is 24/24.
- Gate 1B: drive 0001 frames 10–107 and drive 0091 frames 10–339, each at H10 and H5.
  Acceptance is 856/856.
- The 20 identical overlapping memberships are satisfied by one canonical observation, yielding
  860 unique live comparisons.

There is no tolerance, skipped condition, reduced corpus, or ROS-derived expected reference.
Chronological replay rebuilds history naturally. Local state under `.local/m6c-r3/` is atomic and
may skip only PASS records matching the frozen protocol commit, measurement implementation,
manifest SHA, condition, expected/observed SHA, point count, history depth, and timestamp.

Any Gate 1 failure stops detector execution and closes R3 negatively.

## Gate 2 — unchanged parity-v2 semantic envelope

Gate 2 starts only after Gate 1A is 24/24, Gate 1B is 856/856, and the unique population is
860/860. It verifies the checkpoint, ONNX, and 40k engine hashes without rebuilding them. Each
sentinel is replayed through the actual live chain and must reproduce its projected manifest SHA
before inference.

The frozen population is drive 0001 frames 10, 83, 11, and 15 at H10/H5, plus drive 0091 frame 10
at H10/H5: ten conditions, with no substitutions.

Reference means frozen accepted M6b DetectionFrames. Candidate means DetectionFrames produced
from byte-exact projected/live ROS inputs. The inspected unchanged identities are:

| Component | SHA256 |
|---|---|
| `configs/detection/m2_parity_v2.yaml` | `91e7cde19076c6452d9ff8e0fefc893a6d429622ed30c2da88127d29d4418df0` |
| `parity_v2.py` Stage 1 evaluator | `24fd8c7bcf8ee74049682ecd7d93989f4d62736eaeb35033155c0115281c38b4` |
| `parity_validation.py` sample analyzer | `37652e464a785174170240e99d593cd9d00a8362008537e182ad0e2b0a83d7f0` |
| `parity.py` matcher | `1be52b850ba5f41e1abf96e83923c1f4dbe65a5a2c592a4e6bb4185dc7e83c00` |

The unchanged Stage 1 contract is:

- export score `>= 0.25`, high confidence `>= 0.30`, inclusive diagnostic band `[0.20, 0.30]`;
- class-wise one-to-one matching, reference descending score then stable sort key, maximum-IoU
  unmatched same-class candidate, minimum BEV IoU `0.50`;
- per-condition count difference `<= max(1, ceil(0.05 × reference exported count))` and aggregate
  relative count difference `<= 0.05`;
- bidirectional high-confidence coverage each `>= 0.99`;
- independent `>= 0.99` pass fraction for XY `<= 0.25 m`, absolute Z `<= 0.25 m`, maximum L/W/H
  relative error `<= 0.05`, score difference `<= 0.05`, and axis yaw modulo pi `<= 5°`;
- heading-direction agreement `>= 0.99`, where disagreement is full circular yaw difference
  greater than `90°`;
- zero high-confidence class-name mismatches.

The continuous-outlier union is descriptive, not another gate. Every Stage 1 check must pass.
Stage 2 forensics runs only after failure and may explain but never change the result.

Parity-v2 was accepted for FP32/TensorRT differences on identical model-ready inputs. R3 inherits
it as the project's existing semantic deployment envelope for a different perturbation source. It
was not statistically derived for quaternion-projection input noise. A pass means only that the
representation-induced detector change stayed inside the inherited envelope; it does not make
that perturbation experimentally equivalent to FP16 arithmetic.

## Final Detection3DArray contract

For all ten candidates, the published Detection3DArray must represent its own candidate
DetectionFrame exactly under the accepted conversion: timestamp, frame, detection count, class,
score, center, L/W/H, and orientation. Per-detection headers must be exact. Velocity remains absent.
This is not a frozen-M6b message byte comparison. Acceptance is 10/10 without tolerance.

## Frozen implementation files

| Implementation | SHA256 |
|---|---|
| Gate 1 harness | `97068272cac11d26d83da0b3d81839ba74ef1c6a2510773bb512a0985529988d` |
| Gate 2 harness | `e3cd25c5932a9d4baa0a2ff43d10c0bc61617115282d45b58c34bfc9fd5c0fcf` |
| Historical detector support harness | `44ca8737c57478da56da6df5469674d978297e011f84f915a34bb8379122b20d` |
| R3 progress contract | `5ceac7ab5c69a091bc7e6f56f99d45a02c7684e74179bc14b5fa8646f3343d96` |
| Projected-reference builder | `4930e34ace88f6a4c6d8c45a45a930efa797c3cf7ceb3f671126040e778659b4` |
| Matrix-to-quaternion source | `d128ea170c00a9c8459f336145e15ec8293576dd4ccfe0ecbbcb86140949bee3` |
| KITTI ROS replay node | `32122a000f9f650ae061893e425bdc09acae61a2777dec4f9ded133c2b0138e9` |
| Live multi-sweep node | `715debf2a7ea000575a04cd1d63f6c13e54124f67149a810adbe8bc9d872a22e` |

## Execution and closeout

The order is fixed:

1. Run Gate 1 and commit compact evidence without semantic code changes.
2. Only on complete Gate 1 PASS, run Gate 2 and the Detection3DArray contract.
3. Preserve the final compact result below 5 MB, with no raw points, tensors, or private paths.
4. Document the full R2 → D1 → feasibility → R3 chronology.
5. Close M6c and M6 positively or negatively; do not tune or create R4.

No performance campaign, model change, threshold change, engine rebuild, release, merge, or next
milestone is authorized by this protocol.
