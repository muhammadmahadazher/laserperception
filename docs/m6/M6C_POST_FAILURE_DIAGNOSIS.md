# M6c post-failure diagnostic D1

**PROTOCOL R2 STATUS: FAILED.** This document is diagnostic evidence only. It does not revise the
R2 exact-equality rule, create R3, or make M6c ready.

The frozen R2 run at `0a8419978d265571b51f943ffc797b5fcc78c4ca` remains frame 0 PASS,
frame 1 FAIL, and Gate A stopped at 1 PASS / 1 FAIL / 22 pending. Gate B was not started and the
remaining detector sentinels were not run. D1 was preregistered at
`6a00cdc8fc2fa950ca7f8a4bf4261fdeeefbc6d9` before any new detector execution.

The initial ladder at `34d976f22acec713ac756ba48dc226d61d9a1142` explained the preserved transform
before downstream execution. The authorized downstream measurement ran at
`0da293d16a705c58e2d341b1d7379a0c93d6aabd`. A later CPU/ROS-only audit at
`6b768af6a4e6f6de30c83eac462334ccc19e8826` corrected the diagnostic artifact to retain T1's
pre-cast float64 matrix before builder storage; it changed no production runtime and executed no
network. The matrices and statistics below are from that final corrected transform-only run.

## Finding

The preserved frame-1 ROS transform is fully explained by multiple numerical representation
boundaries. Direct WSL matrix arithmetic first differs from the frozen Windows float32 transform
by one translation ULP. Normalizing the accepted, slightly non-orthonormal pose rotations into
unit quaternions adds the dominant float32 point difference. Real tf2 then adds only binary64
differences below float32 resolution, and final builder storage introduces rounding in its
binary64 view without changing the serialized float32 transform or model-ready output.

The complete ladder ends at the exact transform and model-ready hashes recorded in the immutable
R2 failure. The mechanism classifications are:

- `PLATFORM_ARITHMETIC_PRESENT`;
- `UNIT_QUATERNION_PROJECTION_PRESENT`;
- `TF2_ADDITIONAL_DIVERGENCE_PRESENT_BELOW_FLOAT32`;
- `FLOAT32_STORAGE_ROUNDING_PRESENT`.

The input pose rotations had determinant approximately `1.0000000287` and maximum orthonormality
residual approximately `8.49e-8`. Independent quaternion-to-matrix reconstruction produced unit
determinant to rounding, residual at most `2.22e-16`, and identical rotations for `q` and `-q`.
The measured result therefore supports this precise statement: **unit-quaternion representation
projects the serialized rotation onto an orthonormal proper-rotation representation.**

## Frame-1 transform ladder

The condition was `2011_09_26_drive_0001`, frame `0000000001`, H10 with actual history depth 1.
Each comparison below is to the immediately preceding stage.

| Stage | Float32 SHA256 | Prior-stage float32 exact | Differing elements | Max abs delta | ULP median / p95 / p99 / max |
|---|---|---:|---:|---:|---|
| T0 frozen Windows canonical | `c0c66df42379...` | reference | — | — | — |
| T1 WSL direct matrix | `2ed714460e5c...` | no | 1 | `9.31323e-10` | 1 / 1 / 1 / 1 |
| T2 WSL unit-quaternion composition | `a57ade3532cc...` | no | 6 | `1.19209e-7` | 1.5 / 62.75 / 78.95 / 83 |
| T3 real tf2 relative transform | `a57ade3532cc...` | yes | 0 | 0 | 0 / 0 / 0 / 0 |
| T4 final builder storage | `a57ade3532cc...` | yes | 0 | 0 | 0 / 0 / 0 / 0 |

The float64 comparisons remain distinct: T0→T1 differed in 12 values with maximum
`1.26751e-8`; T1→T2 differed in 12 with maximum `7.67689e-8`; T2→T3 differed in 10 with maximum
`4.44089e-16`; and T3→T4 differed in 12 with maximum `5.67137e-8`. T2, T3, and T4 nevertheless
serialize to the same float32 transform hash, which is also the preserved R2 ROS transform hash.

By propagated model-ready values, the unit-quaternion boundary was dominant: T0→T1 affected 294
float32 values with maximum `7.45e-9`, while T1→T2 affected 77,354 values with maximum
`1.90735e-6`. T2→T3 and T3→T4 affected none after float32 storage.

## Frame-1 point and voxel consequence

Expected and ROS outputs retained identical point count, shape, dtype, current-sweep rows, and
time-lag values. Among 118,525 historical rows, 64,629 rows differed: 24,785 X values, 47,070 Y
values, 5,499 Z values, and no time-lag values. Across the 77,354 nonzero differences, absolute
delta median was `2.38419e-7`, p95/p99 `4.76837e-7`, and maximum `1.90735e-6`.

The distinction between voxel structure and feature values is important:

| Boundary | Frame-1 result |
|---|---|
| Range-mask membership | exact; 237,342 retained in both, zero changes |
| Per-point discrete voxel coordinates | exact; zero points changed |
| Candidate pillars | 22,971 in both; key set exact |
| Retained pillars | 22,971 in both; key set and order exact |
| `coors` | exact |
| `num_points` | exact |
| Retained point membership/order | exact |
| Voxel feature bytes | **not exact**; 64,903 values differed, max `1.90735e-6` |

Thus frame 1 preserves discrete structure and membership, but not the coordinate-bearing voxel
feature bytes. D1 does not relabel this as an exact pass or introduce a tolerance.

## One-frame downstream diagnostic

Because the transform ladder exactly reproduced the preserved R2 ROS transform, D1 proceeded to
the one preregistered downstream condition: frame `0000000010`, H10. Before interpreting the ROS
variant, the frozen model-ready control ran in the same session through unchanged `exact_fast` and
the existing 40k TensorRT engine (`2e790b1c...`). It reproduced the accepted model-ready, voxel,
raw-output, 301-detection hash, and complete DetectionFrame payload: **control PASS**.

The tracked sentinel has a frozen Windows-worktree byte identity, including CRLF line endings. An
initial attempt using the LF-normalized WSL checkout stopped before runtime initialization and ran
no network. The actual diagnostic received the verified external frozen bytes with SHA256
`e80e803f...`; neither the tracked sentinel nor its scientific contents were changed.

Equivalence did not recover in the ROS variant:

| Boundary | Exact? | Measured consequence |
|---|---:|---|
| Model-ready XYZT | no | 1,510,648 values; max `7.62939e-6` |
| Range mask | yes | 1,312,220 retained in both |
| Discrete voxel coordinates | no | 6 points changed coordinates |
| Candidate pillar keys | yes | 41,437 in both |
| Retained pillar keys/order | yes | 40,000 in both |
| Retained point membership/order | no | membership and order changed |
| `exact_fast` coors | yes | accepted hash reproduced |
| `exact_fast` `num_points` | no | hash differed |
| Voxel feature values | no | 829,042 values; max `2.13492` |
| TensorRT `cls_score` | no | 1,369,451 values; max `0.09375` |
| TensorRT `bbox_pred` | no | 2,587,890 values; max `0.0683594` |
| TensorRT `dir_cls_pred` | no | 692,888 values; max `0.0336914` |
| DetectionFrame | no | expected 301 detections, observed 302 |
| Detection3DArray semantics | no | expected 301 detections, observed 302 |

The 2.13 voxel maximum is not characterized as a small representation perturbation: the six
coordinate crossings changed retained point membership and per-pillar counts. Raw tensor deltas
are descriptive only, not a new parity gate. Because DetectionFrame cardinality differed,
index-aligned field deltas are not claimed. The ROS message contract still does not expose or
overload `velocity_xy`.

## Scope and interpretation

M6b had already documented the Windows/WSL one-ULP reconstruction boundary and prospectively
froze canonical float32 sweep transforms for its detector measurement. That was valid for M6b,
where reconstructing poses through ROS was not the test. D1 shows why the same workaround cannot
silently establish M6c: real ROS/tf2 necessarily represents rotation as a unit quaternion, and the
accepted serialized KITTI rotations are not perfectly orthonormal.

This diagnosis does not invalidate M6a or M6b, but it also does not rescue R2. Potential future
protocol designs include (A) a same-platform WSL-derived reference that tests full pose derivation
through ROS or (B) publishing frozen canonical transforms as pose data, which would no longer
retest OXTS derivation. A separately justified semantic downstream gate would be another possible
design, but this single D1 condition did not recover DetectionFrame semantics. Selecting or
implementing any option requires explicit owner authorization and a prospective protocol; D1
chooses none.

Compact evidence:

- [`post_failure_tf_representation.json`](../../benchmarks/m6c/diagnostics/post_failure_tf_representation.json)
- [`post_failure_downstream_frame10_h10.json`](../../benchmarks/m6c/diagnostics/post_failure_downstream_frame10_h10.json)

M6a remains complete, M6b remains complete, M6c remains **NOT READY**, and M6 remains in progress.
