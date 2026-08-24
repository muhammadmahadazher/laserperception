# M6c KITTI Raw ROS exactness result

> **Preserved historical result:** R2 remains a failed protocol. The final M6c outcome is
> documented in [M6C_RESULTS_R3.md](M6C_RESULTS_R3.md).

Status: **M6c NOT READY — M6A ROS INPUT EXACTNESS FAILED**

M6c stopped at Gate A under prospective Protocol R2, commit
`0a8419978d265571b51f943ffc797b5fcc78c4ca`. No detector inference, Gate B corpus
campaign, performance measurement, governance completion, or downstream milestone was started.

## What passed and failed

The current-only frame 0 reproduced the frozen M6a model-ready payload exactly. Frame 1 was the
first condition containing historical data. Its ROS result retained the exact official timestamp,
H1 history depth, 237,342-point count, float32 `(237342, 4)` shape, current-sweep rows, and time-lag
column, but the XYZT bytes differed:

| Frame | History | Expected SHA256 | Observed SHA256 | Result |
|---|---:|---|---|---|
| `0000000000` | 0 | `68e5350355a3a284fdc8477a9e6222fd48d6f3eeada07d5bac5e20adcf3dac23` | same | PASS |
| `0000000001` | 1 | `4088c7ca546aa4b9a00f485153d4a00fd7ed92cde1e7c70f3a24bb6ab883bf7e` | `5bd1d66a1cfe553ae91493b7eb48f36233afe0947f8ab096576f40d2557f16f7` | **FAIL** |

Gate A therefore ended at 1 PASS, 1 FAIL, and 22 pending. Gate B remained 0/856 and was not
started. The ten detector sentinels remained unexecuted.

## First differing boundary

The first difference is the historical-sweep transform after the real ROS unit-quaternion TF
representation. The frozen expected float32 transform and observed tf2-derived transform differed
in 6 values, with a maximum absolute difference of one float32 ULP
(`1.1920928955078125e-7`). That propagated to 64,629 historical output rows and 77,354 float32 XYZ
values. The largest absolute XYZ differences were approximately `1.91e-6`, `1.91e-6`, and
`2.38e-7`; time lag remained exact.

Two numerical boundaries are visible:

1. The compact M6a/M6b hashes preserve transforms originally serialized on Windows. Recomputing
   frame 1 directly from `KittiRawSequence` in WSL produced a different hash (`7da7dfe8...`), which
   is why M6b had frozen the float32 transform records for portable detector execution.
2. ROS TF must encode rotation as a unit quaternion. Encoding the accepted OXTS/calibration pose
   through real tf2 produced the observed `5bd1d66a...` payload and did not reproduce the frozen
   affine transform byte-for-byte.

This is not treated as a tolerance pass. The full compact diagnostic is
[`gate_a_failure_frame_0000000001.json`](../../benchmarks/m6c/diagnostics/gate_a_failure_frame_0000000001.json).

## Preserved pre-output chronology

Before Protocol R2, two harness invocations stopped before publishing any PointCloud2 or evaluating
a gate:

- the original topic token began with a number and ROS rejected it;
- the first pose helper used a stricter rotation validation bound than the accepted KITTI adapter.

Both were documented prospectively in the protocol before the Gate A run. Protocol R2 then retained
exact output equality and produced the failure above.

## Independence and limitations

The replay adapter alone used `KittiRawSequence` to decode source bytes and obtain OXTS-derived
poses. `LaserPerceptionMultiSweepNode` consumed only published PointCloud2 bytes and tf2 and never
called the dataset adapter. The failure therefore occurred at the intended independent live ROS
transform boundary.

The result does not invalidate M6a or M6b. Those remain accepted offline evidence under their frozen
contracts. It shows that M6c has not established byte-exact equivalence between their frozen
float32 transform products and the ROS unit-quaternion representation.

No tolerance was adopted, no transform was tuned, no frozen source evidence changed, and the R2
run performed no detector/model/ONNX/engine/threshold/performance work after the failed gate. M6c
remains not ready; M6 remains in progress and M5 remains conditional/inactive.

The separately preregistered
[`post-failure diagnostic D1`](M6C_POST_FAILURE_DIAGNOSIS.md) isolates the platform,
unit-quaternion, tf2, and storage boundaries and follows exactly one authorized frame downstream.
It preserves this R2 failure and does not create a replacement success protocol.
