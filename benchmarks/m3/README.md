# M3 ROS 2 evidence

M3 is complete and has one canonical final record:
[`results/rtx4060_ros2_humble_exact_tensorrt_fp16.json`](results/rtx4060_ros2_humble_exact_tensorrt_fp16.json).
It does not replace or modify any M1/M2 result. Historical M3A, M3B-V1, and M3B-V2 diagnostics
remain preserved below because the progression from failure to exact integration is scientifically
material.

## Final production policy and correctness

The deployed ROS configuration explicitly selects:

```yaml
voxelization_mode: exact_fast
provenance_mode: live
```

Historical M2/evidence behavior remains `official`/`full`. Exact-fast initialization is fail-closed
and never falls back to `deterministic=False`. The supported LaserPerception implementation uses
the pinned MMCV dynamic-coordinate CUDA operation plus PyTorch grouping to reproduce the pinned
deterministic hard-voxel outputs exactly; it is not an upstream MMDetection3D implementation and
adds no custom CUDA or C++.

At exact measurement commit `a129b3507597b25f44ab1a833562f68883ebe8ce`:

| Production gate | Result |
|---|---:|
| Official vs exact-fast voxels/num_points/coors | **81/81 bit-exact** |
| Frozen raw TensorRT outputs | **20/20 exact** |
| Frozen final DetectionFrames | **20/20 exact** |
| PointCloud2 point values/hashes and Detection3DArray semantics | **20/20 exact** |
| Low-rate W1 PointCloud2 → exact-fast runtime → Detection3DArray smoke | **1/1 pass** |

The external correctness JSON SHA256 is
`000ba4bd15bc4349a0df29a2252819e00326c406e5b1dc0e787c0c060359d388`.

Frozen assets remained:

| Artifact | SHA256 |
|---|---|
| Checkpoint | `f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0` |
| ONNX | `61ce22a8ca31498675c32576bfb94f0093d31dc95d2762f7254bf915a59ecc16` |
| TensorRT engine | `a005f75852097cd9b193750560b214cc3d5237ae9b6c106c7fca3d4fc348714b` |

## Canonical representative W1 result

The final workload is `mini_val` index 42: 10 historical sweeps plus current keyframe,
354,182 points. It uses ROS Humble, `rmw_fastrtps_cpp`, bounded input depth 1 best-effort QoS,
bounded reliable output depth 5, exact-fast, live provenance, and the unchanged TensorRT FP16
engine on the RTX 4060 Laptop GPU.

After a 30-second sustained GPU warmup, the run used 20 message warmups and 200 measured
accepted/output opportunities. Callback timing starts at detector callback entry and ends after
Detection3DArray `publish()` returns. Same-host loopback starts at the source publisher stamp and
ends at sink reception. The immutable replay PointCloud2 payload was constructed once before
timing; only its stamp was refreshed before publication.

### Primary 20 Hz result — FAIL

| Boundary | Count | Mean | Median | P90 | P95 | Min | Max | Population std | >50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Callback | 200 | 77.391 | 75.701 | 85.496 | 89.197 | 63.475 | 258.069 | 14.799 | 200 |
| Loopback | 200 | 138.457 | 134.250 | 158.999 | 165.446 | 93.869 | 458.677 | 42.475 | 200 |

All latency values are milliseconds. Effective offered rate was 19.509 Hz. The replay published
398 messages; the detector received/accepted/published 221, and the sink received all 221. Within
the measured accepted-output window, 159/359 offered inputs dropped. Effective detector output was
10.825 Hz; detector-to-sink drops and final processing backlog were both zero.

The first/second half entry-interval medians were 87.243/90.463 ms and input drops were 77/82.
Both grew, so the system was explicitly classified as falling behind. The telemetry session was
eligible. Representative 20 Hz operation was **not demonstrated**.

The external 20 Hz record SHA256 is
`1bc77d7cbbdc6151b2a9c17815528d7bbedd10cee49c458f60336778f23b3046`.

### Bounded characterization

The user-authorized cap was followed exactly: test 10 Hz first; because it sustained, test 15 Hz;
stop after those two rates.

| Offered | Callback median | Loopback median | Effective output | Measured drops | Result |
|---|---:|---:|---:|---:|---|
| 10 Hz | 65.483 ms | 81.400 ms | 9.949 Hz | 0/200 | **sustained** |
| 15 Hz | 60.215 ms | 121.219 ms | 13.336 Hz | 21/221 | **not sustained** |

Ten-hertz first/second entry medians were 99.998/99.983 ms with zero drops in both halves.
Fifteen-hertz entry medians were 73.946/72.094 ms with 11/10 drops; it did not accumulate growing
half-to-half loss but still failed the offered-rate/loss criteria. The highest tested clean rate
was 10 Hz. No 5 Hz run or further search was performed.

The external 10/15 Hz record SHA256 values are
`39e260bc67405346b2e252ad8b15d8e8e93c861c4884cbf78b19d64095e3c30d` and
`92d59b01d0888c713080dfa461ed9548c423196e693fb28b1b55eb30ff0a6c52`.

M3 is complete despite the failed 20 Hz target. No postprocess, DDS, executor, or custom CUDA
optimization was authorized or performed.

## Historical M3A scene-start failure

The earlier exact-commit diagnostic at
`d54da837602de2924825d3045cb4a17b72c5b7b0` used scene-start index 0 with zero historical
sweeps. It held 19.945 Hz replay but measured 238.255 ms callback median, 303.283 ms loopback
median, 3.990 Hz output, and 875 bounded input drops. It remains diagnostic rather than canonical.

The 20-sample transport gate from that stage is
[`diagnostics/roundtrip_d54da83.json`](diagnostics/roundtrip_d54da83.json), file SHA256
`bea49823e3d8547e405f1ceef1e4a9a2efe20ba9cff85ddf537171bf332c7462`.
The failed-rate record is
[`diagnostics/failed_rate_d54da83.json`](diagnostics/failed_rate_d54da83.json), file SHA256
`47cbd7e58c995ee42a0e4dee4f4d6ac56eaae91baf85425c287847bf5fb5ac43`.

## Historical M3B-V1 rejection

At commit `ad0d38b6e926f3a03b471c192d3e815cd07d34d1`, the
`deterministic=False` candidate reduced direct W1/W2 voxelization-layer medians from
270.937/291.729 ms to 4.782/5.040 ms. It was rejected because saturated voxels selected different
capped point subsets and the required 30-run W2 detector comparison failed the existing axis-yaw
yardstick: 0.989834 pass fraction with 19 failures among 1,869 matched high-confidence detections.

See [`VOXELIZATION_V1.md`](VOXELIZATION_V1.md) and
[`diagnostics/voxelization_v1_ad0d38b.json`](diagnostics/voxelization_v1_ad0d38b.json).
`deterministic=False` remains rejected.

## Historical M3B-V2 exact diagnostic

At commit `85b6488c92eda266f049ff142fc06bdab658d7ed`, V2 matched all 81 voxel
samples, both 30-run W1/W2 repeatability suites, and all 20 detector samples exactly. Full-history
hard-layer medians changed from 238.910 to 1.758 ms on W1 and 261.918 to 1.918 ms on W2. Direct W1
median changed from 333.137 to 43.168 ms under explicit live provenance.

The ~43 ms figure is isolated **direct runtime** evidence, not ROS callback or loopback evidence.
Unchanged MMDeploy postprocessing remained the largest direct live component at about 21 ms and
was not optimized.

The V2 diagnostic record remains
[`diagnostics/deterministic_voxelization_v2_85b6488.json`](diagnostics/deterministic_voxelization_v2_85b6488.json).
Its candidate was subsequently accepted and integrated through a separate production commit and
fresh correctness/ROS measurement; the diagnostic values themselves were not relabeled canonical.
See [`VOXELIZATION_V2.md`](VOXELIZATION_V2.md).

## Scope and deferred work

M3 did not change or add model training, thresholds, point geometry, NMS, ONNX, engine, precision,
postprocess optimization, DDS tuning, executor redesign, custom CUDA, tracking, or raw sensor
history construction. Postprocessing optimization, ROS/DDS/executor work if later needed,
exact-fast tuning, custom CUDA only if justified, INT8, and other detectors remain post-v0.1
backlog.
