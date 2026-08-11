# M3 ROS 2 evidence

M3A measures the ROS-facing interface around the unchanged M2 TensorRT FP16 detector on the
NVIDIA GeForce RTX 4060 Laptop GPU. It does not replace or modify any M1/M2 result.

## Correctness gate

All 20 frozen M2 parity-v2 samples pass the real M3 transport path:

```text
official nuScenes multi-sweep preparation
-> exact Nx4 x/y/z/time_lag array
-> sensor_msgs/PointCloud2 serialization
-> exact subscriber conversion
-> in-memory official MMDetection3D batch
-> official voxelization
-> unchanged TensorRT FP16 engine
-> shared MMDeploy postprocess
-> DetectionFrame
```

The gate requires exact point values/hashes, voxel hashes, raw output hashes/statistics, and final
detections. It passed at implementation commit
`d54da837602de2924825d3045cb4a17b72c5b7b0`; the sanitized record is
[`diagnostics/roundtrip_d54da83.json`](diagnostics/roundtrip_d54da83.json) with file SHA256
`bea49823e3d8547e405f1ceef1e4a9a2efe20ba9cff85ddf537171bf332c7462`.

The frozen round-trip suite preserves the M2 parity-v2 workload distribution: 19 of 20 samples have
10 historical sweeps plus the current keyframe, while `mini_val` index 0 is a scene start with zero
historical sweeps.

## Preregistered performance protocol

- hardware: RTX 4060 Laptop GPU;
- runtime: TensorRT 8.6.1 FP16, frozen engine SHA256
  `a005f75852097cd9b193750560b214cc3d5237ae9b6c106c7fca3d4fc348714b`;
- input: model-ready nuScenes v1.0-mini `mini_val` index 0 repeated (scene start, zero historical
  sweeps);
- cadence: synthetic 20 Hz stress replay, not native annotated-keyframe timing;
- warmups/measured: 20/200;
- input QoS: volatile best-effort keep-last depth 1;
- output QoS: volatile reliable keep-last depth 5; and
- optional markers disabled for the measured canonical Detection3DArray interface.

Boundary A starts at detector callback entry and ends immediately after the detections publisher's
`publish()` call. Boundary B is same-host ROS loopback from the replay's preserved ROS stamp to a
separate sink's reception time. Boundary B is not sensor-to-actuator latency.

## M3A rate-gate disposition

The exact-commit diagnostic at `d54da837602de2924825d3045cb4a17b72c5b7b0` failed every
rate/deadline criterion while the replay itself held 19.945 Hz:

| Boundary | Count | Mean | Median | P90 | P95 | Min | Max | Population std | >50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Callback processing | 200 | 239.026 ms | 238.255 ms | 267.840 ms | 274.637 ms | 191.421 ms | 344.951 ms | 22.306 ms | 200 (100%) |
| Same-host loopback | 200 | 312.519 ms | 303.283 ms | 348.865 ms | 352.550 ms | 266.044 ms | 468.693 ms | 30.699 ms | 200 (100%) |

The replay published 1,096 messages; the detector received/accepted/published 221 and the sink
received all 221 outputs. Bounded best-effort input QoS discarded 875 stale messages. There was no
rejection, detector-to-sink transport loss, or final processing backlog. Effective output rate was
3.990 Hz (19.95% of requested sensor rate).

Therefore:

- there is no accepted/canonical M3 result under `benchmarks/m3/results/`;
- M3A status is `FAIL — review required`;
- M3B-V1 was subsequently authorized as a diagnostic-only hard-voxelization experiment;
- no candidate voxelizer was adopted and no postprocessing or ROS/DDS optimization was attempted;
  and
- the M3B-V1 repeatability evidence requires reviewer inspection before any production change.

The sanitized failed record is
[`diagnostics/failed_rate_d54da83.json`](diagnostics/failed_rate_d54da83.json), file SHA256
`47cbd7e58c995ee42a0e4dee4f4d6ac56eaae91baf85425c287847bf5fb5ac43`. These values are diagnostic,
not accepted M3 performance.

## M3B-V1 hard-voxelization diagnosis

The exact-commit diagnostic at `ad0d38b6e926f3a03b471c192d3e815cd07d34d1` isolated the
official deterministic hard voxelizer from an in-memory `deterministic=False` candidate. Both used
the frozen hard-voxel settings, model, engine, samples, postprocess, and ROS-independent detector
path. The detailed report is [`VOXELIZATION_V1.md`](VOXELIZATION_V1.md); its sanitized record is
[`diagnostics/voxelization_v1_ad0d38b.json`](diagnostics/voxelization_v1_ad0d38b.json), file SHA256
`98315416fa148a52ed14f734f923661cb70a70c8c032549fe54bd0dbf4354423`.

The candidate reduced the direct voxelization-layer median from 270.937 to 4.782 ms on W1 and from
291.729 to 5.040 ms on W2. However, the projected no-provenance-hash end-to-end medians remained
98.910 and 106.112 ms, respectively, and did not demonstrate 20 Hz. More importantly, 30-run W2
repeatability against the deterministic reference failed the existing detector yardstick: the
box-axis-yaw pass fraction was 0.989834, below 0.99, with 19 tolerance failures among 1,869 matched
high-confidence detections. Saturated voxels retained different point subsets in 75.52% of the
20-sample comparisons.

Therefore the experimental fast voxelizer is **not recommended for production integration**. This
is diagnostic evidence only, not a canonical M3 result, and M3 remains stopped for review.
