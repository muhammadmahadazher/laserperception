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
detections. Generated full evidence stays in the external M3 cache until a sanitized exact-commit
diagnostic record is reviewed.

## Preregistered performance protocol

- hardware: RTX 4060 Laptop GPU;
- runtime: TensorRT 8.6.1 FP16, frozen engine SHA256
  `a005f75852097cd9b193750560b214cc3d5237ae9b6c106c7fca3d4fc348714b`;
- input: model-ready nuScenes v1.0-mini `mini_val` index 0 repeated;
- cadence: synthetic 20 Hz stress replay, not native annotated-keyframe timing;
- warmups/measured: 20/200;
- input QoS: volatile best-effort keep-last depth 1;
- output QoS: volatile reliable keep-last depth 5; and
- optional markers disabled for the measured canonical Detection3DArray interface.

Boundary A starts at detector callback entry and ends immediately after the detections publisher's
`publish()` call. Boundary B is same-host ROS loopback from the replay's preserved ROS stamp to a
separate sink's reception time. Boundary B is not sensor-to-actuator latency.

## M3A rate-gate disposition

The first diagnostic completed all messages without a final backlog or processing-induced loss,
but failed the preregistered callback-median and approximately-20-Hz rate requirements. Therefore:

- there is no accepted/canonical M3 result under `benchmarks/m3/results/`;
- M3A status is `FAIL — review required`;
- M3B is indicated but not authorized;
- no postprocessing or other runtime optimization was attempted; and
- no bottleneck cause is claimed without a separate measured M3 pipeline profile.

An exact-commit sanitized failed diagnostic will be retained under `benchmarks/m3/diagnostics/` for
review. Failed diagnostic values must not be marketed as accepted M3 performance.
