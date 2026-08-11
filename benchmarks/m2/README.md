# M2 benchmark protocol and canonical result

Status: **Repaired canonical measurement complete; PR #3 remains draft for final review.**

The canonical record is
[`results/rtx4060_pytorch_fp32_vs_tensorrt_fp16.json`](results/rtx4060_pytorch_fp32_vs_tensorrt_fp16.json).
It was measured on the NVIDIA GeForce RTX 4060 Laptop GPU at exact implementation commit
`3f240d60569b53a2e4445d34b0905a807cf54879`.

## Runtime roles and exact evidence

- Parity reference: MMDeploy-rewritten PyTorch FP32.
- Performance baseline: native MMDetection3D PyTorch FP32.
- Deployment runtime: TensorRT FP16.
- Common path: identical configured nuScenes preparation, official voxelization, shared MMDeploy
  postprocessing, and DetectionFrame conversion.
- Parity v2: PASS, external JSON SHA256
  `5e8d49ce3847248f2a1a6d28fd92903d80c118de2cdec7b3c08fcab6c2f58853`.
- Native/rewrite fidelity: PASS on 20 samples and 235.2 million bit-identical raw values, external
  JSON SHA256 `1a5ccbad83ebee06178d2dfdafbb830eafe3adb3eb1f55b0523a4a47a01783ad`.
- Checkpoint/ONNX/engine SHA256: `f19d00a3…` / `61ce22a8…` / `a005f758…`, unchanged.
- Benchmark review flags: none.

## Diagnostic component context

The retained diagnostic at commit `4e12374dec8eecaf0e772b2b5776e0b266fbe09e` measured component
medians of 5.567 ms prepare, 8.356 ms voxelize, 20.800 ms native raw, 1910.464 ms rewritten raw,
6.917 ms TensorRT raw, 24.093 ms shared MMDeploy postprocessing, 0.999 ms bbox-head construction,
and 5.160 ms DetectionFrame conversion. These 20-warmup/30-measurement values explain the
bottleneck but are not summed to produce the canonical end-to-end result. No cached or custom
postprocess was implemented.

## Direct end-to-end result — headline

| Runtime | Mean | Median | P90 | P95 | Min | Max | Std. dev. | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Native PyTorch FP32 | 60.007 ms | 59.289 ms | 62.928 ms | 64.945 ms | 55.384 ms | 74.541 ms | 2.701 ms | 16.867 |
| TensorRT FP16 | 45.655 ms | 45.637 ms | 48.210 ms | 48.711 ms | 41.354 ms | 50.457 ms | 2.045 ms | 21.912 |

The direct end-to-end median speedup is **1.299134×**. This boundary runs from sample preparation
through voxelization, the selected network, shared MMDeploy postprocessing, and DetectionFrame.

## Network-only result — secondary

| Runtime | Mean | Median | P90 | P95 | Min | Max | Std. dev. | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Native PyTorch FP32 | 19.449 ms | 19.189 ms | 20.230 ms | 20.696 ms | 18.747 ms | 22.634 ms | 0.714 ms | 52.114 |
| TensorRT FP16 | 6.156 ms | 6.126 ms | 6.402 ms | 6.547 ms | 5.810 ms | 7.327 ms | 0.251 ms | 163.250 |

The network-only median speedup is 3.132564×. Identical precomputed voxel tensors enter each
runtime, and CUDA events stop when the three raw head outputs are available.

## Protocol, memory, and limitations

The run used batch size one, `mini_val` index 0, 10 warmups, and 100 measurements per runtime and
boundary. Runtime blocks were isolated native then TensorRT blocks in one same-session process.
Index 0 is a scene-start workload: it contains the current keyframe and zero accumulated historical
sweeps. The canonical M2 performance measurement therefore does not represent the usual full-history
10-sweep-plus-current input. This workload qualification does not alter the measured values.

Native network/end-to-end PyTorch peak allocated memory was 0.381/0.385 GiB; both peak reserved
values were 0.398 GiB. TensorRT records a 31,519,476-byte serialized engine and 1,212,340,736 bytes
of engine device memory. Comparable process-level GPU memory is `Pending measurement` because the
accounting methods differ.

This is a warm-cache repeated-single-sample microbenchmark, not cold-storage I/O, whole-dataset
sequential throughput, a sensor-throughput guarantee, or production evidence. Most TensorRT
end-to-end latency remains in work shared with native PyTorch outside the network.

## Rejected first benchmark and retained parity disclosures

The run at `e2f9b6babb541d52beaa0bcd58e841a0a56cc851` remains only in
`diagnostics/rejected_e2f9b6b.json` with status `rejected_measurement`. Its 124.297× network and
23.101× end-to-end ratios are not publication evidence. Parity v2 remains valid and PASS.

All preregistered per-metric Stage 1 gates passed. Separately, 8/753 (1.06%) high-confidence matches
exceeded at least one continuous tolerance. The 47.626393° index-50 pedestrian axis-yaw case remains
unexplained and fully retained. Raw `cls_score` p99 absolute difference was 0.056829 with a rare
maximum tail of 0.721187.
