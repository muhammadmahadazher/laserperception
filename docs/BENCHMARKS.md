# Benchmarks

## Detection workload sweep-history qualification

The pinned nuScenes test pipeline requests 10 historical sweeps in addition to the current
keyframe. The actual `mini_val` split has 81 samples: indices 0 and 40 are scene starts with zero
available history, while the other 79 contain all 10 requested historical sweeps plus the current
keyframe. The dataset and configured multi-sweep pipeline are therefore not broken.

Workloads used by the existing evidence differ:

- M1 performance repeatedly measures scene-start index 0, with zero historical sweeps.
- M2 canonical performance repeatedly measures the same scene-start zero-history index 0.
- M2 parity v2 covers 20 frozen samples: 19 full-history samples and scene-start index 0.
- M3 PointCloud2 round-trip correctness uses those same 20 frozen samples.
- The failed M3A 20 Hz stress replay repeatedly uses scene-start zero-history index 0.

These qualifications preserve all existing results and clarify their input weight; they do not
change any timing, parity, engine, model, threshold, precision, or sample selection.

LaserPerception M1 has a real, sanitized FP32 result from the stated RTX 4060 Laptop GPU. The
measurement record is
[`benchmarks/m1/results/rtx4060_laptop_fp32.json`](../benchmarks/m1/results/rtx4060_laptop_fp32.json).
It records benchmark commit `f435f03b0e8cfdaf1b1af5b17d5c4d1e105adf86`, UTC timestamp
`2026-08-10T09:36:11.151600+00:00`, and the complete software, hardware, sample, asset, timing, and
memory provenance.

## M2 — repaired canonical RTX 4060 measurement

The canonical measured record is
[`benchmarks/m2/results/rtx4060_pytorch_fp32_vs_tensorrt_fp16.json`](../benchmarks/m2/results/rtx4060_pytorch_fp32_vs_tensorrt_fp16.json).
It was measured at exact implementation commit
`3f240d60569b53a2e4445d34b0905a807cf54879` on the NVIDIA GeForce RTX 4060 Laptop GPU. The
parity reference remains MMDeploy-rewritten PyTorch FP32; the performance baseline is native
MMDetection3D PyTorch FP32; the deployment runtime is TensorRT FP16.

### 1. Component breakdown and bottleneck context

The retained diagnostic at commit `4e12374dec8eecaf0e772b2b5776e0b266fbe09e` first established
that the native GPU path was healthy and that rewritten eager PyTorch was not a representative
performance denominator. Its 20-warmup/30-measurement component profile was:

| Diagnostic component | Median | P95 |
|---|---:|---:|
| Prepare | 5.567 ms | 6.238 ms |
| Voxelize | 8.356 ms | 9.039 ms |
| Native PyTorch raw | 20.800 ms | 21.786 ms |
| Rewritten eager PyTorch raw | 1910.464 ms | 2462.358 ms |
| TensorRT raw | 6.917 ms | 7.452 ms |
| Shared MMDeploy postprocessing | 24.093 ms | 27.919 ms |
| Bbox-head construction | 0.999 ms | 1.244 ms |
| DetectionFrame conversion | 5.160 ms | 6.043 ms |

These values are diagnostic context, not inputs to the published end-to-end calculation. The shared
MMDeploy postprocessing was the largest individually profiled shared stage, while preparation,
voxelization, and DetectionFrame conversion added further work outside the network. No cached or
custom postprocess was implemented. Independent component medians must not be summed to replace the
direct distributions below.

At the final measurement commit, native and rewritten PyTorch were reconfirmed bit-identical on all
20 frozen samples and 235.2 million raw values. The exact-commit fidelity JSON SHA256 is
`1a5ccbad83ebee06178d2dfdafbb830eafe3adb3eb1f55b0523a4a47a01783ad`.

### 2. Direct end-to-end comparison — headline

| Runtime | Mean | Median | P90 | P95 | Min | Max | Population std. dev. | FPS from median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Native MMDetection3D PyTorch FP32 | 60.007 ms | 59.289 ms | 62.928 ms | 64.945 ms | 55.384 ms | 74.541 ms | 2.701 ms | 16.867 |
| TensorRT FP16 | 45.655 ms | 45.637 ms | 48.210 ms | 48.711 ms | 41.354 ms | 50.457 ms | 2.045 ms | 21.912 |

**Headline: TensorRT FP16 measured a direct 1.299134× end-to-end median speedup over native
PyTorch FP32.** Both paths repeatedly used `mini_val` index 0, a scene-start keyframe with zero
accumulated historical sweeps, and otherwise identical configured preparation, official
voxelization, shared MMDeploy postprocessing, and DetectionFrame conversion. Only the network
runtime differed. Timing used synchronized `time.perf_counter` wall time.

### 3. Network-only comparison — secondary

| Runtime | Mean | Median | P90 | P95 | Min | Max | Population std. dev. | FPS from median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Native MMDetection3D PyTorch FP32 | 19.449 ms | 19.189 ms | 20.230 ms | 20.696 ms | 18.747 ms | 22.634 ms | 0.714 ms | 52.114 |
| TensorRT FP16 | 6.156 ms | 6.126 ms | 6.402 ms | 6.547 ms | 5.810 ms | 7.327 ms | 0.251 ms | 163.250 |

The secondary network-only median speedup is 3.132564×. Both paths received the same precomputed
voxel tensors; CUDA events ended when `cls_score`, `bbox_pred`, and `dir_cls_pred` were available.
The smaller end-to-end gain shows that shared work outside the TensorRT network dominates much of
the deployed latency.

### Exact evidence, memory, and limitations

The full 20-sample parity-v2 suite passed at the same commit with external JSON SHA256
`5e8d49ce3847248f2a1a6d28fd92903d80c118de2cdec7b3c08fcab6c2f58853`. The checkpoint, ONNX, and
engine SHA256 values remained `f19d00a3…`, `61ce22a8…`, and `a005f758…`; no artifact was rebuilt.
All benchmark review flags were empty.

PyTorch peak allocator counters after a reset and one call reported 408,934,400 bytes (0.381 GiB)
allocated and 427,819,008 bytes (0.398 GiB) reserved for the native network, and 413,477,888 bytes
(0.385 GiB) allocated and 427,819,008 bytes (0.398 GiB) reserved for native end to end. TensorRT
records a 31,519,476-byte serialized engine and `ICudaEngine.device_memory_size` of 1,212,340,736
bytes. Comparable process-level GPU memory is `Pending measurement`; these methods are not directly
interchangeable.

The run used batch size one, 10 warmups, and 100 measurements for each runtime and boundary in one
same-session process with isolated native then TensorRT blocks. It is a warm-cache,
repeated-single-sample latency microbenchmark—not cold-storage I/O, whole-dataset sequential
throughput, guaranteed sensor throughput, or production evidence.

### Scientific chronology and retained parity disclosures

- M2 parity v1 failed and remains failed.
- The separately preregistered parity v2 passed on the unchanged engine.
- The first benchmark at `e2f9b6babb541d52beaa0bcd58e841a0a56cc851` was rejected because it
  used MMDeploy-rewritten eager PyTorch as the performance denominator. Its 124.297× network,
  23.101× end-to-end, 2164.527 ms rewritten-network, and 1816.859 ms rewritten-end-to-end values
  are retained only in `benchmarks/m2/diagnostics/rejected_e2f9b6b.json` and are not canonical.
- The diagnostic proved native and rewritten outputs bit-identical, then selected native PyTorch as
  the performance baseline.
- The repaired canonical benchmark above used that approved baseline and direct distributions.

All preregistered per-metric parity-v2 Stage 1 gates passed. Separately, 8/753 (1.06%)
high-confidence matched detections exceeded at least one continuous tolerance; all exceptions remain
in applicable denominators. One index-50 pedestrian match had a 47.626393° box-axis difference whose
causal mechanism was not established. The raw classification tensor had p99 absolute difference
0.056829 and a rare maximum tail of 0.721187. These facts do not create post-hoc gates.

## M1 — PointPillars FP32

| Backend | Model | Dataset | Precision | GPU | Model median | End-to-end median | Model P95 | End-to-end P95 | Model FPS | Peak CUDA memory |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| PyTorch/MMDetection3D 1.4.0 | Official pretrained PointPillars | nuScenes v1.0-mini | FP32 | NVIDIA GeForce RTX 4060 Laptop GPU | 52.896 ms | 55.097 ms | 60.729 ms | 62.568 ms | 18.905 | 0.381 GiB allocated / 0.400 GiB reserved |
| Jetson (conditional M5) | Pending measurement | Pending measurement | Pending measurement | Physical hardware unavailable | Pending measurement | Pending measurement | Pending measurement | Pending measurement | Pending measurement | Pending measurement |

The measured PointPillars asset is
`configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py` with checkpoint SHA256
`f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0`. The model was not trained
by LaserPerception. The run used `mini_val` index 0, sample token
`3e8750f331d7499e9b5123e9eb70f2e2`, explicit FP32, batch size one, 10 warm-up iterations, and 50
measurements per boundary. Warm-ups run before measurements, and every iteration repeats
`mini_val` index 0. This is a scene-start keyframe with zero accumulated historical sweeps. The
end-to-end result is therefore a warm-cache, repeated-single-sample latency microbenchmark—not
cold-storage I/O latency or whole-dataset sequential throughput.

| Boundary | Mean | Median | P90 | P95 | Min | Max | Population std. dev. | FPS from median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Model/GPU | 53.472 ms | 52.896 ms | 59.067 ms | 60.729 ms | 45.788 ms | 64.323 ms | 3.904 ms | 18.905 |
| End to end | 56.272 ms | 55.097 ms | 61.477 ms | 62.568 ms | 51.889 ms | 64.036 ms | 3.044 ms | 18.150 |

Model/GPU timing uses `torch.cuda.Event` around the official MMDetection3D model test step,
including device preprocessing and upstream postprocessing. End-to-end timing uses
`time.perf_counter` with explicit CUDA synchronization from official dataset sample loading and
multi-sweep preprocessing through LaserPerception result conversion. Both exclude environment
setup, downloads, model/checkpoint initialization, visualization, and image writes.

PyTorch peak counters, reset before each measured boundary, reported 409,533,440 bytes (0.381 GiB)
allocated and 429,916,160 bytes (0.400 GiB) reserved. The GPU reported 8,585,216,000 bytes
(7.996 GiB) total memory. These figures are framework memory counters, not board power or total
system consumption.

## Parked Experiment 001 — semantic transfer

| Experiment | Source | Target | Backbone | Input | Normalization | Voxel size | mIoU | Per-class IoU | Peak VRAM | Wall-clock | Commit SHA | Config |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exp001 | SemanticKITTI | DALES | Not implemented | xyz | min_xyz | 0.30 m | Pending measurement | Pending measurement | Pending measurement | Pending measurement | Pending measurement | `configs/experiments/exp001_semkitti_to_dales.yaml` |

The implemented data pipeline uses official SemanticKITTI splits and deterministic DALES 50 m ×
50 m reference cells. No real dataset audit or semantic-segmentation model benchmark has been run.

## Result acceptance criteria

A measured row must come from an actual run and include the immutable commit SHA, config/manifest,
dataset release and split, exact framework versions, official checkpoint source and SHA256, sample
selection, precision, threshold where applicable, environment, hardware, warm-up/run counts,
complete latency statistics, timing boundaries, and memory-measurement method. Private absolute
paths and secrets must be removed. Missing values use `Pending measurement`; illustrative,
upstream-published, or estimated values are not accepted as LaserPerception measurements.
