# Benchmarks

LaserPerception M1 has a real, sanitized FP32 result from the stated RTX 4060 Laptop GPU. The
measurement record is
[`benchmarks/m1/results/rtx4060_laptop_fp32.json`](../benchmarks/m1/results/rtx4060_laptop_fp32.json).
It records benchmark commit `f435f03b0e8cfdaf1b1af5b17d5c4d1e105adf86`, UTC timestamp
`2026-08-10T09:36:11.151600+00:00`, and the complete software, hardware, sample, asset, timing, and
memory provenance.

## M2 — same-session PyTorch FP32 versus TensorRT FP16

The canonical measured record is
[`benchmarks/m2/results/rtx4060_pytorch_fp32_vs_tensorrt_fp16.json`](../benchmarks/m2/results/rtx4060_pytorch_fp32_vs_tensorrt_fp16.json).
It was generated at `2026-08-11T10:27:34.536114+00:00` from measurement commit
`e2f9b6babb541d52beaa0bcd58e841a0a56cc851`.

Parity v1 remains an authoritative failure at commit
`a9314483e0ba7a191866266080c3147f9d902956`. The separately preregistered parity v2 first passed at
`6258d53c89ff8d9ffe2d13393b636f8c00ba9a6c`. After the benchmark integration fix, the complete
20-sample v2 suite passed again at the exact measurement commit with unchanged ONNX and engine.
The reconfirmed external parity JSON SHA256 is
`fbecf5493a34bf840d2a71b1fe1851010110e15b1aee78b23a26d2ef4f880634`.

### Parity disclosure retained with the benchmark

| Evidence | Result |
|---|---|
| Preregistered per-metric Stage 1 gates | **All passed** |
| Distinct high-confidence matches exceeding at least one continuous tolerance | **8/753 (1.06%)** |
| Maximum box-axis yaw difference | **47.626393°**, index 50 pedestrian; causal mechanism not established |
| Raw `cls_score` absolute-difference tail | p99 0.056829; maximum 0.721187 |

The 8/753 figure is a separate union diagnostic, not a post-hoc acceptance gate. The frozen
acceptance rule is per metric, a detection may exceed multiple tolerances, and all eight exceptions
remain in every applicable metric denominator. The rare 47.63° case is not one of the approximately
180° full-heading reversals. The classification tensor aligned closely for the great majority of
elements but has a rare heavier tail relative to p99; its maximum is diagnostic, not a new gate.

### Frozen protocol and timing boundaries

The run repeatedly measured nuScenes v1.0-mini `mini_val` index 0 at batch size one. Each runtime
and boundary received 10 warmups first and then 100 measurements, with PyTorch-first and
TensorRT-first order alternating by iteration.

- Network: common voxelized tensors through raw network outputs required by shared postprocessing;
  CUDA events with per-iteration end-event synchronization.
- End to end: official sample loading and multi-sweep preparation through official voxelization,
  runtime, shared official postprocessing, and `DetectionFrame`; `time.perf_counter` with CUDA
  synchronization before stopping.

Imports, environment/model/engine initialization, checkpoint loading, downloads, visualization,
and JSON/image writes were excluded.

### Measured same-session result

| Runtime | Precision | Network median | End-to-end median | End-to-end P95 | End-to-end FPS |
|---|---|---:|---:|---:|---:|
| MMDeploy-rewritten PyTorch | FP32 | 2164.527 ms | 1816.859 ms | 2552.475 ms | 0.550 |
| TensorRT | FP16 | 17.414 ms | 78.647 ms | 105.017 ms | 12.715 |

**Headline end-to-end median speedup: 23.101×.** The secondary network-only median speedup is
124.297×.

| Network runtime | Mean | Median | P90 | P95 | Min | Max | Population std. dev. | FPS from median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 | 2142.212 ms | 2164.527 ms | 2763.908 ms | 2890.222 ms | 1330.156 ms | 3224.011 ms | 453.387 ms | 0.462 |
| TensorRT FP16 | 26.619 ms | 17.414 ms | 71.383 ms | 73.785 ms | 6.502 ms | 84.383 ms | 23.906 ms | 57.425 |

| End-to-end runtime | Mean | Median | P90 | P95 | Min | Max | Population std. dev. | FPS from median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 | 1856.050 ms | 1816.859 ms | 2448.022 ms | 2552.475 ms | 1160.047 ms | 2725.626 ms | 388.796 ms | 0.550 |
| TensorRT FP16 | 80.014 ms | 78.647 ms | 96.420 ms | 105.017 ms | 48.957 ms | 120.781 ms | 13.928 ms | 12.715 |

This is a warm-cache repeated-single-sample latency microbenchmark. It is not cold-storage I/O
latency, whole-dataset sequential throughput, guaranteed LiDAR sensor throughput, or a production
real-time guarantee.

### Memory and environment

| Metric | Measured value | Definition |
|---|---:|---|
| PyTorch network peak allocated / reserved | 408,934,400 / 713,031,680 bytes | Torch allocator counters after reset for one network call |
| PyTorch end-to-end peak allocated / reserved | 413,477,888 / 713,031,680 bytes | Torch allocator counters after reset for one end-to-end call |
| TensorRT serialized engine | 31,519,476 bytes | External engine file size |
| TensorRT engine device memory | 1,212,340,736 bytes | `ICudaEngine.device_memory_size` |
| Comparable process-level GPU memory | Pending measurement | No common reliable method was used |

Hardware was the NVIDIA GeForce RTX 4060 Laptop GPU with 8,585,216,000 bytes total memory and
NVIDIA driver 610.88 under WSL2 kernel 6.18.33.2. The measured stack was Python 3.10.12,
PyTorch 2.1.0+cu118/CUDA 11.8, MMDeploy 1.3.1 at
`bc75c9d6c8940aa03d0e1e5b5962bd930478ba77`, MMDetection3D 1.4.0, MMCV 2.1.0, MMEngine 0.10.7,
MMDetection 3.2.0, ONNX 1.14.1, and TensorRT 8.6.1.

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
`mini_val` index 0. The end-to-end result is therefore a warm-cache, repeated-single-sample latency
microbenchmark—not cold-storage I/O latency or whole-dataset sequential throughput.

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
