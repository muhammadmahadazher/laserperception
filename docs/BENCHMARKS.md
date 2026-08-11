# Benchmarks

LaserPerception M1 has a real, sanitized FP32 result from the stated RTX 4060 Laptop GPU. The
measurement record is
[`benchmarks/m1/results/rtx4060_laptop_fp32.json`](../benchmarks/m1/results/rtx4060_laptop_fp32.json).
It records benchmark commit `f435f03b0e8cfdaf1b1af5b17d5c4d1e105adf86`, UTC timestamp
`2026-08-10T09:36:11.151600+00:00`, and the complete software, hardware, sample, asset, timing, and
memory provenance.

## M2 — benchmark diagnosis; no canonical result

The run measured at commit e2f9b6babb541d52beaa0bcd58e841a0a56cc851 failed sanity review.
It is preserved only as benchmarks/m2/diagnostics/rejected_e2f9b6b.json with
status rejected_measurement. The reported 124.297× network speedup, 23.101× end-to-end speedup,
2164.527 ms rewritten-PyTorch network median, and 1816.859 ms rewritten-PyTorch end-to-end median
are not accepted M2 results.

The failure does not invalidate parity v2. Parity v2 still establishes that TensorRT FP16
reproduces the MMDeploy-rewritten/exported graph within the preregistered gates. It does not by
itself establish that rewritten eager PyTorch is a representative performance baseline.

### Correct runtime roles

- Parity reference: MMDeploy-rewritten PyTorch FP32.
- Performance baseline: native MMDetection3D PyTorch FP32 on the same precomputed voxel tensors.
- Candidate runtime: TensorRT FP16 with the unchanged ONNX and engine.
- Common work: official preparation, voxelization, existing MMDeploy postprocess, and
  DetectionFrame conversion.

The diagnostic pass first reproduces the original M1 benchmark in the M2 environment, asserts
CUDA device/dtype/shape metadata, compares native and rewritten outputs on all frozen 20 samples,
and profiles every component. A future canonical run requires passing parity and
native-vs-rewritten fidelity evidence from the exact benchmark implementation commit.

### Repaired measurement protocol awaiting review

Any future canonical benchmark will repeatedly measure nuScenes v1.0-mini mini_val index 0 at
batch size one, with 10 warmups and 100 measurements per runtime and boundary. Runtime blocks are
isolated rather than alternated per iteration.

- Network: identical precomputed voxel tensors through raw cls_score, bbox_pred, and dir_cls_pred;
  CUDA-event timing.
- End to end: sample preparation through official voxelization, the selected network, existing
  common MMDeploy postprocess, and DetectionFrame; synchronized wall-clock timing.
- Headline: end-to-end median speedup. Network-only speedup remains secondary.

This remains a warm-cache repeated-single-sample latency microbenchmark, not cold-storage I/O,
whole-dataset sequential throughput, guaranteed sensor throughput, or production evidence. No
repaired canonical benchmark will be run or promoted until the methodology is reviewed.

### Parity-v2 disclosures retained

All preregistered per-metric Stage 1 gates passed. Separately, 8/753 (1.06%) high-confidence
matched detections exceeded at least one continuous tolerance; all exceptions remain in applicable
denominators. One index-50 pedestrian match had a 47.626393° box-axis difference whose causal
mechanism was not established. The raw classification tensor had p99 absolute difference 0.056829
and a rare maximum tail of 0.721187. These facts remain diagnostic and do not create post-hoc gates.

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
