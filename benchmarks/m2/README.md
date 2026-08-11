# M2 same-session benchmark protocol and result

Status: **Measured evidence complete—awaiting PR review/merge.**

The benchmark compares the MMDeploy-rewritten PyTorch FP32 network with the TensorRT FP16 network
in one initialized process. Tracked-result promotion requires full protocol-v2 parity at the exact
current Git commit, the frozen 20-sample set, Stage 1 and overall pass, and the same ONNX and engine
hashes. V1 or malformed parity evidence is rejected.

## Frozen measurement

- Measurement commit: `e2f9b6babb541d52beaa0bcd58e841a0a56cc851`.
- Dataset: nuScenes v1.0-mini, `mini_val` index 0, batch size one.
- Warmups: 10 per runtime and boundary before measurement.
- Measurements: 100 per runtime and boundary.
- Order: alternates PyTorch-first and TensorRT-first by iteration.
- Network boundary: common voxelized tensors to raw tensors consumed by shared postprocessing,
  measured with CUDA events and per-iteration end-event synchronization.
- End-to-end boundary: official sample loading/multi-sweep preparation through official
  voxelization, runtime, shared official postprocessing, and `DetectionFrame`, measured with
  synchronized wall time.
- Headline: end-to-end median speedup; network-only speedup is secondary.

Imports, environment/model/engine initialization, checkpoint loading, downloads, visualization,
and JSON/image writes are excluded.

## Parity binding and disclosures

The exact-commit external parity-v2 JSON passed Stage 1 and overall acceptance on all 20 samples;
its SHA256 is `fbecf5493a34bf840d2a71b1fe1851010110e15b1aee78b23a26d2ef4f880634`.
ONNX SHA256 remained `61ce22a8ca31498675c32576bfb94f0093d31dc95d2762f7254bf915a59ecc16`
and engine SHA256 remained `a005f75852097cd9b193750560b214cc3d5237ae9b6c106c7fca3d4fc348714b`.

| Evidence | Result |
|---|---|
| Preregistered per-metric Stage 1 gates | **All passed** |
| Distinct high-confidence matches exceeding at least one continuous tolerance | **8/753 (1.06%)** |
| Rare box-axis outlier | 47.626393°, index 50 pedestrian; causal mechanism not established |
| Raw `cls_score` absolute-difference tail | p99 0.056829; maximum 0.721187 |

The 8/753 figure is a retained union diagnostic, not a post-hoc gate; all exceptions remain in every
applicable per-metric denominator. The 47.63° axis difference is not one of the approximately 180°
heading reversals and is not labeled harmless, a direction-bit flip, or an NMS survivor swap. The
classification tensor has a rare heavier tail relative to p99; its maximum is diagnostic, not a new
acceptance criterion.

## Canonical measured result

The sanitized tracked record is
`benchmarks/m2/results/rtx4060_pytorch_fp32_vs_tensorrt_fp16.json`.

| Runtime | Precision | Network median | End-to-end median | End-to-end P95 | End-to-end FPS |
|---|---|---:|---:|---:|---:|
| MMDeploy-rewritten PyTorch | FP32 | 2164.527 ms | 1816.859 ms | 2552.475 ms | 0.550 |
| TensorRT | FP16 | 17.414 ms | 78.647 ms | 105.017 ms | 12.715 |

**Headline end-to-end median speedup: 23.101×.** Secondary network-only median speedup: 124.297×.

This is a warm-cache repeated-single-sample latency microbenchmark. It is not cold-storage I/O
latency, whole-dataset sequential throughput, guaranteed LiDAR sensor throughput, or a production
real-time guarantee.

## Memory and result policy

PyTorch network peak allocated/reserved memory was 408,934,400/713,031,680 bytes; end-to-end was
413,477,888/713,031,680 bytes. TensorRT serialized-engine size was 31,519,476 bytes and
`ICudaEngine.device_memory_size` was 1,212,340,736 bytes. These definitions are independent.
Comparable process-level GPU memory remains `Pending measurement`.

The canonical device was the NVIDIA GeForce RTX 4060 Laptop GPU. Raw/debug output remains under
ignored `benchmarks/m2/raw/` or the external M2 cache. ONNX, engines, checkpoints, datasets, and raw
parity dumps are not committed.

See `docs/TENSORRT.md` for permanent v1 failure chronology, complete v2 evidence and individual
outlier details. See `docs/BENCHMARKS.md` for all latency statistics and exact environment.
