# M2 same-session benchmark protocol

Status: **Not run or promoted because the frozen 20-sample FP16 parity gate failed.**

The implemented benchmark compares the MMDeploy-rewritten PyTorch FP32 network with the
TensorRT FP16 network in one initialized process. It refuses tracked-result promotion unless the
full parity JSON passes at the current Git commit, names the exact frozen sample set, and hashes the
same ONNX and engine artifacts.

## Frozen measurement

- Dataset: nuScenes v1.0-mini, `mini_val` index 0, batch size one.
- Warmups: 10 per runtime and boundary before measurement.
- Measurements: 100 per runtime and boundary.
- Order: alternates PyTorch-first and TensorRT-first by iteration.
- Network boundary: common voxelized tensors to raw tensors consumed by shared postprocessing,
  measured with CUDA events and per-iteration synchronization.
- End-to-end boundary: official sample loading/multi-sweep preparation through voxelization,
  runtime, shared postprocessing, and `DetectionFrame`, measured with synchronized wall time.
- Headline: end-to-end median speedup; network-only speedup is secondary.

This is a warm-cache repeated-single-sample latency microbenchmark. It is not cold-storage I/O
latency, whole-dataset sequential throughput, or a sensor-throughput guarantee. PyTorch allocator
counters and TensorRT engine/device-memory requirements are reported separately; they are not a
generic cross-runtime VRAM comparison.

## Result policy

The only promotable path is
`benchmarks/m2/results/rtx4060_pytorch_fp32_vs_tensorrt_fp16.json`, produced on the canonical
NVIDIA GeForce RTX 4060 Laptop GPU after passing parity. No such file exists for the current M2
attempt. Raw/debug output belongs under ignored `benchmarks/m2/raw/` or the external M2 cache.

See `docs/TENSORRT.md` for the exact artifact hashes, parity failure, and architecture-review
disposition.
