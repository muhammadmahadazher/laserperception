# M1 benchmark protocol

Status: **Measured**

The sanitized measured result is
[`results/rtx4060_laptop_fp32.json`](results/rtx4060_laptop_fp32.json). It was measured on the
NVIDIA GeForce RTX 4060 Laptop GPU at commit
`f435f03b0e8cfdaf1b1af5b17d5c4d1e105adf86`, using official nuScenes v1.0-mini validation index 0,
explicit FP32, batch size one, 10 warm-up iterations, and 50 measurements per boundary.

Summary: model median 52.896 ms (18.905 FPS), model P95 60.729 ms; end-to-end median 55.097 ms
(18.150 FPS), end-to-end P95 62.568 ms; peak PyTorch CUDA memory 0.381 GiB allocated and 0.400 GiB
reserved. `docs/BENCHMARKS.md` contains the full concise table; the JSON is authoritative.

## Boundaries

The benchmark reports two deliberately separate FP32, batch-size-one boundaries on `cuda:0`:

1. **Model GPU** uses `torch.cuda.Event` around the official MMDetection3D `test_step`. This includes
   device preprocessing, PointPillars inference, and the upstream head's postprocessing. It excludes
   file loading, the official CPU dataset pipeline, and LaserPerception result conversion.
2. **End to end** uses `time.perf_counter` with an explicit CUDA synchronization. It begins before
   MMDetection3D dataset sample loading and its official multi-sweep preprocessing, and ends after
   model execution and conversion to `DetectionFrame`.

Each boundary receives its own warm-up phase. The committed defaults are 10 warm-up iterations and
50 measured iterations, all repeatedly measuring the documented `mini_val` sample index 0. Because
warm-ups run first and the same sample is repeated, the end-to-end boundary is a warm-cache,
repeated-single-sample latency microbenchmark. It is not cold-storage I/O latency and is not
whole-dataset sequential throughput. Reported statistics are count, mean, median, p90, p95,
minimum, maximum, population standard deviation, and FPS derived from median latency. Peak allocated
and reserved CUDA memory come from PyTorch counters reset before each measured boundary.

## Running the benchmark

After completing `docs/DETECTION_ENVIRONMENT.md` and `docs/DETECTION.md`, run inside the pinned WSL
environment:

```bash
export LASERPERCEPTION_NUSCENES_ROOT=~/datasets/nuscenes
python scripts/detection/benchmark_m1.py
```

The script writes only after both boundaries complete. Its default raw destination,
`benchmarks/m1/raw/pointpillars_fp32.json`, is gitignored. The JSON contains no absolute dataset,
checkpoint, repository, or user paths. Review a completed result for scientific and privacy accuracy
before promoting it to `results/`.

Do not compare the two latency boundaries as though they measure the same work, and do not describe
either as training, TensorRT, FP16, INT8, ROS 2, safety, or production performance.
