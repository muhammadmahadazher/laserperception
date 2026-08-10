# M1 benchmark protocol

Status: **Pending measurement**

No LaserPerception latency result has been recorded yet. The licensed nuScenes v1.0-mini files
were not present in the local dataset root during M1 implementation, so the benchmark correctly
stopped before measurement and wrote no result JSON.

## Boundaries

The benchmark reports two deliberately separate FP32, batch-size-one boundaries on `cuda:0`:

1. **Model GPU** uses `torch.cuda.Event` around the official MMDetection3D `test_step`. This includes
   device preprocessing, PointPillars inference, and the upstream head's postprocessing. It excludes
   file loading, the official CPU dataset pipeline, and LaserPerception result conversion.
2. **End to end** uses `time.perf_counter` with an explicit CUDA synchronization. It begins before
   MMDetection3D dataset sample loading and its official multi-sweep preprocessing, and ends after
   model execution and conversion to `DetectionFrame`.

Each boundary receives its own warm-up phase. The committed defaults are 10 warm-up iterations and
50 measured iterations, all repeating the documented `mini_val` sample index 0. Reported statistics
are count, mean, median, p90, p95, minimum, maximum, population standard deviation, and FPS derived
from median latency. Peak allocated and reserved CUDA memory come from PyTorch counters reset before
each measured boundary.

## Running the benchmark

After completing `docs/DETECTION_ENVIRONMENT.md` and `docs/DETECTION.md`, run inside the pinned WSL
environment:

```bash
export LASERPERCEPTION_NUSCENES_ROOT=/root/datasets/nuscenes
python scripts/detection/benchmark_m1.py
```

The script writes only after both boundaries complete. Its default raw destination,
`benchmarks/m1/raw/pointpillars_fp32.json`, is gitignored. The JSON contains no absolute dataset,
checkpoint, repository, or user paths. Review a completed result for scientific and privacy accuracy
before promoting selected numbers into a version-controlled result document.

Do not compare the two latency boundaries as though they measure the same work, and do not describe
either as training, TensorRT, FP16, INT8, ROS2, or production performance.
