# Benchmarks

No LaserPerception detection benchmark has been measured yet. Values remain **Pending measurement**
until a reproducible run succeeds on the stated hardware and produces a sanitized record.

## M1 — PointPillars FP32

| Model | Dataset | Device | Precision | Model/GPU latency | End-to-end latency | FPS | Peak VRAM | Commit | Manifest |
|---|---|---|---|---|---|---|---|---|---|
| PointPillars (official pretrained asset to be pinned) | nuScenes v1.0-mini | RTX 4060 Laptop GPU | FP32 | Pending measurement | Pending measurement | Pending measurement | Pending measurement | Pending measurement | `configs/detection/m1_pointpillars_nuscenes.yaml` (planned) |

Model/GPU timing will use CUDA events and synchronization. End-to-end sample timing will include
the documented preprocessing, inference, and postprocessing path, while excluding model/checkpoint
initialization, downloads, visualization, and image writes. Warmup and measured counts, complete
software/hardware metadata, checkpoint checksum, sample selection, threshold, and timing boundaries
must accompany any accepted result.

## Parked Experiment 001 — semantic transfer

| Experiment | Source | Target | Backbone | Input | Normalization | Voxel size | mIoU | Per-class IoU | Peak VRAM | Wall-clock | Commit SHA | Config |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exp001 | SemanticKITTI | DALES | Not implemented | xyz | min_xyz | 0.30 m | Pending measurement | Pending measurement | Pending measurement | Pending measurement | Pending measurement | `configs/experiments/exp001_semkitti_to_dales.yaml` |

The implemented data pipeline uses official SemanticKITTI splits and deterministic DALES 50 m ×
50 m reference cells. No real dataset audit or semantic-segmentation model benchmark has been run.

## Result acceptance criteria

A measured row must come from an actual run and include the immutable commit SHA, config/manifest,
dataset release and split, exact framework versions, official checkpoint source and SHA256, sample
selection, precision, threshold, environment, hardware, warmup/run counts, complete latency
statistics, timing boundaries, and memory-measurement method. Private absolute paths and secrets
must be removed. Missing values use `Pending measurement`; illustrative or estimated values are not
accepted.
