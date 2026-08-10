# Benchmarks

No LaserPerception detection benchmark has been measured yet. Values remain **Pending measurement**
until a reproducible run succeeds on the stated hardware and produces a sanitized record.

The real M1 CUDA environment and official PointPillars checkpoint have been initialized successfully
on the target RTX 4060 Laptop GPU. That is an integration check, not a latency measurement or model
quality result.

## M1 — PointPillars FP32

| Backend | Model | Dataset | Precision | GPU | Model median | End-to-end median | P95 | FPS | Peak VRAM |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| PyTorch/MMDetection3D 1.4.0 | Official pretrained PointPillars | nuScenes v1.0-mini | FP32 | RTX 4060 Laptop GPU | Pending measurement | Pending measurement | Pending measurement | Pending measurement | Pending measurement |
| TensorRT (future M2) | PointPillars | nuScenes v1.0-mini | FP16 | RTX 4060 Laptop GPU | Pending measurement | Pending measurement | Pending measurement | Pending measurement | Pending measurement |
| Jetson (conditional M5) | Pending measurement | Pending measurement | Pending measurement | Physical hardware unavailable | Pending measurement | Pending measurement | Pending measurement | Pending measurement | Pending measurement |

The pinned M1 asset is
`configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py` with checkpoint SHA256
`f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0`. The tracked manifest is
`configs/detection/m1_pointpillars_nuscenes.yaml`.

Model/GPU timing uses CUDA events around the official MMDetection3D model test step, including
device preprocessing and upstream postprocessing. End-to-end timing uses a synchronized monotonic
wall clock from official dataset sample loading/multi-sweep preprocessing through LaserPerception
result conversion. Both exclude environment setup, downloads, model/checkpoint initialization,
visualization, and image writes. Defaults are 10 warm-up and 50 measured iterations per boundary.
The complete protocol is in `benchmarks/m1/README.md`.

## Parked Experiment 001 — semantic transfer

| Experiment | Source | Target | Backbone | Input | Normalization | Voxel size | mIoU | Per-class IoU | Peak VRAM | Wall-clock | Commit SHA | Config |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exp001 | SemanticKITTI | DALES | Not implemented | xyz | min_xyz | 0.30 m | Pending measurement | Pending measurement | Pending measurement | Pending measurement | Pending measurement | `configs/experiments/exp001_semkitti_to_dales.yaml` |

The implemented data pipeline uses official SemanticKITTI splits and deterministic DALES 50 m ×
50 m reference cells. No real dataset audit or semantic-segmentation model benchmark has been run.

## Result acceptance criteria

A measured row must come from an actual run and include the immutable commit SHA, config/manifest,
dataset release and split, exact framework versions, official checkpoint source and SHA256, sample
selection, precision, threshold, environment, hardware, warm-up/run counts, complete latency
statistics, timing boundaries, and memory-measurement method. Private absolute paths and secrets
must be removed. Missing values use `Pending measurement`; illustrative, upstream-published, or
estimated values are not accepted as LaserPerception measurements.
