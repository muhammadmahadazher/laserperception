# Benchmarks

No LaserPerception benchmark has been measured yet. Values remain **Pending measurement** until a
reproducible run produces them.

| Experiment | Source | Target | Backbone | Input | Normalization | Voxel size | mIoU | Per-class IoU | Peak VRAM | Wall-clock | Commit SHA | Config |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exp001 | SemanticKITTI | DALES | Not implemented | xyz | min_xyz | 0.30 m | Pending measurement | Pending measurement | Pending measurement | Pending measurement | Pending measurement | `configs/experiments/exp001_semkitti_to_dales.yaml` |

## Result acceptance criteria

A measured row must include the immutable commit SHA and config, dataset releases and splits,
ontology/preprocessing versions, per-class confusion counts and ignore handling, environment and
hardware, seeds and run count, wall-clock boundaries, memory-measurement method, and all deviations.

Missing measurements are written as `Pending measurement`, never as illustrative numbers.
