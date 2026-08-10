# Benchmarks

No LaserPerception benchmark has been measured yet. Values remain **Pending measurement** until a
reproducible run produces them.

| Experiment | Source | Target | Backbone | Input | Normalization | Voxel size | mIoU | Per-class IoU | Peak VRAM | Wall-clock | Commit SHA | Config |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exp001 | SemanticKITTI | DALES | Not implemented | xyz | min_xyz | 0.30 m | Pending measurement | Pending measurement | Pending measurement | Pending measurement | Pending measurement | `configs/experiments/exp001_semkitti_to_dales.yaml` |

## Experiment 001 data-pipeline status

The implemented target policy partitions each DALES tile into configurable, deterministic 50 m ×
50 m reference cells with no overlap, half-open XY boundaries, and skipped-but-counted empty cells.
`min_xyz` is applied per target patch only when explicitly requested. SemanticKITTI uses the official
sequence split manifest and per-scan normalization scope.

| Audit evidence | Status |
|---|---|
| SemanticKITTI source audit | Pending measurement |
| DALES target audit | Pending measurement |
| Shared-ontology ignored fractions | Pending measurement |

No real dataset has been inspected for this table. Adapter tests use only temporary synthetic data.

## Result acceptance criteria

A measured row must include the immutable commit SHA, config, corresponding dataset-audit JSON,
dataset releases and splits,
ontology/preprocessing versions, per-class confusion counts and ignore handling, environment and
hardware, seeds and run count, wall-clock boundaries, memory-measurement method, and all deviations.

Missing measurements are written as `Pending measurement`, never as illustrative numbers.
