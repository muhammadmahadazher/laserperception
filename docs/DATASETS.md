# Datasets and data governance

LaserPerception does not redistribute SemanticKITTI, KITTI, DALES, or future datasets. Data must
remain outside Git and be addressed through configuration or environment variables.

## SemanticKITTI source

Obtain KITTI odometry scans and SemanticKITTI labels from their official sources. Users are
responsible for the current terms and citations of both projects.

- Format and terms: <https://semantic-kitti.org/dataset.html>
- Official development kit: <https://github.com/PRBonn/semantic-kitti-api>
- Suggested variable: `LASERPERCEPTION_SEMANTICKITTI_ROOT`

The loader preserves remission; Experiment 001 excludes it from model features.

## DALES target

Obtain DALES through its official distribution path and review its current terms. Apache-2.0 does
not apply to DALES.

- Paper and class specification: <https://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Varney_DALES_A_Large-Scale_Aerial_LiDAR_Data_Set_for_Semantic_Segmentation_CVPRW_2020_paper.html>
- Suggested variable: `LASERPERCEPTION_DALES_ROOT`

## Directory adapters

### SemanticKITTI hierarchy and splits

`SemanticKITTIDataset` expects the official `sequences/<sequence>/velodyne/*.bin` hierarchy and
paired `labels/*.label` files for labelled splits. Its pinned official manifest comes from
`semantic-kitti.yaml` at commit `a9c749e8124b2243b6eef1b8bcf971a9f1173a2d`:

- train: `00`–`07`, `09`, `10`;
- validation (`valid`): `08`; and
- test: `11`–`21`.

An optional sequence list is an explicit experiment subset and must stay inside the selected official
split. It is not a replacement split. Train/validation labels are required by default; official test
labels are optional. Discovery validates missing sequences, orphan labels, missing labels, and numeric
frame ordering before a scan is loaded.

### DALES distribution assumptions

The DALES paper specifies forty LAS 1.2 tiles with 29 training and 11 test tiles, but it does not
publish a stable filename manifest in the paper. `DalesDataset` therefore requires an explicit
`train` or `test` directory and discovers `.las`/`.laz` files deterministically. The expected 29/11
counts are recorded as provenance but not enforced, so incomplete local subsets can be audited.

The dataset path supports either `<root>/<split>` or `<root>/dales_las/<split>`. A directory that is
itself named `train` or `test` is also accepted. No validation split is fabricated.

## Memory-conscious DALES patches

The dataset path uses the official laspy `LasReader.chunk_iterator()` API instead of `laspy.read()`.
It retains only scaled X, Y, Z, and classification. Optional dimensions may be listed in metadata,
but their arrays are not copied into patch `PointCloud` objects. The simple `load_las()` interchange
API remains unchanged.

One streamed pass partitions each tile into a deterministic grid. The configured 50 m × 50 m size
is a configurable Experiment 001 audit/reference choice, not a DALES requirement or claim of CVGC
equivalence. Grid origin is the LAS header minimum X/Y. Cells use half-open intervals
`[xmin, xmax)` and `[ymin, ymax)`, so boundary points occur in exactly one cell. Patches do not
overlap. Empty cells are skipped and counted. Non-finite points are counted but cannot be assigned.

The adapter crops before any normalization and never maps ontology labels automatically:

```text
LOAD TILE CHUNKS -> CROP/PATCH -> EXPLICIT min_xyz -> EXPLICIT ONTOLOGY MAP -> future voxelization
```

Patch assignment uses float64 scaled coordinates; the resulting `PointCloud` follows the project
float32 canonical representation. One tile is partitioned at a time, and only required dimensions
are retained. The current partition result holds one tile's non-empty raw patches in memory; it does
not create an all-dimension tile cloud or normalize/copy the whole tile before cropping.

## Dataset audit

Run small CPU-only audits before selecting a training subset:

```bash
python -m laserperception.audit semantickitti --split train --sequences 00 --max-samples 5
python -m laserperception.audit dales --split test --max-tiles 1 \
  --normalization min_xyz --json audit-reports/dales-test.json
```

Use `--root` or the documented environment variables. Reports include counts, coordinate ranges,
label histograms, shared-ontology coverage, ignored fraction, adapter version, timestamp, and Git
commit when available. DALES reports also include patch and empty-cell counts. Absolute dataset
roots are omitted. `audit-reports/` is Git-ignored. No real-dataset report is committed by default.

## Mapping provenance

SemanticKITTI IDs come from official `semantic-kitti.yaml`. The DALES paper defines unknown `0`,
ground `1`, vegetation `2`, cars `3`, trucks `4`, power lines `5`, fences `6`, poles `7`, and
buildings `8`. Grouping those verified IDs into six classes is LaserPerception's Experiment 001
policy and is tested in `ontology/mappings.py`.

Synthetic tests generate temporary data and need no public dataset download.
