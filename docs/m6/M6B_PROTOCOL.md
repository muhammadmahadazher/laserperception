# M6b preregistered KITTI Raw cross-domain protocol

Status: **frozen before detector inference**. At this document's commit, no KITTI detector
prediction, score, box, preview, threshold experiment, or screenshot had been generated.

M6b asks what happens when LaserPerception's unchanged nuScenes-trained PointPillars deployment
detector is applied offline to the verified KITTI Raw model-ready inputs established by M6a. It is
a cross-domain characterization, not the official KITTI benchmark, not ROS replay, and not model
development. Poor detector quality is a finding rather than a protocol failure.

The machine-readable freeze is [`configs/m6/kitti_m6b.yaml`](../../configs/m6/kitti_m6b.yaml).
The tracked compact GT/input audit ledger is
[`benchmarks/m6b/diagnostics/pre_inference_input_ledger.json`](../../benchmarks/m6b/diagnostics/pre_inference_input_ledger.json).
The complete pre-inference payload was frozen before inference and is preserved after measurement
as immutable external evidence; the non-normative packaging identity is recorded below.

## Prospective pre-inference portability correction

The first GPU-environment attempt at commit `bb279cf` stopped on the model-ready SHA check before
model-ready preparation, voxelization, raw network execution, postprocessing, or any detector
prediction. The detector runtime had initialized, but no KITTI score or predicted box was produced;
the preregistration barrier therefore remained intact. The complete diagnostic is
[`pre_inference_platform_reproduction.json`](../../benchmarks/m6b/diagnostics/pre_inference_platform_reproduction.json).

Windows reproduced the canonical M6a frame-10 H10 hash exactly. Byte-identical source files in WSL2
produced the same shape, sweep order, and time lags, but one float32 transform translation differed
by one ULP across the two NumPy/OpenBLAS environments. That affected 423 X-coordinate values in one
sweep, with a maximum absolute difference of `1.4901161193847656e-08 m`. Substituting the already
canonical float32 transform reproduced the exact M6a hash in WSL2.

Prospectively, before the first detector prediction, the input ledger therefore freezes each
canonical float32 sweep transform. The unchanged `MultiSweepBuilder` consumes those matrices. H10
and H5 frame sets, histories, points, time lags, pillar records, and model-ready hash commitments do
not change. Every one of the 428 paired H10/H5 inputs must reproduce its frozen hash in WSL2 before
detector inference may begin.

## Frozen detector and input contracts

The checkpoint, ONNX, TensorRT FP16 engine, MMDeploy postprocess, exact-fast voxelizer, voxel
geometry, `max_voxels=40000`, NMS, class scores, and deployment threshold `score >= 0.25` remain
unchanged. M6b uses `exact_fast` with full provenance on `cuda:0`. The two authorized conditions
are:

- H10: current acquisition plus ten historical acquisitions;
- H5: the same current frame plus five historical acquisitions.

H10 versus H5 is a compound temporal-and-density ablation. It changes temporal span, time-lag
values, point count, beam-return density, occupied pillars, and cap pressure simultaneously. M6b
does not attribute any output change to time lag alone.

## Dataset selection frozen without predictions

The canonical `2011_09_26_drive_0001` remains mandatory. The one additional drive was selected
from every official City/Residential drive with validated Raw tracklets by descending valid
Pedestrian labelled-pose count, then descending Pedestrian-tracklet count, descending labelled
frame count, and finally lexicographic drive ID. No detector output informed the selection.

| Drive | Category | Valid Ped poses | Ped tracks | Labelled frames |
|---|---:|---:|---:|---:|
| 0001 | City | 0 | 0 | 73 |
| 0002 | City | 0 | 0 | 23 |
| 0005 | City | 5 | 2 | 91 |
| 0011 | City | 22 | 1 | 129 |
| 0013 | City | 0 | 0 | 52 |
| 0017 | City | 0 | 0 | 9 |
| 0018 | City | 0 | 0 | 167 |
| 0019 | Residential | 71 | 3 | 231 |
| 0020 | Residential | 0 | 0 | 20 |
| 0022 | Residential | 21 | 2 | 367 |
| 0023 | Residential | 0 | 1 | 308 |
| 0035 | Residential | 2 | 2 | 102 |
| 0046 | Residential | 0 | 0 | 31 |
| 0048 | City | 0 | 0 | 21 |
| 0051 | City | 7 | 3 | 252 |
| 0056 | City | 8 | 2 | 176 |
| 0057 | City | 0 | 0 | 91 |
| 0059 | City | 23 | 5 | 188 |
| 0060 | City | 16 | 1 | 38 |
| 0061 | Residential | 0 | 0 | 368 |
| 0064 | Residential | 0 | 0 | 219 |
| 0084 | City | 25 | 3 | 271 |
| **0091** | **City** | **398** | **42** | **268** |

Drive 0091 therefore wins before tie-breaking. Dataset archives remain external. The exact archive
and tracklet hashes are in the config.

## Exact paired frame set and H5 oracle

All uninterrupted frames with full H10 history are evaluated:

- drive 0001 indices 10–107 (98 frames);
- drive 0091 indices 10–339 (330 frames).

The ordered 428-frame list has SHA256
`76bd5f7adac3d892ad2fb00cb9cf5f4f73dd475682ee011b7ea9524060c46c95`. H10 and H5 use this
same list. There are 396 eligible Pedestrian poses on these frames after the frozen validity and
reference-FOV rules, exceeding the preregistered LOW-N floor of 50.

The original full pre-inference ledger recorded every frame's history IDs, source-row counts,
time-lag set, model-ready SHA256, point count, pillar counts, and first-touch provenance. Its frozen
internal audit digest is
`2c41c9b21f9d30016ca22c46f75650e753cfe2a9b825077e715d65803610b480`. The ordered H10 and H5
model-ready hash commitments are respectively
`63f4bd20d33a62948dc9a2593b57509380848cb48980827d0b0352c47fa37469` and
`e5f43d6511d96f6db232c880f94b5464ab5d217f5e5bfdf34bd1626ab8ac7f89`. Ten independent H5
reconstructions were exact for each of the five frozen sentinels.

## Annotation region

Official KITTI Raw documentation limits tracklets to dynamic objects in the reference-camera field
of view, so M6b never reports full-360-degree precision. The reference is rectified camera 0,
using `R_rect_00`, `P_rect_00`, `S_rect_00`, and the official Velodyne-to-camera transform. Image
size is 1242 by 375 pixels.

For each predicted box, all eight corners are transformed to rectified camera 0. An edge crossing
the fixed `z=1e-6 m` near plane contributes its exact intersection point; points at or beyond that
plane are projected. A prediction is in the annotated region when its projected axis-aligned extent
intersects the half-open image rectangle `[0,1242) x [0,375)`. Wholly outside predictions are not
false positives and are counted separately as `outside_annotation_fov_predictions`.

## Ground truth and taxonomy

A Raw pose is eligible only when `state=LABELED (2)`, occlusion is visible or partly occluded
(`0/1`), and truncation is in-image or truncated (`0/1`). The selected XMLs contain no explicit
DontCare regions, so none are synthesized.

The source is an upright KITTI Velodyne box with dimensions `(h,w,l)`, bottom/contact-centre
translation, and `rz` about +Z. Its geometric centre is `(x,y,z+h/2)`, then transformed by:

```text
A = [[0,-1,0],
     [1, 0,0],
     [0, 0,1]]
```

Output size remains `(l,w,h)` and yaw is `normalize(rz + pi/2)`. Analytic fail-first tests cover
zero, +/-pi/2, arbitrary yaw, translation, centre shift, dimension order, basis inversion, and
specifically reject unchanged-yaw and `rz-pi/2` implementations.

Quantitative targets are `Car -> car` and `Pedestrian -> pedestrian`. Real `Van` boxes can explain
unmatched car predictions and real `Person (sitting)` boxes can explain unmatched pedestrian
predictions. This is **benchmark-inspired neighbour ignore semantics**, not an official KITTI Raw
evaluator. `Truck`, `Cyclist`, `Tram`, and `Misc` are unmapped; Cyclist is not a Pedestrian ignore.

## Matching and metrics

The primary geometry is oriented BEV IoU at 0.50, with frozen sensitivity results at 0.30 and 0.70.
Within each frame/class, predictions are processed by descending score then stable detection index.
Each takes the unmatched target with maximum IoU, breaking exact ties by lower GT track ID. After
target matching, an unmatched prediction at or above the same IoU against a real neighbour-ignore
box is explained and excluded from TP/FP. Any remaining inside-FOV prediction is FP.

At score 0.25, each class/condition reports GT and prediction denominators, TP/FP/FN, explained
ignores, outside-FOV predictions, precision, recall, and F1. Clean mapped classes also receive an
all-points score-ranked PR curve and monotonic precision-envelope area; the curve is descriptive
and cannot select a new threshold. Its frozen population is every survivor of the unchanged
MMDetection3D/MMDeploy postprocess, whose pinned `score_thr=0.05` is below the 0.25 deployment
operating point. It is therefore the full ranked curve of postprocessed detector outputs, not a
claim to rank raw anchors below the upstream postprocess floor. Matched pairs report BEV IoU,
centre error, wrapped yaw error,
and score. Independently tested 3D IoU is absent, so M6b will not report it.

Forward-distance strata use model-frame GT-centre Y in `[0,20)`, `[20,35)`, and `[35,50]` metres.
Occlusion/truncation strata require denominator 10. Tracklet IDs summarize per-track labelled and
eligible frames, detection continuity, longest hit/miss runs, matched-score median, and range span;
they do not connect predictions or implement tracking.

## Pillar-cap and mechanism diagnostics

Every H10/H5 frame reports in-range points, candidate, retained, and discarded pillars, overflow
count/fraction, 12 equal 30-degree sectors, four Cartesian quadrants, and 0–20/20–35/35–50 m radial
bins. Each pillar is attributed to the earliest acquisition that touches it (`0=current` through
`10=oldest H10`). Eligible target footprints are intersected against actual 0.25 m pillar cell area,
not centre points alone.

The spatial homogeneity test is a Pearson 2x12 candidate-versus-discarded sector test at alpha
0.01, df 11, with critical statistic 24.7249703113 and all expected cells at least five. Calling
truncation spatially non-uniform additionally requires largest sector drop rate at least twice the
smallest nonzero rate and the aggregate highest-drop sector to be each frame's highest in more than
half of overflow frames. Otherwise only descriptive statistics are reported. These diagnostics can
support association, not detector-quality causation.

## Frozen hypotheses and repeatability

- H1: H10/H5 differences are associated with the compound history shift, not isolated time lag.
- H2: output differences may associate with pillar-cap overflow; no causal claim follows.
- H3: KITTI HDL-64E density/beam geometry is measured context, not independently manipulated.
- H4: mounting, environment, and dataset distribution are contextual, not causally isolated.

Before the full run, H10 inference is repeated ten times on these input/GT-only sentinels:

1. `0001/10` — canonical first full-history frame;
2. `0001/83` — canonical lower-median candidate-pillar frame;
3. `0001/11` — highest-overflow full-history M6a frozen sentinel;
4. `0001/15` — non-overflow frame closest to 40,000 candidate pillars;
5. `0091/10` — lowest-index eligible Pedestrian frame on the selected drive.

Raw `cls_score`, `bbox_pred`, `dir_cls_pred`, and final DetectionFrame content must be exact across
all ten runs. A failure stops characterization.

## Visualization and interpretation freeze

All five sentinels are shown regardless of quality. Post-hoc Car best/median/worst examples use
frame recall at score 0.25/IoU 0.50 only among frames with eligible Car GT, with the deterministic
tie-breaks in config. Additional selections are highest overflow, its closest non-overflow input
comparison, and the frame with largest absolute H10/H5 Car-recall difference. They are explicitly
metric-selected, never hand-picked or called representative.

Results are named **LaserPerception M6b cross-domain Raw-tracklet metrics**, never KITTI benchmark
AP. No result authorizes threshold tuning, detector changes, optimization, ROS replay, M6c, or M5.

## Post-measurement packaging note

This note was added after characterization and is not part of the prospective protocol. The final
tracked ledger is a compact audit artifact containing all 856 condition identities; its SHA256 is
`2413224808b0140856a0e00f884c53bc9b49ec6b545172f5e7fa57d00802dc15`. The immutable full ledger
is external generated evidence under logical name `pre_inference_input_ledger_full.json`, size
5,837,452 bytes, SHA256
`e25b3d62113cc7e8c1fcf736caa68b1ab698f965f007c758ff91d3e498ca6caa`. This packaging change did
not alter the preregistration barrier, frozen transforms, model-ready hashes, or detector results.

## Primary sources

- [KITTI Raw data page](https://www.cvlibs.net/datasets/kitti/raw_data.php)
- [KITTI sensor/dataset paper](https://www.cvlibs.net/publications/Geiger2013IJRR.pdf)
- [KITTI object evaluation page](https://www.cvlibs.net/datasets/kitti/eval_object.php)
