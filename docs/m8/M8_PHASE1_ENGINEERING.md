# M8 Phase 1 engineering integration

> **ENGINEERING TIER ONLY. NO KITTI COMPARATIVE RESULT EXISTS. NO SCIENTIFIC CLAIM IS
> AUTHORIZED.**

PointPillars remains the frozen historical baseline. This integration adds one prospective modern
detector stack without changing or rerunning M1–M7 evidence. DSVT-Pillar was selected strictly by
the feasibility record in [M8_CANDIDATE_DECISION.md](M8_CANDIDATE_DECISION.md).

## External source and environment

The lightweight wheel does not install or vendor OpenPCDet, DSVT, Torch, CUDA, spconv,
torch-scatter, ONNX, or TensorRT. The manual M8 environment is separate and fail-closed.

| Component | Frozen engineering identity |
|---|---|
| DSVT | `Haiyang-W/DSVT@8cfc2a6f23eed0b10aabcdc4768c60b184357061` (`0.6.0+8cfc2a6`) |
| OpenPCDet reference audit | `open-mmlab/OpenPCDet@233f849829b6ac19afb8af8837a0246890908755`, setup-derived `0.6.0+233f849` (not installed) |
| Python / OS | 3.10.12 / Ubuntu 22.04.5 under WSL2 |
| Torch / CUDA | `2.1.0+cu118` / 11.8 |
| spconv / cumm | 2.3.8 / 0.7.11 |
| torch-scatter | `2.1.2+pt21cu118` |
| NumPy / Numba | 1.23.5 / 0.57.1 (isolated compatibility environment) |
| TensorRT | 8.6.1 |
| GPU / driver | NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB / 610.88 |

The DSVT fork carries Apache-2.0. Its official checkpoint remains external, is not owned or
relicensed by LaserPerception, and is not committed. See `THIRD_PARTY_NOTICES.md`.

## Five-feature input contract

M8 consumes a contiguous finite float32 matrix:

```text
[x, y, z, intensity, time_lag]
```

`M8MultiSweepBuilder` uses the accepted source sweep identities, nearest-to-farthest ordering,
float32 transforms, timestamp arithmetic, source-row order, and strict M6 physical range. It
retains the fourth KITTI Raw Velodyne field as reflectance. There is no data-dependent intensity
normalization, synthesis, or scaling; candidate-consumed intensity equals raw intensity byte for
byte. The historical projection is always `m8[:, [0,1,2,4]]`, and the builder fails if that matrix
differs from the unchanged `MultiSweepBuilder` result.

Although the upstream test processor is configured to shuffle point rows, the M8 deployment
adapter prospectively freezes the accepted source-row order and bypasses random inference-time
shuffling. Candidate range masking remains deterministic. This point-order decision was made
without observing KITTI detector outputs.

The input-only full-corpus gate covered 428 H10 and 428 H5 conditions. All 856 projected XYZT
matrices reproduced their frozen M6b SHA256 identities. The candidate range
`[-54,-54,-5,54,54,3]` removed zero points because the common corpus was already frozen by the
stricter M6 physical range `(-50,-50,-5,50,50,3)`.

- Ledger: [`m8_input_projection_ledger.json`](../../benchmarks/m8/diagnostics/m8_input_projection_ledger.json)
- Ledger SHA256: `474e87e34c64d669750d4b6f7a64ac46fc9c5c462693fad79ff7c9547a7f1f7c`
- Gate: H10 428/428 exact; H5 428/428 exact.
- Detector inference during this gate: none.

## Backend, coordinates, classes, and scores

`DsvtBackend.infer(points_xyzit, sample_id=...) -> DetectionFrame` is the only candidate inference
contract. Heavy dependencies are imported only during explicit backend initialization. Startup
verifies the DSVT Git commit, official config hash, checkpoint size/hash, Torch, CUDA, spconv,
torch-scatter, CUDA availability, parameter device, feature schema, and checkpoint identity. There
is no public arbitrary model factory.

The input and output lidar frame is right-handed: X forward, Y left, Z up. OpenPCDet native boxes
are geometric-centre `[x,y,z,dx,dy,dz,heading,vx,vy]`; `dx,dy,dz` map directly to
LaserPerception length/width/height, and heading maps directly to yaw counter-clockwise from +X
about +Z. Analytic 0 and 90-degree box fixtures protect axis and size ordering.

The complete official nuScenes class table remains available in `DetectionFrame`. The future M6
primary mapping is frozen prospectively as `car -> car` and `pedestrian -> pedestrian`. All other
classes are ignored for that primary evaluation. In particular, truck and bus are not merged into
car.

Native scores are preserved. The selected head's dense threshold is 0.0 and its final configured
threshold is 0.1, both at or below the future external scientific threshold 0.25. Therefore no
prediction above 0.25 is discarded internally. No score threshold was tuned on KITTI.

## Official-domain engineering smoke

The official checkpoint loaded 449/449 parameters. nuScenes v1.0-mini validation index 0 produced
a source-format `(207998, 5)` input and 132 finite `(132, 9)` boxes with finite scores and class IDs
in `[1,10]`. This confirms structure and environment only; LaserPerception did not re-benchmark
source-domain accuracy or claim reproduction of upstream leaderboard values.

Ten identical inferences were not byte-exact. Label tensors were exact 10/10, while box tensors
had ten unique byte hashes with maximum absolute difference `0.0110605657`, and score tensors had
ten unique hashes with maximum absolute difference `0.0001267940`. Consequently the converted
DetectionFrames also had ten unique hashes. `pred_boxes` was the first differing tensor. This
nondeterminism is retained for owner review; no future scientific repeatability rule was relaxed.

On the RTX 4060 Laptop session:

- peak CUDA allocated: 1,454,273,536 bytes;
- peak CUDA reserved: 1,904,214,016 bytes;
- host maximum RSS: 1,492,934,656 bytes;
- ten-run wall-clock engineering context: mean 133.394 ms, median 133.030 ms, min 130.657 ms,
  max 140.571 ms.

These timings are not performance claims and are not compared with PointPillars. The full record is
[`dsvt_source_domain_smoke.json`](../../benchmarks/m8/diagnostics/dsvt_source_domain_smoke.json).

## TensorRT deployment audit

The official DSVT repository's checked-in `tools/deploy.py` exports only the DSVT 3D transformer
backbone boundary. It is hard-coded for the Waymo pillar config (`d_model=192`, set size 36), while
the selected official nuScenes config uses `d_model=128`, set size 90. LaserPerception adapted only
the shape/config binding—not model mathematics—to smoke the same official boundary with the
selected checkpoint:

```text
after DynPillarVFE and DSVT InputLayer
  -> four DSVT transformer blocks
  -> boundary output
```

ONNX opset 14 export/check passed; TensorRT 8.6.1 parsed, built, and deserialized an external FP16
engine for the observed `(3687,128)` source-domain pillar shape. The ONNX and engine were not
committed. This establishes a concrete selected-config partial deployment route, not detector
parity or end-to-end deployment evidence.

The separate linked `DSVT-AI-TRT` repository is end-to-end-oriented but documents a different
one-sweep retraining path, so it was audited rather than treated as a drop-in runtime for this
official checkpoint. Its audited commit was `15b31c39b1727746507dddbec562ae23bab5dbab`.

TensorRT warnings are retained: INT64-to-INT32 casting/clamping, FP16 LayerNorm overflow risk,
eight FP32 infinity weights converted to FP16 infinity, and 93 subnormal FP16 weights. A future
owner-approved deployment protocol must address fidelity; this engineering smoke makes no parity
claim. The evidence is
[`dsvt_deployment_smoke.json`](../../benchmarks/m8/diagnostics/dsvt_deployment_smoke.json).

The upstream `13.8 ms` figure is an RTX 3090 partial-boundary figure that excludes InputLayer. The
upstream paper/repository separately reports approximately `37 ms / 27 Hz` for fully deployed
DSVT-Pillar. Neither number is LaserPerception latency, and no deployment latency was measured in
this task.

## Owner-gated scientific tier — drafts only

### P1-S1 — DRAFT, SCIENTIFIC MEASUREMENT NOT AUTHORIZED

A future owner-reviewed protocol may preregister V2 H10/H5 inference at score >= 0.25 and oriented
BEV IoU >= 0.50 on the frozen 66 Car and 396 Pedestrian eligible GT. It must describe the result as
a **frozen detector-stack comparison under the same cross-domain corpus and evaluation protocol**;
the stacks differ in architecture, training recipe, checkpoint, framework, feature contract, and
postprocessing. Frozen PointPillars results are reused, not rerun.

### P1-S2 — DRAFT, SCIENTIFIC MEASUREMENT NOT AUTHORIZED

Only after V2 A/E results are frozen may a separate protocol define V2-specific paired partitions
and normalized recovery gates for M7 B/C/D/F replication. This checkpoint does consume encoded
timestamp as a point feature, so lag replication is technically applicable. Every future V2
condition must first prove its `[x,y,z,time]` projection matches the frozen M7 XYZT hash. No
B/C/D/F inference occurred in this engineering task.
