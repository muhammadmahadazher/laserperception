# M2 ONNX and TensorRT FP16 deployment

Status: **M2 evidence complete—awaiting PR review/merge; parity v1 failed and parity v2 passed.**

## Parity v1 — measured failure

Gate 0 passed, all 81 `mini_val` samples were profiled, ONNX export/checking passed, and the
official TensorRT FP16 engine built and executed. The unchanged 20-sample parity suite then failed
four locked high-confidence guards at implementation commit
`a9314483e0ba7a191866266080c3147f9d902956`. At that stage M2 remained partial. This authoritative
v1 result remains failed, and no benchmark was run or promoted from v1 evidence.

| Evidence | Result |
|---|---|
| ONNX | SHA256 `61ce22a8ca31498675c32576bfb94f0093d31dc95d2762f7254bf915a59ecc16`, 60,711,828 bytes, opset 11, full checker pass |
| TensorRT engine | SHA256 `a005f75852097cd9b193750560b214cc3d5237ae9b6c106c7fca3d4fc348714b`, 31,519,476 bytes, FP16 requested, INT8 disabled |
| Engine build | 30.419959327 s on the RTX 4060 Laptop GPU; 1,212,340,736 bytes reported device memory |
| Parity JSON | External SHA256 `f6474c365f8fc3d8595db813d1d23258574e4d0448e73473d68bf817f297e534`; status `fail` |

The final run completed all fixed indices. Counts and coverage passed: 883 PyTorch versus 885
TensorRT detections at 0.25; PyTorch-to-TensorRT coverage was 1.0 and the reverse was
0.9986737401. Z and class guards passed. XY, per-dimension size, yaw, and score guards failed:

| Metric | Median | Maximum | Limit | Result |
|---|---:|---:|---:|---|
| XY center displacement | 0.001374 m | 0.978381 m | 0.25 m | Fail |
| Absolute Z difference | 0.000858 m | 0.105435 m | 0.25 m | Pass |
| Per-dimension relative size error | 0.000264 | 0.084264 | 0.05 | Fail |
| Circular yaw difference | 0.035954° | 179.963955° | 5° | Fail |
| Absolute score difference | 0.001046 | 0.168615 | 0.05 | Fail |

All seven threshold-edge crossings were retained:

| Index | Class | PyTorch score | TensorRT score |
|---:|---|---:|---:|
| 25 | bicycle | 0.2504397333 | 0.2494472563 |
| 33 | car | 0.2505155504 | 0.2490817457 |
| 33 | truck | 0.2488429695 | 0.2505458593 |
| 42 | pedestrian | 0.2474979907 | 0.2509127855 |
| 58 | truck | 0.2497554421 | 0.2531217933 |
| 71 | car | 0.2495126724 | 0.2542311251 |
| 80 | car | 0.2562966049 | 0.2440025359 |

The failure is classified as `network_numerical_difference`: both runtimes consume the same
voxel objects/hashes, every sample fits the verified profile and bindings, and both outputs enter
the same official postprocessing function. The threshold-edge crossings are recorded separately;
the count guards still pass.

Bounded diagnostics did not change the protocol. An identical official FP16 rebuild reproduced
the failure despite different TensorRT tactics/engine hashes. A TensorRT FP32 diagnostic removed
the XY/yaw/score failures but still exceeded the dimension guard (maximum 0.066682). Building and
running FP16 with TF32 disabled reproduced the original failed guards. The MMDeploy debugging
timebox is exhausted, so no layer-precision override, model change, threshold relaxation, or
benchmark promotion was attempted.

## Architecture review and parity v2

Architecture review found that v1 applied hard per-box maxima after discontinuous decisions,
including the two-class direction argmax and NMS. The microscopic v1 medians and rare large maxima
motivated a separately versioned protocol; they do not invalidate v1 and do not make it pass. The
PointPillars direction head adds pi to final yaw according to its selected direction class, so a
near-tied two-logit argmax can yield an approximately 180-degree heading change even when the
rectangular box axes remain nearly identical. Such a result is described as **geometrically
axis-equivalent but heading-divergent**, not harmless.

Parity v2 is frozen in `configs/detection/m2_parity_v2.yaml` before its first run. It keeps the
same 20 indices, checkpoint, upstream commits, ONNX, FP16 engine, preprocessing, voxelization,
postprocessing, class-wise matching, 0.50 BEV IoU, score thresholds, count guards, 0.99 coverage,
and numerical tolerances. Its Stage 1 gate changes only the aggregate continuous rule:

- at least 99% of high-confidence matched detections must meet the unchanged 0.25 m XY tolerance;
- at least 99% must meet the unchanged 0.25 m absolute-Z tolerance;
- at least 99% must have all three L/W/H relative errors at or below the unchanged 5% tolerance;
- at least 99% must meet the unchanged 0.05 score tolerance;
- at least 99% must meet the unchanged 5° modulo-pi geometric box-axis yaw tolerance;
- final heading/direction agreement must be at least 99%; and
- class-name mismatches remain forbidden.

Every failed detection stays in its metric denominator and is recorded. A distinct-outlier count
also counts one matched detection once when it violates multiple continuous metrics. Full circular
heading error remains separately visible from modulo-pi axis error.

The low-cost direction diagnostic covers all anchors and the union of the official `nms_pre`
candidate pools selected by either runtime. It reports direction argmax disagreements and PyTorch
and TensorRT winning-logit margins. Raw `cls_score`, `bbox_pred`, and `dir_cls_pred` differences
report count, median, p95, p99, maximum, and mean with shape/dtype consistency.

Stage 1 alone determines v2 PASS/FAIL. Detailed pre-NMS survivor provenance is required only for
targeted Stage 2 diagnosis after a Stage 1 failure; no output is labeled an NMS survivor swap
without competing-candidate, suppression, survivor, and ordering evidence. The first v2 run must
reuse engine SHA256 `a005f75852097cd9b193750560b214cc3d5237ae9b6c106c7fca3d4fc348714b`.
No mixed precision or rebuild was authorized. The first passing v2 run then awaited benchmark review.

### Parity v2 — Stage 1 PASS

The first full v2 run completed at implementation commit
`6258d53c89ff8d9ffe2d13393b636f8c00ba9a6c` using the unchanged ONNX and FP16 engine. The
external `parity_v2.json` has SHA256
`4b29211e52d4e6e14f379d8aebfd7561341c2fd15f625c31d61ed6b86f5dc15c`; its protocol-config
SHA256 is `c26fa7a67289c64c607707141a7d6721a2821d8fccbaa54ee1401c6c03a721bc`.
All 20 frozen indices completed. The engine was not rebuilt and no layer precision changed.

Before benchmarking, the integration fix was frozen as exact measurement commit
`e2f9b6babb541d52beaa0bcd58e841a0a56cc851`. The complete 20-sample v2 suite passed again and
generated a new external parity JSON with SHA256
`fbecf5493a34bf840d2a71b1fe1851010110e15b1aee78b23a26d2ef4f880634`. ONNX SHA256
`61ce22a8ca31498675c32576bfb94f0093d31dc95d2762f7254bf915a59ecc16`, engine SHA256
`a005f75852097cd9b193750560b214cc3d5237ae9b6c106c7fca3d4fc348714b`, and checkpoint SHA256
`f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0` remained unchanged.
Stage 1 passed with the same summary below, and Stage 2 was not required.

Both count guards passed with the same 883 PyTorch and 885 TensorRT exported detections seen in v1.
High-confidence coverage was 750/750 (1.0) PyTorch-to-TensorRT and 753/754
(0.9986737401) TensorRT-to-PyTorch. All 753 high-confidence matches were retained:

| Stage 1 metric | Pass count | Pass fraction | P99 | Maximum | Unchanged tolerance | Result |
|---|---:|---:|---:|---:|---:|---|
| XY center | 749/753 | 0.9946879150 | 0.084380 m | 0.978381 m | 0.25 m | Pass |
| Absolute Z | 753/753 | 1.0 | 0.024166 m | 0.105435 m | 0.25 m | Pass |
| All L/W/H per detection | 750/753 | 0.9960159363 | 0.020420 | 0.084264 | 0.05 | Pass |
| Absolute score | 751/753 | 0.9973439575 | 0.008387 | 0.168615 | 0.05 | Pass |
| Axis yaw modulo pi | 751/753 | 0.9973439575 | 1.433365° | 47.626393° | 5° | Pass |
| Heading/direction agreement | 751/753 | 0.9973439575 | — | 179.963955° full-heading error | 0.99 agreement | Pass |
| Class-name mismatches | 0 | — | — | — | zero | Pass |

Maxima remain diagnostics and every failed detection remains recorded.

| Separate retained-exception diagnostic | Result |
|---|---:|
| High-confidence matches exceeding at least one continuous tolerance | **8/753 (1.06%)** |
| Fraction | **0.0106241700** |

**All preregistered per-metric Stage 1 gates passed. Separately, 8/753 (1.06%) high-confidence
matched detections exceeded at least one continuous tolerance; these exceptions are retained in all
applicable metric denominators.** This does not contradict the preregistered pass because the frozen
acceptance rule is per metric, not a union-of-all-errors gate. One match may exceed several metrics.
No post-hoc union gate was introduced. The seven threshold-edge crossings remain separately
recorded.

One rare high-confidence matched detection had a 47.63° box-axis difference:

| Index | Class | XY difference | Axis-yaw difference | Full-heading difference | Relative L/W/H size difference | Score difference | BEV IoU |
|---:|---|---:|---:|---:|---:|---:|---:|
| 50 | pedestrian | 0.182466 m | 47.626393° | 47.626393° | 0.044505 / 0.027293 / 0.012218 | 0.000269 | 0.508165 |

This is not one of the approximately 180° heading reversals listed below: its box axis itself differs
materially. Its causal mechanism was not established; it is not labeled harmless, a direction-bit
flip, or an NMS survivor swap, and it remains fully retained in the parity statistics.

Two final matched detections were geometrically axis-equivalent but heading-divergent:

| Index | Class | Full heading difference | Axis difference modulo pi |
|---:|---|---:|---:|
| 4 | pedestrian | 179.818776° | 0.181224° |
| 80 | car | 179.963955° | 0.036045° |

The final detection contract has no anchor provenance, so these two final flips are not assigned
specific anchor direction classes or logits. The separately measured raw direction-head populations
are diagnostic and are not claimed as causal links to those final boxes:

| Population | Anchors | Argmax disagreements | Disagreement fraction |
|---|---:|---:|---:|
| All anchors | 11,200,000 | 48,180 | 0.0043017857 |
| Official `nms_pre` union | 20,114 | 35 | 0.0017400815 |

For all-anchor disagreements, PyTorch/TensorRT winning-margin medians were
0.00676051/0.00672913, p95 values were 0.03467448/0.03463020, p99 values were
0.05970200/0.06039749, and maxima were 0.23807795/0.31640625. For decision-relevant
disagreements, the corresponding medians were 0.01447149/0.00955200, p95 values were
0.05222656/0.06074982, p99 values were 0.05962952/0.08803162, and maxima were
0.05977241/0.09899902. The external JSON also retains p90 values and all 35 decision-relevant
disagreement records.

Raw network absolute differences across all samples were:

| Tensor | Count | Median | P95 | P99 | Maximum | Mean |
|---|---:|---:|---:|---:|---:|---:|
| `cls_score` | 112,000,000 | 0.007844925 | 0.032186508 | 0.056829453 | 0.721186638 | 0.011085223 |
| `bbox_pred` | 100,800,000 | 0.000608385 | 0.007675864 | 0.018442094 | 0.427443326 | 0.001848617 |
| `dir_cls_pred` | 22,400,000 | 0.003826864 | 0.017819986 | 0.030509621 | 0.173043281 | 0.005795913 |

The raw classification tensor was closely aligned for the great majority of elements (p99 absolute
difference 0.0568) but showed a rare heavier tail (maximum 0.7212). The maximum is not described as
harmless and was not promoted into a new acceptance criterion. Final detector outputs nevertheless
satisfied the preregistered parity-v2 acceptance criteria.

All raw tensor shapes and dtypes were consistent between runtimes and across all 20 samples. Stage 2
was not required, so no pre-NMS survivor tracing was performed and no NMS-swap or other causal
labels were assigned. Review later authorized the same-commit benchmark documented below.

M2 deploys the exact M1 pretrained PointPillars model through ONNX and TensorRT FP16. It does not
train, simplify, or replace the model. The frozen scientific configuration is
`configs/detection/m2_pointpillars_tensorrt.yaml`. The byte-identical historical v1 policy is
`configs/detection/m2_parity_v1.yaml`; the preregistered current policy is
`configs/detection/m2_parity_v2.yaml`.

## Upstream support and pins

MMDeploy 1.3.1 lists PointPillars on nuScenes with TensorRT as supported. Its worked mmdet3d
conversion example uses CenterPoint with ONNX Runtime, so this exact target is officially listed
but less thoroughly demonstrated. LaserPerception must validate the pinned combination before
claiming M2 success.

- MMDeploy tag: `v1.3.1`
- MMDeploy commit: `bc75c9d6c8940aa03d0e1e5b5962bd930478ba77`
- Official deployment config:
  `configs/mmdet3d/voxel-detection/voxel-detection_tensorrt_dynamic-nus-64x4.py`
- TensorRT target: 8.6.1 with CUDA 11.8
- ONNX opset: 11, inherited from the pinned MMDeploy configuration
- MMDetection3D: 1.4.0 at `fe25f7a51d36e3702f961e198894580d83c4387b`
- PointPillars checkpoint SHA256:
  `f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0`

Primary upstream references:

- [MMDeploy v1.3.1 release](https://github.com/open-mmlab/mmdeploy/releases/tag/v1.3.1)
- [Pinned mmdet3d support matrix](https://github.com/open-mmlab/mmdeploy/blob/bc75c9d6c8940aa03d0e1e5b5962bd930478ba77/docs/en/04-supported-codebases/mmdet3d.md)
- [Pinned nuScenes 64×4 TensorRT config](https://github.com/open-mmlab/mmdeploy/blob/bc75c9d6c8940aa03d0e1e5b5962bd930478ba77/configs/mmdet3d/voxel-detection/voxel-detection_tensorrt_dynamic-nus-64x4.py)
- [Pinned official Python voxel detector](https://github.com/open-mmlab/mmdeploy/blob/bc75c9d6c8940aa03d0e1e5b5962bd930478ba77/mmdeploy/codebase/mmdet3d/deploy/voxel_detection_model.py)
- [NVIDIA TensorRT 8.6.1 installation guide](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/install-guide/index.html)

## Frozen deployment boundary

The ONNX/TensorRT graph is deliberately not raw point cloud to final detections:

```text
official nuScenes multi-sweep preparation
    -> official MMDetection3D voxelization outside the engine
    -> MMDeploy-rewritten PointPillars network
    -> TensorRT FP16 network outputs
    -> official MMDeploy/MMDetection3D postprocessing outside the engine
    -> LaserPerception DetectionFrame
```

MMDeploy v1.3.1 does not expose `VoxelDetectionModel.voxelize`. Its equivalent official path uses
the MMDetection3D `Det3DDataPreprocessor` from the voxel task's `create_input` flow. It does expose
`VoxelDetectionModel.postprocess`, which rebuilds the configured head and invokes the upstream
`predict_by_feat` implementation. M2 must use those exact pinned implementations or their direct
equivalents; it must not independently implement voxelization, anchor decoding, NMS, or score
handling.

The checked ONNX graph has dynamic `FLOAT` `voxels` (`N×64×4`), `INT32` `num_points` (`N`), and
`INT32` `coors` (`N×4`) inputs. Its `FLOAT` outputs are `cls_score0` (`1×140×200×200`),
`bbox_pred0` (`1×126×200×200`), and `dir_cls_pred0` (`1×28×200×200`). These are observed graph
bindings, not guessed profile shapes.

## Gate 0: TensorRT smoke test

Before any PointPillars export attempt, the isolated M2 environment must prove TensorRT itself can:

1. import the Python bindings and report exact versions;
2. see CUDA device 0;
3. build and serialize a trivial FP16-capable network;
4. deserialize it and create an execution context; and
5. execute one inference whose output matches the expected values.

If Gate 0 fails, PointPillars/MMDeploy work stops until the environment is repaired.

## Shape profiling

The preprocessing-only profiler scanned all 81 observed `mini_val` samples. Voxel counts were
4,352 minimum, 18,207 p50, 20,085 p90, 20,544 p95, and 22,546 maximum. The frozen input profile is
4,352/18,207/30,000 voxels for minimum/optimum/maximum, with corresponding `num_points` and `coors`
shapes. The maximum retains the official MMDeploy 30,000-voxel bound, exceeds the measured 25%
headroom target, and remains below the upstream 40,000 validation limit. Every parity sample fit.

## Frozen samples and shared execution

The parity set is fixed at indices 0, 4, 8, 12, 16, 21, 25, 29, 33, 37, 42, 46, 50, 54, 58, 63,
67, 71, 75, and 80. Final boxes are matched class-wise, deterministically and one-to-one, with a
minimum candidate BEV IoU of 0.50. The exported threshold is 0.25, the symmetric high-confidence
guard is 0.30, and the threshold-edge diagnostic band is 0.20–0.30 inclusive.

The exact v1 and v2 acceptance semantics live in their versioned parity configurations. V2 does
not loosen any numerical threshold: it preregisters a 99% per-detection aggregate rule, separates
geometric axis yaw from direction agreement, and records every exception. A Stage 1 failure
triggers only targeted Stage 2 forensics; a Stage 1 pass stops parity investigation and waits for
review before the benchmark.

## Performance evidence

After parity reconfirmation at measurement commit
`e2f9b6babb541d52beaa0bcd58e841a0a56cc851`, the reviewer-authorized benchmark compared the
MMDeploy-rewritten PyTorch FP32 network and TensorRT FP16 in one initialized process. The canonical
record is
`benchmarks/m2/results/rtx4060_pytorch_fp32_vs_tensorrt_fp16.json`.

The run repeatedly measured nuScenes v1.0-mini `mini_val` index 0 at batch size one. Each runtime
and boundary received 10 warmups first and 100 measured iterations, with runtime order alternating
each iteration.

| Runtime | Precision | Network median | End-to-end median | End-to-end P95 | End-to-end FPS |
|---|---|---:|---:|---:|---:|
| MMDeploy-rewritten PyTorch | FP32 | 2164.527 ms | 1816.859 ms | 2552.475 ms | 0.550 |
| TensorRT | FP16 | 17.414 ms | 78.647 ms | 105.017 ms | 12.715 |

**Headline end-to-end median speedup: 23.101×.** The secondary network-only median speedup is
124.297×. The historical M1 result was not used in either ratio.

| Network runtime | Mean | Median | P90 | P95 | Min | Max | Population std. dev. | FPS from median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 | 2142.212 ms | 2164.527 ms | 2763.908 ms | 2890.222 ms | 1330.156 ms | 3224.011 ms | 453.387 ms | 0.462 |
| TensorRT FP16 | 26.619 ms | 17.414 ms | 71.383 ms | 73.785 ms | 6.502 ms | 84.383 ms | 23.906 ms | 57.425 |

| End-to-end runtime | Mean | Median | P90 | P95 | Min | Max | Population std. dev. | FPS from median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 | 1856.050 ms | 1816.859 ms | 2448.022 ms | 2552.475 ms | 1160.047 ms | 2725.626 ms | 388.796 ms | 0.550 |
| TensorRT FP16 | 80.014 ms | 78.647 ms | 96.420 ms | 105.017 ms | 48.957 ms | 120.781 ms | 13.928 ms | 12.715 |

Network timing begins after common voxelization and ends when raw outputs required by shared
postprocessing are available; it uses CUDA events with per-iteration end-event synchronization.
End-to-end timing uses synchronized wall time from official sample loading and multi-sweep
preparation through official voxelization, runtime, shared official postprocessing, and
`DetectionFrame`. Imports, environment/model/engine initialization, checkpoint loading, downloads,
visualization, and JSON/image writes are excluded.

This is a warm-cache repeated-single-sample latency microbenchmark. It is not cold-storage I/O
latency, whole-dataset sequential throughput, guaranteed LiDAR sensor throughput, or a production
real-time guarantee.

| Memory metric | Measured value |
|---|---:|
| PyTorch network peak allocated / reserved | 408,934,400 / 713,031,680 bytes |
| PyTorch end-to-end peak allocated / reserved | 413,477,888 / 713,031,680 bytes |
| TensorRT serialized engine | 31,519,476 bytes |
| TensorRT `ICudaEngine.device_memory_size` | 1,212,340,736 bytes |
| Comparable process-level GPU memory | Pending measurement |

PyTorch rows are allocator counters after reset for one call in the initialized process. TensorRT
rows are the serialized file size and exact engine API metric; they are independent definitions,
not a generic cross-runtime VRAM comparison.

## Artifact and portability policy

ONNX and TensorRT engine files are external cache artifacts and are never committed. Tracked
evidence records logical names, SHA256, sizes, tool versions, graph bindings, profiles, and build
environment without private absolute paths.

A serialized TensorRT engine is tied to its build/runtime environment and is not a generally
portable model file. The repository will provide reproducible build instructions rather than an
engine download. The ONNX and engine metadata above are measured build evidence. M2 evidence is
complete and awaiting PR review/merge; serialized ONNX and TensorRT binaries remain external.

MMDeploy integration is limited to five materially distinct failed attempts or approximately six
focused hours, whichever comes first. At that boundary the exact failure and attempts are recorded
and implementation stops for architecture review. M2 does not permit custom LaserPerception CUDA
plugins, altered anchors, reduced ranges, removed layers, another checkpoint, or another detector.
