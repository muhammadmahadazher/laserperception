# M2 ONNX and TensorRT FP16 deployment

Status: **Partial M2 implementation; ONNX/engine pass, frozen FP16 parity fails.**

## Measured M2 outcome

Gate 0 passed, all 81 `mini_val` samples were profiled, ONNX export/checking passed, and the
official TensorRT FP16 engine built and executed. The unchanged 20-sample parity suite then failed
four locked high-confidence guards at implementation commit
`a9314483e0ba7a191866266080c3147f9d902956`. M2 is therefore partial and requires architecture
review; no benchmark was run or promoted.

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

M2 deploys the exact M1 pretrained PointPillars model through ONNX and TensorRT FP16. It does not
train, simplify, or replace the model. The frozen scientific configuration is
`configs/detection/m2_pointpillars_tensorrt.yaml`; the acceptance policy is
`configs/detection/m2_parity.yaml`.

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

## Frozen parity protocol

The parity set is fixed at indices 0, 4, 8, 12, 16, 21, 25, 29, 33, 37, 42, 46, 50, 54, 58, 63,
67, 71, 75, and 80. Final boxes are matched class-wise, deterministically and one-to-one, with a
minimum candidate BEV IoU of 0.50. The exported threshold is 0.25, the symmetric high-confidence
guard is 0.30, and the threshold-edge diagnostic band is 0.20–0.30 inclusive.

The exact count, coverage, center, size, yaw, score, and class-name acceptance limits live in the
parity configuration. They cannot be relaxed after observing FP16 output. Failed evidence must be
classified as preprocessing mismatch, profile/binding failure, network numerical difference,
postprocessing mismatch, or threshold-edge difference.

## Performance protocol

M2 will remeasure the MMDeploy-rewritten PyTorch FP32 network and TensorRT FP16 in the same session,
using the same voxelized inputs and common postprocessing. It will not use the archived M1 latency
to calculate speedup. Each runtime and boundary receives 10 warmups and 100 measurements at batch
size one on `mini_val` index 0.

The headline result is end-to-end detection speedup. Network-only speedup is reported separately
because voxelization and postprocessing remain outside TensorRT. Both are warm-cache,
repeated-single-sample latency microbenchmarks—not cold-storage latency, whole-dataset throughput,
or a sensor-throughput guarantee.

PyTorch allocator counters and TensorRT engine/context memory are independent metrics. A generic
cross-runtime VRAM comparison remains `Pending measurement` unless a reliable, consistently defined
process-level method is available.

## Artifact and portability policy

ONNX and TensorRT engine files are external cache artifacts and are never committed. Tracked
evidence records logical names, SHA256, sizes, tool versions, graph bindings, profiles, and build
environment without private absolute paths.

A serialized TensorRT engine is tied to its build/runtime environment and is not a generally
portable model file. The repository will provide reproducible build instructions rather than an
engine download. The ONNX and engine metadata above are measured build evidence. M2 itself
remains partial because parity failed; no same-session benchmark result is accepted or committed.

MMDeploy integration is limited to five materially distinct failed attempts or approximately six
focused hours, whichever comes first. At that boundary the exact failure and attempts are recorded
and implementation stops for architecture review. M2 does not permit custom LaserPerception CUDA
plugins, altered anchors, reduced ranges, removed layers, another checkpoint, or another detector.
