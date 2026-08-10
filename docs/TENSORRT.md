# M2 ONNX and TensorRT FP16 deployment

Status: **Protocol frozen; implementation and measurements pending.**

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

The exported graph's actual input/output names, shapes, and dtypes remain `Pending measurement`
until an exported ONNX graph has been inspected. The expected 64-points-by-4-features family comes
from the frozen M1 configuration, not from a guessed engine profile.

## Gate 0: TensorRT smoke test

Before any PointPillars export attempt, the isolated M2 environment must prove TensorRT itself can:

1. import the Python bindings and report exact versions;
2. see CUDA device 0;
3. build and serialize a trivial FP16-capable network;
4. deserialize it and create an execution context; and
5. execute one inference whose output matches the expected values.

If Gate 0 fails, PointPillars/MMDeploy work stops until the environment is repaired.

## Shape profiling

Before the final engine is built, a preprocessing-only command will scan all 81 observed
`mini_val` samples. It will record minimum, p50, p90, p95, and maximum for every dynamic input
dimension. Profile selection must combine those observations with the upstream hard limits:
64 points per voxel, four point features, and at most 40,000 validation voxels. Every frozen parity
sample must fit before parity begins. Profile overflow is an engine-configuration failure, not a
numerical-parity failure.

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
engine download. No result becomes measured evidence until the real artifact, parity, and
same-session benchmark have completed successfully.

MMDeploy integration is limited to five materially distinct failed attempts or approximately six
focused hours, whichever comes first. At that boundary the exact failure and attempts are recorded
and implementation stops for architecture review. M2 does not permit custom LaserPerception CUDA
plugins, altered anchors, reduced ranges, removed layers, another checkpoint, or another detector.
