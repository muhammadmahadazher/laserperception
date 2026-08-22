# M6b-R1 structural 40,000-voxel TensorRT profile remediation

Status: preregistered prospective remediation; M6b remains blocked.

## Scientific chronology

The frozen M6b protocol reproduced all 856 H10/H5 model-ready inputs before the historical
TensorRT engine rejected a valid `exact_fast` tensor shape. The rejected engine has a 30,000-voxel
profile maximum while the unchanged validation voxelizer contract permits 40,000 voxels. No KITTI
network output or detector prediction was produced. The blocker remains recorded in
`benchmarks/m6b/diagnostics/failed_engine_shape_preflight.json`; this protocol does not rewrite it.

M6b-R1 is a prospective compatibility remediation. It may build exactly one second FP16 TensorRT
engine from the byte-identical M2 ONNX. It does not authorize M6b evaluation, prediction on either
evaluation drive, tuning, performance work, M6c, or a release.

## Structural invariant and frozen artifact change

A deployment engine declared compatible with a deterministic voxelizer contract must have a
TensorRT profile maximum at least as large as the maximum tensor that voxelizer may emit. The
prospective engine therefore freezes these shapes for all three inputs:

| Binding | Minimum | Optimum | Maximum |
| --- | --- | --- | --- |
| `voxels` | `[4352, 64, 4]` | `[18207, 64, 4]` | `[40000, 64, 4]` |
| `num_points` | `[4352]` | `[18207]` | `[40000]` |
| `coors` | `[4352, 4]` | `[18207, 4]` | `[40000, 4]` |

Only the insufficient maximum changes. The optimum remains the historical nuScenes-derived 18,207
shape; it is not retuned to the KITTI distribution. The candidate logical identity is
`engines/m6/pointpillars_fp16_profile40k.engine`. The original 30k engine remains immutable and
canonical for historical M2 through v0.2.0 evidence.

The source ONNX must have SHA256
`61ce22a8ca31498675c32576bfb94f0093d31dc95d2762f7254bf915a59ecc16`. The checkpoint, ONNX graph,
MMDeploy rewrite, FP16 precision, workspace setting, model, voxelizer, postprocess, thresholds, and
TensorRT/MMDeploy generation path remain unchanged. A source hash or load-bearing toolchain mismatch
stops the build.

## Build and serialized-engine gate

The build uses MMDeploy 1.3.1 at commit
`bc75c9d6c8940aa03d0e1e5b5962bd930478ba77`, TensorRT 8.6.1, the official
`voxel-detection_tensorrt_dynamic-nus-64x4.py` configuration, `onnx2tensorrt`, FP16, no INT8, and
CUDA device 0. The historical builder defaults must continue to select the historical M2 manifest.

The serialized candidate must expose exactly `voxels`, `num_points`, `coors`, `cls_score0`,
`bbox_pred0`, and `dir_cls_pred0`, with the exact frozen profile above. Build evidence records the
engine hash and size, inspector hash, build environment and duration, workspace, and serialized
`getDeviceMemorySize`/`device_memory_size` value. The old engine's recorded device-memory size is
1,212,340,736 bytes; both values are retained for capacity review on the 8 GB reference GPU.

## Gate 1: frozen nuScenes parity and same-session characterization

Before any KITTI network execution, the candidate runs on the exact frozen 20-sample nuScenes
parity-v2 suite. Rewritten PyTorch FP32 versus candidate TensorRT FP16 uses the unchanged M2
parity-v2 samples, thresholds, matching, metrics, direction diagnostics, and Stage 1 acceptance.
Stage 1 must pass.

In the same session, the original 30k and candidate 40k engines run on the identical frozen tensors.
Raw classification, box, and direction differences plus final DetectionFrame counts and matched
differences are reported as characterization only. This same-session comparison avoids confounding
profile effects with documented cross-session performance and numerical variability. Exact equality
is recorded if it occurs but is not required.

## Gate 2: frozen non-evaluation KITTI parity

The only authorized KITTI source is non-evaluation drive `2011_09_30_drive_0016`. Before network
execution, all eligible H10 frames receive input-only point counts, model-ready hashes, and retained
voxel counts. Twelve frames are frozen deterministically from nearest-rank quantiles 0, 10, 20, 30,
40, 50, 60, 70, 80, 90, 95, and 100 percent, using lower frame index for ties. Duplicate quantiles
are filled by unused frames with greatest voxel-count distance from the selected set, again using
lower index for ties.

The frozen set must contain at least one frame at or below 30,000 voxels, at least four above 30,000,
and at least one at or above 39,000 (or exactly 40,000). Otherwise the gate stops without choosing a
new drive. Rewritten PyTorch FP32 versus candidate TensorRT FP16 then uses identical voxel tensors
and unchanged parity-v2 acceptance. No GT metric or M6b evaluation is run.

## Repeatability gate

After Gate 2 passes, candidate raw inference runs five times on each of two frozen drive-0016 frames:

- the highest-voxel selected frame, covering the expanded envelope; and
- the selected frame nearest the 18,207 optimum, tie-breaking by lower frame index.

`cls_score`, `bbox_pred`, and `dir_cls_pred` must each have one exact SHA256 across all five runs for
both frames. Any difference stops the remediation; no tolerance is adopted.

## Stop boundary

No network output may be produced for M6b evaluation drives `2011_09_26_drive_0001` or
`2011_09_26_drive_0091`. Passing the build, parity, characterization, and repeatability gates permits
only a draft prospective `M6B_PROTOCOL_R2_DRAFT.md`. Owner approval is still required before the
428-frame M6b characterization resumes.
