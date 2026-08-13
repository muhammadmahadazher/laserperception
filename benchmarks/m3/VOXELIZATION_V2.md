# M3B-V2 exact deterministic voxelization diagnostic

Status: **diagnostic measurement complete; candidate subsequently accepted and integrated through a separate production commit.**

The measurement was made at exact implementation commit
85b6488c92eda266f049ff142fc06bdab658d7ed. The structured result is
[diagnostics/deterministic_voxelization_v2_85b6488.json](diagnostics/deterministic_voxelization_v2_85b6488.json),
with committed-file SHA256
dea0e6ba0590b91de12e5978e401f9dd1481f176b7a5c15fda753d6cdbd8a79d. The original external
record SHA256 was e96d30be2feb86bb1ef7eb72c17721fd289eed95dd3d842e09f520f83f720da5;
the committed copy changes only two occurrences of the generic WSL Windows-PowerShell bridge path
to a placeholder. The protocol was frozen before measurement in
[configs/detection/m3b_deterministic_voxelization_v2.yaml](../../configs/detection/m3b_deterministic_voxelization_v2.yaml),
SHA256 9d0babfc7ae71ea6ce77cfb110ee02de07ea3d54a9b3e14c7fdfb7789309a8d6.

This file remains diagnostic evidence rather than canonical M3 performance. The candidate was
subsequently accepted, integrated into the M3 ROS production path, and revalidated at commit
`a129b3507597b25f44ab1a833562f68883ebe8ce`; the V2 timing values remain unchanged and diagnostic.

## Scope and immutable assets

V2 did not change the model, weights, checkpoint, ONNX, TensorRT engine, point-cloud range, voxel
size, voxel limits, sweep pipeline, postprocess, or ROS/DDS transport. The frozen artifact SHA256
values were:

| Artifact | SHA256 |
|---|---|
| PointPillars checkpoint | f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0 |
| ONNX | 61ce22a8ca31498675c32576bfb94f0093d31dc95d2762f7254bf915a59ecc16 |
| TensorRT FP16 engine | a005f75852097cd9b193750560b214cc3d5237ae9b6c106c7fca3d4fc348714b |

The measured environment was Ubuntu 22.04 under WSL2, Python 3.10.12, PyTorch 2.1.0+cu118,
MMCV 2.1.0, MMDetection 3.2.0, MMDetection3D 1.4.0, MMEngine 0.10.7, CUDA runtime 11.8, and an
NVIDIA GeForce RTX 4060 Laptop GPU with driver 610.88.

## Independently verified deterministic-reference semantics

The pinned MMCV source was inspected and hashed before candidate implementation. The reference:

1. computes integer voxel coordinates and rejects invalid/out-of-range points;
2. creates voxels in first-occurrence order from the original point stream;
3. retains points inside a voxel in original input order;
4. keeps the first 64 points per voxel;
5. keeps the first newly encountered voxels until max_voxels;
6. emits coordinates in z/y/x order; and
7. leaves unused hard-voxel slots zero-filled.

The evidence records hashes for the Python wrapper, PyTorch dispatch, CUDA launcher, and CUDA
kernel source. No unverified fast-deterministic upstream flag was assumed.

## Exact-fast candidate design

The experimental candidate uses the pinned MMCV dynamic voxel-coordinate CUDA operation followed
only by PyTorch tensor operations:

    input points + original indices
    -> pinned dynamic voxel coordinates
    -> unique composite (voxel key, original index) ordering
    -> group by z/y/x key
    -> order groups by first original index
    -> apply max_voxels
    -> retain first 64 original-order points
    -> zero-filled voxels / num_points / coors

The composite ordering key makes within-voxel order explicit and does not depend on sort stability
for equal keys. The candidate adds no custom CUDA, C++, or TensorRT plugin and does not use
`deterministic=False`.

## Hard correctness gates

All required gates passed at the measurement commit:

| Gate | Result |
|---|---:|
| Complete ordered mini_val exact voxel suite | **81/81 PASS** |
| W1 index 42 candidate input repeatability | **30/30 exact** |
| W2 index 49 candidate input repeatability | **30/30 exact** |
| W1 raw TensorRT output repeatability | **30/30 exact** |
| W2 raw TensorRT output repeatability | **30/30 exact** |
| Frozen detector samples | **20/20 exact raw outputs and final frames** |

For every one of the 81 samples, reference and candidate had identical voxel counts, shapes,
dtypes, values, output voxel order, within-voxel point order, zero padding, and SHA256 for
voxels, coors, and num_points. There was no first mismatch. Both 30-run full-history suites
matched their deterministic reference input hashes on every run; raw TensorRT hashes were also
identical on every run. The frozen detector suite observed exact raw-network and final
DetectionFrame equality, so the unchanged diagnostic yardstick passed without using tolerances
to excuse an input difference.

## Explicit provenance policy

`provenance_mode=full|live` is separate from voxelization:

- `full` is the historical default and retains exact per-frame voxel hashes;
- `live` omits full tensor SHA256 work and explicitly labels its lightweight semantic metadata; and
- evidence, parity, native/rewrite diagnostics, and existing callers continue to use full unless
  they explicitly request otherwise.

Core/evidence behavior continues to default to `full`. The final deployed ROS YAML explicitly
requests `live`, while detection values remained exact between full and live in every W0/W1/W2
reference and candidate check. The option does not change the model, network inputs, TensorRT
outputs, postprocess, or detections.

## Measurement-session eligibility

Each performance boundary used 20 warmups and 100 measured observations after a 30.201-second
sustained alternating W1/W2 reference/candidate GPU warmup. Timing used synchronized
time.perf_counter() wall clock. Reference and candidate ran in isolated blocks in the same
session.

The machine was externally confirmed on AC power and switched from its existing Balanced plan to
its existing Ultimate Performance plan immediately before the run; Balanced was restored after the
run. The in-run WSL-to-Windows PowerShell query returned an explicit Exec format error, so the
raw JSON retains that limitation instead of pretending the host check succeeded internally.

The preregistered 0.5-second telemetry sampler recorded 588/588 available samples during warmup and
measured blocks:

| Signal | Observed |
|---|---:|
| Performance state | P0: 585, P3: 3 |
| GPU temperature | 67–78 °C; median 74 °C |
| SM clock | 2160–2640 MHz; median 2557.5 MHz |
| Memory clock | 8001 MHz throughout |
| Power draw | 24.36–81.20 W; median 50.185 W |
| GPU utilization | 0–100%; median 96% |
| Memory utilization | 2–39%; median 6% |
| Reported power limit | unavailable under this WSL interface |

All 12 layer/preprocessing/direct reference-candidate telemetry pairs were assessable, shared
observed performance states, and had no obvious material state mismatch. The session was accepted
by the frozen paired-state rule with zero rejection reasons. Supported clocks were visible, but
application-clock reporting was deprecated and no clock lock or setting change was attempted.

## W0/W1/W2 timing

These are medians in milliseconds. The hard-layer boundary receives the same pre-collated CUDA
points. Complete preprocessing starts from the prepared model-ready sample before collation and
ends with batch-padded voxel tensors. Direct E2E starts with the model-ready point cloud and ends
with a DetectionFrame carrying the selected provenance metadata.

| Workload | Points | Voxels | Layer ref | Layer candidate | Speedup | Preprocess ref | Preprocess candidate | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W0, index 0, scene start | 33,587 | 4,352 | 6.450 | 1.365 | 4.724× | 6.883 | 1.832 | 3.757× |
| W1, index 42, full history | 354,182 | 18,207 | 238.910 | 1.758 | 135.907× | 270.029 | 2.701 | 99.992× |
| W2, index 49, full history | 346,073 | 20,085 | 261.918 | 1.918 | 136.579× | 260.886 | 2.657 | 98.188× |

Full-history voxelization is therefore materially faster while preserving bit-exact outputs.

### Direct TensorRT E2E

| Workload | Full ref | Full candidate | Live ref | Live candidate | Candidate full classification | Candidate live classification |
|---|---:|---:|---:|---:|---|---|
| W0 | 36.489 | 31.687 | 34.181 | 29.256 | ≤50 ms | ≤50 ms |
| W1 | 333.137 | 55.416 | 316.012 | 43.168 | 50–75 ms, close | ≤50 ms |
| W2 | 319.162 | 57.854 | 305.229 | 45.971 | 50–75 ms, close | ≤50 ms |

The honest direct-path conclusion is provenance-dependent. With historical full hashes, W1 and W2
are close but do not demonstrate 20 Hz. With explicit live provenance, both medians are below
50 ms and demonstrate direct-path 20 Hz feasibility in this eligible isolated diagnostic session.
This is not a ROS callback or sustained ROS rate result.

## Same-session candidate component ledger

The ledger instruments each stage separately and synchronizes between stages, so its total is a
separate, slightly more instrumented measurement than the direct E2E table. Component medians
should not be arithmetically summed as though medians were one shared iteration.

| Workload / mode | Prepare | Collate | Voxelize | TensorRT raw | Postprocess | Semantic conversion | Provenance | Residual | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W1 full | 3.143 | 0.857 | 2.574 | 14.554 | 21.411 | 1.301 | 12.446 | 0.006 | 56.896 |
| W1 live | 3.014 | 0.814 | 2.438 | 14.644 | 21.797 | 1.326 | 0.064 | 0.005 | 45.155 |
| W2 full | 3.175 | 0.840 | 2.624 | 16.664 | 21.480 | 1.018 | 13.620 | 0.006 | 60.125 |
| W2 live | 2.966 | 0.829 | 2.669 | 16.988 | 21.232 | 1.037 | 0.055 | 0.005 | 46.786 |

After exact-fast voxelization, unchanged MMDeploy postprocessing is the largest individual live
component. It was measured, not optimized. Full provenance adds approximately 12.4–13.6 ms; live
provenance reduces that stage to about 0.06 ms.

## Decision and subsequent integration

V2 satisfied its exactness, repeatability, detector-fidelity, and meaningful full-history
acceleration conditions and was accepted for the separately authorized final M3 integration.
The V2 diagnostic itself did not become canonical performance: its candidate was integrated later,
revalidated through the actual production path on all 81 voxel samples and frozen 20 detector/ROS
samples, and then measured through ROS at exact commit
`a129b3507597b25f44ab1a833562f68883ebe8ce`.

The final ROS result is reported separately in
[`results/rtx4060_ros2_humble_exact_tensorrt_fp16.json`](results/rtx4060_ros2_humble_exact_tensorrt_fp16.json).
It did not sustain 20 Hz; bounded characterization sustained 10 Hz and did not sustain 15 Hz.
M3B-V1 `deterministic=False` remains rejected. V2 and final integration added no custom CUDA,
changed no model or artifacts, rebuilt no engine, exported no ONNX, optimized no postprocess or
ROS/DDS path, and did not begin M4.
