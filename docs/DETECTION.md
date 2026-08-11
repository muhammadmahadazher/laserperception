# PointPillars detection

M1 wraps the upstream pretrained MMDetection3D PointPillars checkpoint used by LaserPerception. The
model was not trained by LaserPerception. This milestone is inference-only and preserves the
official nuScenes ten-class taxonomy and multi-sweep preprocessing pipeline.

## M2 deployment status

M2 preserves this exact M1 model and routes its network through the pinned official MMDeploy
1.3.1 ONNX/TensorRT configuration. Official multi-sweep preparation and MMDetection3D voxelization
remain outside the engine; both rewritten PyTorch FP32 and TensorRT outputs use the same official
MMDeploy postprocessing and LaserPerception conversion.

The implementation passed TensorRT Gate 0, profiled all 81 `mini_val` samples, checked the exported
ONNX graph, and built/ran the FP16 engine. Parity v1 failed its hard maximum XY, per-dimension size,
yaw, and score guards and remains failed. The separately preregistered v2 Stage 1 passed every
per-metric fraction, count, coverage, direction, and class gate using the same 20 samples and
unchanged engine, then passed again at exact measurement commit
`e2f9b6babb541d52beaa0bcd58e841a0a56cc851`. The benchmark from that commit was subsequently rejected because rewritten eager PyTorch was
not a valid performance baseline. Parity v2 remains valid; M2 benchmark diagnosis is in progress. Exact hashes, metrics, exceptions, and protocol chronology
are in `docs/TENSORRT.md`.

After activating the M2 environment, the reproducible sequence is:

```bash
python scripts/detection/check_m2_tensorrt.py
python scripts/detection/profile_m2_voxels.py
python scripts/detection/export_m2_onnx.py
python scripts/detection/build_m2_tensorrt.py
python scripts/detection/validate_m2_parity.py
python scripts/detection/benchmark_m2.py
```

The parity-v2 command exits zero for a passing full suite and writes external `parity_v2.json`.
`benchmark_m2.py` requires protocol-v2 passing evidence from the exact current commit and exact
ONNX/engine hashes. Benchmarking requires a passing native-vs-rewritten fidelity diagnosis from the exact current
commit. Native MMDetection3D PyTorch is the performance baseline; no replacement canonical result
is authorized yet. `LASERPERCEPTION_M2_CACHE` selects the external
cache; its default is `~/.cache/laserperception`.

## M1 status

M1 is complete and merged. The official converter observed 323 training and 81 validation
samples. FP32 inference on `mini_val` index 0 (token `3e8750f331d7499e9b5123e9eb70f2e2`)
produced 182 raw detections and nine at the fixed 0.25 display threshold: four cars and five genuine
model-predicted pedestrians, with best pedestrian score 0.403. The original 1800×1800 headless BEV
was generated at `artifacts/m1/pedestrian_sample0_bev.png` and remains intentionally ignored.

The sanitized measured benchmark is
`benchmarks/m1/results/rtx4060_laptop_fp32.json`; see `docs/BENCHMARKS.md` for its concise table and
exact timing boundaries.

## Official assets

- MMDetection3D 1.4.0, commit `fe25f7a51d36e3702f961e198894580d83c4387b`
- Config: `configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py`
- Checkpoint: `hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth`
- Checkpoint SHA256: `f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0`
- [Official checkpoint source](https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth)

The tracked source of truth is `configs/detection/m1_pointpillars_nuscenes.yaml`. The checkpoint is
an external upstream artifact and is never committed. Published model-zoo quality numbers, if any,
belong to the upstream project and are not LaserPerception measurements.

## Obtain nuScenes v1.0-mini

Review and comply with the [official nuScenes terms](https://www.nuscenes.org/terms-of-use) and use
only the official distribution. The [official nuScenes tutorial](https://www.nuscenes.org/public/tutorials/nuscenes_tutorial.html)
publishes the mini archive used here:

```bash
mkdir -p ~/datasets/nuscenes
curl -L --fail -o ~/datasets/nuscenes/v1.0-mini.tgz \
  https://www.nuscenes.org/data/v1.0-mini.tgz
tar -xf ~/datasets/nuscenes/v1.0-mini.tgz -C ~/datasets/nuscenes
export LASERPERCEPTION_NUSCENES_ROOT=~/datasets/nuscenes
```

Before preparation, the root must contain at least:

```text
nuscenes/
├── maps/
├── samples/
│   └── LIDAR_TOP/
├── sweeps/
│   └── LIDAR_TOP/
└── v1.0-mini/
```

Do not add nuScenes files or archives to this repository. nuScenes is governed by its own terms,
is not redistributed by LaserPerception, and is not relicensed by Apache-2.0. Its inclusion here
does not imply endorsement by Motional or the nuScenes maintainers.

## Prepare the official ten-sweep pipeline

Activate the environment from `docs/DETECTION_ENVIRONMENT.md`, then run:

```bash
python scripts/detection/prepare_nuscenes_mini.py \
  --data-root "$LASERPERCEPTION_NUSCENES_ROOT"
```

The wrapper resolves the MMDetection3D checkout through `LASERPERCEPTION_M1_CACHE`, falling back to
`~/.cache/laserperception`, unless `--mmdet3d-root` explicitly overrides it. It validates that both
data and generated metadata remain outside the repository, verifies the exact upstream commit,
creates the upstream-documented `data/nuscenes` symlink in that external checkout for the v1.4 info
updater, and invokes the installed official `tools/create_data.py` with
`--version v1.0-mini --max-sweeps 10`. It does not copy or rewrite the upstream converter. A dry run
is available with `--dry-run`.

## Run one inference

```bash
python scripts/detection/run_m1_inference.py \
  --data-root "$LASERPERCEPTION_NUSCENES_ROOT" \
  --split mini_val \
  --index 0
```

The backend validates CUDA, framework versions, and checkpoint SHA256; loads the model in eval mode
on `cuda:0`; explicitly disables autocast; runs the official multi-sweep test pipeline; and prints a
deterministically ordered table. `--json artifacts/m1/sample.json` writes an optional ignored,
framework-independent result. Score filtering occurs only after raw output conversion.

## Detection contract and conventions

`Detection3D` stores the geometric center as `(x, y, z)`, dimensions as `(length, width, height)`,
yaw in radians, score, official class ID/name, and optional `(vx, vy)`. `DetectionFrame` stores a
stable tuple of detections, sample token, coordinate frame, and copied metadata.

For M1, the frame is nuScenes `LIDAR_TOP`: X forward, Y left, Z up. Positive yaw is
counter-clockwise from positive X when viewed from above. At zero yaw, length is parallel to X and
width to Y. The public contract contains no MMDetection3D tensors or classes.

Raw class names are exactly: `car`, `truck`, `trailer`, `bus`, `construction_vehicle`, `bicycle`,
`motorcycle`, `pedestrian`, `traffic_cone`, and `barrier`. M1 does not introduce a `cyclist` mapping.

## Render an original headless BEV

```bash
python scripts/detection/render_m1_bev.py \
  --data-root "$LASERPERCEPTION_NUSCENES_ROOT" \
  --split mini_val \
  --index 0 \
  --output artifacts/m1/pointpillars_bev.png
```

The original Matplotlib renderer uses the noninteractive `Agg` backend and draws cropped LiDAR
points, oriented XY box projections, heading, upstream class labels, scores, origin, axes, and a
legend. PNG and SVG are supported. The committed visualization threshold is 0.25; changing it does
not alter model inference or benchmark timing.

To find, rather than fabricate, a sample with a qualifying prediction:

```bash
python scripts/detection/find_m1_sample.py \
  --data-root "$LASERPERCEPTION_NUSCENES_ROOT" \
  --class-name pedestrian \
  --min-score 0.25 \
  --max-samples 100
```

The scan is deterministic, bounded, uses exact upstream class equality, and never lowers the
threshold. Render the reported token/index with `render_m1_bev.py`. If no match is found, report that
fact rather than adding a ground-truth or hand-authored box.

## Benchmark

```bash
python scripts/detection/benchmark_m1.py \
  --data-root "$LASERPERCEPTION_NUSCENES_ROOT"
```

See `benchmarks/m1/README.md` for the two timing boundaries and statistics. The script writes a
sanitized ignored JSON only after both measurement sections finish. Initialization, downloads,
checkpoint loading, visualization, and image saving are outside both boundaries.

## Limitations and safety

This path is a single upstream model on one small demo dataset, with batch size one and no accuracy
evaluation by LaserPerception. It is not evidence of generalization, production readiness, or safe
operation. LaserPerception v0.1 is for research, benchmarking and demo use. It is NOT
safety-certified and must not be treated as a certified perception system for operation around
humans.
