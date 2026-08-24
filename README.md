# LaserPerception

> Reproducible 3D LiDAR detection with raw ROS 2 ingestion, time-aware multi-sweep
> reconstruction, TensorRT FP16, and exact deterministic voxelization.

[![CI](https://github.com/muhammadmahadazher/laserperception/actions/workflows/ci.yml/badge.svg)](https://github.com/muhammadmahadazher/laserperception/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-0.2.0-4c1.svg)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

LaserPerception v0.2.0 is an open-source research toolkit that runs one frozen, official pretrained
PointPillars detector on nuScenes, deploys its network through TensorRT FP16, preserves deterministic
voxel semantics with an exact fast path, and publishes 3D detections through ROS 2 Humble. It can
accept either model-ready temporal input or compatible raw single-sweep XYZ `PointCloud2` plus a
valid time-aware TF tree. LaserPerception did **not** train PointPillars or introduce a new detector
architecture.

![Real W1 ROS 2 replay output with predicted 3D boxes in RViz2.](docs/assets/v0_1_ros_demo.png)

*Real W1 ROS 2 replay output with predicted 3D boxes in RViz2.*

**Historical model-ready performance:** on representative full-history W1 (10 historical sweeps
plus current, 354,182 points), 10 Hz was the highest tested clean sustained ROS rate. Fifteen hertz
and 20 Hz were not sustained. M4.5 raw ingestion was correctness/integration work and has no new
throughput claim.

[Run the v0.2 detection/ROS quickstart](docs/QUICKSTART_V0_2.md) ·
[Read the v0.2.0 release notes](docs/releases/v0.2.0.md) ·
[Inspect the benchmark evidence](docs/BENCHMARKS.md)

**New raw ingestion path:** compatible scalar float32 XYZ `PointCloud2` plus time-aware tf2 feeds a
bounded current-plus-ten-sweep builder and then the unchanged model-ready detector. The frozen raw
ROS suite matched accepted model-ready inputs, voxel tensors, TensorRT tensors, detections, and ROS
message semantics exactly on 20/20 samples. [Read the raw ROS contract](docs/RAW_LIDAR_ROS2.md) or
the [multi-sweep evidence](docs/MULTISWEEP.md).

**Cross-domain KITTI Raw study:** the frozen detector's Car recall changed from 0.242 under H10 to
0.727 under H5 without fine-tuning. H10/H5 is a compound temporal-and-density ablation, and the
preregistered 40,000-voxel-cap hypothesis was not supported as the primary corpus-wide explanation.
Final ROS integration reproduced 860/860 same-platform projected references exactly and passed the
unchanged detector semantic envelope on ten frozen sentinels; the original R2 byte-exactness failure
remains preserved. [Read the M6 technical note](docs/m6/M6_CROSS_DOMAIN_TECHNICAL_NOTE.md) or the
[final M6c result](docs/m6/M6C_RESULTS_R3.md).

## What v0.2.0 does—and what was measured

| Engineering story | Shipped behavior | Measured evidence |
|---|---|---|
| TensorRT deployment | Frozen pretrained PointPillars, pinned MMDeploy export, TensorRT 8.6.1 FP16 | Parity-v2 gates passed; repaired scene-start M2 median was 59.289 ms native PyTorch vs 45.637 ms TensorRT end to end (1.2991×) |
| Deterministic voxelization | `exact_fast` preserves pinned official deterministic retained-point semantics | 81/81 validation samples bit-exact; frozen raw/final detector outputs exact; W1 hard layer 238.910 → 1.758 ms (≈136×, hard layer only) |
| ROS 2 integration | Model-ready multi-sweep `PointCloud2` → `Detection3DArray` plus RViz/Foxglove markers | Historical model-ready W1 sustained 10 Hz cleanly; 15 Hz and 20 Hz were not sustained |
| Raw multi-sweep ingestion | Compatible float32 XYZ `PointCloud2` + time-aware TF → bounded history → same model-ready detector | M4.5a 81/81 inputs exact; M4.5b complete raw ROS detector chain exact on 20/20 frozen samples; no new rate campaign |

The accepted M3B-V2 direct W1 live diagnostic changed from about 333 ms to 43.168 ms. The ~43 ms
figure is **direct runtime evidence, not ROS callback or loopback latency**. The ≈136× ratio applies
only to the hard voxel layer, not whole LiDAR inference.

## Reproducibility scope

Correctness claims—exact-fast 81/81 bit-exact voxel outputs, frozen detector exactness, and ROS
message-contract correctness—are semantic/software evidence intended to be reproducible when the
pinned software stack and inputs are reproduced.

Performance claims are measurements from **one system**: an NVIDIA GeForce RTX 4060 Laptop GPU,
WSL2, driver 610.88, and the pinned CUDA/TensorRT/OpenMMLab environment. **Timings are measurements
of this specific environment, not portable hardware capability guarantees.** They do not promise
10 Hz on every RTX 4060 laptop or equivalent behavior on another GPU, Windows native, native Linux,
Jetson, or another environment.

## Current architecture

```mermaid
flowchart TD
    R["v0.2: raw single-sweep PointCloud2"] --> T["Time-aware tf2 + bounded live history"]
    T --> A["Model-ready multi-sweep PointCloud2"]
    A --> B["exact_fast deterministic voxelization"]
    B --> C["Frozen TensorRT FP16 PointPillars network"]
    C --> D["Unchanged MMDeploy postprocess"]
    D --> E["Framework-independent DetectionFrame"]
    E --> F["Detection3DArray"]
    E --> G["RViz / Foxglove markers"]
```

v0.2 supports both boundaries. The v0.1-compatible path begins at model-ready `x`, `y`, `z`, and
`time_lag`. The new path begins one boundary earlier: compatible raw single-sweep messages require
scalar float32 XYZ and a valid time-indexed TF tree. The builder uses the current acquisition time as
the target time and preserves historical acquisition stamps; a repeated frame name at different
times is not assumed to be an identity transform. Extra fields are ignored by the frozen detector.

Two explicit policies preserve historical evidence and deployed semantics:

| Use | Voxelization | Provenance |
|---|---|---|
| Historical/core evidence | `official` pinned deterministic hard voxelization | `full` tensor hashes |
| ROS deployment | `exact_fast` LaserPerception implementation | `live` lightweight metadata |

`exact_fast` uses the pinned MMCV dynamic-coordinate CUDA operation plus PyTorch tensor grouping.
It is a LaserPerception deployment optimization, not an upstream MMDetection3D implementation. It
fails closed; there is no silent fallback to `deterministic=False`.

## Quickstart

### Lightweight CPU package

The Python wheel supports Python 3.10–3.13 and does not depend on CUDA, PyTorch, OpenMMLab,
TensorRT, or ROS 2:

```bash
git clone https://github.com/muhammadmahadazher/laserperception.git
cd laserperception
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

```python
import numpy as np
from laserperception import PointCloud, __version__

cloud = PointCloud(xyz=np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
print(__version__, len(cloud), cloud.xyz.dtype)
```

### GPU detector and ROS demo

The validated deployment stack is Ubuntu 22.04 under WSL2, Python 3.10, CUDA 11.8, TensorRT 8.6.1,
and ROS 2 Humble. nuScenes, the official checkpoint, ONNX, and TensorRT engine are external and are
not committed or included in the wheel.

Follow the [v0.2.0 quickstart](docs/QUICKSTART_V0_2.md). The existing model-ready W1 replay and
actual predicted boxes remain available through the compatibility wrapper:

```bash
bash scripts/run_v0_1_demo.sh
```

The v0.2 raw XYZ plus time-aware TF replay uses the accepted M4.5b launch:

```bash
ros2 launch laserperception_ros m45b_raw_multisweep.launch.py
```

The wrapper and launch use the existing detector path. Missing prerequisites fail closed; no
licensed dataset is silently downloaded and no engine is rebuilt. The raw replay is a nuScenes
reproducibility example, not a physical-sensor validation or a new performance measurement.

## Three evidence-backed engineering stories

### 1. TensorRT without changing the detector

M2 froze the pretrained checkpoint, ONNX, engine, samples, thresholds, and deployment boundary.
The parity reference is MMDeploy-rewritten PyTorch FP32; the performance baseline is native
MMDetection3D PyTorch FP32. Rewritten eager PyTorch is deliberately **not** the speedup denominator.

Parity v1 failed and remains preserved. The separately preregistered parity v2 passed all Stage 1
per-metric gates on the unchanged 20-sample suite and engine. The first M2 benchmark was rejected
because it used the wrong denominator; the repaired result is the only canonical M2 performance
record.

### 2. Exact fast voxelization after rejecting the easy shortcut

Official deterministic hard voxelization dominated representative full-history latency. Upstream
`deterministic=False` was fast, but saturated voxels retained different point subsets and frozen
repeatability exposed observable detection changes, so M3B-V1 rejected it.

M3B-V2 then implemented `ExactDeterministicVoxelizer` without custom CUDA/C++. It reproduced all
81 validation voxel tensors bit-for-bit, repeated exactly on W1/W2, and retained exact raw TensorRT
and final detection outputs on the frozen suite. Its direct diagnostics are useful bottleneck
evidence, but they are not ROS timing.

### 3. Honest representative ROS behavior

The final canonical M3 workload is `mini_val` index 42 with 10 historical sweeps plus the current
keyframe. The full run records callback entry through `publish()` return, same-host publisher-stamp
to sink reception, drops, effective output rate, GPU telemetry, and first/second-half backlog
behavior.

- 10 Hz: sustained cleanly, 9.949 Hz useful output, 0/200 measured input drops.
- 15 Hz: not sustained, about 13.34 Hz useful output, 21/221 drops.
- 20 Hz: not sustained, about 10.83 Hz useful output, 159/359 drops; entry intervals and drops grew
  between run halves.

This is behavior consistent with overload in the measured ROS configuration. The evidence does not
establish DDS, executor, or thermal behavior as a single cause.

## Benchmark map

The release separates workloads and evidence types rather than combining incompatible sessions:

- **M1:** historical FP32 warm-cache scene-start index 0, zero historical sweeps.
- **M2:** canonical same-session native PyTorch FP32 vs TensorRT FP16 on that same scene-start input.
- **M3B-V2:** direct diagnostic evidence for exact-fast correctness and component latency.
- **M3:** canonical ROS result on representative full-history W1 index 42.

Start with [BENCHMARKS.md](docs/BENCHMARKS.md), then inspect the sanitized records under
[`benchmarks/`](benchmarks/). The rejected M2 run, failed M3A rate test, rejected M3B-V1 candidate,
and accepted M3B-V2 diagnostics remain visible in the scientific chronology.

## Known issues and limitations

Observed issues include material GPU session-to-session timing variability, an uncontrolled later
M1-style reproduction that differed from the archived M1 result, and decreasing useful output under
15/20 Hz offered-rate overload. Causes were not isolated. See the separate
[historical v0.1 known issues](docs/releases/v0.1.0.md#known-issues).

v0.2.0 supports the model-ready interface and compatible raw PointCloud2 plus valid time-aware TF.
It does not prove arbitrary sensor calibration, localization, odometry, intra-scan deskew,
physical-LiDAR accuracy, or plug-and-play support for any LiDAR. Training, tracking, camera fusion,
a second detector, INT8, and Jetson measurements remain absent. LaserPerception is research/demo
software, not a safety-certified perception system. See the
[v0.2.0 limitations](docs/releases/v0.2.0.md#limitations).

## Repository map

```text
src/laserperception/   Lightweight CPU core and optional detector/deployment surface
ros2/                  Isolated ROS 2 Humble package, config, launch, and native tests
scripts/                Setup, evidence, validation, and demo entry points
configs/                Frozen detector/deployment protocols and parked experiment config
benchmarks/             Sanitized canonical results and intentionally retained diagnostics
docs/                   Quickstart, architecture, evidence, roadmap, and release notes
tests/                  Synthetic CPU regression suite
```

## Parked experimental infrastructure

The earlier SemanticKITTI-to-DALES `PointCloud`, I/O, transforms, ontology, adapters, and audit
pipeline remain tested and supported, but they are not the current detection release line. No
segmentation model or training result exists; those fields remain `Pending measurement`.

## Safety, citation, and licensing

Predictions can be missed, misclassified, or poorly localized. Do not treat LaserPerception as a
certified component for operation around people or vehicles.

Original LaserPerception code is [Apache-2.0](LICENSE). That license does not relicense nuScenes,
external weights, TensorRT engines, ROS/OpenMMLab/NVIDIA components, datasets, or papers. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Cite LaserPerception v0.2.0 and, where
reproducibility matters, the exact commit; no DOI is claimed. Citation metadata is in
[CITATION.cff](CITATION.cff).

Questions and contributions: [CONTRIBUTING.md](CONTRIBUTING.md) ·
[GitHub Discussions](https://github.com/muhammadmahadazher/laserperception/discussions) ·
[Security policy](SECURITY.md)
