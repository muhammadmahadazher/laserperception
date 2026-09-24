# Frequently asked questions

## What is LaserPerception?

LaserPerception is an open-source 3D LiDAR perception and deployment-engineering toolkit. It
combines CPU-safe data/model contracts, deterministic tracking and semantic evaluation with
optional detector, TensorRT, ROS 2, and external-GPU workflows. See [project status](PROJECT_STATUS.md).

## Is it a detector framework?

It provides a small framework-independent `DetectionFrame` contract, reviewed model manifests,
input validation, execution plans, and guarded backends. It does not replace MMDetection3D or
DSVT/OpenPCDet; optional backends wrap pinned upstream runtimes.

## Which models are supported?

The released historical path uses the official pretrained MMDetection3D PointPillars model on
nuScenes. M8 integrates official pretrained DSVT-Pillar + TransFusion as an active research
candidate. Its accepted three-process S1 primary raw measurement is published without a scientific
interpretation or winner claim.

## Did LaserPerception train PointPillars or DSVT?

No. Both checkpoints are official upstream pretrained artifacts. LaserPerception records their
sources and hashes and does not redistribute or relicense them.

## Does the wheel require CUDA?

No. The core wheel depends only on lightweight CPU packages. PyTorch, CUDA, OpenMMLab,
DSVT/OpenPCDet, TensorRT, and ROS 2 are optional isolated environments.

## Can I use it CPU-only?

Yes. Model and adapter discovery, input inspection, dry-run planning, tracking, semantic-result
inspection/evaluation, worker planning, and the synthetic examples run on CPU. Start with the
[CPU-first quickstart](QUICKSTART_PERCEPTION.md).

## Does it support ROS 2?

Yes. The historical deployment path targets ROS 2 Humble and supports model-ready replay plus
compatible raw XYZ `PointCloud2` with time-aware tf2 accumulation. ROS is an optional environment.

## Does it support raw LiDAR?

It supports eight reviewed input paths, including KITTI Velodyne, SemanticKITTI, LAS/LAZ, DALES,
KITTI Raw, nuScenes `LIDAR_TOP`, and a PointCloud2 XYZ boundary. Each adapter states its exact
features and limitations in [DATA_ADAPTERS.md](DATA_ADAPTERS.md).

## What does “multi-sweep” mean?

Historical acquisitions are transformed into the current sensor frame and concatenated with a
time-lag feature. Correctness depends on acquisition timestamps and poses, not merely frame names.
See [MULTISWEEP.md](MULTISWEEP.md).

## Is tracking implemented?

Yes. P2 provides deterministic class-aware constant-XY-velocity tracking over explicitly timed,
precomputed `DetectionFrame` input. It is CPU-only infrastructure; no detector-plus-tracker
end-to-end benchmark is claimed. See [TRACKING.md](TRACKING.md).

## Is segmentation implemented?

Semantic result and evaluation infrastructure is implemented: immutable row-aligned results,
taxonomy metadata, identity-bound serialization, and confusion/IoU metrics. No production
segmentation model is included. The earlier SemanticKITTI-to-DALES experiment remains parked with
`Pending measurement` results. See [SEMANTIC_SEGMENTATION.md](SEMANTIC_SEGMENTATION.md).

## What is M8?

M8 is the active Detector V2 research milestone using DSVT-Pillar + TransFusion. Engineering
integration and a frozen S1 protocol exist, but the accepted three-pass primary A2/E2 measurement
is pending. See [M8 external-runtime status](m8/M8_S1_EXTERNAL_RUNTIME_STATUS.md).

## Why are failed experiments retained?

Failed, rejected, incomplete, and negative records protect scientific chronology and prevent later
repairs from erasing evidence. See [FAILURE_INDEX.md](FAILURE_INDEX.md).

## What benchmark claims are canonical?

Only records identified as canonical in [BENCHMARKS.md](BENCHMARKS.md). Diagnostic, rejected,
external, and incomplete records are explicitly separate. Hardware measurements are not portable
guarantees.

## What is the OmniLink evaluation?

It is an independent synthetic evaluation of historical v0.3.0 using OmniSim. It verified the
multi-sweep transform/reconstruction convention and found no valid intended traffic-cone match at
threshold 0.25 in either authored case. It is a negative domain-gap result, not dataset-level
accuracy or a LaserPerception-run benchmark. See the [full record](external/OMNILINK_OMNISIM_EVALUATION.md).

## Is LaserPerception production-ready or autonomy-safety certified?

No. The project makes no production-readiness, deployment-safety, autonomous-driving
certification, universality, or SOTA claim.

## What data and models are included?

The repository and wheel do not contain datasets, checkpoints, ONNX files, TensorRT engines, or
third-party raw evaluation captures. Users obtain those assets from their official sources under
their own terms.

## Is an external GPU provider required?

No provider is required by the software design. CPU development works locally; optional scientific
GPU execution uses separately selected, qualified, authorized workers. RunPod has been used in M8
operations, but the tooling remains provider-neutral.

## Can companies use the Apache-2.0 code commercially?

Apache-2.0 permits commercial use of LaserPerception's own code subject to its terms. It does not
grant rights to third-party datasets, models, software, or captures; review those terms separately.
