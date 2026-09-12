# Vision — evidence before deployment claims

LaserPerception is an open-source 3D LiDAR object-detection and deployment-engineering toolkit. Its
released v0.3.0 path deliberately freezes one official pretrained PointPillars detector, verifies
TensorRT FP16 deployment, preserves deterministic voxel semantics, exposes detections through ROS
2, and can reconstruct model-ready temporal input from compatible raw XYZ PointCloud2 plus
time-aware TF. The project values auditable correctness, honest hardware-specific measurements,
and visible failure records over broad or unqualified performance claims.

The accepted release line is a research, benchmarking, and demonstration toolkit—not a novel
detector, a trained LaserPerception model, a production-ready autonomy stack, or a safety-certified
system.

## Evidence-gated sequence

1. M0: project direction and governance transition — complete.
2. M1: pretrained PointPillars, nuScenes v1.0-mini, RTX 4060 FP32 evidence, and BEV output —
   complete.
3. M2: ONNX/TensorRT FP16, parity/fidelity evidence, and repaired performance baseline — complete.
4. M3: ROS 2 interface, exact-fast deployment, correctness gates, and representative W1 rate
   evidence — complete.
5. M4: v0.1.0 release engineering — complete.
6. M4.5: offline and live raw-sweep reconstruction, time-aware TF, and exact detector-chain
   evidence — complete.
7. M4.6: v0.2.0 release engineering for the accepted M4.5 capability — complete.
8. M5: conditional physical Jetson measurements only if hardware is available and the owner
   activates the milestone; currently inactive.
9. M6: KITTI Raw offline reconstruction, frozen-detector characterization, and projected-reference
   ROS validation — complete.
10. v0.3.0: release engineering for the completed M6 state — complete.
11. M7: controlled history-mechanism study — complete, owner-approved, and frozen. Encoded-lag
    compression substantially explained the measured Car gap under the frozen criterion, but
    class/range limitations prevent adopting a universal deployment policy. See
    [the frozen interpretation](m7/M7_RESULTS.md).
12. M8 Detector V2: active, with DSVT-Pillar plus TransFusion selected. P1-E is complete and the
    S1 protocol is frozen. Retired-runtime Stage R is complete; retired-runtime primary A2/E2
    calls remain zero. S1 is paused pending a newly qualified external GPU runtime and fresh
    runtime-specific authorization. See [the roadmap](ROADMAP.md).

Training, additional detectors, INT8, tracking, camera fusion, custom CUDA, localization, sensor
calibration automation, and ROS/DDS optimization are not part of v0.3.0. M8's separately authorized
DSVT engineering integration does not rewrite that release or establish cross-domain superiority,
M7 replication, or end-to-end DSVT TensorRT parity. Zero-intensity, S2, and training remain
unauthorized/not started as recorded in the current operational policy.

## Long-term platform direction — proposed, not authorized implementation

The long-term aim is to make LiDAR perception accessible through a unified, sensor-conscious
framework, analogous in developer experience to computer-vision libraries. Simple discovery and
prediction interfaces should expose their sensor, coordinate, feature, and runtime assumptions
rather than hide incompatibilities. This is a direction, not a claim of universal sensor support.

Future capability layers are roadmap targets:

- universal point-cloud and temporal ingestion, with explicit sensor/dataset adapters;
- canonical coordinate and feature contracts;
- 3D object detection across deliberately integrated upstream backends;
- semantic segmentation and instance/panoptic-style point understanding;
- multi-object tracking;
- learned point/object/scene embeddings and scene-level understanding; and
- eventually language-queryable LiDAR representations, subject to separate research validation.

Detection contracts, raw ingestion, temporal reconstruction, and reproducibility provide existing
foundations. The broader layers above are not implemented platform features. LaserPerception
should own interoperability and a stable developer API while using proven upstream models, not
duplicate whole training frameworks. The [platform RFC](PERCEPTION_PLATFORM_RFC.md) proposes
incremental interfaces around both frozen stacks; each implementation or scientific study still
requires explicit owner scope.

Normal development is CPU-capable and local. GitHub holds tracked source/history, Drive holds
private and large state, and separately selected external GPU workers provide disposable execution
capacity when authorized. Codex Cloud is not required; provider choice remains open. The
[compute workflow](CLOUD_WORKFLOW.md) preserves the full runtime-specific scientific gates.

## Parked semantic-transfer research

The SemanticKITTI-to-DALES Experiment 001 infrastructure remains valuable, tested, and supported.
Its semantic-segmentation model and benchmark remain unimplemented, and all result fields remain
`Pending measurement`. It is not the active detection release line.
