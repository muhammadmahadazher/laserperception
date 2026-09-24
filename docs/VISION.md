# Vision

LaserPerception aims to make 3D LiDAR perception work inspectable from input bytes through model
contracts, outputs, deployment boundaries, and scientific evidence. The project favors explicit
semantics, deterministic CPU tooling, optional isolated accelerators, and preservation of failed
results over broad unsupported claims.

## Implemented foundation

- A framework-independent `DetectionFrame` with explicit geometry and taxonomy.
- Eight reviewed input adapters with metadata-first discovery and local inspection.
- Model manifests, compatibility validation, deterministic plans, and guarded backend APIs.
- The historical official pretrained PointPillars path with TensorRT FP16 and ROS 2.
- Time-aware multi-sweep reconstruction from model-ready and compatible raw PointCloud2 inputs.
- Deterministic CPU tracking over timed saved detections.
- Immutable semantic point results and CPU confusion/IoU evaluation.
- Provider-neutral worker artifacts, persistence, qualification, and authorization boundaries.

## Active research

M8 studies the official pretrained DSVT-Pillar + TransFusion candidate under a prospectively frozen
S1 protocol. Engineering readiness does not imply scientific completion: the accepted primary A2/E2
comparison is pending. Current details belong in [PROJECT_STATUS.md](PROJECT_STATUS.md) rather than
being duplicated here.

## Future capability

Future work may add a real segmentation model, broader learned representations, camera fusion,
training, hardware-specific optimization, physical-sensor evaluation, and production operations.
None is advertised as present. New work must retain explicit dataset/model licensing, exact input
contracts, bounded claims, and failure evidence.

LaserPerception does not claim SOTA, universal sensor support, production readiness, autonomous
driving safety, or ownership of upstream pretrained models.
