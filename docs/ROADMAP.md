# Roadmap

Current operational state is centralized in [PROJECT_STATUS.md](PROJECT_STATUS.md). This roadmap
records milestone order without rewriting frozen protocols or evidence.

| Milestone | Status | Scope |
|---|---|---|
| M1 | Historical, complete | Official pretrained PointPillars FP32 integration and evidence |
| M2 | Historical, complete | TensorRT FP16 parity and repaired performance comparison |
| M3 | Historical, complete | ROS 2 deployment and deterministic exact-fast voxelization |
| M4.5 | Historical, complete | NumPy multi-sweep builder and raw PointCloud2/tf2 boundary |
| M5 | Conditional, inactive | Jetson work requires hardware and separate activation |
| M6 | Historical, complete | KITTI Raw cross-domain and projected-reference ROS validation |
| M7 | Historical, complete | Controlled history-mechanism measurement |
| M8 | Active | DSVT-Pillar + TransFusion Detector V2 research |

## M6 — complete

M6 remains closed and historical. Its final projected-reference ROS integration reproduced
860/860 unique live conditions exactly while preserving the original R2 failure and diagnosis.
M5 remains conditional and inactive; completing M6 did not activate it.

## v0.4.0 platform release

v0.4.0 releases engineering already merged since v0.3.0:

- P0 model manifests, registry, exact input contracts, validation, and CPU dry-run planning;
- P1 guarded PointPillars and DSVT backend construction;
- P2 deterministic CPU multi-object tracking over timed saved detections;
- P3 immutable semantic results and CPU confusion/IoU evaluation;
- P4 eight reviewed data-adapter paths and local inspection;
- provider-neutral worker and Drive-backed persistence tooling; and
- M8 external-runtime readiness and fail-closed preflight/evidence improvements.

## M8 current S1 status

The implementation is ready for the frozen campaign, but the accepted primary measurement is
pending. External A40 qualification and a fresh 10-process/140-call Stage R completed. A later
primary attempt ended incomplete after 779 attempted conditions and contributed zero accepted
canonical calls. PRs #41–#45 corrected replay, revalidation, and failure-evidence behavior.

Future execution remains bound to commit
`6994d72c3e7691a86116d1417ac3ae08256d163f` and receipt SHA256
`bef4c55575581aefe8f477e32d1b394f40823a0b0858c66c3ac5c1fae141ec4d`. Current attempts are
blocked by external A40 capacity, not missing scientific code. The v0.4.0 release does not rebind
the campaign.

Zero-intensity is unauthorized. S2 and training have not started. No dates are promised.

## Future directions

Potential future work includes completing M8 primary measurement, segmentation-model integration,
broader learned representations, camera fusion, physical-sensor validation, deployment
optimization on specifically available hardware, and productionization. Each requires explicit
scope, evidence gates, and applicable hardware/runtime authorization.
