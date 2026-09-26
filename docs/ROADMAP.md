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

## M8 — active

M8 selected DSVT-Pillar with a TransFusion head through P1-E and froze the P1-S1 protocol. See
[PROJECT_STATUS.md](PROJECT_STATUS.md) for the current execution state, authorization boundaries,
frozen identities, and operational blockers. The accepted three-process S1 primary raw
measurement is published in [M8_S1_MEASUREMENT_RAW.md](m8/M8_S1_MEASUREMENT_RAW.md), and its
separate [scientific interpretation](m8/M8_S1_INTERPRETATION.md) is complete. It records the
positive descriptive Car direction and severe Pedestrian transfer failure without causal or
winner claims. The supporting external-runtime ledger is maintained in
[M8_S1_EXTERNAL_RUNTIME_STATUS.md](m8/M8_S1_EXTERNAL_RUNTIME_STATUS.md).
The separate [zero-intensity raw measurement](m8/M8_S1_ZERO_INTENSITY_RAW.md) completed three
accepted processes and 2,568 calls. Its
[scientific interpretation](m8/M8_S1_ZERO_INTENSITY_INTERPRETATION.md) found no Pedestrian recovery
and persistence of the positive H5-over-H10 Car direction, without an intensity-causality claim.
The prospective [S2 scientific protocol](m8/M8_S2_PROTOCOL.md) and S1-derived paired partitions
are frozen. Its [CPU-only input freeze](m8/M8_S2_INPUT_FREEZE.md) records the complete ledger,
exact 1,712/1,712 M7 XYZT identity proof, and fresh-adapter replay. Runtime binding and separate
inference authorization remain pending. S2 detector execution is unstarted.

## Future directions

The next M8 step is external-runtime qualification and binding under the frozen S2 gates;
inference requires separate owner authorization. Other future work includes segmentation-model integration, broader
learned representations, camera fusion, physical-sensor validation, deployment optimization on
specifically available hardware, and productionization. Each requires explicit
scope, evidence gates, and applicable hardware/runtime authorization.
