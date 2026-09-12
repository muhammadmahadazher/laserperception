# RFC: a sensor-conscious perception platform

Status: **architecture proposal only; not implemented and not scientific authorization**.

This RFC proposes a developer-facing interoperability layer around proven upstream models.
LaserPerception would own stable contracts, reproducibility, model discovery, and a simple
perception API. It would not become a monolithic clone of MMDetection3D or OpenPCDet, duplicate
their training systems, or replace validated model mathematics merely to unify interfaces.

The [vision](VISION.md) describes the long-term direction; the [roadmap](ROADMAP.md) records
authorized work. Every implementation increment below requires separate owner scope. No training,
new detector, segmentation, tracking, language model, service, or scientific run starts here.

## Existing foundations and limits

| Existing foundation | Preserve | Proposed extension, not shipped |
|---|---|---|
| `PointCloud` | Float32 `(N, 3)` XYZ, optional labels, separate attributes and metadata; readers preserve data | Typed sensor/feature/frame descriptors around the existing object |
| `RawSweep`, `MultiSweepBuilder`, `ModelReadyPointCloud` | Exact timestamp arithmetic, source rows, sweep order, transform casts, range semantics | Explicit temporal envelope usable by deliberately integrated adapters |
| `DetectionFrame` / `Detection3D` | Frame, native classes/scores, geometric center, LWH, yaw, optional velocity and deterministic ordering | Versioned result families for other tasks |
| PointPillars release stack | Frozen M1–M7 evidence and accepted GPU/ROS deployment contracts | Compatibility adapter without changing execution |
| M8 DSVT stack | Frozen five-feature input, candidate and checkpoint identities, scientific authorization barriers | Capability/manifest view around the existing backend |
| Dataset and evaluation adapters | Explicit dataset splits, annotation rules, provenance and failure records | Task-specific evaluation interfaces |

M8 provides a selected DSVT-Pillar/TransFusion engineering integration and a partial TensorRT
route; it does not establish end-to-end TensorRT parity or a completed primary S1 comparison.
Parked segmentation adapters are not a segmentation model. Neither existing stack implies
plug-and-play support for arbitrary sensors.

## 1. Input and data layer

Keep `PointCloud` small. An additive typed envelope should describe:

- acquisition identity, timestamp integer value, unit, clock domain, and reference time;
- coordinate-frame identifier, handedness, axes, length units, and transform direction;
- point features by name, dtype, shape, unit, source and missing-value policy, including intensity,
  ring or return identifiers when available, without inventing unavailable features;
- calibration/pose references with version/hash and time validity, rather than embedding a vendor
  SDK object; and
- source dataset/sensor adapter identity and provenance for every explicit transformation.

A proposed temporal container holds an ordered current sweep and historical sweeps with their
individual stamps, source-row identities, and pose references. It must distinguish acquisition
time, arrival time, and model-relative elapsed time. Same frame names at different timestamps do
not imply an identity transform. Motion compensation must name its pose source and reference time;
per-point deskew is a separate, presently unimplemented capability.

Dataset/sensor adapters decode data faithfully. Transforms explicitly convert coordinate bases,
units, and features; they return provenance and source-row mappings whenever rows are filtered.
They must not silently normalize intensity, synthesize timestamps, crop, shuffle, or select history.
Missing required intensity fails validation unless a particular model manifest explicitly permits
a reviewed missing-feature policy. The frozen M8 primary requires raw reflectance; this RFC does
not authorize its separately gated zero-intensity intervention.

Universal ingestion is a target for extensible contracts, not a promise that all sensors can supply
the inputs or calibration a model needs. Do not force the historical nuScenes multi-sweep path
through the parked single-scan abstraction or alter frozen builders to satisfy a new container.

## 2. Model and backend abstraction

Propose a framework-independent, typed `Protocol` with task-specific specializations. Metadata
inspection is CPU-only and must never instantiate a model, import a heavy backend, or enumerate
devices. Execution requires an explicit execution context; construction cannot silently choose a
GPU. An illustrative signature, **not a current public API**, is:

```python
class DetectionBackend(Protocol):
    def describe(self) -> ModelManifest: ...
    def validate_input(self, input: PerceptionInput) -> ValidationReport: ...
    def predict(self, input: PerceptionInput, *, context: ExecutionContext) -> DetectionFrame: ...
    def close(self) -> None: ...
```

Names other than the existing `DetectionFrame` are proposed types. A common metadata protocol
should expose capability declarations; distinct detection, segmentation, embedding, and tracking
protocols should give callers narrow result types. Avoid a single backend method returning an
untyped dictionary or silently guessing the requested task.

Each model manifest should bind an immutable model ID/version to:

- upstream repository/commit, config/checkpoint byte sizes and hashes, source and license notices;
- supported tasks, class taxonomy/version and output schema versions;
- required point feature names/order/dtypes/units, coordinate basis, range, history and ordering;
- preprocessing/postprocessing identities and internal score policy;
- optional framework/runtime requirements and supported execution targets, without probing them;
- deployment completeness (for example, eager detector versus partial engine), precision and
  capacity constraints, with evidence references rather than inferred guarantees; and
- evaluation provenance and known limitations, separate from source-reported model metrics.

Public requests use device-independent contracts and an explicit target identifier. Backend
adapters handle framework objects and device transfers internally. Device-independent API design
does not promise CPU inference for a GPU-only model or equivalent numerical behavior on all
devices. Unsupported inputs, tasks, artifacts or runtime targets fail before model initialization.
Existing M8 authorization checks remain mandatory beneath any future API and CLI wrappers.

## 3. Task layer and typed outputs

Retain `DetectionFrame` unchanged. Add result families only with an owner-approved implementation
and serialization contract:

| Proposed family | Required semantics |
|---|---|
| Semantic point result | Per-source-row class IDs, taxonomy/version, ignored/unknown distinction, optional scores |
| Instance/panoptic point result | Per-row instance membership, semantic classes, explicit unassigned value, thing/stuff policy |
| Track result | Stable IDs within a declared sequence scope, timestamp/frame, source detections, lifecycle state and association provenance |
| Embedding result | Point/object/scene granularity, source identities, model/version, vector dimension, dtype, normalization and similarity convention |
| Scene result | Typed entities and spatial/temporal relationships with evidence references and uncertainty; an extensible future schema |

All families must declare frame, axes, units, sample/sequence identity, and provenance. Geometry
conversions must explicitly preserve the existing right-handed X-forward/Y-left/Z-up basis,
geometric center, length-width-height and counter-clockwise yaw from +X when that contract applies.
Any alternative basis needs an explicit adapter. Never silently reinterpret length and width.

Serialization should use a versioned envelope plus typed payloads, stable ordering and float
encoding, explicit absent values, and no implicit non-finite JSON numbers. Array payloads should
have typed shapes, byte order, hashes and source-row mappings. Schema migrations must be explicit;
new serializers cannot regenerate old evidence or claim identical hashes for changed bytes.
Deterministic serialization means identical input values serialize identically; it does not imply
that a numerically nondeterministic detector produces identical values across runs.

## 4. Pipeline and entry points

```text
ingest -> explicit transform -> validated model input -> backend
       -> declared postprocess -> optional tracker -> export
```

A future Python API composes those stages using typed contracts. A CLI should invoke that same
pipeline, not duplicate its model, voxelization or postprocessing logic. A ROS 2 adapter should
preserve acquisition timestamps, time-aware TF and source headers while delegating the core work.
A future service/API boundary could serialize the same versioned requests/results, with explicit
resource limits, cancellation and failure accounting; no network service is implemented here.

Distinguish backend-owned model postprocessing from optional display/export filtering. Record
thresholds and transformations without letting presentation settings redefine benchmark execution.
An optional tracker consumes detections through a separately versioned association policy and
cannot modify detector evidence. Streaming implementations must define bounded history, ordering,
backpressure and dropped-input accounting; these are future designs, not new ROS behavior.

## 5. Model registry and discovery

The registry should initially be static, reviewed manifests distributed with source or explicitly
selected by the user. Discovery reads metadata without downloads, framework imports or hardware
inspection. Model IDs resolve to exact versions, never silently to changing checkpoint contents.
Weight retrieval must be explicit, license-aware, external to the core wheel and hash-verified.
Manifest metadata must not execute arbitrary code or install arbitrary dependencies.

Illustrative future CLI commands — **proposed, not implemented**:

```text
laserperception models list
laserperception inspect <model>
laserperception predict ...
```

`inspect` would explain required features, classes, runtime requirements, artifact provenance and
limitations before execution. `predict` would require an explicit model/target and preserve all
applicable runtime/scientific authorization gates. Model availability is not execution permission.

## 6. Evaluation and reproducibility

Dataset-specific adapters define splits, sensor conventions, annotation eligibility, ignored
classes and incomplete-label limitations. Task-specific evaluators own metrics: detection matching,
segmentation labels, instance matching, tracking identity continuity or embedding retrieval.
Do not reuse one task's metric as a substitute for another or imply an official benchmark score
from a benchmark-inspired adapter. Future evaluator implementations require their own review.

Every evaluated run should bind repository/config/model/input/evaluator versions and hashes,
runtime identity, precision, feature transformations, seed/policy state, thresholds, timing
boundaries and failure/call accounting. Repeated runtime realizations are distinct from independent
dataset samples. Preserve each accepted and failed attempt and keep raw evidence separate from
interpretation. Frozen M6/M7/M8 protocols remain authoritative over proposed general abstractions.

## 7. Remote compute boundary

Core installation, imports, manifest inspection and synthetic contract tests remain CPU-only.
Heavy framework backends are optional and isolated; GPUs are explicitly authorized execution
targets, not core dependencies. Follow the [compute workflow](CLOUD_WORKFLOW.md): local development,
GitHub tracked state, Drive private/large state, and disposable external GPU workers.

Transport integrity and scientific authorization are separate checks. A runtime must be newly
qualified and bound before its authorized measurements; the retired runtime's primary authorization
cannot be reused. Neither a model registry nor an execution context can bypass this boundary.
Provider selection, credentials, and workspace roots remain external configuration.

## 8. Language-queryable LiDAR — research hypothesis

A possible long-term path combines learned point/object/scene embeddings, a structured scene
representation with spatial and temporal references, and a language-alignment layer for retrieval
or grounded queries. A query result would need to identify supporting points/objects, model
provenance and uncertainty rather than invent semantic certainty from a language response.

Whether such representations generalize across sensor characteristics, domains and query wording
is an open research question. Grounding, retrieval quality, hallucination, calibration, licensing
and annotation requirements would need a separately preregistered evaluation. No embeddings,
language alignment, language model training or query service are implemented or authorized here.

## 9. Incremental migration and review gates

1. **Inventory, then metadata:** owner-scope a manifest/capability design against existing contracts.
   Prove discovery imports no GPU frameworks and performs no device discovery or network action.
2. **Thin detection adapters:** wrap existing PointPillars and DSVT entry points without changing
   their frozen feature construction or model execution. CPU fixtures test coordinate/feature
   validation and missing-dependency failures. Any real parity gate needs separate authorization.
3. **Shared developer entry points:** introduce a Python/CLI layer only after adapter contract and
   failure-path review; preserve historical wrappers and scientific runners as versioned paths.
4. **Additional tasks:** separately scope typed result families, models and evaluations. Do not
   bundle segmentation, tracking, embeddings or training into an interface cleanup.
5. **Services or language research:** proceed only after stable contracts and explicit owner scope,
   with their own reproducibility and evaluation criteria.

Adoption should be additive and reversible: retain original entry points and frozen manifests,
compare adapter behavior using CPU fixtures first, and leave historical scientific code untouched
unless a separately reviewed need justifies a change. Architecture consistency is not a reason to
rewrite frozen evidence code. No dates, universal support, performance or accuracy promises follow
from this RFC.
