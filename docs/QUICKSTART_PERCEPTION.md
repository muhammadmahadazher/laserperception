# Perception platform quickstart

LaserPerception exposes reviewed model metadata, exact LiDAR input contracts, deterministic dry-run
plans, and an explicitly gated detection API. These operations are CPU-safe until a caller invokes
`predict()` with a complete execution context, owner authorization, and external artifacts.

The metadata examples below inspect and plan work; the final journey also performs real CPU tracking
and semantic evaluation of synthetic/precomputed results. They do not discover hardware, import a detector framework,
download weights, or run inference.

## 1. List reviewed models

```console
laserperception models list
```

The current registry contains the historical nuScenes PointPillars model and the M8 DSVT-Pillar with
TransFusion research candidate. Registry presence describes a reviewed identity; it is not execution
authorization.

Use deterministic JSON when integrating with tooling:

```console
laserperception models list --json
laserperception models inspect dsvt-pillar-transfusion-m8 --json
```

## 2. Inspect and validate an input contract

An `InputDescription` carries complete ordered feature metadata plus coordinate and temporal
contracts. It contains no point bytes, so it is suitable for planning.

```python
from laserperception.perception import InputDescription, load_model

model = load_model("pointpillars-nuscenes-v0.3")
manifest = model.describe()
input_description = InputDescription(
    schema_version="1.0",
    sample_id="planned-sample",
    frame_id="lidar",
    payload_kind="model_ready_xyzt",
    features=manifest.input_features,
    coordinates=manifest.coordinates,
    temporal=manifest.temporal,
    source="explicit model-ready input",
)

report = model.validate(input_description)
assert report.valid
```

Validation is exact for execution: feature order, feature semantics, coordinates, temporal policy,
and payload kind must match the selected thin adapter. It does not synthesize intensity or time
lag, reorder points, or transform coordinates.

Save the description for the CLI:

```python
from pathlib import Path

Path("pointpillars-input.json").write_text(input_description.to_json(), encoding="utf-8")
```

## 3. Produce a dry-run execution plan

Every target field is explicit. The following command describes a future external runtime; it does
not inspect this computer or contact that runtime.

```console
laserperception predict \
  --model pointpillars-nuscenes-v0.3 \
  --input pointpillars-input.json \
  --runtime-target external_cuda \
  --precision fp32 \
  --device cuda:0 \
  --runtime-id planned-external-worker \
  --task-id planned-prediction \
  --execution-commit aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --deterministic \
  --dry-run
```

The JSON plan reports:

- input, coordinate, temporal, runtime, precision, and device compatibility;
- expected optional dependencies and model artifacts;
- whether an authorization reference was supplied;
- every static blocker;
- `hardware_probed: false`, unverified dependency/artifact states, and `executable: false`.

Dry run never claims live executability. `ready_for_guarded_initialization` means only that metadata
references are complete enough for `predict()` to begin strict file and authorization checks.

## 4. Understand the execution boundary

`PerceptionInput` wraps an existing model-ready payload by reference. It does not copy the point
array merely to satisfy the generic API.

```python
from laserperception.perception import PerceptionInput

# model_ready_cloud is an existing validated ModelReadyPointCloud.
request = PerceptionInput(input_description, model_ready_cloud)
```

Actual PointPillars prediction additionally requires:

- an explicitly selected external CUDA runtime and exact `cuda:0` device;
- the eager FP32 deterministic route supported by this P1 adapter;
- the accepted config and checkpoint byte identities, plus a clean upstream checkout at the
  pinned MMDetection3D commit with its config at the recorded relative path;
- an owner authorization bound to runtime, task, commit, device, precision, input description,
  exact input payload, config, checkpoint, and upstream commit; and
- the isolated pinned MMDetection3D runtime.

Supply the checkout through `ModelResources.upstream_root` (or `--upstream-root` when planning),
with explicit config and checkpoint paths. No checkout is searched for or downloaded.

Only after these checks does the adapter import and delegate to
`Mmdet3dBackend.prepare_model_ready_points()` and `run_prepared()`. It returns that exact
`DetectionFrame`; presentation filtering remains separate.

M8 DSVT discovery, validation, and planning are available. Generic DSVT execution deliberately
fails closed. The existing frozen S1 runner must first own static binding checks, exact owner
authorization, live runtime-policy verification, canonical input selection, and `AtomicAttempt`
call accounting before it can supply an accounted session. An authorization path alone never
unlocks DSVT.

The fake backend lives under `laserperception.perception.backends.fake` for CPU tests and examples.
It returns an empty frame marked `non_scientific` and is not part of the reviewed model registry.

For external artifact preparation and qualification planning, see
[External workers](EXTERNAL_WORKERS.md). Scientific execution still requires fresh authorization
for the exact selected runtime.
## 5. Discover ingestion and inspect a real local input

```console
laserperception data adapters list
laserperception data adapters inspect semantickitti --json
laserperception data inspect sample.las --model dsvt-pillar-transfusion-m8 --json
laserperception data inspect scan.bin --adapter kitti-velodyne --json
```

No detector is loaded. Inspection reports actual stored features and labels; missing time_lag,
unverified coordinates and single-scan history are preparation blockers. It does not manufacture
model compatibility. See [data adapters](DATA_ADAPTERS.md) for explicit dataset options and sequences.

## 6. Run a complete synthetic CPU journey

Create a new Drive-backed local output directory for a small reusable synthetic fixture:

```console
python examples/perception_cpu_journey.py .local/cpu-example
laserperception data inspect .local/cpu-example/synthetic-dales.las --json
laserperception track .local/cpu-example/detections.jsonl --sequence-id synthetic-journey --json
laserperception semantic inspect .local/cpu-example/prediction.json --json
laserperception semantic evaluate .local/cpu-example/prediction.json .local/cpu-example/ground-truth.json --json
```

The script requires a fresh directory and preserves generated LAS/result/JSONL files. It demonstrates
adapter-to-PointCloud-to-explicit-semantic-GT evaluation, alongside precomputed DetectionFrame-to-tracker
composition. Synthetic predictions are fixtures, not a segmentation model or measured benchmark.
All timestamps, sample identities and coordinate declarations are explicit. No GPU is needed.
Tracking consumes the versioned timed envelope around the existing DetectionFrame representation.
Semantic evaluation rejects mismatched sample/frame/taxonomy/source-row identities before metrics.
For your own results, see [tracking](TRACKING.md) and [semantic evaluation](SEMANTIC_SEGMENTATION.md).
