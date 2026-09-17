# Data adapters and CPU ingestion

P4 unifies discovery of implemented readers. It adds wrappers and metadata, with no new sensor SDK,
universal format claim, model initialization or scientific experiment. Existing readers stay intact.
`laserperception.data.registry` has deterministic list/get/inspect and feature/task filters; duplicate
IDs fail. DataAdapterManifest and the DataAdapter protocol describe raw input capability rather than
execution permission. Listing and manifest inspection never load data or import optional frameworks.

| Adapter | Actual input/API | Preserved output and limitations |
|---|---|---|
| kitti-velodyne | Four-column little-endian float32 .bin | PointCloud XYZ/remission; optional explicit packed semantic/instance label file; no clock |
| semantickitti | Official split/subset sequences directory | Existing numeric ordering and native labels; point scans lazy; no clock/pose/calibration loader |
| las | .las through existing laspy reader | Scaled float32 XYZ, classification and all stored dimensions; CRS retained; axes/units unverified |
| laz | .laz through the same reader | Same data semantics; optional decompressor required |
| dales | Explicit train/test directory or explicitly selected LAS/LAZ tile | Full tile retains attributes; existing streaming chunks retain XYZ/classification only |
| kitti-raw | Synchronized drive plus explicit date root | Native XYZ/remission, exact nanoseconds; calibration/OXTS read at construction, points lazy |
| nuscenes-lidar-top | Five-float raw .bin with explicit source ID/time | XYZ/intensity/ring_index; no directory discovery or implicit multi-sweep reconstruction |
| pointcloud2-xyz | PointCloud2Layout plus separate SourceHeader | API-only raw XYZ boundary; finite rows retained in order; original source-row indices unavailable |

Stored LAS dimensions depend on its point format. LAS GPS time alone does not establish acquisition
clock semantics. DALES identity/taxonomy is explicit; a generic LAS file does not imply DALES labels.
The nuScenes fifth column is ring_index, never time_lag. KITTI remission remains named remission.
No wrapper normalizes, fabricates features/timestamps, sorts points or applies model-axis rotations.
Historical multi-sweep builders and scientific frozen-input paths remain separate and unchanged.

## Discovery, loading and inspection

```console
laserperception data adapters list --json
laserperception data adapters inspect kitti-raw --json
laserperception data inspect sample.las --model dsvt-pillar-transfusion-m8 --json
laserperception data inspect scan.bin --adapter kitti-velodyne --label-path scan.label --json
laserperception data inspect DATA_ROOT --adapter semantickitti --split valid --sequence 08 --index 0 --json
laserperception data inspect DATA_ROOT --adapter dales --split train --index 0 --json
laserperception data inspect DRIVE_ROOT --adapter kitti-raw --date-root DATE_ROOT --index 10 --json
laserperception data inspect scan.bin --adapter nuscenes-lidar-top --timestamp-microseconds 123456 --sample-id acquisition-001 --json
```

Automatic selection accepts only .las/.laz as generic formats. A .bin file is ambiguous regardless of
filename or byte divisibility: supply an adapter. Other extensions/directories require explicit
registered context. There is no PCD, rosbag, camera, vendor SDK or arbitrary dataset reader.

`load_input(path, adapter_id=..., options=InputOptions(...))` returns a typed LoadedInput reusing the
canonical PointCloud. FileDataAdapter implements the protocol. `inspect_input` and
`inspect_loaded_input` report actual dtypes/shapes, labels, count, coordinates/time facts and optional
model compatibility through the existing `validate_input` function. Unsupported FeatureSpec dtypes
stay in attribute metadata without coercion. Omitted full coordinate contracts remain explicitly
unverified, so `compatibility_verified` is false. Raw scans lack prepared time_lag/history and are not
execution-ready for either current detector manifest. Metadata validation never imports a detector.

## Lazy sequences and composition

`data.sequences.SemanticKITTISequence`, `DalesSequence` and `KittiRawDataSequence` wrap the existing
readers behind DatasetSequence/SampleRef. Iterating references does not load points. Indices are
bounded non-negative integers. DALES sample IDs include split and relative path, avoiding equal-stem
collisions. `DalesSequence.iter_chunks()` exposes exact contiguous source_row_start/source_row_stop;
it keeps the reader's XYZ/classification-only behavior. Spatial patches still require explicit
patch-local identities or known original row maps. PointCloud2 finite filtering does not provide an
original source-row map; counts must not be used to invent one.

Adapters compose with semantic.datasets.ground_truth_from_point_cloud and evaluate_semantics. Existing
precomputed DetectionFrames independently compose with track_sequence; ingestion does not run a
detector. `examples/perception_cpu_journey.py` creates reusable synthetic fixtures and exercises
inspection, semantic evaluation and tracking. See the [CPU quickstart](QUICKSTART_PERCEPTION.md).
