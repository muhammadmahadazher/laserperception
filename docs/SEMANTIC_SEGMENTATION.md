# Semantic point results and CPU evaluation

P3 implements semantic result/evaluation infrastructure. There is no production segmentation model,
training run, accuracy claim or new scientific experiment. Existing Experiment 001 mappings remain
unchanged and explicitly group selected native dataset labels into its geometry-only six-class policy.
Unrelated taxonomies are never automatically equated.

## Result and taxonomy contracts

`SemanticPointFrame` is immutable. It records sample/frame IDs, the existing CoordinateContract,
source-point count, signed int64 class IDs, complete SemanticTaxonomy identity/version/definitions,
SemanticProvenance and optional float64 confidence in [0,1]. Arrays are defensive bytes-backed copies;
callers cannot make them writable. Unknown prediction IDs are retained as supplied, not remapped.
Custom taxonomies may use noncontiguous native IDs. SemanticClass declares ID, canonical name and
ignored status; unknown_id is a separate reserved unknown marker. All undeclared predictions count
as unknown. Native taxonomies require explicit complete definitions from the caller.

Without a source map, result length must equal source-point count and rows mean 0..N-1. Filtered
results require bounded unique strictly increasing source_rows. No helper silently sorts rows or
transforms coordinates. Point-only contracts should mark irrelevant box/yaw conventions as
`not_applicable`; sensor axes/units remain an explicit caller declaration.

## Exact evaluation rules

`evaluate_semantics(prediction, ground_truth)` first checks sample ID, frame ID, complete coordinate
contract, source count, complete taxonomy and exact source rows. Different filtered subsets fail.
GT labels must be evaluable or explicitly ignored; unknown GT fails. Ignored GT rows do not contribute
to confusion, accuracy or prediction-unknown counts. Ignored/undeclared predictions on valid GT
contribute to a final unknown column and count as false negatives and incorrect predictions.

The confusion matrix is K by K+1: taxonomy-sorted evaluable GT classes as rows and the same prediction
classes plus unknown as columns. Per-class TP is the diagonal; FP is column sum minus TP; FN is row
sum minus TP; support is row sum. IoU=TP/(TP+FP+FN). Zero union yields None and its class ID is explicitly
listed as excluded from mIoU. Classes absent in GT but receiving false positives have IoU=0 and remain
in the mean. Accuracy is total TP / valid GT rows. Empty/all-ignored input yields None for mIoU and
accuracy. These are generic semantic metrics, not an official benchmark submission or score.

## Persistence and CLI

`save_semantic_frame(frame, "result.json")` creates int64 labels/rows and optional float64 confidence
NPY sidecars first, then a `laserperception.semantic-point-frame.v1` JSON envelope. References contain
portable relative paths, dtype, shape, byte size and full-file SHA256. Loaders reject traversal/symlink
escapes, changed bytes, mismatched dtype/shape, pickle and trailing bytes. Existing files are not
overwritten; incomplete write files remain available for diagnosis. Storage does not download inputs.

`frame.to_json()` and `save_semantic_frame(..., inline=True)` allow at most 4096 inline rows. JSON rejects
unknown fields, duplicate keys and non-finite values. Large results require sidecars.

```console
laserperception semantic inspect result.json --json
laserperception semantic evaluate prediction.json ground-truth.json --json
python examples/semantic_evaluation.py
```

The typed evaluation record is deterministic JSON and includes all excluded classes and unknown
counts. Inspection reports identities and counts without dumping large per-point arrays.

## Existing reader integration

`semantic.datasets.ground_truth_from_point_cloud()` retains raw labels unless an explicit mapping is
selected. `semantickitti_ground_truth(dataset, index, frame_id=..., coordinates=...)` uses the existing
SemanticKITTIDataset/KITTI reader, records sequence/frame sample identity and explicitly maps native
IDs. `dales_ground_truth(tile_path, ...)` uses the full LAS reader and DALES mapping. Neither mutates
reader output. Missing labels fail clearly.

Full tiles retain original rows. Existing spatial patches do not carry original tile mappings: use
an explicit patch-local sample identity or supply a known source map; never infer one from lengths.
The mappings exclude unsupported native labels as the historical IGNORE_ID=-1. No datasets are
redistributed or downloaded. Synthetic fixtures test perfect/wrong/ignored/unknown predictions,
alignment, exact metrics, immutable arrays, sidecar identity, both readers and CLI import boundaries.
