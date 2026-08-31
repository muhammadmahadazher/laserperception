#!/usr/bin/env python3
"""Run the manual source-domain DSVT engineering smoke and repeatability record."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import resource
import statistics
import time
from pathlib import Path

import numpy as np

from laserperception.detection.m8_backend import (
    DsvtBackend,
    dsvt_predictions_to_detection_frame,
)
from laserperception.detection.m8_input import M8PointCloud


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _frame_sha256(frame: dict[str, object]) -> str:
    payload = json.dumps(frame, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--nuscenes-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    backend = DsvtBackend(
        manifest_path=args.manifest,
        upstream_root=args.upstream_root,
        checkpoint_path=args.checkpoint,
    )
    datasets = __import__("pcdet.datasets", fromlist=["build_dataloader"])
    cfg = backend._cfg
    cfg.DATA_CONFIG.DATA_PATH = str(args.nuscenes_root)
    cfg.DATA_CONFIG.VERSION = "v1.0-mini"
    cfg.DATA_CONFIG.INFO_PATH["test"] = ["nuscenes_infos_10sweeps_val.pkl"]
    cfg.DATA_CONFIG.BALANCED_RESAMPLING = False
    cfg.DATA_CONFIG.DATA_PROCESSOR[1].SHUFFLE_ENABLED["test"] = False
    logger = logging.getLogger("laserperception.m8.source_smoke")
    dataset, _, _ = datasets.build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=0,
        logger=logger,
        training=False,
    )
    source = M8PointCloud(np.ascontiguousarray(dataset[0]["points"], dtype=np.float32))
    torch = backend._torch
    with torch.inference_mode():
        for _ in range(2):
            backend._predict_arrays(source.points)
        torch.cuda.reset_peak_memory_stats(0)
        outputs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        frame_hashes: list[str] = []
        wall_ms: list[float] = []
        for repeat in range(10):
            started = time.perf_counter()
            boxes, scores, labels, dropped = backend._predict_arrays(source.points)
            wall_ms.append((time.perf_counter() - started) * 1_000.0)
            if dropped:
                raise RuntimeError("official-domain prepared input unexpectedly lost points")
            outputs.append((boxes, scores, labels))
            frame = dsvt_predictions_to_detection_frame(
                boxes, scores, labels, sample_id=f"source-repeat-{repeat}"
            )
            # The sample ID is intentionally excluded from semantic repeatability.
            semantic = dict(frame.to_dict())
            semantic["sample_id"] = "source"
            frame_hashes.append(_frame_sha256(semantic))

    names = ("pred_boxes", "pred_scores", "pred_labels")
    reference = outputs[0]
    tensor_results: dict[str, object] = {}
    first_differing = None
    for index, name in enumerate(names):
        arrays = [output[index] for output in outputs]
        same_shapes = all(array.shape == arrays[0].shape for array in arrays)
        exact = same_shapes and all(np.array_equal(arrays[0], array) for array in arrays[1:])
        maximum = None
        if same_shapes:
            maximum = max(
                float(
                    np.max(np.abs(array.astype(np.float64) - reference[index].astype(np.float64)))
                )
                if array.size
                else 0.0
                for array in arrays[1:]
            )
        tensor_results[name] = {
            "shape": list(arrays[0].shape),
            "dtype": str(arrays[0].dtype),
            "exact_10_of_10": exact,
            "maximum_absolute_difference": maximum,
            "unique_sha256": len({_sha256(array) for array in arrays}),
        }
        if not exact and first_differing is None:
            first_differing = name

    boxes, scores, labels = outputs[0]
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise RuntimeError("source-domain predictions are not finite")
    if np.any((labels < 1) | (labels > 10)):
        raise RuntimeError("source-domain class ID is outside [1, 10]")
    record = {
        "schema_version": "1.0",
        "status": "m8_phase1_source_domain_engineering_smoke_pass",
        "scientific_accuracy_measurement": False,
        "source_domain_accuracy_rebenchmarked": False,
        "sample": "nuScenes v1.0-mini validation index 0",
        "input": {
            "shape": list(source.points.shape),
            "dtype": str(source.points.dtype),
            "device_at_backend": "cuda:0",
            "sha256": _sha256(source.points),
        },
        "identity": backend.identity,
        "prediction": {
            "count": int(scores.size),
            "boxes_shape": list(boxes.shape),
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
            "scores_finite": True,
            "boxes_finite": True,
            "class_ids_valid": True,
        },
        "repeatability": {
            "repeats": 10,
            "raw_tensors": tensor_results,
            "first_differing_tensor": first_differing,
            "detection_frame_exact_10_of_10": len(set(frame_hashes)) == 1,
            "detection_frame_unique_sha256": len(set(frame_hashes)),
        },
        "resources": {
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            "host_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "single_sample_wall_ms_engineering_context": {
                "mean": statistics.fmean(wall_ms),
                "median": statistics.median(wall_ms),
                "min": min(wall_ms),
                "max": max(wall_ms),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
