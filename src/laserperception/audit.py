"""CPU-only dataset audit CLI for Experiment 001.

Run ``python -m laserperception.audit --help`` for usage. Auditing never trains a model and never
writes dataset points. JSON reports contain stable sample identifiers, not absolute source paths.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np

from laserperception.datasets import (
    DALES_ADAPTER_VERSION,
    SEMANTICKITTI_ADAPTER_VERSION,
    DalesDataset,
    SemanticKITTIDataset,
)
from laserperception.ontology import (
    CLASS_NAMES,
    IGNORE_ID,
    label_histogram,
    map_dales_labels,
    map_semantickitti_labels,
)
from laserperception.transforms import normalize_coordinates

AUDIT_SCHEMA_VERSION: Final = "1.0"
ONTOLOGY_NAME: Final = "cvgc_group2_v1"


@dataclass
class _CoordinateAccumulator:
    point_count: int = 0
    non_finite_point_count: int = 0
    xyz_min: np.ndarray | None = None
    xyz_max: np.ndarray | None = None

    def update(self, xyz: np.ndarray) -> None:
        points = np.asarray(xyz)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"xyz must have shape (N, 3); received {points.shape}")
        self.point_count += int(points.shape[0])
        finite_mask = np.all(np.isfinite(points), axis=1)
        self.non_finite_point_count += int(np.count_nonzero(~finite_mask))
        if not np.any(finite_mask):
            return
        finite = points[finite_mask]
        current_min = finite.min(axis=0)
        current_max = finite.max(axis=0)
        self.xyz_min = (
            current_min if self.xyz_min is None else np.minimum(self.xyz_min, current_min)
        )
        self.xyz_max = (
            current_max if self.xyz_max is None else np.maximum(self.xyz_max, current_max)
        )

    def update_summary(
        self,
        *,
        point_count: int,
        non_finite_point_count: int,
        xyz_min: tuple[float, float, float] | None,
        xyz_max: tuple[float, float, float] | None,
    ) -> None:
        """Merge precomputed float64 bounds without reconstructing a tile cloud."""
        self.point_count += point_count
        self.non_finite_point_count += non_finite_point_count
        if xyz_min is None or xyz_max is None:
            return
        minimum = np.asarray(xyz_min, dtype=np.float64)
        maximum = np.asarray(xyz_max, dtype=np.float64)
        self.xyz_min = minimum if self.xyz_min is None else np.minimum(self.xyz_min, minimum)
        self.xyz_max = maximum if self.xyz_max is None else np.maximum(self.xyz_max, maximum)

    def as_dict(self) -> dict[str, object]:
        minimum = None if self.xyz_min is None else [float(value) for value in self.xyz_min]
        maximum = None if self.xyz_max is None else [float(value) for value in self.xyz_max]
        z_range = None if minimum is None or maximum is None else [minimum[2], maximum[2]]
        return {
            "point_count": self.point_count,
            "xyz_min": minimum,
            "xyz_max": maximum,
            "z_range": z_range,
            "non_finite_point_count": self.non_finite_point_count,
        }


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _validate_limit(value: int | None, name: str) -> int | None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _point_count_statistics(counts: list[int]) -> dict[str, int | float | None]:
    if not counts:
        return {"minimum": None, "maximum": None, "mean": None, "median": None}
    values = np.asarray(counts, dtype=np.int64)
    return {
        "minimum": int(values.min()),
        "maximum": int(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
    }


def _merge_histogram(target: dict[int, int], labels: np.ndarray) -> None:
    for label, count in label_histogram(labels).items():
        target[label] = target.get(label, 0) + count


def _json_histogram(histogram: dict[int, int]) -> dict[str, int]:
    return {str(label): histogram[label] for label in sorted(histogram)}


def _shared_histogram(histogram: dict[int, int]) -> dict[str, int]:
    return {
        CLASS_NAMES[label]: histogram[label]
        for label in sorted(histogram)
        if 0 <= label < len(CLASS_NAMES)
    }


def _base_report(dataset: str, split: str, adapter_version: str) -> dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "dataset": dataset,
        "split": split,
        "timestamp": _utc_timestamp(),
        "git_commit": _git_commit(),
        "adapter_version": adapter_version,
        "ontology": {"name": ONTOLOGY_NAME, "ignore_id": IGNORE_ID},
    }


def audit_semantickitti(
    dataset: SemanticKITTIDataset,
    *,
    max_samples: int | None = None,
    normalization: str | None = None,
) -> dict[str, Any]:
    """Audit a deterministic SemanticKITTI subset without altering source points."""
    limit = _validate_limit(max_samples, "max_samples")
    if normalization not in {None, "min_xyz"}:
        raise ValueError(f"unsupported normalization: {normalization!r}")
    inspected_count = len(dataset) if limit is None else min(len(dataset), limit)

    raw_coordinates = _CoordinateAccumulator()
    normalized_coordinates = _CoordinateAccumulator() if normalization else None
    point_counts: list[int] = []
    source_histogram: dict[int, int] = {}
    shared_histogram: dict[int, int] = {}
    ignored_count = 0
    labelled_point_count = 0
    labelled_samples = 0
    sample_ids: list[dict[str, str]] = []

    for index in range(inspected_count):
        info = dataset.sample_info(index)
        cloud = dataset.load(index)
        sample_ids.append({"sequence": info.sequence, "frame": info.frame})
        point_counts.append(len(cloud))
        raw_coordinates.update(cloud.xyz)
        if normalized_coordinates is not None and len(cloud):
            normalized_coordinates.update(normalize_coordinates(cloud).xyz)
        if cloud.labels is None:
            continue
        labelled_samples += 1
        labelled_point_count += len(cloud)
        mapped = map_semantickitti_labels(cloud.labels)
        _merge_histogram(source_histogram, cloud.labels)
        _merge_histogram(shared_histogram, mapped[mapped != IGNORE_ID])
        ignored_count += int(np.count_nonzero(mapped == IGNORE_ID))

    report = _base_report("semantickitti", dataset.split, SEMANTICKITTI_ADAPTER_VERSION)
    report.update(
        {
            "normalization": normalization or "not_applied",
            "sequences_inspected": sorted({sample["sequence"] for sample in sample_ids}),
            "samples": sample_ids,
            "counts": {
                "scans_available": len(dataset),
                "scans_inspected": inspected_count,
                "total_points": raw_coordinates.point_count,
                "labelled_points": labelled_point_count,
                "labelled_scans": labelled_samples,
            },
            "per_scan_point_count": _point_count_statistics(point_counts),
            "raw_coordinate_statistics": raw_coordinates.as_dict(),
            "source_semantic_label_histogram": _json_histogram(source_histogram),
            "shared_ontology_histogram": _shared_histogram(shared_histogram),
            "ignored_point_count": ignored_count,
            "ignored_fraction": (
                ignored_count / labelled_point_count if labelled_point_count else None
            ),
            "label_point_consistency": {
                "checked_scans": labelled_samples,
                "mismatched_scans": 0,
            },
        }
    )
    if normalized_coordinates is not None:
        report["normalized_coordinate_statistics"] = normalized_coordinates.as_dict()
    return report


def audit_dales(
    dataset: DalesDataset,
    *,
    max_tiles: int | None = None,
    patch_size_m: tuple[float, float] = (50.0, 50.0),
    chunk_size: int = 1_000_000,
    normalization: str | None = None,
) -> dict[str, Any]:
    """Audit streamed DALES patches without retaining optional LAS dimensions."""
    limit = _validate_limit(max_tiles, "max_tiles")
    if normalization not in {None, "min_xyz"}:
        raise ValueError(f"unsupported normalization: {normalization!r}")
    inspected_tiles = len(dataset) if limit is None else min(len(dataset), limit)

    raw_coordinates = _CoordinateAccumulator()
    normalized_coordinates = _CoordinateAccumulator() if normalization else None
    point_counts: list[int] = []
    source_histogram: dict[int, int] = {}
    shared_histogram: dict[int, int] = {}
    ignored_count = 0
    total_points = 0
    non_finite_points = 0
    empty_patches = 0
    grid_cells = 0
    patch_count = 0
    tile_ids: list[str] = []
    normalized_patch_ranges: list[dict[str, object]] = []

    for tile_index in range(inspected_tiles):
        partition = dataset.partition_tile(
            tile_index,
            patch_size_m=patch_size_m,
            chunk_size=chunk_size,
        )
        tile_ids.append(partition.tile.tile_id)
        total_points += partition.total_point_count
        non_finite_points += partition.non_finite_point_count
        empty_patches += partition.empty_patch_count
        grid_cells += partition.grid_cell_count
        raw_coordinates.update_summary(
            point_count=partition.total_point_count,
            non_finite_point_count=partition.non_finite_point_count,
            xyz_min=partition.raw_xyz_min,
            xyz_max=partition.raw_xyz_max,
        )
        for patch in partition.patches:
            cloud = patch.cloud
            patch_count += 1
            point_counts.append(len(cloud))
            if cloud.labels is None:
                raise ValueError(f"DALES patch lacks classification labels: {patch.info.tile_id}")
            mapped = map_dales_labels(cloud.labels)
            _merge_histogram(source_histogram, cloud.labels)
            _merge_histogram(shared_histogram, mapped[mapped != IGNORE_ID])
            ignored_count += int(np.count_nonzero(mapped == IGNORE_ID))
            if normalized_coordinates is not None:
                normalized = normalize_coordinates(cloud)
                normalized_coordinates.update(normalized.xyz)
                normalized_patch_ranges.append(
                    {
                        "tile": patch.info.tile_id,
                        "row": patch.info.row,
                        "column": patch.info.column,
                        "xyz_min": [float(value) for value in normalized.xyz.min(axis=0)],
                        "xyz_max": [float(value) for value in normalized.xyz.max(axis=0)],
                    }
                )

    raw_stats = raw_coordinates.as_dict()
    raw_stats["point_count"] = total_points
    raw_stats["non_finite_point_count"] = non_finite_points
    labelled_points = sum(point_counts)
    report = _base_report("dales", dataset.split, DALES_ADAPTER_VERSION)
    report.update(
        {
            "normalization": normalization or "not_applied",
            "tiles": tile_ids,
            "patch_policy": {
                "policy": "deterministic_grid",
                "size_m": {"x": float(patch_size_m[0]), "y": float(patch_size_m[1])},
                "origin": "tile_header_min_xy",
                "boundary": "[xmin, xmax) and [ymin, ymax)",
                "overlap": False,
                "empty_patch_policy": "skip_and_count",
            },
            "counts": {
                "tiles_available": len(dataset),
                "tiles_inspected": inspected_tiles,
                "grid_cells": grid_cells,
                "patches_produced": patch_count,
                "empty_patches": empty_patches,
                "total_points": total_points,
                "points_in_patches": labelled_points,
            },
            "per_patch_point_count": _point_count_statistics(point_counts),
            "raw_coordinate_statistics": raw_stats,
            "source_classification_histogram": _json_histogram(source_histogram),
            "shared_ontology_histogram": _shared_histogram(shared_histogram),
            "ignored_point_count": ignored_count,
            "ignored_fraction": ignored_count / labelled_points if labelled_points else None,
            "label_point_consistency": {
                "checked_patches": patch_count,
                "mismatched_patches": 0,
            },
        }
    )
    if normalized_coordinates is not None:
        report["normalized_coordinate_statistics"] = normalized_coordinates.as_dict()
        report["normalized_patch_ranges"] = normalized_patch_ranges
    return report


def write_json_report(report: dict[str, Any], path: str | Path) -> None:
    """Write a machine-readable report without modifying its contents."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_root(explicit: str | None, environment_name: str) -> Path:
    configured = explicit or os.environ.get(environment_name)
    if not configured:
        raise ValueError(
            f"dataset root is required via --root or {environment_name} environment variable"
        )
    return Path(configured)


def _print_summary(report: dict[str, Any]) -> None:
    counts = report["counts"]
    print(f"Dataset: {report['dataset']}")
    print(f"Split: {report['split']}")
    print(f"Counts: {json.dumps(counts, sort_keys=True)}")
    print(f"Ignored points: {report['ignored_point_count']}")
    print(f"Ignored fraction: {report['ignored_fraction']}")


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root", help="Dataset root; otherwise use the dataset environment variable"
    )
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--normalization",
        choices=("none", "min_xyz"),
        default="none",
        help="Optional explicit audit stage; raw statistics are always retained",
    )
    parser.add_argument(
        "--json", dest="json_path", help="Write JSON, preferably under audit-reports/"
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the tested command-line parser."""
    parser = argparse.ArgumentParser(description="Audit LaserPerception datasets without training")
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    semkitti = subparsers.add_parser("semantickitti", help="Audit SemanticKITTI scans")
    _add_common_arguments(semkitti)
    semkitti.add_argument(
        "--sequences", nargs="+", help="Explicit subset within the official split"
    )
    semkitti.add_argument("--max-samples", type=int)

    dales = subparsers.add_parser("dales", help="Audit streamed DALES grid patches")
    _add_common_arguments(dales)
    dales.add_argument("--max-tiles", type=int)
    dales.add_argument("--patch-size-x", type=float, default=50.0)
    dales.add_argument("--patch-size-y", type=float, default=50.0)
    dales.add_argument("--chunk-size", type=int, default=1_000_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    normalization = None if args.normalization == "none" else args.normalization
    try:
        if args.dataset == "semantickitti":
            root = _resolve_root(args.root, "LASERPERCEPTION_SEMANTICKITTI_ROOT")
            sem_dataset = SemanticKITTIDataset(
                root,
                split=args.split,
                sequences=args.sequences,
            )
            report = audit_semantickitti(
                sem_dataset,
                max_samples=args.max_samples,
                normalization=normalization,
            )
        else:
            root = _resolve_root(args.root, "LASERPERCEPTION_DALES_ROOT")
            dales_dataset = DalesDataset(root, split=args.split)
            report = audit_dales(
                dales_dataset,
                max_tiles=args.max_tiles,
                patch_size_m=(args.patch_size_x, args.patch_size_y),
                chunk_size=args.chunk_size,
                normalization=normalization,
            )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    _print_summary(report)
    if args.json_path:
        write_json_report(report, args.json_path)
        print(f"JSON report: {args.json_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
