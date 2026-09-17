"""Explicit label mapping helpers; existing raw readers stay unchanged."""

from pathlib import Path
from typing import Literal

import numpy as np

from laserperception.core import PointCloud
from laserperception.datasets.semantickitti import SemanticKITTIDataset
from laserperception.io.las import load_las
from laserperception.ontology import map_dales_labels, map_semantickitti_labels
from laserperception.ontology.taxonomy import EXPERIMENT_001_TAXONOMY, SemanticTaxonomy
from laserperception.perception.contracts import CoordinateContract

from .types import SemanticPointFrame, SemanticProvenance


def ground_truth_from_point_cloud(
    cloud: PointCloud,
    *,
    sample_id: str,
    frame_id: str,
    coordinates: CoordinateContract,
    taxonomy: SemanticTaxonomy,
    mapping: Literal["semantickitti", "dales"] | None = None,
    source_rows: np.ndarray | None = None,
    source_point_count: int | None = None,
) -> SemanticPointFrame:
    if cloud.labels is None:
        raise ValueError("point cloud has no semantic labels")
    labels = cloud.labels
    if mapping not in (None, "semantickitti", "dales"):
        raise ValueError("unsupported explicit dataset mapping")
    if mapping is not None:
        if taxonomy != EXPERIMENT_001_TAXONOMY:
            raise ValueError("dataset mappings target only Experiment 001 taxonomy")
        labels = (
            map_semantickitti_labels(labels)
            if mapping == "semantickitti"
            else map_dales_labels(labels)
        )
    return SemanticPointFrame(
        sample_id,
        frame_id,
        coordinates,
        len(cloud) if source_point_count is None else source_point_count,
        labels,
        taxonomy,
        SemanticProvenance(
            "laserperception", "ground-truth:" + (mapping or "native"), "PointCloud.labels"
        ),
        source_rows,
    )


def semantickitti_ground_truth(
    dataset: SemanticKITTIDataset, index: int, *, frame_id: str, coordinates: CoordinateContract
) -> SemanticPointFrame:
    info = dataset.sample_info(index)
    return ground_truth_from_point_cloud(
        dataset.load(index),
        sample_id=f"semantickitti:{info.sequence}:{info.frame}",
        frame_id=frame_id,
        coordinates=coordinates,
        taxonomy=EXPERIMENT_001_TAXONOMY,
        mapping="semantickitti",
    )


def dales_ground_truth(
    tile_path: str | Path, *, sample_id: str, frame_id: str, coordinates: CoordinateContract
) -> SemanticPointFrame:
    return ground_truth_from_point_cloud(
        load_las(tile_path),
        sample_id=sample_id,
        frame_id=frame_id,
        coordinates=coordinates,
        taxonomy=EXPERIMENT_001_TAXONOMY,
        mapping="dales",
    )
