"""NumPy semantic evaluation; alignment precedes every metric calculation."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from laserperception.ontology.taxonomy import SemanticTaxonomy
from laserperception.perception.serialization import JsonRecord

from .types import SemanticPointFrame


@dataclass(frozen=True)
class SemanticClassMetrics(JsonRecord):
    class_id: int
    class_name: str
    true_positive: int
    false_positive: int
    false_negative: int
    support: int
    iou: float | None


@dataclass(frozen=True)
class SemanticEvaluation(JsonRecord):
    taxonomy: SemanticTaxonomy
    confusion_matrix: tuple[tuple[int, ...], ...]
    class_metrics: tuple[SemanticClassMetrics, ...]
    evaluated_points: int
    ignored_ground_truth_points: int
    unknown_prediction_points: int
    excluded_mean_iou_class_ids: tuple[int, ...]
    mean_iou: float | None
    overall_accuracy: float | None
    schema_version: Literal["laserperception.semantic-evaluation.v1"] = (
        "laserperception.semantic-evaluation.v1"
    )


def evaluate_semantics(
    prediction: SemanticPointFrame, ground_truth: SemanticPointFrame
) -> SemanticEvaluation:
    """Rows are GT classes; columns are predictions plus a final unknown bucket."""
    for name in ("sample_id", "frame_id", "coordinates", "source_point_count", "taxonomy"):
        if getattr(prediction, name) != getattr(ground_truth, name):
            raise ValueError(f"semantic alignment mismatch: {name}")
    assert prediction.source_rows is not None and ground_truth.source_rows is not None
    if not np.array_equal(prediction.source_rows, ground_truth.source_rows):
        raise ValueError("semantic alignment mismatch: source_rows")
    classes = tuple(c for c in ground_truth.taxonomy.classes if not c.ignored)
    ignored_ids = [c.class_id for c in ground_truth.taxonomy.classes if c.ignored]
    gt = ground_truth.class_ids
    ignored = np.isin(gt, ignored_ids)
    valid_gt = np.isin(gt, [c.class_id for c in classes])
    if not np.all(valid_gt | ignored):
        raise ValueError("ground truth contains an undeclared or unknown class ID")
    gt_index = np.full(gt.shape, -1, dtype=np.int64)
    pred_index = np.full(gt.shape, len(classes), dtype=np.int64)
    for i, cls in enumerate(classes):
        gt_index[gt == cls.class_id] = i
        pred_index[prediction.class_ids == cls.class_id] = i
    matrix = np.zeros((len(classes), len(classes) + 1), dtype=np.int64)
    np.add.at(matrix, (gt_index[valid_gt], pred_index[valid_gt]), 1)
    metrics = []
    excluded = []
    for i, cls in enumerate(classes):
        tp = int(matrix[i, i])
        support = int(matrix[i].sum())
        fp = int(matrix[:, i].sum()) - tp
        fn = support - tp
        union = tp + fp + fn
        iou = tp / union if union else None
        if iou is None:
            excluded.append(cls.class_id)
        metrics.append(SemanticClassMetrics(cls.class_id, cls.name, tp, fp, fn, support, iou))
    ious = [m.iou for m in metrics if m.iou is not None]
    total = int(valid_gt.sum())
    return SemanticEvaluation(
        ground_truth.taxonomy,
        tuple(tuple(int(v) for v in row) for row in matrix),
        tuple(metrics),
        total,
        int(ignored.sum()),
        int(matrix[:, -1].sum()),
        tuple(excluded),
        sum(ious) / len(ious) if ious else None,
        sum(m.true_positive for m in metrics) / total if total else None,
    )
