"""CPU-only deterministic matching and metrics for M2 backend parity."""

from __future__ import annotations

from dataclasses import dataclass
from math import degrees, pi, sqrt

import numpy as np

from laserperception.detection.geometry import bev_corners
from laserperception.detection.types import Detection3D


@dataclass(frozen=True, slots=True)
class DetectionMatch:
    """One class-wise, one-to-one reference/candidate detection match."""

    reference: Detection3D
    candidate: Detection3D
    bev_iou: float

    @property
    def center_displacement_3d_m(self) -> float:
        return sqrt(
            sum(
                (candidate - reference) ** 2
                for reference, candidate in zip(
                    self.reference.center_xyz, self.candidate.center_xyz, strict=True
                )
            )
        )

    @property
    def center_displacement_xy_m(self) -> float:
        return sqrt(
            sum(
                (candidate - reference) ** 2
                for reference, candidate in zip(
                    self.reference.center_xyz[:2], self.candidate.center_xyz[:2], strict=True
                )
            )
        )

    @property
    def center_displacement_z_absolute_m(self) -> float:
        return abs(self.candidate.center_xyz[2] - self.reference.center_xyz[2])

    @property
    def dimension_relative_error_lwh(self) -> tuple[float, float, float]:
        values = tuple(
            abs(candidate - reference) / reference
            for reference, candidate in zip(
                self.reference.size_lwh, self.candidate.size_lwh, strict=True
            )
        )
        return values[0], values[1], values[2]

    @property
    def circular_yaw_difference_degrees(self) -> float:
        difference = (self.candidate.yaw_rad - self.reference.yaw_rad + pi) % (2.0 * pi) - pi
        return degrees(abs(difference))

    @property
    def confidence_score_absolute_difference(self) -> float:
        return abs(self.candidate.score - self.reference.score)

    @property
    def class_equal(self) -> bool:
        return (
            self.reference.class_id == self.candidate.class_id
            and self.reference.class_name == self.candidate.class_name
        )

    def to_dict(self) -> dict[str, object]:
        """Return the frozen M2 matched-pair metrics as JSON-compatible data."""

        return {
            "reference": self.reference.to_dict(),
            "candidate": self.candidate.to_dict(),
            "bev_iou": self.bev_iou,
            "center_displacement_3d_m": self.center_displacement_3d_m,
            "center_displacement_xy_m": self.center_displacement_xy_m,
            "center_displacement_z_absolute_m": self.center_displacement_z_absolute_m,
            "dimension_relative_error_lwh": list(self.dimension_relative_error_lwh),
            "circular_yaw_difference_degrees": self.circular_yaw_difference_degrees,
            "confidence_score_absolute_difference": (self.confidence_score_absolute_difference),
            "class_equal": self.class_equal,
        }


@dataclass(frozen=True, slots=True)
class MatchingResult:
    """Deterministic class-wise matching result."""

    matches: tuple[DetectionMatch, ...]
    unmatched_reference: tuple[Detection3D, ...]
    unmatched_candidate: tuple[Detection3D, ...]


def oriented_bev_iou(first: Detection3D, second: Detection3D) -> float:
    """Return exact convex-polygon BEV IoU without an optional geometry backend."""

    first_corners = bev_corners(first)
    second_corners = bev_corners(second)
    intersection = _convex_clip(first_corners, second_corners)
    intersection_area = _polygon_area(intersection)
    first_area = first.size_lwh[0] * first.size_lwh[1]
    second_area = second.size_lwh[0] * second.size_lwh[1]
    union_area = first_area + second_area - intersection_area
    if union_area <= 0.0:
        return 0.0
    return float(min(1.0, max(0.0, intersection_area / union_area)))


def match_detections(
    reference: tuple[Detection3D, ...],
    candidate: tuple[Detection3D, ...],
    *,
    minimum_bev_iou: float = 0.50,
) -> MatchingResult:
    """Greedily match by reference score order and maximum unmatched class-wise IoU."""

    if not 0.0 <= minimum_bev_iou <= 1.0:
        raise ValueError("minimum_bev_iou must be between 0 and 1")
    ordered_reference = tuple(sorted(reference, key=Detection3D.sort_key))
    ordered_candidate = tuple(sorted(candidate, key=Detection3D.sort_key))
    unmatched_indices = set(range(len(ordered_candidate)))
    matches: list[DetectionMatch] = []
    unmatched_reference: list[Detection3D] = []

    for reference_detection in ordered_reference:
        options = [
            (
                oriented_bev_iou(reference_detection, ordered_candidate[index]),
                ordered_candidate[index].sort_key(),
                index,
            )
            for index in unmatched_indices
            if ordered_candidate[index].class_id == reference_detection.class_id
            and ordered_candidate[index].class_name == reference_detection.class_name
        ]
        if not options:
            unmatched_reference.append(reference_detection)
            continue
        best_iou, _, best_index = min(options, key=lambda item: (-item[0], item[1]))
        if best_iou < minimum_bev_iou:
            unmatched_reference.append(reference_detection)
            continue
        candidate_detection = ordered_candidate[best_index]
        unmatched_indices.remove(best_index)
        matches.append(
            DetectionMatch(
                reference=reference_detection,
                candidate=candidate_detection,
                bev_iou=best_iou,
            )
        )

    return MatchingResult(
        matches=tuple(matches),
        unmatched_reference=tuple(unmatched_reference),
        unmatched_candidate=tuple(ordered_candidate[index] for index in sorted(unmatched_indices)),
    )


def _polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x_values = points[:, 0]
    y_values = points[:, 1]
    return float(
        0.5 * abs(np.dot(x_values, np.roll(y_values, -1)) - np.dot(y_values, np.roll(x_values, -1)))
    )


def _convex_clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    output = np.asarray(subject, dtype=np.float64)
    for edge_index in range(len(clip)):
        edge_start = clip[edge_index]
        edge_end = clip[(edge_index + 1) % len(clip)]
        current = output
        if len(current) == 0:
            break
        clipped: list[np.ndarray] = []
        previous = current[-1]
        previous_inside = _inside(previous, edge_start, edge_end)
        for point in current:
            point_inside = _inside(point, edge_start, edge_end)
            if point_inside:
                if not previous_inside:
                    clipped.append(_intersection(previous, point, edge_start, edge_end))
                clipped.append(point)
            elif previous_inside:
                clipped.append(_intersection(previous, point, edge_start, edge_end))
            previous = point
            previous_inside = point_inside
        output = np.asarray(clipped, dtype=np.float64).reshape(-1, 2)
    return output


def _inside(point: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray) -> bool:
    edge = edge_end - edge_start
    relative = point - edge_start
    return bool(edge[0] * relative[1] - edge[1] * relative[0] >= -1e-12)


def _intersection(
    line_start: np.ndarray,
    line_end: np.ndarray,
    edge_start: np.ndarray,
    edge_end: np.ndarray,
) -> np.ndarray:
    line = line_end - line_start
    edge = edge_end - edge_start
    denominator = line[0] * edge[1] - line[1] * edge[0]
    if abs(denominator) <= 1e-15:
        return line_end
    delta = edge_start - line_start
    parameter = (delta[0] * edge[1] - delta[1] * edge[0]) / denominator
    return np.asarray(line_start + parameter * line, dtype=np.float64)
