"""Deterministic, headless bird's-eye-view detection rendering."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from laserperception.detection.geometry import bev_corners
from laserperception.detection.types import Detection3D, DetectionFrame

_CLASS_COLORS = {
    "car": "#4C78A8",
    "truck": "#F58518",
    "trailer": "#E45756",
    "bus": "#72B7B2",
    "construction_vehicle": "#B279A2",
    "bicycle": "#FF9DA6",
    "motorcycle": "#9D755D",
    "pedestrian": "#54A24B",
    "traffic_cone": "#EECA3B",
    "barrier": "#BAB0AC",
}


@dataclass(frozen=True, slots=True)
class BevRenderData:
    """Validated arrays and detections selected for one BEV render."""

    points_xy: np.ndarray
    detections: tuple[Detection3D, ...]


def prepare_bev_render_data(
    points_xyz: np.ndarray,
    frame: DetectionFrame,
    *,
    min_score: float,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    max_points: int,
) -> BevRenderData:
    """Crop and deterministically subsample points, and filter boxes by score."""

    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points_xyz must have shape (N, 3+)")
    if not np.isfinite(points).all():
        raise ValueError("points_xyz must contain only finite values")
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    threshold = float(min_score)
    if not isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("min_score must be finite and between 0 and 1")
    x_min, x_max = _validate_limits(x_limits, "x_limits")
    y_min, y_max = _validate_limits(y_limits, "y_limits")

    visible = (
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
    )
    points_xy = points[visible, :2]
    if len(points_xy) > max_points:
        indices = np.linspace(0, len(points_xy) - 1, max_points, dtype=np.int64)
        points_xy = points_xy[indices]
    return BevRenderData(
        points_xy=points_xy.copy(),
        detections=frame.filtered(threshold).detections,
    )


def render_bev(
    points_xyz: np.ndarray,
    frame: DetectionFrame,
    output_path: str | Path,
    *,
    min_score: float = 0.25,
    x_limits: tuple[float, float] = (-50.0, 50.0),
    y_limits: tuple[float, float] = (-50.0, 50.0),
    max_points: int = 120_000,
    dpi: int = 180,
) -> Path:
    """Write an original headless PNG or SVG BEV plot and return its path."""

    destination = Path(output_path)
    if destination.suffix.lower() not in {".png", ".svg"}:
        raise ValueError("output_path must end in .png or .svg")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    data = prepare_bev_render_data(
        points_xyz,
        frame,
        min_score=min_score,
        x_limits=x_limits,
        y_limits=y_limits,
        max_points=max_points,
    )
    pyplot, patches, lines = _load_matplotlib()

    figure, axis = pyplot.subplots(figsize=(10, 10), constrained_layout=True)
    figure.patch.set_facecolor("#111318")
    axis.set_facecolor("#111318")
    if len(data.points_xy):
        axis.scatter(
            data.points_xy[:, 0],
            data.points_xy[:, 1],
            s=0.12,
            c="#D6DAE0",
            alpha=0.35,
            linewidths=0,
            rasterized=destination.suffix.lower() == ".svg",
        )

    present_classes: set[str] = set()
    for detection in data.detections:
        color = _CLASS_COLORS.get(detection.class_name, "#FFFFFF")
        present_classes.add(detection.class_name)
        corners = bev_corners(detection)
        axis.add_patch(
            patches.Polygon(corners, closed=True, fill=False, edgecolor=color, linewidth=1.35)
        )
        center_x, center_y = detection.center_xyz[:2]
        heading = np.array([np.cos(detection.yaw_rad), np.sin(detection.yaw_rad)])
        front = np.array([center_x, center_y]) + heading * detection.size_lwh[0] / 2.0
        axis.plot([center_x, front[0]], [center_y, front[1]], color=color, linewidth=1.2)
        axis.text(
            center_x,
            center_y,
            f"{detection.class_name} {detection.score:.2f}",
            color=color,
            fontsize=5.5,
            ha="center",
            va="bottom",
        )

    axis.scatter([0.0], [0.0], marker="+", s=85, c="#FFFFFF", linewidths=1.3, label="LiDAR")
    handles = [
        lines.Line2D([0], [0], color=_CLASS_COLORS.get(name, "#FFFFFF"), lw=2, label=name)
        for name in sorted(present_classes)
    ]
    handles.append(lines.Line2D([0], [0], marker="+", color="#FFFFFF", lw=0, label="LiDAR origin"))
    axis.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.75)
    axis.set(
        xlim=x_limits,
        ylim=y_limits,
        aspect="equal",
        xlabel="x forward (m)",
        ylabel="y left (m)",
        title=(
            f"PointPillars / nuScenes — {frame.sample_id}\n"
            f"score ≥ {min_score:.2f}; {len(data.detections)} detections"
        ),
    )
    axis.grid(color="#525866", alpha=0.25, linewidth=0.5)
    axis.tick_params(colors="#E5E7EB")
    axis.xaxis.label.set_color("#E5E7EB")
    axis.yaxis.label.set_color("#E5E7EB")
    axis.title.set_color("#FFFFFF")
    for spine in axis.spines.values():
        spine.set_color("#737986")

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi, facecolor=figure.get_facecolor())
    pyplot.close(figure)
    return destination


def _validate_limits(values: tuple[float, float], name: str) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} must contain two values")
    lower, upper = (float(value) for value in values)
    if not isfinite(lower) or not isfinite(upper) or lower >= upper:
        raise ValueError(f"{name} must be finite and increasing")
    return lower, upper


def _load_matplotlib() -> tuple[Any, Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import lines, patches, pyplot
    except (ImportError, RuntimeError) as error:
        raise RuntimeError(
            "BEV rendering requires the optional visualization dependencies; "
            "install LaserPerception with the 'viz' extra"
        ) from error
    return pyplot, patches, lines
