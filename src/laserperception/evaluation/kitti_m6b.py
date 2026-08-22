"""Preregistered CPU-only geometry and scoring helpers for KITTI Raw M6b.

The module deliberately has no PyTorch, MMDetection3D, MMDeploy, TensorRT, or ROS
dependency.  It parses official KITTI Raw tracklets, performs the frozen
Velodyne-to-model box conversion, applies reference-camera visibility, and
computes deterministic one-to-one BEV matches.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as et
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from laserperception.datasets.kitti_raw import KITTI_TO_MODEL_ROTATION
from laserperception.detection.types import Detection3D

TARGET_TYPES = {"Car": "car", "Pedestrian": "pedestrian"}
NEIGHBOUR_IGNORE_TYPES = {"Van": "car", "Person (sitting)": "pedestrian"}
VALID_STATES = frozenset({2})
VALID_OCCLUSIONS = frozenset({0, 1})
VALID_TRUNCATIONS = frozenset({0, 1})
BOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


@dataclass(frozen=True, slots=True)
class KittiTrackletPose:
    """One official KITTI Raw tracklet pose in native Velodyne coordinates."""

    track_id: int
    frame_index: int
    object_type: str
    height: float
    width: float
    length: float
    translation_xyz: tuple[float, float, float]
    rotation_xyz: tuple[float, float, float]
    state: int
    occlusion: int
    truncation: int

    def __post_init__(self) -> None:
        numeric = (
            self.height,
            self.width,
            self.length,
            *self.translation_xyz,
            *self.rotation_xyz,
        )
        if self.track_id < 0 or self.frame_index < 0:
            raise ValueError("track and frame identifiers must be non-negative")
        if not self.object_type.strip():
            raise ValueError("object_type must be non-empty")
        if not np.isfinite(np.asarray(numeric, dtype=np.float64)).all():
            raise ValueError("tracklet geometry must contain only finite values")
        if min(self.height, self.width, self.length) <= 0.0:
            raise ValueError("tracklet dimensions must be positive")

    @property
    def valid_labelled_pose(self) -> bool:
        """Apply the frozen GT-only eligibility predicate."""

        return (
            self.state in VALID_STATES
            and self.occlusion in VALID_OCCLUSIONS
            and self.truncation in VALID_TRUNCATIONS
        )

    @property
    def evaluation_role(self) -> str:
        """Return ``target``, ``neighbour_ignore``, or ``unmapped``."""

        if self.object_type in TARGET_TYPES:
            return "target"
        if self.object_type in NEIGHBOUR_IGNORE_TYPES:
            return "neighbour_ignore"
        return "unmapped"

    @property
    def evaluation_class(self) -> str | None:
        """Return the frozen nuScenes evaluation class where mapped."""

        return TARGET_TYPES.get(self.object_type) or NEIGHBOUR_IGNORE_TYPES.get(self.object_type)


@dataclass(frozen=True, slots=True)
class M6bGroundTruthBox:
    """One converted KITTI box in the frozen PointPillars model frame."""

    track_id: int
    frame_index: int
    source_type: str
    evaluation_role: str
    class_name: str | None
    center_xyz: tuple[float, float, float]
    size_lwh: tuple[float, float, float]
    yaw_rad: float


@dataclass(frozen=True, slots=True)
class KittiReferenceCamera:
    """Rectified camera-0 projection contract from official Raw calibration."""

    projection: np.ndarray
    rectified_camera_from_velodyne: np.ndarray
    image_size_wh: tuple[int, int]
    near_plane_metres: float = 1e-6

    def __post_init__(self) -> None:
        projection = np.asarray(self.projection, dtype=np.float64)
        transform = np.asarray(self.rectified_camera_from_velodyne, dtype=np.float64)
        if projection.shape != (3, 4):
            raise ValueError("projection must have shape (3, 4)")
        if transform.shape != (4, 4):
            raise ValueError("rectified_camera_from_velodyne must have shape (4, 4)")
        if not np.isfinite(projection).all() or not np.isfinite(transform).all():
            raise ValueError("camera matrices must contain only finite values")
        if self.image_size_wh[0] <= 0 or self.image_size_wh[1] <= 0:
            raise ValueError("image dimensions must be positive")
        if not math.isfinite(self.near_plane_metres) or self.near_plane_metres <= 0.0:
            raise ValueError("near plane must be finite and positive")
        object.__setattr__(self, "projection", projection.copy())
        object.__setattr__(self, "rectified_camera_from_velodyne", transform.copy())

    @classmethod
    def from_date_root(cls, date_root: str | Path) -> KittiReferenceCamera:
        """Load official reference-camera fields from a KITTI Raw date root."""

        root = Path(date_root)
        camera = _read_keyed_floats(root / "calib_cam_to_cam.txt")
        velo = _read_keyed_floats(root / "calib_velo_to_cam.txt")
        projection = _matrix(camera, "P_rect_00", (3, 4))
        rectification = _matrix(camera, "R_rect_00", (3, 3))
        raw_transform = np.eye(4, dtype=np.float64)
        raw_transform[:3, :3] = _matrix(velo, "R", (3, 3))
        raw_transform[:3, 3] = _vector(velo, "T", 3)
        rectified = np.eye(4, dtype=np.float64)
        rectified[:3, :3] = rectification
        size = _vector(camera, "S_rect_00", 2)
        return cls(
            projection,
            rectified @ raw_transform,
            (int(size[0]), int(size[1])),
        )


@dataclass(frozen=True, slots=True)
class MatchRecord:
    """One deterministic prediction disposition at one BEV IoU threshold."""

    prediction_index: int
    score: float
    disposition: str
    gt_track_id: int | None
    bev_iou: float


@dataclass(frozen=True, slots=True)
class MatchSummary:
    """Counts and ordered dispositions for one class/threshold evaluation."""

    true_positives: int
    false_positives: int
    false_negatives: int
    ignored_predictions: int
    records: tuple[MatchRecord, ...]

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0


def parse_kitti_tracklets(path: str | Path) -> tuple[KittiTrackletPose, ...]:
    """Parse official Boost-serialized Raw tracklets without a devkit dependency."""

    root = et.parse(Path(path)).getroot()
    container = root.find("tracklets")
    if container is None:
        raise ValueError("KITTI tracklet XML lacks a tracklets container")
    expected = _required_int(container, "count")
    track_items = container.findall("item")
    if len(track_items) != expected:
        raise ValueError("KITTI tracklet count does not match serialized items")
    result: list[KittiTrackletPose] = []
    for track_id, item in enumerate(track_items):
        object_type = _required_text(item, "objectType")
        height = _required_float(item, "h")
        width = _required_float(item, "w")
        length = _required_float(item, "l")
        first_frame = _required_int(item, "first_frame")
        poses = item.find("poses")
        if poses is None:
            raise ValueError("KITTI tracklet lacks poses")
        pose_items = poses.findall("item")
        if len(pose_items) != _required_int(poses, "count"):
            raise ValueError("KITTI pose count does not match serialized items")
        for offset, pose in enumerate(pose_items):
            result.append(
                KittiTrackletPose(
                    track_id=track_id,
                    frame_index=first_frame + offset,
                    object_type=object_type,
                    height=height,
                    width=width,
                    length=length,
                    translation_xyz=tuple(
                        _required_float(pose, name) for name in ("tx", "ty", "tz")
                    ),
                    rotation_xyz=tuple(_required_float(pose, name) for name in ("rx", "ry", "rz")),
                    state=_required_int(pose, "state"),
                    occlusion=_required_int(pose, "occlusion"),
                    truncation=_required_int(pose, "truncation"),
                )
            )
    return tuple(result)


def convert_tracklet_pose(pose: KittiTrackletPose) -> M6bGroundTruthBox:
    """Convert official h/w/l bottom-centre Velodyne geometry to model axes."""

    if abs(pose.rotation_xyz[0]) > 1e-12 or abs(pose.rotation_xyz[1]) > 1e-12:
        raise ValueError("M6b supports official upright Raw tracklets only")
    bottom = np.asarray(pose.translation_xyz, dtype=np.float64)
    centre_velodyne = bottom + np.array([0.0, 0.0, pose.height / 2.0])
    centre_model = KITTI_TO_MODEL_ROTATION @ centre_velodyne
    return M6bGroundTruthBox(
        track_id=pose.track_id,
        frame_index=pose.frame_index,
        source_type=pose.object_type,
        evaluation_role=pose.evaluation_role,
        class_name=pose.evaluation_class,
        center_xyz=tuple(float(value) for value in centre_model),
        size_lwh=(pose.length, pose.width, pose.height),
        yaw_rad=normalize_angle(pose.rotation_xyz[2] + math.pi / 2.0),
    )


def native_box_corners(pose: KittiTrackletPose) -> np.ndarray:
    """Return eight native Velodyne corners for reference-camera visibility."""

    length = pose.length / 2.0
    width = pose.width / 2.0
    local = np.array(
        [
            [length, width, 0.0],
            [-length, width, 0.0],
            [-length, -width, 0.0],
            [length, -width, 0.0],
            [length, width, pose.height],
            [-length, width, pose.height],
            [-length, -width, pose.height],
            [length, -width, pose.height],
        ],
        dtype=np.float64,
    )
    yaw = pose.rotation_xyz[2]
    rotation = np.array(
        [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0, 0, 1]],
        dtype=np.float64,
    )
    return local @ rotation.T + np.asarray(pose.translation_xyz, dtype=np.float64)


def model_box_corners(box: Detection3D | M6bGroundTruthBox) -> np.ndarray:
    """Return eight model-frame corners for prediction visibility."""

    length, width, height = box.size_lwh
    centre = np.asarray(box.center_xyz, dtype=np.float64)
    half_z = height / 2.0
    local = np.array(
        [
            [length / 2, width / 2, -half_z],
            [-length / 2, width / 2, -half_z],
            [-length / 2, -width / 2, -half_z],
            [length / 2, -width / 2, -half_z],
            [length / 2, width / 2, half_z],
            [-length / 2, width / 2, half_z],
            [-length / 2, -width / 2, half_z],
            [length / 2, -width / 2, half_z],
        ],
        dtype=np.float64,
    )
    yaw = box.yaw_rad
    rotation = np.array(
        [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0, 0, 1]],
        dtype=np.float64,
    )
    return local @ rotation.T + centre


def visible_in_reference_camera(
    corners_velodyne: np.ndarray,
    camera: KittiReferenceCamera,
) -> bool:
    """Apply the frozen near-plane-clipped projected-extent intersection rule."""

    corners = np.asarray(corners_velodyne, dtype=np.float64)
    if corners.shape != (8, 3) or not np.isfinite(corners).all():
        raise ValueError("box corners must be a finite (8, 3) array")
    homogeneous = np.column_stack([corners, np.ones(8, dtype=np.float64)])
    camera_points = (camera.rectified_camera_from_velodyne @ homogeneous.T).T[:, :3]
    retained = [point for point in camera_points if point[2] >= camera.near_plane_metres]
    for first, second in BOX_EDGES:
        a = camera_points[first]
        b = camera_points[second]
        a_front = a[2] >= camera.near_plane_metres
        b_front = b[2] >= camera.near_plane_metres
        if a_front == b_front:
            continue
        fraction = (camera.near_plane_metres - a[2]) / (b[2] - a[2])
        retained.append(a + fraction * (b - a))
    if not retained:
        return False
    points = np.asarray(retained, dtype=np.float64)
    projected = (camera.projection @ np.column_stack([points, np.ones(len(points))]).T).T
    valid = projected[:, 2] > 0.0
    if not bool(valid.any()):
        return False
    image = projected[valid, :2] / projected[valid, 2:3]
    minimum = image.min(axis=0)
    maximum = image.max(axis=0)
    width, height = camera.image_size_wh
    return bool(
        maximum[0] >= 0.0 and maximum[1] >= 0.0 and minimum[0] < width and minimum[1] < height
    )


def model_to_native_corners(corners_model: np.ndarray) -> np.ndarray:
    """Rotate model-frame corners back into native KITTI Velodyne axes."""

    corners = np.asarray(corners_model, dtype=np.float64)
    if corners.shape != (8, 3):
        raise ValueError("model corners must have shape (8, 3)")
    return corners @ KITTI_TO_MODEL_ROTATION


def bev_iou(
    first: Detection3D | M6bGroundTruthBox,
    second: Detection3D | M6bGroundTruthBox,
) -> float:
    """Return deterministic oriented rectangle intersection-over-union."""

    first_polygon = _bev_corners(first)
    second_polygon = _bev_corners(second)
    intersection = _clip_convex(first_polygon, second_polygon)
    intersection_area = _polygon_area(intersection)
    union = _polygon_area(first_polygon) + _polygon_area(second_polygon) - intersection_area
    return intersection_area / union if union > 0.0 else 0.0


def match_detections(
    predictions: Sequence[Detection3D],
    targets: Sequence[M6bGroundTruthBox],
    neighbour_ignores: Sequence[M6bGroundTruthBox],
    *,
    class_name: str,
    iou_threshold: float,
    score_threshold: float = 0.25,
) -> MatchSummary:
    """Score predictions with frozen score order, IoU tie-break, and ignore handling."""

    if class_name not in {"car", "pedestrian"}:
        raise ValueError("M6b quantitative classes are car and pedestrian")
    if not 0.0 < iou_threshold <= 1.0 or not 0.0 <= score_threshold <= 1.0:
        raise ValueError("thresholds must lie in their probability ranges")
    class_targets = sorted(
        (box for box in targets if box.class_name == class_name),
        key=lambda box: box.track_id,
    )
    class_ignores = tuple(box for box in neighbour_ignores if box.class_name == class_name)
    ordered = sorted(
        (
            (index, prediction)
            for index, prediction in enumerate(predictions)
            if prediction.class_name == class_name and prediction.score >= score_threshold
        ),
        key=lambda item: (-item[1].score, item[0]),
    )
    unmatched = set(range(len(class_targets)))
    records: list[MatchRecord] = []
    for prediction_index, prediction in ordered:
        candidates = sorted(
            ((bev_iou(prediction, class_targets[index]), index) for index in unmatched),
            key=lambda item: (-item[0], class_targets[item[1]].track_id),
        )
        best_iou, best_index = candidates[0] if candidates else (0.0, -1)
        if best_index >= 0 and best_iou >= iou_threshold:
            unmatched.remove(best_index)
            records.append(
                MatchRecord(
                    prediction_index,
                    prediction.score,
                    "true_positive",
                    class_targets[best_index].track_id,
                    best_iou,
                )
            )
            continue
        ignore_iou = max((bev_iou(prediction, box) for box in class_ignores), default=0.0)
        if ignore_iou >= iou_threshold:
            records.append(
                MatchRecord(
                    prediction_index,
                    prediction.score,
                    "ignored_neighbour",
                    None,
                    ignore_iou,
                )
            )
        else:
            records.append(
                MatchRecord(prediction_index, prediction.score, "false_positive", None, best_iou)
            )
    true_positives = sum(record.disposition == "true_positive" for record in records)
    false_positives = sum(record.disposition == "false_positive" for record in records)
    ignored = sum(record.disposition == "ignored_neighbour" for record in records)
    return MatchSummary(true_positives, false_positives, len(unmatched), ignored, tuple(records))


def normalize_angle(angle: float) -> float:
    """Normalize radians into the half-open interval ``[-pi, pi)``."""

    if not math.isfinite(angle):
        raise ValueError("angle must be finite")
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _bev_corners(box: Detection3D | M6bGroundTruthBox) -> np.ndarray:
    length, width, _ = box.size_lwh
    local = np.array(
        [
            [length / 2, width / 2],
            [-length / 2, width / 2],
            [-length / 2, -width / 2],
            [length / 2, -width / 2],
        ],
        dtype=np.float64,
    )
    cosine, sine = math.cos(box.yaw_rad), math.sin(box.yaw_rad)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    return local @ rotation.T + np.asarray(box.center_xyz[:2], dtype=np.float64)


def _clip_convex(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    output = [point.copy() for point in subject]
    for index, edge_start in enumerate(clip):
        edge_end = clip[(index + 1) % len(clip)]
        input_points = output
        output = []
        if not input_points:
            break
        previous = input_points[-1]
        for current in input_points:
            current_inside = _left_of(edge_start, edge_end, current) >= -1e-12
            previous_inside = _left_of(edge_start, edge_end, previous) >= -1e-12
            if current_inside:
                if not previous_inside:
                    output.append(_line_intersection(previous, current, edge_start, edge_end))
                output.append(current)
            elif previous_inside:
                output.append(_line_intersection(previous, current, edge_start, edge_end))
            previous = current
    return np.asarray(output, dtype=np.float64).reshape(-1, 2)


def _left_of(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> float:
    edge = end - start
    relative = point - start
    return float(edge[0] * relative[1] - edge[1] * relative[0])


def _line_intersection(
    first: np.ndarray,
    second: np.ndarray,
    edge_start: np.ndarray,
    edge_end: np.ndarray,
) -> np.ndarray:
    direction = second - first
    edge = edge_end - edge_start
    denominator = direction[0] * edge[1] - direction[1] * edge[0]
    if abs(denominator) < 1e-15:
        return second.copy()
    offset = edge_start - first
    fraction = (offset[0] * edge[1] - offset[1] * edge[0]) / denominator
    return first + fraction * direction


def _polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    return (
        abs(
            float(
                np.dot(points[:, 0], np.roll(points[:, 1], -1))
                - np.dot(points[:, 1], np.roll(points[:, 0], -1))
            )
        )
        / 2.0
    )


def _read_keyed_floats(path: Path) -> Mapping[str, tuple[float, ...]]:
    values: dict[str, tuple[float, ...]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or ":" not in raw:
            continue
        key, text = raw.split(":", 1)
        if key == "calib_time":
            continue
        try:
            values[key] = tuple(float(value) for value in text.split())
        except ValueError as error:
            raise ValueError(f"calibration field {key} is not numeric") from error
    return values


def _matrix(
    values: Mapping[str, tuple[float, ...]], key: str, shape: tuple[int, int]
) -> np.ndarray:
    expected = shape[0] * shape[1]
    if key not in values or len(values[key]) != expected:
        raise ValueError(f"calibration field {key} must contain {expected} values")
    return np.asarray(values[key], dtype=np.float64).reshape(shape)


def _vector(values: Mapping[str, tuple[float, ...]], key: str, size: int) -> np.ndarray:
    if key not in values or len(values[key]) != size:
        raise ValueError(f"calibration field {key} must contain {size} values")
    return np.asarray(values[key], dtype=np.float64)


def _required_text(parent: et.Element, key: str) -> str:
    element = parent.find(key)
    if element is None or element.text is None or not element.text.strip():
        raise ValueError(f"KITTI tracklet field {key} is missing")
    return element.text.strip()


def _required_float(parent: et.Element, key: str) -> float:
    try:
        return float(_required_text(parent, key))
    except ValueError as error:
        raise ValueError(f"KITTI tracklet field {key} is not numeric") from error


def _required_int(parent: et.Element, key: str) -> int:
    try:
        return int(_required_text(parent, key))
    except ValueError as error:
        raise ValueError(f"KITTI tracklet field {key} is not an integer") from error
