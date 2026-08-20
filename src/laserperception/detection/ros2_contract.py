"""ROS-independent contracts for the M3 model-ready point-cloud interface."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from math import cos, isfinite, sin

import numpy as np

POINT_FIELD_FLOAT32 = 7
MODEL_READY_FIELD_NAMES = ("x", "y", "z", "time_lag")
RAW_XYZ_FIELD_NAMES = ("x", "y", "z")
_POINT_FIELD_SIZES = {1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 4, 7: 4, 8: 8}


@dataclass(frozen=True, slots=True)
class PointFieldLayout:
    """ROS PointField metadata without importing ROS."""

    name: str
    offset: int
    datatype: int
    count: int = 1


@dataclass(frozen=True, slots=True)
class PointCloud2Layout:
    """The byte-layout portion of a PointCloud2 message."""

    height: int
    width: int
    fields: tuple[PointFieldLayout, ...]
    is_bigendian: bool
    point_step: int
    row_step: int
    data: bytes


@dataclass(frozen=True, slots=True)
class TimeStamp:
    """A ROS-compatible timestamp value."""

    sec: int
    nanosec: int

    def __post_init__(self) -> None:
        if isinstance(self.sec, bool) or not isinstance(self.sec, int):
            raise TypeError("stamp.sec must be an integer")
        if isinstance(self.nanosec, bool) or not isinstance(self.nanosec, int):
            raise TypeError("stamp.nanosec must be an integer")
        if not 0 <= self.nanosec < 1_000_000_000:
            raise ValueError("stamp.nanosec must be in [0, 1000000000)")


@dataclass(frozen=True, slots=True)
class SourceHeader:
    """Frame and acquisition time copied from one input message."""

    frame_id: str
    stamp: TimeStamp

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str) or not self.frame_id.strip():
            raise ValueError("frame_id must be a non-empty string")
        object.__setattr__(self, "frame_id", self.frame_id.strip())


@dataclass(frozen=True, slots=True)
class RawPointCloudXYZ:
    """Decoded finite raw XYZ rows plus deterministic filtering counts."""

    points_xyz: np.ndarray
    source_point_count: int
    invalid_point_count: int

    def __post_init__(self) -> None:
        points = np.asarray(self.points_xyz)
        if points.dtype != np.dtype(np.float32):
            raise TypeError("raw XYZ points must have dtype float32")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("raw XYZ points must have shape (N, 3)")
        if not np.isfinite(points).all():
            raise ValueError("retained raw XYZ points must contain only finite values")
        for name, value in (
            ("source_point_count", self.source_point_count),
            ("invalid_point_count", self.invalid_point_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.source_point_count != len(points) + self.invalid_point_count:
            raise ValueError("raw XYZ point counts are inconsistent")
        object.__setattr__(self, "points_xyz", np.ascontiguousarray(points).copy())


@dataclass(frozen=True, slots=True)
class ModelReadyPointCloud:
    """Validated PointPillars input points in ``x, y, z, time_lag`` order."""

    points_xyzt: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points_xyzt)
        if points.dtype != np.dtype(np.float32):
            raise TypeError("model-ready points must have dtype float32")
        if points.ndim != 2 or points.shape[1] != 4:
            raise ValueError("model-ready points must have shape (N, 4)")
        if points.shape[0] == 0:
            raise ValueError("model-ready points must be non-empty")
        if not np.isfinite(points).all():
            raise ValueError("model-ready points must contain only finite values")
        object.__setattr__(self, "points_xyzt", np.ascontiguousarray(points).copy())

    @property
    def sha256(self) -> str:
        """Hash the exact native-endian contiguous float32 point representation."""

        return hashlib.sha256(self.points_xyzt.tobytes(order="C")).hexdigest()


@dataclass(frozen=True, slots=True)
class QuaternionRecord:
    """A normalized quaternion in x, y, z, w order."""

    x: float
    y: float
    z: float
    w: float


@dataclass(frozen=True, slots=True)
class PoseRecord:
    """Pure representation of a geometry_msgs Pose."""

    position_xyz: tuple[float, float, float]
    orientation: QuaternionRecord


@dataclass(frozen=True, slots=True)
class HypothesisRecord:
    """One detector class/score hypothesis and its pose."""

    class_id: str
    score: float
    pose: PoseRecord


@dataclass(frozen=True, slots=True)
class DetectionRecord:
    """Pure representation of one vision_msgs Detection3D."""

    header: SourceHeader
    center: PoseRecord
    size_xyz: tuple[float, float, float]
    results: tuple[HypothesisRecord, ...]
    tracking_id: str = ""


@dataclass(frozen=True, slots=True)
class DetectionArrayRecord:
    """Pure representation of one vision_msgs Detection3DArray."""

    header: SourceHeader
    detections: tuple[DetectionRecord, ...]


def decode_model_ready_pointcloud(layout: PointCloud2Layout) -> ModelReadyPointCloud:
    """Decode required named PointCloud2 fields with row padding and arbitrary ordering."""

    _validate_layout(layout)
    by_name = {field.name: field for field in layout.fields}
    missing = [name for name in MODEL_READY_FIELD_NAMES if name not in by_name]
    if missing:
        raise ValueError(f"PointCloud2 is missing required field(s): {', '.join(missing)}")
    required = [by_name[name] for name in MODEL_READY_FIELD_NAMES]
    for field in required:
        if field.datatype != POINT_FIELD_FLOAT32 or field.count != 1:
            raise ValueError(f"PointCloud2 field {field.name} must be one float32 value")

    endian = ">" if layout.is_bigendian else "<"
    dtype = np.dtype(
        {
            "names": list(MODEL_READY_FIELD_NAMES),
            "formats": [f"{endian}f4"] * 4,
            "offsets": [field.offset for field in required],
            "itemsize": layout.point_step,
        }
    )
    rows: list[np.ndarray] = []
    for row_index in range(layout.height):
        structured: np.ndarray = np.ndarray(
            shape=(layout.width,),
            dtype=dtype,
            buffer=layout.data,
            offset=row_index * layout.row_step,
        )
        rows.append(
            np.column_stack(tuple(structured[name] for name in MODEL_READY_FIELD_NAMES)).astype(
                np.float32, copy=False
            )
        )
    return ModelReadyPointCloud(np.ascontiguousarray(np.concatenate(rows, axis=0)))


def decode_raw_xyz_pointcloud(layout: PointCloud2Layout) -> RawPointCloudXYZ:
    """Decode finite float32 XYZ rows while preserving PointCloud2 row order."""

    _validate_layout(layout)
    by_name = {field.name: field for field in layout.fields}
    missing = [name for name in RAW_XYZ_FIELD_NAMES if name not in by_name]
    if missing:
        raise ValueError(f"PointCloud2 is missing required field(s): {', '.join(missing)}")
    required = [by_name[name] for name in RAW_XYZ_FIELD_NAMES]
    for field in required:
        if field.datatype != POINT_FIELD_FLOAT32 or field.count != 1:
            raise ValueError(f"PointCloud2 field {field.name} must be one float32 value")

    endian = ">" if layout.is_bigendian else "<"
    dtype = np.dtype(
        {
            "names": list(RAW_XYZ_FIELD_NAMES),
            "formats": [f"{endian}f4"] * 3,
            "offsets": [field.offset for field in required],
            "itemsize": layout.point_step,
        }
    )
    rows: list[np.ndarray] = []
    for row_index in range(layout.height):
        structured: np.ndarray = np.ndarray(
            shape=(layout.width,),
            dtype=dtype,
            buffer=layout.data,
            offset=row_index * layout.row_step,
        )
        rows.append(
            np.column_stack(tuple(structured[name] for name in RAW_XYZ_FIELD_NAMES)).astype(
                np.float32, copy=False
            )
        )
    points = np.ascontiguousarray(np.concatenate(rows, axis=0))
    finite = np.isfinite(points).all(axis=1)
    retained = np.ascontiguousarray(points[finite])
    return RawPointCloudXYZ(
        retained,
        source_point_count=len(points),
        invalid_point_count=int(np.count_nonzero(~finite)),
    )


def model_ready_pointcloud_layout(
    points: ModelReadyPointCloud | np.ndarray,
    *,
    field_order: Sequence[str] = MODEL_READY_FIELD_NAMES,
) -> PointCloud2Layout:
    """Serialize model-ready points into a canonical unorganized PointCloud2 layout."""

    cloud = points if isinstance(points, ModelReadyPointCloud) else ModelReadyPointCloud(points)
    names = tuple(field_order)
    if len(names) != 4 or set(names) != set(MODEL_READY_FIELD_NAMES):
        raise ValueError("field_order must contain x, y, z, and time_lag exactly once")
    ordered = np.column_stack(
        tuple(cloud.points_xyzt[:, MODEL_READY_FIELD_NAMES.index(name)] for name in names)
    ).astype("<f4", copy=False)
    point_step = 4 * len(names)
    return PointCloud2Layout(
        height=1,
        width=len(cloud.points_xyzt),
        fields=tuple(
            PointFieldLayout(name=name, offset=index * 4, datatype=POINT_FIELD_FLOAT32)
            for index, name in enumerate(names)
        ),
        is_bigendian=False,
        point_step=point_step,
        row_step=point_step * len(cloud.points_xyzt),
        data=ordered.tobytes(order="C"),
    )


def yaw_to_quaternion(yaw_rad: float) -> QuaternionRecord:
    """Convert x-forward/y-left/z-up yaw to a normalized +Z quaternion."""

    yaw = float(yaw_rad)
    if not isfinite(yaw):
        raise ValueError("yaw must be finite")
    half = yaw / 2.0
    quaternion = QuaternionRecord(0.0, 0.0, sin(half), cos(half))
    norm = sum(value * value for value in (quaternion.x, quaternion.y, quaternion.z, quaternion.w))
    if not abs(norm - 1.0) <= 1e-12:
        raise RuntimeError("yaw quaternion is not normalized")
    return quaternion


def detection_frame_to_records(frame: object, header: SourceHeader) -> DetectionArrayRecord:
    """Map a LaserPerception DetectionFrame into ROS-independent message records."""

    detections = getattr(frame, "detections", None)
    if not isinstance(detections, tuple):
        raise TypeError("frame must expose a tuple of detections")
    records: list[DetectionRecord] = []
    for detection in detections:
        center_xyz = (
            float(detection.center_xyz[0]),
            float(detection.center_xyz[1]),
            float(detection.center_xyz[2]),
        )
        size_lwh = (
            float(detection.size_lwh[0]),
            float(detection.size_lwh[1]),
            float(detection.size_lwh[2]),
        )
        pose = PoseRecord(center_xyz, yaw_to_quaternion(float(detection.yaw_rad)))
        records.append(
            DetectionRecord(
                header=header,
                center=pose,
                size_xyz=size_lwh,
                results=(
                    HypothesisRecord(
                        class_id=str(detection.class_name),
                        score=float(detection.score),
                        pose=pose,
                    ),
                ),
                tracking_id="",
            )
        )
    return DetectionArrayRecord(header=header, detections=tuple(records))


def _validate_layout(layout: PointCloud2Layout) -> None:
    for name, value in (("height", layout.height), ("width", layout.width)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"PointCloud2 {name} must be a positive integer")
    if isinstance(layout.point_step, bool) or not isinstance(layout.point_step, int):
        raise ValueError("PointCloud2 point_step must be an integer")
    if layout.point_step <= 0:
        raise ValueError("PointCloud2 point_step must be positive")
    minimum_row_step = layout.width * layout.point_step
    if layout.row_step < minimum_row_step:
        raise ValueError("PointCloud2 row_step is smaller than width * point_step")
    if len(layout.data) != layout.height * layout.row_step:
        raise ValueError("PointCloud2 data length does not equal height * row_step")
    names: set[str] = set()
    for field in layout.fields:
        if not isinstance(field.name, str) or not field.name:
            raise ValueError("PointCloud2 fields must have non-empty names")
        if field.name in names:
            raise ValueError(f"PointCloud2 contains duplicate field {field.name}")
        names.add(field.name)
        size = _POINT_FIELD_SIZES.get(field.datatype)
        if size is None or field.count <= 0 or field.offset < 0:
            raise ValueError(f"PointCloud2 field {field.name} has invalid metadata")
        if field.offset + size * field.count > layout.point_step:
            raise ValueError(f"PointCloud2 field {field.name} exceeds point_step")
