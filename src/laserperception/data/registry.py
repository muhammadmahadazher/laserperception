"""Deterministic metadata discovery with no reader imports or dataset access."""

from collections.abc import Iterable

from laserperception.perception.contracts import PerceptionTask

from .contracts import DataAdapterManifest


class DataAdapterRegistry:
    def __init__(self, manifests: Iterable[DataAdapterManifest] = ()) -> None:
        self._adapters: dict[str, DataAdapterManifest] = {}
        for manifest in manifests:
            self.register(manifest)

    def register(self, manifest: DataAdapterManifest) -> None:
        if not isinstance(manifest, DataAdapterManifest):
            raise TypeError("registry requires DataAdapterManifest")
        if manifest.adapter_id in self._adapters:
            raise ValueError("duplicate data adapter ID")
        self._adapters[manifest.adapter_id] = manifest

    def list_adapters(self) -> tuple[DataAdapterManifest, ...]:
        return tuple(self._adapters[k] for k in sorted(self._adapters))

    def get(self, adapter_id: str) -> DataAdapterManifest:
        try:
            return self._adapters[adapter_id]
        except KeyError:
            raise ValueError(
                f"unknown data adapter {adapter_id!r}; use data adapters list"
            ) from None

    def inspect(self, adapter_id: str) -> DataAdapterManifest:
        return self.get(adapter_id)

    def filter_by_feature(self, feature: str) -> tuple[DataAdapterManifest, ...]:
        return tuple(m for m in self.list_adapters() if feature in m.supplied_features)

    def filter_by_task(self, task: PerceptionTask) -> tuple[DataAdapterManifest, ...]:
        return tuple(m for m in self.list_adapters() if task in m.tasks)


def builtin_registry() -> DataAdapterRegistry:
    def manifest(
        adapter_id: str,
        name: str,
        category: str,
        formats: tuple[str, ...],
        features: tuple[str, ...],
        labels: str,
        temporal: str,
        calibration: tuple[str, ...],
        coords: str,
        optional: tuple[str, ...],
        indexing: str,
        streaming: str,
        limitations: tuple[str, ...],
    ) -> DataAdapterManifest:
        return DataAdapterManifest.from_dict(
            {
                "adapter_id": adapter_id,
                "display_name": name,
                "category": category,
                "formats": formats,
                "supplied_features": features,
                "tasks": ("detection_3d", "semantic_segmentation")
                if labels != "none"
                else ("detection_3d",),
                "label_semantics": labels,
                "temporal_support": temporal,
                "calibration_requirements": calibration,
                "coordinate_semantics": coords,
                "optional_dependencies": optional,
                "indexing": indexing,
                "streaming": streaming,
                "limitations": limitations,
            }
        )

    lidar = "native sensor XYZ; forward/left/up in metres; no transform"
    las = (
        "stored scaled XYZ; CRS retained when present; axes and units require explicit verification"
    )
    return DataAdapterRegistry(
        (
            manifest(
                "kitti-velodyne",
                "KITTI Velodyne scan",
                "file",
                (".bin",),
                ("x", "y", "z", "remission"),
                "optional explicit SemanticKITTI packed label file",
                "none",
                (),
                lidar,
                (),
                "single file",
                "one scan",
                ("No timestamp, pose or multi-sweep preparation; .bin is ambiguous.",),
            ),
            manifest(
                "semantickitti",
                "SemanticKITTI directory",
                "dataset",
                ("official sequences directory",),
                ("x", "y", "z", "remission", "instance_id"),
                "native semantic uint16 and instance uint16 IDs",
                "deterministic numeric frame order; no acquisition clock",
                (),
                lidar,
                (),
                "official split/subset; lazy point loading",
                "one scan at a time",
                ("No pose/calibration reader; instance IDs only when labels exist.",),
            ),
            manifest(
                "las",
                "LAS point file",
                "file",
                (".las",),
                ("x", "y", "z", "intensity", "stored_dimensions"),
                "stored classification; taxonomy unverified",
                "stored GPS time if present; clock interpretation unverified",
                (),
                las,
                (),
                "single file",
                "full file",
                ("Stored dimensions depend on point format; full file loaded.",),
            ),
            manifest(
                "laz",
                "LAZ compressed point file",
                "file",
                (".laz",),
                ("x", "y", "z", "intensity", "stored_dimensions"),
                "stored classification; taxonomy unverified",
                "stored GPS time if present; clock interpretation unverified",
                (),
                las,
                ("laspy LAZ backend (laserperception[laz])",),
                "single file",
                "full file",
                ("Optional decompressor required; stored dimensions depend on point format.",),
            ),
            manifest(
                "dales",
                "DALES LAS/LAZ tiles",
                "dataset",
                ("explicit train/test directory", ".las", ".laz"),
                ("x", "y", "z", "classification", "stored_dimensions"),
                "native DALES classification; explicit mapping only",
                "spatial tiles; no temporal sequence",
                (),
                las,
                ("LAZ backend for compressed tiles",),
                "deterministic tile paths",
                "existing chunks retain only XYZ/classification",
                (
                    "Full tile path retains LAS attributes; chunk path does not.",
                    "Spatial patches lack original source-row maps.",
                ),
            ),
            manifest(
                "kitti-raw",
                "KITTI Raw synchronized drive",
                "dataset",
                ("KITTI Raw date/drive directory",),
                ("x", "y", "z", "remission"),
                "none",
                "exact acquisition nanoseconds and OXTS poses",
                ("three date calibration files", "OXTS and timestamp files"),
                lidar,
                (),
                "contiguous indexed frames; OXTS/calibration read at construction",
                "lazy point scans",
                ("Native points only; no implicit model-axis rotation or reconstruction.",),
            ),
            manifest(
                "nuscenes-lidar-top",
                "nuScenes raw LIDAR_TOP acquisition",
                "file",
                (".bin",),
                ("x", "y", "z", "intensity", "ring_index"),
                "none",
                "explicit caller-supplied microsecond timestamp",
                (),
                lidar,
                (),
                "one file",
                "one acquisition",
                (
                    "Fifth raw column is ring_index, never time_lag.",
                    "No nuScenes dataset discovery or multi-sweep preparation.",
                ),
            ),
            manifest(
                "pointcloud2-xyz",
                "PointCloud2 raw XYZ layout",
                "message",
                ("PointCloud2Layout + SourceHeader",),
                ("x", "y", "z"),
                "none",
                "explicit separate message header",
                (),
                "header frame; caller must establish axes and units",
                (),
                "API message boundary",
                "one message",
                (
                    "Decoder removes non-finite rows without reordering; "
                    "original row indices unavailable.",
                    "No PCD, rosbag or vendor SDK file reader.",
                ),
            ),
        )
    )


registry = builtin_registry()
