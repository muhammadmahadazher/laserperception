"""Thin reader wrappers; automatic format selection is deliberately narrow."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from laserperception.core import PointCloud

from .contracts import DataAdapterManifest, InputOptions, LoadedInput
from .registry import registry


def select_adapter(path: str | Path, adapter_id: str | None = None) -> DataAdapterManifest:
    if adapter_id is not None:
        return registry.get(adapter_id)
    suffix = Path(path).suffix.lower()
    if suffix in (".las", ".laz"):
        return registry.get(suffix[1:])
    if suffix == ".bin":
        raise ValueError("ambiguous .bin input; supply --adapter explicitly")
    raise ValueError("unsupported or ambiguous input; use an explicit registered adapter")


def load_input(
    path: str | Path, *, adapter_id: str | None = None, options: InputOptions | None = None
) -> LoadedInput:
    options = options or InputOptions()
    manifest = select_adapter(path, adapter_id)
    target = Path(path)
    if manifest.adapter_id == "las" and target.suffix.lower() != ".las":
        raise ValueError("LAS adapter requires .las; use laz for .laz")
    if manifest.adapter_id == "laz" and target.suffix.lower() != ".laz":
        raise ValueError("LAZ adapter requires .laz")
    timestamp = None
    sample_id = options.sample_id or target.resolve().as_posix()
    if manifest.adapter_id in ("las", "laz") or (
        manifest.adapter_id == "dales" and target.is_file()
    ):
        try:
            from laserperception.io.las import load_las

            cloud = load_las(target)
        except ImportError as error:
            raise RuntimeError(
                "LAS/LAZ reader dependency missing; install laserperception[laz] for LAZ"
            ) from error
        except Exception as error:
            if target.suffix.lower() == ".laz" and type(error).__module__.startswith("laspy"):
                raise RuntimeError(
                    "LAZ decompression failed; install laserperception[laz] and verify the file"
                ) from error
            raise
    elif manifest.adapter_id == "kitti-velodyne":
        from laserperception.io.kitti import load_kitti_bin

        if target.suffix.lower() != ".bin":
            raise ValueError("KITTI Velodyne requires .bin")
        cloud = load_kitti_bin(target, label_path=options.label_path)
    elif manifest.adapter_id == "nuscenes-lidar-top":
        if target.suffix.lower() != ".bin":
            raise ValueError("nuScenes raw input requires .bin")
        if target.stat().st_size == 0 or target.stat().st_size % 20:
            raise ValueError("nuScenes raw file must contain complete 20-byte records")
        if options.timestamp_microseconds is None or options.sample_id is None:
            raise ValueError(
                "nuScenes raw input requires explicit timestamp_microseconds and sample_id"
            )
        from laserperception.detection.multisweep import RawSweep

        sweep = RawSweep.from_nuscenes_file(
            target,
            timestamp_microseconds=options.timestamp_microseconds,
            source_id=options.sample_id,
        )
        cloud = PointCloud(
            sweep.points[:, :3],
            attributes={"intensity": sweep.points[:, 3], "ring_index": sweep.points[:, 4]},
            metadata={
                "source_id": sweep.source_id,
                "timestamp_microseconds": sweep.timestamp_microseconds,
                "coordinates_normalized": False,
            },
        )
        timestamp = sweep.timestamp_microseconds * 1000
    elif manifest.adapter_id == "semantickitti":
        if options.split is None:
            raise ValueError("SemanticKITTI directory requires explicit split")
        from laserperception.datasets.semantickitti import SemanticKITTIDataset

        from .sequences import SemanticKITTISequence

        sequence = SemanticKITTISequence(
            SemanticKITTIDataset(target, split=options.split, sequences=options.sequences)
        )
        info = sequence.sample_info(options.index)
        cloud = sequence.load(options.index)
        sample_id = info.sample_id
    elif manifest.adapter_id == "dales":
        if options.split is None:
            raise ValueError("DALES directory requires explicit split")
        from laserperception.datasets.dales import DalesDataset

        from .sequences import DalesSequence

        tiles = DalesSequence(DalesDataset(target, split=options.split))
        info = tiles.sample_info(options.index)
        cloud = tiles.load(options.index)
        sample_id = info.sample_id
    elif manifest.adapter_id == "kitti-raw":
        if options.date_root is None:
            raise ValueError("KITTI Raw drive requires explicit date_root")
        from laserperception.datasets.kitti_raw import KittiRawSequence

        from .sequences import KittiRawDataSequence

        raw = KittiRawDataSequence(KittiRawSequence(options.date_root, target))
        info = raw.sample_info(options.index)
        cloud = raw.load(options.index)
        sample_id = info.sample_id
        timestamp = info.timestamp_ns
    else:
        raise ValueError("PointCloud2 is an API message boundary; use load_pointcloud2_xyz")
    cloud.metadata.update(adapter_id=manifest.adapter_id, sample_id=sample_id)
    return LoadedInput(manifest, cloud, sample_id, timestamp)


@dataclass(frozen=True)
class FileDataAdapter:
    adapter_id: str

    def describe(self) -> DataAdapterManifest:
        return registry.get(self.adapter_id)

    def load(self, path: str | Path, options: InputOptions | None = None) -> LoadedInput:
        return load_input(path, adapter_id=self.adapter_id, options=options)


def load_pointcloud2_xyz(layout: "PointCloud2Layout", header: "SourceHeader") -> LoadedInput:
    from laserperception.detection.ros2_contract import decode_raw_xyz_pointcloud

    decoded = decode_raw_xyz_pointcloud(layout)
    timestamp = header.stamp.sec * 1_000_000_000 + header.stamp.nanosec
    sample_id = f"{header.frame_id}:{timestamp}"
    cloud = PointCloud(
        decoded.points_xyz,
        metadata={
            "adapter_id": "pointcloud2-xyz",
            "sample_id": sample_id,
            "frame_id": header.frame_id,
            "timestamp_ns": timestamp,
            "source_point_count": decoded.source_point_count,
            "invalid_point_count": decoded.invalid_point_count,
            "original_source_row_mapping": "unavailable after finite filtering",
            "coordinates_normalized": False,
        },
    )
    return LoadedInput(registry.get("pointcloud2-xyz"), cloud, sample_id, timestamp)


if TYPE_CHECKING:
    from laserperception.detection.ros2_contract import PointCloud2Layout, SourceHeader
