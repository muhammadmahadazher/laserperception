"""Lazy canonical PointCloud access over existing dataset readers."""

from collections.abc import Iterator
from pathlib import Path

from laserperception.core import PointCloud
from laserperception.datasets.dales import DalesDataset
from laserperception.datasets.kitti_raw import KittiRawSequence
from laserperception.datasets.semantickitti import SemanticKITTIDataset

from .contracts import SampleRef


def _check_index(index: int, count: int) -> None:
    if type(index) is not int or not 0 <= index < count:
        raise ValueError("dataset index must be an integer within sequence bounds")


class SemanticKITTISequence:
    def __init__(self, dataset: SemanticKITTIDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def sample_info(self, index: int) -> SampleRef:
        _check_index(index, len(self))
        info = self.dataset.sample_info(index)
        return SampleRef(
            "semantickitti",
            f"semantickitti:{info.sequence}:{info.frame}",
            index,
            info.scan_path,
            info.label_path,
        )

    def __iter__(self) -> Iterator[SampleRef]:
        for i in range(len(self)):
            yield self.sample_info(i)

    def load(self, index: int) -> PointCloud:
        _check_index(index, len(self))
        cloud = self.dataset.load(index)
        cloud.metadata.update(
            adapter_id="semantickitti", sample_id=self.sample_info(index).sample_id
        )
        return cloud


class DalesSequence:
    def __init__(self, dataset: DalesDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def sample_info(self, index: int) -> SampleRef:
        _check_index(index, len(self))
        info = self.dataset.tile_info(index)
        return SampleRef("dales", f"dales:{info.split}:{info.relative_path}", index, info.tile_path)

    def __iter__(self) -> Iterator[SampleRef]:
        for i in range(len(self)):
            yield self.sample_info(i)

    def load(self, index: int) -> PointCloud:
        _check_index(index, len(self))
        from laserperception.io.las import load_las

        cloud = load_las(self.sample_info(index).path)
        cloud.metadata.update(adapter_id="dales", sample_id=self.sample_info(index).sample_id)
        return cloud

    def iter_chunks(self, index: int, *, chunk_size: int = 1_000_000) -> Iterator[PointCloud]:
        """Keep XYZ/labels only, exposing exact contiguous tile source-row offsets."""
        _check_index(index, len(self))
        offset = 0
        for cloud in self.dataset.iter_tile_chunks(index, chunk_size=chunk_size):
            cloud.metadata.update(
                source_row_start=offset,
                source_row_stop=offset + len(cloud),
                sample_id=self.sample_info(index).sample_id,
            )
            offset += len(cloud)
            yield cloud


class KittiRawDataSequence:
    def __init__(self, sequence: KittiRawSequence) -> None:
        self.sequence = sequence

    def __len__(self) -> int:
        return len(self.sequence)

    def sample_info(self, index: int) -> SampleRef:
        _check_index(index, len(self))
        stamp = self.sequence.timestamps[index]
        path = Path(self.sequence.drive_root) / "velodyne_points/data" / f"{index:010d}.bin"
        return SampleRef(
            "kitti-raw",
            f"{self.sequence.drive_root.name}:{index:010d}",
            index,
            path,
            timestamp_ns=stamp.nanoseconds,
        )

    def __iter__(self) -> Iterator[SampleRef]:
        for i in range(len(self)):
            yield self.sample_info(i)

    def load(self, index: int) -> PointCloud:
        _check_index(index, len(self))
        frame = self.sequence.frame(index)
        return PointCloud(
            frame.points_xyzi[:, :3],
            attributes={"remission": frame.points_xyzi[:, 3]},
            metadata={
                "adapter_id": "kitti-raw",
                "sample_id": self.sample_info(index).sample_id,
                "timestamp_ns": frame.timestamp.nanoseconds,
                "coordinate_frame": "native KITTI Velodyne",
                "coordinates_normalized": False,
            },
        )
