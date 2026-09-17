"""Static ingestion metadata and lightweight lazy sequence contracts."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from laserperception.core import PointCloud
from laserperception.perception.contracts import PerceptionTask
from laserperception.perception.serialization import JsonRecord
from laserperception.perception.validation import ValidationReport


@dataclass(frozen=True)
class DataAdapterManifest(JsonRecord):
    adapter_id: str
    display_name: str
    category: Literal["file", "dataset", "message"]
    formats: tuple[str, ...]
    supplied_features: tuple[str, ...]
    tasks: tuple[PerceptionTask, ...]
    label_semantics: str
    temporal_support: str
    calibration_requirements: tuple[str, ...]
    coordinate_semantics: str
    optional_dependencies: tuple[str, ...]
    indexing: str
    streaming: str
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.formats or not self.supplied_features or not self.tasks:
            raise ValueError("adapter formats, features and tasks must be non-empty")
        for values in (self.formats, self.supplied_features, self.tasks):
            if len(set(values)) != len(values):
                raise ValueError("duplicate adapter capability")


@dataclass(frozen=True)
class InputOptions:
    label_path: Path | None = None
    timestamp_microseconds: int | None = None
    sample_id: str | None = None
    split: str | None = None
    sequences: tuple[str, ...] | None = None
    index: int = 0
    date_root: Path | None = None

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("input index must be non-negative integer")
        if self.timestamp_microseconds is not None and (
            type(self.timestamp_microseconds) is not int
            or not 0 <= self.timestamp_microseconds < 2**63 // 1000
        ):
            raise ValueError("timestamp_microseconds must fit non-negative nanoseconds")
        if self.sample_id is not None and not self.sample_id.strip():
            raise ValueError("sample identity cannot be empty")


@dataclass(frozen=True)
class LoadedInput:
    adapter: DataAdapterManifest
    cloud: PointCloud
    sample_id: str
    timestamp_ns: int | None


class DataAdapter(Protocol):
    def describe(self) -> DataAdapterManifest: ...
    def load(self, path: str | Path, options: InputOptions | None = None) -> LoadedInput: ...


@dataclass(frozen=True)
class SampleRef:
    adapter_id: str
    sample_id: str
    index: int
    path: Path
    label_path: Path | None = None
    timestamp_ns: int | None = None


class DatasetSequence(Protocol):
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[SampleRef]: ...
    def sample_info(self, index: int) -> SampleRef: ...
    def load(self, index: int) -> PointCloud: ...


@dataclass(frozen=True)
class PointAttributeInfo(JsonRecord):
    name: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class InputInspection(JsonRecord):
    adapter: DataAdapterManifest
    sample_id: str
    point_count: int
    attributes: tuple[PointAttributeInfo, ...]
    labels_available: bool
    label_dtype: str | None
    coordinate_semantics: str
    timestamp_ns: int | None
    model_id: str | None
    compatibility: ValidationReport | None
    compatibility_verified: bool
    limitations: tuple[str, ...]
