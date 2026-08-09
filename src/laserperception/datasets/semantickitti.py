"""SemanticKITTI directory discovery using the official sequence splits.

Structure and splits are pinned to the official SemanticKITTI API configuration:
https://github.com/PRBonn/semantic-kitti-api/blob/a9c749e8124b2243b6eef1b8bcf971a9f1173a2d/config/semantic-kitti.yaml
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from laserperception.core import PointCloud
from laserperception.io import load_kitti_bin

SEMANTICKITTI_ADAPTER_VERSION: Final = "semantickitti-directory-v1"
SEMANTICKITTI_SPLITS = MappingProxyType(
    {
        "train": ("00", "01", "02", "03", "04", "05", "06", "07", "09", "10"),
        "valid": ("08",),
        "test": ("11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"),
    }
)


@dataclass(frozen=True)
class SemanticKITTISample:
    """Paths and stable identifiers for one SemanticKITTI scan."""

    dataset: str
    split: str
    sequence: str
    frame: str
    scan_path: Path
    label_path: Path | None


def _normalize_sequence(sequence: str | int) -> str:
    value = str(sequence)
    if not value.isdigit():
        raise ValueError(f"SemanticKITTI sequence must be numeric; received {sequence!r}")
    return value.zfill(2)


def _frame_sort_key(path: Path) -> tuple[int, str]:
    if not path.stem.isdigit():
        raise ValueError(f"SemanticKITTI frame name must be numeric; received {path.name!r}")
    return int(path.stem), path.stem


class SemanticKITTIDataset:
    """Resolve and load scans from an official SemanticKITTI directory hierarchy.

    ``sequences`` is an optional experiment subset and must remain within the selected official
    split. Omitting it requires every official split sequence to exist. Labels are required by
    default for train/validation and optional for the official test split.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        sequences: tuple[str | int, ...] | list[str | int] | None = None,
        require_labels: bool | None = None,
    ) -> None:
        self.root = Path(root).expanduser()
        if not self.root.is_dir():
            raise FileNotFoundError(f"SemanticKITTI root does not exist: {self.root}")
        if split not in SEMANTICKITTI_SPLITS:
            supported = ", ".join(SEMANTICKITTI_SPLITS)
            raise ValueError(f"unsupported SemanticKITTI split {split!r}; choose {supported}")
        self.split = split
        self.require_labels = split != "test" if require_labels is None else require_labels

        official_sequences = SEMANTICKITTI_SPLITS[split]
        if sequences is None:
            selected_sequences = official_sequences
        else:
            selected_sequences = tuple(_normalize_sequence(sequence) for sequence in sequences)
            if not selected_sequences:
                raise ValueError("SemanticKITTI sequence subset cannot be empty")
            if len(set(selected_sequences)) != len(selected_sequences):
                raise ValueError("SemanticKITTI sequence subset contains duplicates")
            invalid = sorted(set(selected_sequences) - set(official_sequences))
            if invalid:
                raise ValueError(f"sequences {invalid} do not belong to official {split!r} split")
            selected_sequences = tuple(
                sequence for sequence in official_sequences if sequence in selected_sequences
            )
        self.sequences = selected_sequences

        sequences_root = self.root / "sequences"
        if not sequences_root.is_dir() and self.root.name == "sequences":
            sequences_root = self.root
        if not sequences_root.is_dir():
            raise FileNotFoundError("SemanticKITTI root must contain a 'sequences' directory")
        self._sequences_root = sequences_root
        self._samples = self._discover_samples()

    def _discover_samples(self) -> tuple[SemanticKITTISample, ...]:
        samples: list[SemanticKITTISample] = []
        for sequence in self.sequences:
            sequence_dir = self._sequences_root / sequence
            if not sequence_dir.is_dir():
                raise FileNotFoundError(f"missing SemanticKITTI sequence directory: {sequence}")
            scan_dir = sequence_dir / "velodyne"
            if not scan_dir.is_dir():
                raise FileNotFoundError(
                    f"missing SemanticKITTI velodyne directory for sequence {sequence}"
                )
            scans = sorted(scan_dir.glob("*.bin"), key=_frame_sort_key)
            if not scans:
                raise FileNotFoundError(
                    f"no SemanticKITTI .bin scans found for sequence {sequence}"
                )

            label_dir = sequence_dir / "labels"
            labels = (
                {path.stem: path for path in label_dir.glob("*.label")}
                if label_dir.is_dir()
                else {}
            )
            scan_stems = {path.stem for path in scans}
            orphan_labels = sorted(set(labels) - scan_stems)
            missing_labels = sorted(scan_stems - set(labels))
            if orphan_labels:
                raise ValueError(f"sequence {sequence} has labels without scans: {orphan_labels}")
            if self.require_labels and missing_labels:
                raise ValueError(f"sequence {sequence} has scans without labels: {missing_labels}")

            for scan_path in scans:
                samples.append(
                    SemanticKITTISample(
                        dataset="semantickitti",
                        split=self.split,
                        sequence=sequence,
                        frame=scan_path.stem,
                        scan_path=scan_path,
                        label_path=labels.get(scan_path.stem),
                    )
                )
        return tuple(samples)

    def __len__(self) -> int:
        """Return the number of deterministically discovered scans."""
        return len(self._samples)

    def sample_info(self, index: int) -> SemanticKITTISample:
        """Return provenance for one sample without loading its points."""
        return self._samples[index]

    def load(self, index: int) -> PointCloud:
        """Load one raw scan through the existing KITTI I/O layer."""
        sample = self.sample_info(index)
        return load_kitti_bin(sample.scan_path, label_path=sample.label_path)
