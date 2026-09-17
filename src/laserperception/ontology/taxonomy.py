"""Versioned taxonomy descriptions using the existing Experiment 001 policy."""

from dataclasses import dataclass

from laserperception.perception.serialization import JsonRecord

from .mappings import CLASS_NAMES, IGNORE_ID, SharedClass


@dataclass(frozen=True)
class SemanticClass(JsonRecord):
    class_id: int
    name: str
    ignored: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if not -(2**63) <= self.class_id < 2**63:
            raise ValueError("taxonomy class ID exceeds signed int64")


@dataclass(frozen=True)
class SemanticTaxonomy(JsonRecord):
    taxonomy_id: str
    version: str
    classes: tuple[SemanticClass, ...]
    unknown_id: int = -2

    def __post_init__(self) -> None:
        super().__post_init__()
        if not -(2**63) <= self.unknown_id < 2**63:
            raise ValueError("taxonomy unknown ID exceeds signed int64")
        ids = [c.class_id for c in self.classes]
        if len(set(ids)) != len(ids) or len({c.name for c in self.classes}) != len(ids):
            raise ValueError("taxonomy class IDs and names must be unique")
        if not any(not c.ignored for c in self.classes):
            raise ValueError("taxonomy requires an evaluable class")
        if self.unknown_id in ids:
            raise ValueError("unknown ID must differ from declared class IDs")
        object.__setattr__(self, "classes", tuple(sorted(self.classes, key=lambda c: c.class_id)))


EXPERIMENT_001_TAXONOMY = SemanticTaxonomy(
    "laserperception.experiment-001.six-class",
    "1",
    tuple(SemanticClass(int(c), CLASS_NAMES[int(c)]) for c in SharedClass)
    + (SemanticClass(IGNORE_ID, "Ignored", True),),
)
