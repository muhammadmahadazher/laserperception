"""Deterministic local model discovery."""

from collections.abc import Iterable
from importlib.resources import files

from .contracts import PerceptionTask
from .manifests import ModelManifest


class ModelRegistry:
    def __init__(self, manifests: Iterable[ModelManifest]) -> None:
        records = tuple(manifests)
        if not all(isinstance(m, ModelManifest) for m in records):
            raise TypeError("registry requires ModelManifest records")
        self._models = {m.model_id: m for m in records}
        if len(self._models) != len(records):
            raise ValueError("duplicate model ID")

    def list_models(self) -> tuple[ModelManifest, ...]:
        return tuple(self._models[key] for key in sorted(self._models))

    def get(self, model_id: str) -> ModelManifest:
        try:
            return self._models[model_id]
        except KeyError:
            raise ValueError(
                f"unknown model {model_id!r}; use 'laserperception models list'"
            ) from None

    def inspect(self, model_id: str) -> dict[str, object]:
        return self.get(model_id).to_dict()

    def filter_by_task(self, task: PerceptionTask) -> tuple[ModelManifest, ...]:
        return tuple(m for m in self.list_models() if task in m.tasks)

    def filter_by_runtime(self, target: str) -> tuple[ModelManifest, ...]:
        return tuple(m for m in self.list_models() if target in m.capabilities.runtime_targets)


def builtin_registry() -> ModelRegistry:
    resources = files("laserperception.perception").joinpath("models")
    return ModelRegistry(
        ModelManifest.from_json(p.read_text(encoding="utf-8"))
        for p in sorted(resources.iterdir(), key=lambda p: p.name)
        if p.name.endswith(".json")
    )
