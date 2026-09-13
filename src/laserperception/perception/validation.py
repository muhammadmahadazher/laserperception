"""CPU-only compatibility checks; no transformation or feature synthesis."""

from dataclasses import dataclass, fields

from .contracts import CoordinateContract, FeatureSpec, TemporalContract
from .manifests import ModelManifest
from .serialization import JsonRecord


@dataclass(frozen=True)
class ValidationReport(JsonRecord):
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    missing_required_features: tuple[str, ...]
    extra_features: tuple[str, ...]
    coordinate_incompatibilities: tuple[str, ...]
    temporal_incompatibilities: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors


def validate_input(
    manifest: ModelManifest,
    features: tuple[FeatureSpec, ...],
    *,
    coordinates: CoordinateContract | None = None,
    temporal: TemporalContract | None = None,
) -> ValidationReport:
    """Compare supplied metadata; omitted frame/history remain explicitly unverified."""
    available = {f.name: f for f in features}
    expected = {f.name: f for f in manifest.input_features}
    missing = tuple(
        f.name for f in manifest.input_features if f.required and f.name not in available
    )
    extra = tuple(sorted(set(available) - set(expected)))
    errors = [f"missing required feature: {name}" for name in missing]
    warnings = [f"extra feature requires explicit preparation: {name}" for name in extra]
    if len(available) != len(features):
        errors.append("duplicate input feature names")
    for name in available.keys() & expected.keys():
        for attribute in ("dtype", "unit", "position", "semantics", "source"):
            if getattr(available[name], attribute) != getattr(expected[name], attribute):
                errors.append(f"feature {name}: incompatible {attribute}")
    coordinate_errors: list[str] = []
    temporal_errors: list[str] = []
    if coordinates is None:
        warnings.append("coordinate compatibility not checked")
    else:
        coordinate_errors = [
            f.name
            for f in fields(coordinates)
            if getattr(coordinates, f.name) != getattr(manifest.coordinates, f.name)
        ]
    if temporal is None:
        warnings.append("temporal compatibility not checked")
    else:
        temporal_errors = [
            f.name
            for f in fields(temporal)
            if f.name not in {"min_history", "max_history"}
            and getattr(temporal, f.name) != getattr(manifest.temporal, f.name)
        ]
        if (
            not manifest.temporal.min_history
            <= temporal.min_history
            <= temporal.max_history
            <= manifest.temporal.max_history
        ):
            temporal_errors.append("history_range")
    errors.extend(f"coordinate mismatch: {n}" for n in coordinate_errors)
    errors.extend(f"temporal mismatch: {n}" for n in temporal_errors)
    return ValidationReport(
        tuple(sorted(errors)),
        tuple(warnings),
        missing,
        extra,
        tuple(coordinate_errors),
        tuple(temporal_errors),
    )
