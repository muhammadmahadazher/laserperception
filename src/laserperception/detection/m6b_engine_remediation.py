"""Fail-closed contracts for the prospective M6b structural engine remediation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

HISTORICAL_ENGINE_LOGICAL_NAME = "engines/pointpillars_fp16.engine"
CANDIDATE_ENGINE_LOGICAL_NAME = "engines/m6/pointpillars_fp16_profile40k.engine"
EXPECTED_ONNX_SHA256 = "61ce22a8ca31498675c32576bfb94f0093d31dc95d2762f7254bf915a59ecc16"
EXPECTED_PROFILE_COUNTS = {"min": 4352, "opt": 18207, "max": 40000}
M6B_EVALUATION_DRIVES = frozenset({"2011_09_26_drive_0001", "2011_09_26_drive_0091"})
NON_EVALUATION_DRIVE = "2011_09_30_drive_0016"
PARITY_QUANTILES = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)
HISTORICAL_MANIFEST_RELATIVE = Path("configs/detection/m2_pointpillars_tensorrt.yaml")


def _required_record_int(record: Mapping[str, object], key: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"third-drive frame {key} must be an integer")
    return value


def resolve_build_manifest_path(repository_root: str | Path, explicit: str | Path | None) -> Path:
    """Resolve an explicit manifest while preserving the historical M2 default."""

    root = Path(repository_root).resolve()
    if explicit is None:
        return root / HISTORICAL_MANIFEST_RELATIVE
    value = Path(explicit).expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def load_engine_manifest(path: str | Path) -> dict[str, Any]:
    """Load a YAML engine manifest and require a mapping at its root."""

    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("engine manifest root must be a mapping")
    return dict(value)


def profile_shapes(manifest: Mapping[str, Any]) -> dict[str, dict[str, list[int]]]:
    """Return normalized TensorRT input shapes from one manifest."""

    profile = manifest.get("profile")
    if not isinstance(profile, Mapping):
        raise ValueError("engine manifest profile must be a mapping")
    result: dict[str, dict[str, list[int]]] = {}
    for name in ("voxels", "num_points", "coors"):
        result[name] = {}
        for source, destination in (
            ("selected_min_shapes", "min_shape"),
            ("selected_opt_shapes", "opt_shape"),
            ("selected_max_shapes", "max_shape"),
        ):
            shapes = profile.get(source)
            if not isinstance(shapes, Mapping) or name not in shapes:
                raise ValueError(f"engine manifest is missing {source}.{name}")
            shape = shapes[name]
            if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
                raise ValueError(f"engine profile shape {source}.{name} must be a sequence")
            result[name][destination] = [int(value) for value in shape]
    return result


def validate_candidate_manifest(
    candidate: Mapping[str, Any], historical: Mapping[str, Any]
) -> dict[str, dict[str, list[int]]]:
    """Validate the sole authorized structural-profile change against historical M2."""

    candidate_onnx = candidate["artifacts"]["onnx"]
    historical_onnx = historical["artifacts"]["onnx"]
    if candidate_onnx["sha256"] != EXPECTED_ONNX_SHA256:
        raise ValueError("candidate manifest does not require the frozen M2 ONNX SHA256")
    if candidate_onnx["sha256"] != historical_onnx["sha256"]:
        raise ValueError("candidate and historical manifests do not identify the same ONNX")
    candidate_engine = candidate["artifacts"]["engine"]
    historical_engine = historical["artifacts"]["engine"]
    if candidate_engine["logical_name"] == historical_engine["logical_name"]:
        raise ValueError("candidate engine logical identity must differ from historical M2")
    if candidate_engine["logical_name"] != CANDIDATE_ENGINE_LOGICAL_NAME:
        raise ValueError("candidate engine logical identity is not the preregistered name")
    if historical_engine["logical_name"] != HISTORICAL_ENGINE_LOGICAL_NAME:
        raise ValueError("historical M2 engine logical identity changed")

    candidate_shapes = profile_shapes(candidate)
    historical_shapes = profile_shapes(historical)
    expected = {
        "voxels": {
            "min_shape": [4352, 64, 4],
            "opt_shape": [18207, 64, 4],
            "max_shape": [40000, 64, 4],
        },
        "num_points": {
            "min_shape": [4352],
            "opt_shape": [18207],
            "max_shape": [40000],
        },
        "coors": {
            "min_shape": [4352, 4],
            "opt_shape": [18207, 4],
            "max_shape": [40000, 4],
        },
    }
    if candidate_shapes != expected:
        raise ValueError("candidate TensorRT profile is not exactly 4352/18207/40000")
    for name in expected:
        if candidate_shapes[name]["min_shape"] != historical_shapes[name]["min_shape"]:
            raise ValueError("candidate minimum differs from historical M2")
        if candidate_shapes[name]["opt_shape"] != historical_shapes[name]["opt_shape"]:
            raise ValueError("candidate optimum differs from historical M2")
    historical_maximum = historical_shapes["voxels"]["max_shape"][0]
    if historical_maximum != 30000:
        raise ValueError("historical M2 maximum no longer equals 30000")
    voxel_contract = candidate.get("voxel_contract")
    if not isinstance(voxel_contract, Mapping):
        raise ValueError("candidate voxel contract must be a mapping")
    upstream = voxel_contract.get("upstream_max_voxels")
    if not isinstance(upstream, Mapping):
        raise ValueError("candidate upstream voxel limits must be a mapping")
    validation_maximum = int(upstream["validation"])
    if candidate_shapes["voxels"]["max_shape"][0] < validation_maximum:
        raise ValueError("candidate profile maximum is below the voxelizer validation maximum")
    return candidate_shapes


def reject_evaluation_drive(drive_id: str) -> None:
    """Refuse any prospective M6b-R1 network execution on evaluation drives."""

    if drive_id in M6B_EVALUATION_DRIVES:
        raise ValueError(f"M6b-R1 forbids network output for evaluation drive {drive_id}")
    if drive_id != NON_EVALUATION_DRIVE:
        raise ValueError(f"M6b-R1 authorizes only non-evaluation drive {NON_EVALUATION_DRIVE}")


def select_third_drive_frames(
    frames: Sequence[Mapping[str, object]], *, required_count: int = 12
) -> list[dict[str, object]]:
    """Select the preregistered quantile-spanning non-evaluation frame set."""

    if required_count != len(PARITY_QUANTILES):
        raise ValueError("M6b-R1 requires exactly 12 third-drive parity frames")
    normalized = [
        {
            **dict(frame),
            "frame_index": _required_record_int(frame, "frame_index"),
            "voxel_count": _required_record_int(frame, "voxel_count"),
        }
        for frame in frames
    ]
    if len(normalized) < required_count:
        raise ValueError("third drive has fewer than 12 eligible full-history frames")
    if len({record["frame_index"] for record in normalized}) != len(normalized):
        raise ValueError("third-drive frame indices must be unique")
    by_count = sorted(normalized, key=lambda record: (record["voxel_count"], record["frame_index"]))
    selected: list[dict[str, object]] = []
    selected_indices: set[int] = set()
    for quantile in PARITY_QUANTILES:
        rank = 1 if quantile == 0 else math.ceil((quantile / 100) * len(by_count))
        target_count = _required_record_int(by_count[rank - 1], "voxel_count")
        chosen = min(
            (record for record in normalized if record["voxel_count"] == target_count),
            key=lambda record: _required_record_int(record, "frame_index"),
        )
        index = _required_record_int(chosen, "frame_index")
        if index not in selected_indices:
            selected.append({**chosen, "selection_source": f"nearest_rank_q{quantile}"})
            selected_indices.add(index)

    while len(selected) < required_count:
        selected_counts = [_required_record_int(record, "voxel_count") for record in selected]
        unused = [
            record
            for record in normalized
            if _required_record_int(record, "frame_index") not in selected_indices
        ]
        chosen = min(
            unused,
            key=lambda record: (
                -min(
                    abs(_required_record_int(record, "voxel_count") - selected_count)
                    for selected_count in selected_counts
                ),
                _required_record_int(record, "frame_index"),
            ),
        )
        selected.append({**chosen, "selection_source": "greatest_unused_voxel_distance"})
        selected_indices.add(_required_record_int(chosen, "frame_index"))

    selected.sort(key=lambda record: _required_record_int(record, "frame_index"))
    counts = [_required_record_int(record, "voxel_count") for record in selected]
    if not any(count <= 30000 for count in counts):
        raise ValueError("THIRD-DRIVE PROFILE-COVERAGE INSUFFICIENT: no frame <=30000")
    if sum(count > 30000 for count in counts) < 4:
        raise ValueError("THIRD-DRIVE PROFILE-COVERAGE INSUFFICIENT: fewer than four frames >30000")
    if not any(count >= 39000 for count in counts):
        raise ValueError("THIRD-DRIVE PROFILE-COVERAGE INSUFFICIENT: no frame >=39000")
    return selected


def select_repeatability_frames(
    selected: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Choose expanded-envelope and mid-profile repeatability frames deterministically."""

    if not selected:
        raise ValueError("repeatability selection requires at least one frame")
    highest = min(
        selected,
        key=lambda record: (
            -_required_record_int(record, "voxel_count"),
            _required_record_int(record, "frame_index"),
        ),
    )
    optimum = EXPECTED_PROFILE_COUNTS["opt"]
    mid_range = min(
        selected,
        key=lambda record: (
            abs(_required_record_int(record, "voxel_count") - optimum),
            _required_record_int(record, "frame_index"),
        ),
    )
    return {"highest_shape": dict(highest), "mid_range_near_opt": dict(mid_range)}
