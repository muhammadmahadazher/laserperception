"""Generate the complete offline M6c R3 projected-reference population."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from laserperception.datasets.kitti_raw import KittiRawSequence
from laserperception.detection.mmdet3d_backend import sha256_file
from laserperception.detection.multisweep import LidarPose, RawSweep
from laserperception.evaluation.m6c_projected_reference import (
    build_projected_reference_from_sources,
)

BASE_MAIN_SHA = "ebbbc0bbc4423e3be476abcd1165f75a136fa54c"
M6A_DRIVE = "2011_09_26_drive_0001"
SOURCE_IDENTITIES = {
    "m6a": (
        "benchmarks/m6a/results/kitti_raw_offline_reconstruction.json",
        "a62da9808079994d971c4d47bbdf04f2c50d44ddcfd0c7958f7534512552155b",
    ),
    "m6b": (
        "benchmarks/m6b/results/kitti_raw_cross_domain_characterization.json",
        "b9d47120aabb38733d79f987f16277ebab626d2b7c13159c40c30a68b76c1d26",
    ),
    "m6b_input_ledger": (
        "benchmarks/m6b/diagnostics/pre_inference_input_ledger.json",
        "2413224808b0140856a0e00f884c53bc9b49ec6b545172f5e7fa57d00802dc15",
    ),
    "r2_failure": (
        "benchmarks/m6c/diagnostics/gate_a_failure_frame_0000000001.json",
        "fd319dfcb13da570c62d83b4475531bbbd735845808225ebffee4afe03f1cea4",
    ),
    "d1_transform": (
        "benchmarks/m6c/diagnostics/post_failure_tf_representation.json",
        "07ea0434fb5833c96d8e6c619a8459cb43c30bbde97d5cfdba96ac8288f3db5d",
    ),
    "d1_downstream": (
        "benchmarks/m6c/diagnostics/post_failure_downstream_frame10_h10.json",
        "6346a9d0f9916ea4c6e2abb4e7f9c58587a49a5f3b4cbe7ac9d2a6b4b2c3cd3c",
    ),
    "r3_feasibility": (
        "benchmarks/m6c/diagnostics/r3_projected_reference_feasibility.json",
        "b3d2503ed513d258fe2526c162e8e53a51df509a38c6a258a248fcbe29be6b4b",
    ),
    "detector_sentinels": (
        "benchmarks/m6c/preregistration/detector_sentinels.json",
        "e80e803fe8923a52ced1bdecd17841a4cf915352a56ea5427607746409cf3be3",
    ),
}
IMPLEMENTATION_PATHS = {
    "generator": "benchmarks/m6c/generate_projected_reference_manifest.py",
    "projected_reference": "src/laserperception/evaluation/m6c_projected_reference.py",
    "representation": "src/laserperception/evaluation/m6c_representation.py",
    "matrix_to_quaternion": "src/laserperception/datasets/kitti_ros_replay.py",
    "multisweep_builder": "src/laserperception/detection/multisweep.py",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=_root(), check=True, capture_output=True, text=True
    ).stdout.strip()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def _atomic_write(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _verify_identity(implementation_commit: str) -> None:
    if _git("rev-parse", "HEAD") != implementation_commit:
        raise RuntimeError("projected references require the exact implementation commit")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("projected-reference generation requires a clean tracked worktree")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_MAIN_SHA, implementation_commit],
        cwd=_root(),
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError("projected-reference implementation does not descend from base main")


def _verify_sources(
    *, detector_sentinels_bytes: Path | None = None
) -> dict[str, dict[str, object]]:
    verified: dict[str, dict[str, object]] = {}
    for name, (relative, expected) in SOURCE_IDENTITIES.items():
        path = (
            detector_sentinels_bytes
            if name == "detector_sentinels" and detector_sentinels_bytes is not None
            else _root() / relative
        )
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"frozen source identity mismatch: {name}")
        verified[name] = {
            "path": relative,
            "sha256": observed,
            "size_bytes": path.stat().st_size,
            "verification_source": (
                "external canonical tracked bytes"
                if name == "detector_sentinels" and detector_sentinels_bytes is not None
                else "current worktree"
            ),
        }
    return verified


def _condition_key(drive: str, frame: int, condition: str) -> str:
    return f"{drive}/{frame:010d}|{condition}"


def _condition_plan(
    m6a: Mapping[str, Any], m6b: Mapping[str, Any]
) -> tuple[list[dict[str, object]], dict[str, Mapping[str, object]]]:
    memberships: dict[str, set[str]] = defaultdict(set)
    conditions: dict[str, dict[str, object]] = {}
    m6a_frames = m6a["offline_reconstruction"]["frames"]
    for record in m6a_frames:
        frame = int(record["frame_index"])
        key = _condition_key(M6A_DRIVE, frame, "H10")
        memberships[key].add("Gate1A")
        conditions[key] = {
            "drive": M6A_DRIVE,
            "frame_index": frame,
            "condition": "H10",
            "requested_history_depth": 10,
        }
    original_m6b: dict[str, Mapping[str, object]] = {}
    for record in m6b["conditions"]:
        drive = str(record["drive"])
        frame = int(str(record["frame"]))
        condition = str(record["condition"])
        key = _condition_key(drive, frame, condition)
        memberships[key].add("Gate1B")
        conditions[key] = {
            "drive": drive,
            "frame_index": frame,
            "condition": condition,
            "requested_history_depth": int(record["history_depth"]),
        }
        original_m6b[key] = record
    if len(memberships) != 860 or len(original_m6b) != 856:
        raise RuntimeError("projected-reference corpus cardinality changed")
    gate_a = sum("Gate1A" in value for value in memberships.values())
    gate_b = sum("Gate1B" in value for value in memberships.values())
    overlap = sum(value == {"Gate1A", "Gate1B"} for value in memberships.values())
    if (gate_a, gate_b, overlap) != (24, 856, 20):
        raise RuntimeError("projected-reference gate membership changed")
    plan: list[dict[str, object]] = []
    for key, condition in conditions.items():
        plan.append({**condition, "key": key, "gate_membership": sorted(memberships[key])})
    plan.sort(
        key=lambda value: (
            str(value["drive"]),
            int(value["frame_index"]),
            -int(value["requested_history_depth"]),
        )
    )
    return plan, original_m6b


def _transform_sha256(transforms: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for matrix in transforms:
        digest.update(np.ascontiguousarray(matrix, dtype=np.float32).tobytes(order="C"))
    return digest.hexdigest()


def _generate_records(
    data_root: Path, plan: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    by_drive: dict[str, dict[int, list[Mapping[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for condition in plan:
        by_drive[str(condition["drive"])][int(condition["frame_index"])].append(condition)
    records: list[dict[str, object]] = []
    for drive, frame_plan in by_drive.items():
        date_root = data_root / "2011_09_26"
        sequence = KittiRawSequence(date_root, date_root / f"{drive}_sync")
        sweeps: dict[int, RawSweep] = {}
        poses: dict[int, LidarPose] = {}
        maximum_frame = max(frame_plan)
        for frame_index in range(maximum_frame + 1):
            sweeps[frame_index] = sequence.frame(frame_index).to_raw_sweep()
            poses[frame_index] = sequence.lidar_pose(frame_index)
            oldest_retained = frame_index - 10
            for stale in tuple(index for index in sweeps if index < oldest_retained):
                del sweeps[stale]
                del poses[stale]
            for condition in frame_plan.get(frame_index, ()):
                requested_depth = int(condition["requested_history_depth"])
                projected = build_projected_reference_from_sources(
                    current_index=frame_index,
                    history_depth=requested_depth,
                    sweeps=sweeps,
                    poses=poses,
                )
                point_count = len(projected.point_cloud.points_xyzt)
                record = {
                    "key": str(condition["key"]),
                    "gate_membership": list(condition["gate_membership"]),
                    "drive": drive,
                    "frame": f"{frame_index:010d}",
                    "condition": str(condition["condition"]),
                    "requested_history_depth": requested_depth,
                    "expected_actual_history_depth": len(projected.historical_indices),
                    "official_timestamp_nanoseconds": sequence.timestamps[frame_index].nanoseconds,
                    "point_count": point_count,
                    "shape": [point_count, 4],
                    "dtype": "float32",
                    "model_ready_sha256": projected.point_cloud.sha256,
                    "projected_transform_count": len(projected.transforms),
                    "projected_transforms_sha256": _transform_sha256(
                        [value.lidar2sensor for value in projected.transforms]
                    ),
                }
                records.append(record)
                if len(records) == 1 or len(records) % 25 == 0:
                    print(
                        f"projected reference {len(records)}/860: {record['key']} "
                        f"points={point_count}",
                        flush=True,
                    )
    records.sort(key=lambda value: str(value["key"]))
    if len(records) != 860 or len({str(value["key"]) for value in records}) != 860:
        raise RuntimeError("projected-reference generation did not produce 860 unique identities")
    return records


def _population_statistics(
    records: Sequence[Mapping[str, object]],
    original: Mapping[str, Mapping[str, object]],
    *,
    condition: str | None,
) -> dict[str, object]:
    selected = [
        record
        for record in records
        if "Gate1B" in record["gate_membership"]
        and (condition is None or record["condition"] == condition)
    ]
    deltas: list[int] = []
    sha_equal = 0
    point_count_equal = 0
    total_projected = 0
    total_original = 0
    for record in selected:
        source = original[str(record["key"])]
        projected_count = int(record["point_count"])
        original_count = int(source["point_count"])
        delta = projected_count - original_count
        deltas.append(delta)
        total_projected += projected_count
        total_original += original_count
        sha_equal += str(record["model_ready_sha256"]) == str(source["model_ready_input_sha256"])
        point_count_equal += projected_count == original_count
    nonzero_absolute = [abs(value) for value in deltas if value]
    return {
        "conditions_compared": len(selected),
        "model_ready_sha_identical": sha_equal,
        "model_ready_sha_different": len(selected) - sha_equal,
        "point_count_identical": point_count_equal,
        "point_count_different": len(selected) - point_count_equal,
        "signed_point_count_delta_min": min(deltas, default=0),
        "signed_point_count_delta_max": max(deltas, default=0),
        "maximum_absolute_point_count_delta": max(nonzero_absolute, default=0),
        "median_absolute_nonzero_point_count_delta": (
            float(median(nonzero_absolute)) if nonzero_absolute else None
        ),
        "total_projected_points": total_projected,
        "total_original_points": total_original,
    }


def _validate_sha_strings(value: object, *, location: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if "sha256" in str(key).lower() and isinstance(child, str):
                if SHA256_PATTERN.fullmatch(child) is None:
                    raise RuntimeError(f"invalid SHA256 at {child_location}: {child}")
            _validate_sha_strings(child, location=child_location)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _validate_sha_strings(child, location=f"{location}[{index}]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--characterization-output", type=Path, required=True)
    parser.add_argument("--detector-sentinels-bytes", type=Path)
    parser.add_argument("--ros-distro", default="Humble")
    parser.add_argument("--rmw-implementation", default="rmw_fastrtps_cpp")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _verify_identity(args.implementation_commit)
    root = _root()
    source_artifacts = _verify_sources(
        detector_sentinels_bytes=(
            args.detector_sentinels_bytes.expanduser().resolve()
            if args.detector_sentinels_bytes is not None
            else None
        )
    )
    m6a = _load(root / SOURCE_IDENTITIES["m6a"][0])
    m6b_ledger = _load(root / SOURCE_IDENTITIES["m6b_input_ledger"][0])
    plan, original_m6b = _condition_plan(m6a, m6b_ledger)
    records = _generate_records(args.data_root.expanduser().resolve(), plan)
    implementation = {
        name: {
            "path": relative,
            "sha256": sha256_file(root / relative),
        }
        for name, relative in IMPLEMENTATION_PATHS.items()
    }
    environment = {
        "platform": platform.platform(),
        "os_name": os.name,
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "ros_distro_for_live_measurement": args.ros_distro,
        "rmw_implementation_for_live_measurement": args.rmw_implementation,
        "project_commit": args.implementation_commit,
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "FROZEN_PROJECTED_REFERENCE_IDENTITIES_BEFORE_LIVE_R3",
        "artifact_role": "M6c-only same-platform projected offline references; no live output",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "branch": "feat/m6c-kitti-ros-exactness",
        "base_main": BASE_MAIN_SHA,
        "environment": environment,
        "source_artifacts": source_artifacts,
        "implementation": implementation,
        "matrix_to_quaternion_identity": {
            "qualified_name": (
                "laserperception.datasets.kitti_ros_replay.model_lidar_pose_to_world_transform"
            ),
            "source_file_sha256": implementation["matrix_to_quaternion"]["sha256"],
        },
        "independence": {
            "ros_initialized": False,
            "tf2_used": False,
            "ros_messages_used": False,
            "live_builder_node_used": False,
            "detector_executed": False,
            "gpu_initialized": False,
        },
        "population": {
            "unique_conditions": 860,
            "gate_1a_memberships": 24,
            "gate_1b_memberships": 856,
            "overlap_memberships": 20,
        },
        "conditions": records,
    }
    characterization: dict[str, object] = {
        "schema_version": 1,
        "status": "DESCRIPTIVE_ONLY_NOT_A_GATE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_commit": args.implementation_commit,
        "source_artifacts": source_artifacts,
        "comparison": "projected R3 reference minus frozen original M6b model-ready identity",
        "populations": {
            "H10": _population_statistics(records, original_m6b, condition="H10"),
            "H5": _population_statistics(records, original_m6b, condition="H5"),
            "total": _population_statistics(records, original_m6b, condition=None),
        },
        "scope": {
            "ros_outputs_observed": False,
            "detector_executed": False,
            "gpu_initialized": False,
            "point_count_equality_is_gate": False,
        },
    }
    _validate_sha_strings(manifest)
    _validate_sha_strings(characterization)
    _atomic_write(args.manifest_output.expanduser().resolve(), manifest)
    _atomic_write(args.characterization_output.expanduser().resolve(), characterization)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "unique_conditions": len(records),
                "characterization": characterization["populations"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
