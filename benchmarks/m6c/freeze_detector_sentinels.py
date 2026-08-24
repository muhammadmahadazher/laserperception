"""Freeze M6c detector sentinels from immutable accepted M6b evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from laserperception.datasets.kitti_raw import KittiRawSequence
from laserperception.detection.mmdet3d_backend import sha256_file

M6A_SHA256 = "a62da9808079994d971c4d47bbdf04f2c50d44ddcfd0c7958f7534512552155b"
M6B_COMPACT_SHA256 = "b9d47120aabb38733d79f987f16277ebab626d2b7c13159c40c30a68b76c1d26"
M6B_LEDGER_SHA256 = "2413224808b0140856a0e00f884c53bc9b49ec6b545172f5e7fa57d00802dc15"
M6B_FULL_SHA256 = "87870b2aa0cc2a91d39331afc8154fdad0c8c796f1cabfb4f8530a3eb106de27"
M6B_FULL_LEDGER_SHA256 = "e25b3d62113cc7e8c1fcf736caa68b1ab698f965f007c758ff91d3e498ca6caa"
SENTINEL_FRAMES = (
    "2011_09_26_drive_0001/0000000010",
    "2011_09_26_drive_0001/0000000083",
    "2011_09_26_drive_0001/0000000011",
    "2011_09_26_drive_0001/0000000015",
    "2011_09_26_drive_0091/0000000010",
)
CONDITIONS = ("H10", "H5")


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON mapping in {path.name}")
    return value


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _require(path: Path, expected: str) -> dict[str, object]:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"frozen artifact SHA256 mismatch: {path.name}")
    return {
        "path": path.relative_to(_root()).as_posix(),
        "sha256": actual,
        "size_bytes": path.stat().st_size,
    }


def _execution_identity(execution: Mapping[str, object]) -> dict[str, object]:
    return {
        "model_ready_sha256": execution["model_ready_sha256"],
        "point_count": execution["point_count"],
        "history_indices": execution["history_indices"],
        "voxel_count": execution["voxel_count"],
        "voxel_hashes": execution["voxel_hashes"],
        "raw_output_hashes": execution["raw_output_hashes"],
        "detection_frame_sha256": execution["detection_frame_sha256"],
        "detection_count": execution["detection_count_all_postprocessed_scores"],
    }


def _sentinel_record(
    *,
    frame_id: str,
    condition: str,
    compact_condition: Mapping[str, object],
    full_frame: Mapping[str, object],
    checkpoint: Mapping[str, object],
    checkpoint_path: Path,
    sequence: KittiRawSequence,
) -> dict[str, object]:
    if checkpoint.get("status") != "COMPLETE":
        raise RuntimeError(f"accepted M6b checkpoint is incomplete: {frame_id} {condition}")
    if checkpoint.get("frame_id") != frame_id or checkpoint.get("condition") != condition:
        raise RuntimeError(f"accepted M6b checkpoint identity mismatch: {frame_id} {condition}")
    execution = checkpoint.get("execution")
    full_execution = full_frame.get("execution")
    detection_frame = checkpoint.get("detection_frame")
    if not isinstance(execution, Mapping) or not isinstance(full_execution, Mapping):
        raise RuntimeError("accepted M6b execution record is malformed")
    if not isinstance(detection_frame, Mapping):
        raise RuntimeError("accepted M6b DetectionFrame payload is unavailable")
    checkpoint_identity = _execution_identity(execution)
    full_identity = _execution_identity(full_execution)
    if checkpoint_identity != full_identity:
        raise RuntimeError(f"checkpoint/full-result identity mismatch: {frame_id} {condition}")
    if checkpoint_identity["model_ready_sha256"] != compact_condition["model_ready_input_sha256"]:
        raise RuntimeError(f"checkpoint/compact-input identity mismatch: {frame_id} {condition}")
    if checkpoint_identity["point_count"] != compact_condition["point_count"]:
        raise RuntimeError(f"checkpoint/compact-input point-count mismatch: {frame_id} {condition}")
    if checkpoint_identity["voxel_count"] != compact_condition["voxel_count"]:
        raise RuntimeError(f"checkpoint/compact-input voxel-count mismatch: {frame_id} {condition}")
    frame_hash = _canonical_sha256(detection_frame)
    if frame_hash != checkpoint_identity["detection_frame_sha256"]:
        raise RuntimeError(
            f"checkpoint DetectionFrame payload hash mismatch: {frame_id} {condition}"
        )
    drive_id, frame_text = frame_id.split("/", 1)
    frame_index = int(frame_text)
    timestamp = sequence.timestamps[frame_index]
    return {
        "condition": condition,
        "drive": drive_id,
        "frame": frame_text,
        "frame_id": frame_id,
        "history_depth": int(compact_condition["history_depth"]),
        "official_timestamp": {
            "text": timestamp.original_text,
            "nanoseconds": timestamp.nanoseconds,
            "builder_microseconds": timestamp.microseconds,
        },
        "expected": checkpoint_identity,
        "detection_frame": dict(detection_frame),
        "accepted_checkpoint": {
            "logical_path": (
                f".local/m6b-r2/predictions/{drive_id}/{frame_text}_{condition.lower()}.json"
            ),
            "sha256": sha256_file(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--full-result",
        type=Path,
        default=_root()
        / ".local/m6b-r2/evidence/kitti_raw_cross_domain_characterization_full.json",
    )
    parser.add_argument(
        "--full-ledger",
        type=Path,
        default=_root() / ".local/m6b-r2/evidence/pre_inference_input_ledger_full.json",
    )
    parser.add_argument(
        "--checkpoint-root", type=Path, default=_root() / ".local/m6b-r2/predictions"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_root() / "benchmarks/m6c/preregistration/detector_sentinels.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _root()
    source_paths = {
        "m6a_compact": root / "benchmarks/m6a/results/kitti_raw_offline_reconstruction.json",
        "m6b_compact": root / "benchmarks/m6b/results/kitti_raw_cross_domain_characterization.json",
        "m6b_compact_input_ledger": root
        / "benchmarks/m6b/diagnostics/pre_inference_input_ledger.json",
        "m6b_full_result": args.full_result.expanduser().resolve(),
        "m6b_full_input_ledger": args.full_ledger.expanduser().resolve(),
    }
    expected_hashes = {
        "m6a_compact": M6A_SHA256,
        "m6b_compact": M6B_COMPACT_SHA256,
        "m6b_compact_input_ledger": M6B_LEDGER_SHA256,
        "m6b_full_result": M6B_FULL_SHA256,
        "m6b_full_input_ledger": M6B_FULL_LEDGER_SHA256,
    }
    sources: dict[str, object] = {}
    for name, path in source_paths.items():
        if name.startswith("m6b_full"):
            actual = sha256_file(path)
            if actual != expected_hashes[name]:
                raise RuntimeError(f"frozen artifact SHA256 mismatch: {name}")
            sources[name] = {
                "logical_path": f".local/m6b-r2/evidence/{path.name}",
                "sha256": actual,
                "size_bytes": path.stat().st_size,
            }
        else:
            sources[name] = _require(path, expected_hashes[name])

    compact_ledger = _load(source_paths["m6b_compact_input_ledger"])
    full_result = _load(source_paths["m6b_full_result"])
    compact_by_key = {
        f"{record['drive']}/{record['frame']}|{record['condition']}": record
        for record in compact_ledger["conditions"]
    }
    full_by_key = {
        f"{record['frame_id']}|{condition}": record
        for condition in CONDITIONS
        for record in full_result["frame_results"][condition]
    }
    data_root = args.data_root.expanduser().resolve()
    date_root = data_root / "2011_09_26"
    sequences = {
        drive: KittiRawSequence(date_root, date_root / f"{drive}_sync")
        for drive in {frame.split("/", 1)[0] for frame in SENTINEL_FRAMES}
    }
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    sentinels: list[dict[str, object]] = []
    for frame_id in SENTINEL_FRAMES:
        drive, frame_text = frame_id.split("/", 1)
        for condition in CONDITIONS:
            key = f"{frame_id}|{condition}"
            if key not in compact_by_key or key not in full_by_key:
                raise RuntimeError(f"sentinel is absent from accepted M6b evidence: {key}")
            checkpoint_path = checkpoint_root / drive / f"{frame_text}_{condition.lower()}.json"
            checkpoint = _load(checkpoint_path)
            sentinels.append(
                _sentinel_record(
                    frame_id=frame_id,
                    condition=condition,
                    compact_condition=compact_by_key[key],
                    full_frame=full_by_key[key],
                    checkpoint=checkpoint,
                    checkpoint_path=checkpoint_path,
                    sequence=sequences[drive],
                )
            )

    result = {
        "schema_version": 1,
        "artifact_role": "M6c preregistered detector identities; no M6c inference performed",
        "status": "FROZEN_BEFORE_M6C_DETECTOR_EXECUTION",
        "source_artifacts": sources,
        "selection": {
            "frames": list(SENTINEL_FRAMES),
            "conditions": list(CONDITIONS),
            "condition_count": len(sentinels),
            "selected_without_M6c_detector_results": True,
        },
        "ros_output_contract": {
            "velocity_exposed": False,
            "decision": (
                "vision_msgs/Detection3DArray carries class, score, pose, and size in the current "
                "LaserPerception conversion; velocity_xy is intentionally not overloaded."
            ),
        },
        "oracle_independence": {
            "shared": [
                "KittiRawSequence supplies frozen point-axis conversion and OXTS sensor poses",
                "MultiSweepBuilder defines retained-row and transform arithmetic",
            ],
            "independent_ros_boundaries_under_test": [
                "raw PointCloud2 byte serialization and decoding",
                "tf2 transport and cross-time lookup_transform_full",
                "LiveSweepHistory selection",
                "model-ready PointCloud2 serialization and decoding",
            ],
            "builder_node_calls_KittiRawSequence": False,
        },
        "sentinels": sentinels,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if output.stat().st_size >= 5_000_000:
        raise RuntimeError("M6c sentinel preregistration exceeds the 5 MB tracked-file hard stop")
    print(
        f"froze {len(sentinels)} M6c detector conditions; "
        f"sha256={sha256_file(output)} size={output.stat().st_size}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
