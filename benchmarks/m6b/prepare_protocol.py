"""Generate the GT/input-only M6b discovery ledger before detector inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from laserperception.datasets.kitti_raw import (
    KittiRawSequence,
    select_m6a_reconstruction_frames,
)
from laserperception.detection.multisweep import MultiSweepBuilder, MultiSweepBuilderConfig
from laserperception.evaluation.kitti_m6b import (
    KittiReferenceCamera,
    native_box_corners,
    parse_kitti_tracklets,
    visible_in_reference_camera,
)
from laserperception.evaluation.m6b_input_oracle import (
    freeze_sweep_transforms,
    reconstruct_from_frozen_transforms,
)
from laserperception.evaluation.m6b_pillars import analyze_pillars

CITY_DRIVES = (
    "2011_09_26_drive_0001",
    "2011_09_26_drive_0002",
    "2011_09_26_drive_0005",
    "2011_09_26_drive_0011",
    "2011_09_26_drive_0013",
    "2011_09_26_drive_0017",
    "2011_09_26_drive_0018",
    "2011_09_26_drive_0048",
    "2011_09_26_drive_0051",
    "2011_09_26_drive_0056",
    "2011_09_26_drive_0057",
    "2011_09_26_drive_0059",
    "2011_09_26_drive_0060",
    "2011_09_26_drive_0084",
    "2011_09_26_drive_0091",
)
RESIDENTIAL_DRIVES = (
    "2011_09_26_drive_0019",
    "2011_09_26_drive_0020",
    "2011_09_26_drive_0022",
    "2011_09_26_drive_0023",
    "2011_09_26_drive_0035",
    "2011_09_26_drive_0046",
    "2011_09_26_drive_0061",
    "2011_09_26_drive_0064",
)
CANONICAL_DRIVE = "2011_09_26_drive_0001"
EXPECTED_SELECTED_DRIVE = "2011_09_26_drive_0091"
EXPECTED_PRE_AUGMENT_LEDGER_SHA256 = (
    "3b1a452e7056d6493978b496359d7299795153f57800bb6d2edf14e774eefcf8"
)


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _tracklet_path(tracklet_root: Path, drive: str) -> Path:
    return tracklet_root / drive / "2011_09_26" / f"{drive}_sync" / "tracklet_labels.xml"


def _census(tracklet_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for category, drives in (("City", CITY_DRIVES), ("Residential", RESIDENTIAL_DRIVES)):
        for drive in drives:
            path = _tracklet_path(tracklet_root, drive)
            poses = parse_kitti_tracklets(path)
            valid_pedestrian = [
                pose
                for pose in poses
                if pose.object_type == "Pedestrian" and pose.valid_labelled_pose
            ]
            labelled_frames = {pose.frame_index for pose in poses if pose.valid_labelled_pose}
            records.append(
                {
                    "drive_id": drive,
                    "category": category,
                    "tracklet_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "valid_pedestrian_labelled_poses": len(valid_pedestrian),
                    "pedestrian_tracklets": len(
                        {pose.track_id for pose in poses if pose.object_type == "Pedestrian"}
                    ),
                    "total_labelled_frame_count": len(labelled_frames),
                    "maximum_labelled_frame_index": max(
                        (pose.frame_index for pose in poses), default=-1
                    ),
                    "tracklet_pose_count": len(poses),
                }
            )
    return sorted(records, key=lambda item: str(item["drive_id"]))


def _select_drive(census: Sequence[dict[str, object]]) -> str:
    ranked = sorted(
        census,
        key=lambda item: (
            -int(item["valid_pedestrian_labelled_poses"]),
            -int(item["pedestrian_tracklets"]),
            -int(item["total_labelled_frame_count"]),
            str(item["drive_id"]),
        ),
    )
    selected = str(ranked[0]["drive_id"])
    if selected != EXPECTED_SELECTED_DRIVE:
        raise RuntimeError(f"GT-only selection produced unexpected drive {selected}")
    return selected


def _drive_record(
    date_root: Path,
    drive_id: str,
    camera: KittiReferenceCamera,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    drive_root = date_root / f"{drive_id}_sync"
    sequence = KittiRawSequence(date_root, drive_root)
    poses = parse_kitti_tracklets(drive_root / "tracklet_labels.xml")
    by_frame: dict[int, list[object]] = {}
    for pose in poses:
        by_frame.setdefault(pose.frame_index, []).append(pose)
    h10_builder = MultiSweepBuilder(MultiSweepBuilderConfig(max_historical_sweeps=10))
    h5_builder = MultiSweepBuilder(MultiSweepBuilderConfig(max_historical_sweeps=5))
    frames: list[dict[str, object]] = []
    for frame_index in range(10, len(sequence)):
        frozen_transforms = freeze_sweep_transforms(sequence, frame_index)
        h10 = reconstruct_from_frozen_transforms(
            sequence,
            frame_index,
            frozen_transforms,
            builder=h10_builder,
        )
        h5 = reconstruct_from_frozen_transforms(
            sequence,
            frame_index,
            frozen_transforms,
            builder=h5_builder,
        )
        h10_points = h10.point_cloud.points_xyzt
        h5_points = h5.point_cloud.points_xyzt
        h10_pillars = analyze_pillars(h10_points)
        h5_pillars = analyze_pillars(h5_points)
        frame_poses = by_frame.get(frame_index, [])
        valid_targets = [
            pose
            for pose in frame_poses
            if pose.valid_labelled_pose
            and pose.evaluation_role == "target"
            and visible_in_reference_camera(native_box_corners(pose), camera)
        ]
        frame = {
            "frame_id": f"{drive_id}/{frame_index:010d}",
            "frame_index": frame_index,
            "eligible_target_counts": {
                "car": sum(pose.evaluation_class == "car" for pose in valid_targets),
                "pedestrian": sum(pose.evaluation_class == "pedestrian" for pose in valid_targets),
            },
            "eligible_target_track_ids": sorted({pose.track_id for pose in valid_targets}),
            "frozen_sweep_transforms": list(frozen_transforms),
            "h10": {
                "selected_indices": list(h10.selected_indices),
                "source_counts": list(h10.source_counts),
                "point_count": int(len(h10_points)),
                "time_lag_values": [float(value) for value in np.unique(h10_points[:, 3])],
                "model_ready_sha256": _sha256(h10_points),
                "pillars": h10_pillars.summary(),
            },
            "h5": {
                "selected_indices": list(h5.selected_indices),
                "source_counts": list(h5.source_counts),
                "point_count": int(len(h5_points)),
                "time_lag_values": [float(value) for value in np.unique(h5_points[:, 3])],
                "model_ready_sha256": _sha256(h5_points),
                "pillars": h5_pillars.summary(),
            },
        }
        frames.append(frame)
        if frame_index == 10 or frame_index % 25 == 0 or frame_index == len(sequence) - 1:
            print(
                f"{drive_id} {frame_index:03d}/{len(sequence) - 1}: "
                f"H10={h10_pillars.candidate_count} H5={h5_pillars.candidate_count}",
                flush=True,
            )

    valid_labelled = [pose for pose in poses if pose.valid_labelled_pose]
    valid_visible = [
        pose
        for pose in valid_labelled
        if pose.evaluation_role in {"target", "neighbour_ignore"}
        and visible_in_reference_camera(native_box_corners(pose), camera)
    ]
    drive_record = {
        "drive_id": drive_id,
        "frame_count": len(sequence),
        "evaluation_frame_indices": list(range(10, len(sequence))),
        "evaluation_frame_count": len(frames),
        "tracklet_sha256": hashlib.sha256(
            (drive_root / "tracklet_labels.xml").read_bytes()
        ).hexdigest(),
        "valid_labelled_pose_count": len(valid_labelled),
        "valid_reference_fov_mapped_pose_count": len(valid_visible),
        "valid_reference_fov_target_counts": {
            "car": sum(
                pose.evaluation_role == "target" and pose.evaluation_class == "car"
                for pose in valid_visible
            ),
            "pedestrian": sum(
                pose.evaluation_role == "target" and pose.evaluation_class == "pedestrian"
                for pose in valid_visible
            ),
        },
        "valid_reference_fov_neighbour_ignore_counts": {
            "van_for_car": sum(pose.object_type == "Van" for pose in valid_visible),
            "person_sitting_for_pedestrian": sum(
                pose.object_type == "Person (sitting)" for pose in valid_visible
            ),
        },
    }
    return drive_record, frames


def _first_unused(
    ordered: Sequence[dict[str, object]],
    selected: set[str],
) -> dict[str, object]:
    for frame in ordered:
        frame_id = str(frame["frame_id"])
        if frame_id not in selected:
            return frame
    raise RuntimeError("sentinel selection exhausted candidates")


def _sentinels(
    frames: Sequence[dict[str, object]],
    canonical_m6a_indices: Sequence[int],
) -> list[dict[str, object]]:
    canonical = [frame for frame in frames if str(frame["frame_id"]).startswith(CANONICAL_DRIVE)]
    selected: set[str] = set()
    result: list[dict[str, object]] = []

    def add(role: str, frame: dict[str, object]) -> None:
        selected.add(str(frame["frame_id"]))
        result.append(
            {
                "role": role,
                "frame_id": frame["frame_id"],
                "h10_candidate_pillars": frame["h10"]["pillars"]["candidate_occupied_pillars"],
                "h10_overflow_count": frame["h10"]["pillars"]["overflow_count"],
            }
        )

    add("canonical_first_full_history", canonical[0])
    canonical_ranked = sorted(
        canonical,
        key=lambda frame: (
            int(frame["h10"]["pillars"]["candidate_occupied_pillars"]),
            int(frame["frame_index"]),
        ),
    )
    add(
        "canonical_lower_median_candidate_pillars",
        _first_unused(
            canonical_ranked[(len(canonical_ranked) - 1) // 2 :] + canonical_ranked, selected
        ),
    )
    m6a = [frame for frame in canonical if int(frame["frame_index"]) in canonical_m6a_indices]
    m6a_ranked = sorted(
        m6a,
        key=lambda frame: (
            -int(frame["h10"]["pillars"]["overflow_count"]),
            -int(frame["h10"]["pillars"]["candidate_occupied_pillars"]),
            int(frame["frame_index"]),
        ),
    )
    add("highest_overflow_full_history_m6a_sentinel", _first_unused(m6a_ranked, selected))
    non_overflow = sorted(
        (frame for frame in frames if not bool(frame["h10"]["pillars"]["overflow"])),
        key=lambda frame: (
            abs(40_000 - int(frame["h10"]["pillars"]["candidate_occupied_pillars"])),
            str(frame["frame_id"]),
        ),
    )
    add("non_overflow_closest_to_40000", _first_unused(non_overflow, selected))
    pedestrian = sorted(
        (
            frame
            for frame in frames
            if str(frame["frame_id"]).startswith(EXPECTED_SELECTED_DRIVE)
            and int(frame["eligible_target_counts"]["pedestrian"]) > 0
        ),
        key=lambda frame: int(frame["frame_index"]),
    )
    add("second_drive_first_eligible_pedestrian", _first_unused(pedestrian, selected))
    if len(result) != 5 or len(selected) != 5:
        raise RuntimeError("repeatability sentinel selection must produce five unique frames")
    return result


def _verify_h5_repeatability(
    date_root: Path,
    sentinels: Sequence[dict[str, object]],
    repetitions: int = 10,
) -> list[dict[str, object]]:
    builder = MultiSweepBuilder(MultiSweepBuilderConfig(max_historical_sweeps=5))
    sequences: dict[str, KittiRawSequence] = {}
    results: list[dict[str, object]] = []
    for sentinel in sentinels:
        drive, raw_index = str(sentinel["frame_id"]).split("/", 1)
        if drive not in sequences:
            sequences[drive] = KittiRawSequence(date_root, date_root / f"{drive}_sync")
        hashes = []
        for _ in range(repetitions):
            result = sequences[drive].reconstruct(int(raw_index), builder=builder)
            hashes.append(_sha256(result.point_cloud.points_xyzt))
        exact = len(set(hashes)) == 1
        if not exact:
            raise RuntimeError(f"H5 input oracle repeatability failed for {sentinel['frame_id']}")
        results.append(
            {
                "frame_id": sentinel["frame_id"],
                "repetitions": repetitions,
                "model_ready_sha256": hashes[0],
                "exact": exact,
            }
        )
    return results


def _augment_existing_ledger(date_root: Path, output: Path) -> None:
    existing = json.loads(output.read_text(encoding="utf-8"))
    canonical = (json.dumps(existing, indent=2, sort_keys=True) + "\n").encode()
    if hashlib.sha256(canonical).hexdigest() != EXPECTED_PRE_AUGMENT_LEDGER_SHA256:
        raise RuntimeError("existing pre-inference ledger does not match the frozen semantic hash")
    if existing.get("detector_inference_performed") is not False:
        raise RuntimeError("refusing to augment a ledger after detector inference")

    sequences: dict[str, KittiRawSequence] = {}
    frames = existing.get("frames")
    if not isinstance(frames, list):
        raise RuntimeError("existing pre-inference ledger has no frame list")
    for position, frame in enumerate(frames, start=1):
        if not isinstance(frame, dict):
            raise RuntimeError("existing pre-inference ledger contains a malformed frame")
        drive, raw_index = str(frame["frame_id"]).split("/", 1)
        if drive not in sequences:
            sequences[drive] = KittiRawSequence(date_root, date_root / f"{drive}_sync")
        frame["frozen_sweep_transforms"] = list(
            freeze_sweep_transforms(sequences[drive], int(raw_index))
        )
        if position == 1 or position % 100 == 0 or position == len(frames):
            print(f"froze transforms for {position}/{len(frames)} frames", flush=True)

    existing["portable_input_reproduction"] = {
        "policy": "frozen_float32_sweep_transforms_then_unchanged_multisweep_builder",
        "reason": "pre-inference cross-platform BLAS/LAPACK one-ULP transform finding",
        "semantics_changed": False,
    }
    encoded = json.dumps(existing, indent=2, sort_keys=True) + "\n"
    forbidden = (str(Path.home()), "J:\\", "/root/")
    if any(value in encoded for value in forbidden):
        raise RuntimeError("refusing to write discovery data containing a private absolute path")
    output.write_text(encoded, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tracklet-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--augment-existing", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    date_root = args.data_root.expanduser().resolve() / "2011_09_26"
    if args.augment_existing:
        _augment_existing_ledger(date_root, args.output)
        return 0
    tracklet_root = args.tracklet_root.expanduser().resolve()
    census = _census(tracklet_root)
    selected = _select_drive(census)
    camera = KittiReferenceCamera.from_date_root(date_root)
    all_frames: list[dict[str, object]] = []
    drives: list[dict[str, object]] = []
    sequences: dict[str, KittiRawSequence] = {}
    for drive in (CANONICAL_DRIVE, selected):
        record, frames = _drive_record(date_root, drive, camera)
        drives.append(record)
        all_frames.extend(frames)
        sequences[drive] = KittiRawSequence(date_root, date_root / f"{drive}_sync")
    m6a_indices = select_m6a_reconstruction_frames(sequences[CANONICAL_DRIVE].ego_to_global_poses)
    sentinels = _sentinels(all_frames, m6a_indices)
    h5_repeats = _verify_h5_repeatability(date_root, sentinels)
    pedestrian_count = sum(
        int(frame["eligible_target_counts"]["pedestrian"]) for frame in all_frames
    )
    record = {
        "schema_version": 1,
        "status": "pre_inference_gt_and_input_discovery_complete",
        "detector_inference_performed": False,
        "portable_input_reproduction": {
            "policy": "frozen_float32_sweep_transforms_then_unchanged_multisweep_builder",
            "reason": "pre-inference cross-platform BLAS/LAPACK one-ULP transform finding",
            "semantics_changed": False,
        },
        "category_source": "official_KITTI_Raw_page_City_and_Residential_groupings",
        "candidate_census": census,
        "selection_order": [
            "descending_valid_pedestrian_labelled_poses",
            "descending_pedestrian_tracklets",
            "descending_total_labelled_frame_count",
            "lexicographic_drive_id",
        ],
        "selected_drives": [CANONICAL_DRIVE, selected],
        "reference_camera": {
            "camera": "rectified_camera_0",
            "image_size_wh": list(camera.image_size_wh),
            "near_plane_metres": camera.near_plane_metres,
        },
        "drives": drives,
        "evaluation_frame_count": len(all_frames),
        "pedestrian_sample_size_floor": 50,
        "eligible_pedestrian_evaluation_poses": pedestrian_count,
        "pedestrian_low_n": pedestrian_count < 50,
        "canonical_m6a_reconstruction_indices": list(m6a_indices),
        "repeatability_sentinels": sentinels,
        "h5_input_oracle_repeatability": h5_repeats,
        "frames": all_frames,
    }
    encoded = json.dumps(record, indent=2, sort_keys=True) + "\n"
    forbidden = (str(Path.home()), "J:\\", "/root/")
    if any(value in encoded for value in forbidden):
        raise RuntimeError("refusing to write discovery data containing a private absolute path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(f"selected={selected} frames={len(all_frames)} ped={pedestrian_count}")
    print(f"sentinels={[item['frame_id'] for item in sentinels]}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
