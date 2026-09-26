"""CPU-only derivation of prospective S2 pose partitions from accepted S1 archives.

This reads existing evidence; it does not construct inputs or import a detector runtime.
The output is a proposal for owner review, never an inference authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PRIMARY_COMMIT = "6994d72c3e7691a86116d1417ac3ae08256d163f"
MANIFEST_PATH = ROOT / "benchmarks/m8/results/m8_s1_measurement_manifest.json"
PRIMARY_RAW_PATH = ROOT / "benchmarks/m8/results/m8_s1_primary_raw.json"
PROTOCOL_PATH = ROOT / "benchmarks/m8/preregistration/m8_s1_protocol.json"
CLASSES = {"car": 66, "pedestrian": 396}
ARMS = {"H10": "a2", "H5": "e2"}
CATEGORIES = ("shared", "e2_only", "a2_only", "neither", "unstable")
GT_ID = re.compile(r"^(2011_09_26_drive_(?:0001|0091))/track_([0-9]+)$")
CONDITION_FILE = re.compile(r"/conditions/([0-9]{4})\.json$")


def canonical_json_bytes(value: Any) -> bytes:
    """Use the frozen M7 canonical JSON convention, including its final LF."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _member(tar: tarfile.TarFile, suffix: str) -> tarfile.TarInfo:
    matches = [item for item in tar.getmembers() if item.isfile() and item.name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one archive member ending {suffix!r}")
    return matches[0]


def _read_member(tar: tarfile.TarFile, item: tarfile.TarInfo) -> bytes:
    stream = tar.extractfile(item)
    if stream is None:
        raise ValueError(f"archive member is unreadable: {item.name}")
    with stream:
        return stream.read()


def _member_sha256(tar: tarfile.TarFile, item: tarfile.TarInfo) -> str:
    stream = tar.extractfile(item)
    if stream is None:
        raise ValueError(f"archive member is unreadable: {item.name}")
    digest = hashlib.sha256()
    with stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_ids() -> list[str]:
    return [
        f"2011_09_26_drive_{drive}/{frame:010d}"
        for drive, frames in (("0001", range(10, 108)), ("0091", range(10, 340)))
        for frame in frames
    ]


def _pose_key(frame_id: str, observation: dict[str, Any]) -> tuple[str, int, int]:
    match = GT_ID.fullmatch(observation["gt_identity"])
    if match is None:
        raise ValueError("invalid canonical GT identity")
    drive_id, track_id = match.groups()
    frame_index = observation["frame_index"]
    if frame_id != f"{drive_id}/{frame_index:010d}" or observation["track_id"] != int(track_id):
        raise ValueError("GT identity, frame, and track ID disagree")
    return drive_id, frame_index, int(track_id)


def _validate_condition(
    record: dict[str, Any],
    *,
    expected_id: str,
    process_uuid: str,
    pass_index: int,
    states: dict[str, dict[tuple[str, int, int], dict[str, list[int | None]]]],
) -> None:
    if (
        record.get("status") != "COMPLETE"
        or record.get("condition_id") != expected_id
        or record["identity"]["process_uuid"] != process_uuid
        or record["identity"]["runtime_commit"] != PRIMARY_COMMIT
    ):
        raise ValueError("condition identity or completion mismatch")
    frame_id, history = expected_id.rsplit("/", 1)
    payload = record["payload"]
    if payload["frame_id"] != frame_id or payload["history"] != history:
        raise ValueError("condition payload identity mismatch")
    if payload["evaluator_provenance"]["score_threshold"] != 0.25:
        raise ValueError("unexpected score threshold")
    arm = ARMS[history]
    for class_name in CLASSES:
        class_record = payload["classes"][class_name]
        observations = class_record["target_observations"]
        threshold = class_record["thresholds"]["0.50"]
        matched_list = threshold["matched_gt_identity_set"]
        matched = set(matched_list)
        if (
            len(matched) != len(matched_list)
            or len(observations) != class_record["eligible_GT_count"]
            or len(matched) != threshold["true_positives"]
            or matched != {obs["gt_identity"] for obs in observations if obs["matched"]}
        ):
            raise ValueError("matched-GT evidence disagrees with target observations or TP count")
        for observation in observations:
            key = _pose_key(frame_id, observation)
            slot = states[class_name].setdefault(key, {"a2": [None] * 3, "e2": [None] * 3})
            if slot[arm][pass_index] is not None:
                raise ValueError(f"duplicate GT pose in {arm} process {pass_index + 1}: {key}")
            slot[arm][pass_index] = int(observation["gt_identity"] in matched)


def read_accepted_passes(archive_paths: list[Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate all three archived passes against the tracked acceptance manifest."""
    if len(archive_paths) != 3:
        raise ValueError("exactly three accepted primary archives are required")
    manifest_bytes = MANIFEST_PATH.read_bytes()
    primary_bytes = PRIMARY_RAW_PATH.read_bytes()
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    manifest = json.loads(manifest_bytes)
    primary = json.loads(primary_bytes)
    protocol = json.loads(protocol_bytes)
    if (
        manifest["scientific_execution"]["runtime_commit"] != PRIMARY_COMMIT
        or manifest["accepted_passes"] is None
        or len(manifest["accepted_passes"]) != 3
        or len(primary["passes"]) != 3
        or protocol["evaluator"]["score_threshold"] != 0.25
    ):
        raise ValueError("tracked S1 evidence binding mismatch")
    states: dict[str, dict[tuple[str, int, int], dict[str, list[int | None]]]] = {
        class_name: {} for class_name in CLASSES
    }
    sources: list[dict[str, Any]] = []
    expected_ids = [f"{frame}/{history}" for frame in _frame_ids() for history in ARMS]
    for index, (path, accepted, compact_pass) in enumerate(
        zip(archive_paths, manifest["accepted_passes"], primary["passes"], strict=True)
    ):
        if (
            accepted["pass"] != index + 1
            or compact_pass["pass"] != index + 1
            or compact_pass["process_uuid"] != accepted["process_uuid"]
            or path.stat().st_size != accepted["archive_bytes"]
            or file_sha256(path) != accepted["archive_sha256"]
        ):
            raise ValueError(f"archive/pass identity mismatch for process {index + 1}")
        with tarfile.open(path, "r:gz") as archive:
            final_item = _member(archive, "/final_pass_manifest.json")
            final_bytes = _read_member(archive, final_item)
            final = json.loads(final_bytes)
            if (
                len(final_bytes) != accepted["final_manifest_bytes"]
                or sha256(final_bytes) != accepted["final_manifest_sha256"]
                or final["status"] != "COMPLETE"
                or final["accepted_canonical_calls"] != 856
                or final["failed_calls"] != 0
                or final["identity"]["process_uuid"] != accepted["process_uuid"]
                or final["identity"]["runtime_commit"] != PRIMARY_COMMIT
                or final["result_sha256"] != accepted["final_result_sha256"]
                or final["completed_condition_ids"] != expected_ids
                or len(final["condition_file_sha256"]) != 856
            ):
                raise ValueError(f"final manifest mismatch for process {index + 1}")
            raw_item = _member(archive, "/raw_pass.json")
            if (
                raw_item.size != accepted["raw_pass_bytes"]
                or _member_sha256(archive, raw_item) != accepted["raw_pass_file_sha256"]
                or compact_pass["raw_pass_result_sha256"] != accepted["raw_pass_result_sha256"]
            ):
                raise ValueError(f"raw-pass hash mismatch for process {index + 1}")
            condition_members = [
                item
                for item in archive.getmembers()
                if item.isfile() and CONDITION_FILE.search(item.name)
            ]
            members = {
                int(match.group(1)): item
                for item in condition_members
                if (match := CONDITION_FILE.search(item.name))
            }
            if (
                len(condition_members) != 856
                or len(members) != 856
                or set(members) != set(range(856))
            ):
                raise ValueError("archive does not contain exactly 856 condition files")
            counts: Counter[tuple[str, str]] = Counter()
            for condition_index, expected_id in enumerate(expected_ids):
                condition_bytes = _read_member(archive, members[condition_index])
                if sha256(condition_bytes) != final["condition_file_sha256"][condition_index]:
                    raise ValueError(f"condition file SHA mismatch at {condition_index}")
                record = json.loads(condition_bytes)
                _validate_condition(
                    record,
                    expected_id=expected_id,
                    process_uuid=accepted["process_uuid"],
                    pass_index=index,
                    states=states,
                )
                for class_name in CLASSES:
                    counts[(class_name, record["payload"]["history"])] += record["payload"][
                        "classes"
                    ][class_name]["thresholds"]["0.50"]["true_positives"]
            for class_name, expected_gt in CLASSES.items():
                for history, arm in ARMS.items():
                    compact_tp = compact_pass["arms"]["A2" if arm == "a2" else "E2"]["classes"][
                        class_name
                    ]["thresholds"]["0.50"]["true_positives"]
                    if counts[(class_name, history)] != compact_tp:
                        raise ValueError("archived condition TP counts differ from compact S1")
                    seen = sum(slot[arm][index] is not None for slot in states[class_name].values())
                    if seen != expected_gt:
                        raise ValueError("missing eligible GT identities in accepted process")
            sources.append(
                {
                    "pass": index + 1,
                    "process_uuid": accepted["process_uuid"],
                    "attempt_id": accepted["attempt_id"],
                    "archive_logical_filename": path.name,
                    "archive_bytes": accepted["archive_bytes"],
                    "archive_sha256": accepted["archive_sha256"],
                    "raw_pass_bytes": accepted["raw_pass_bytes"],
                    "raw_pass_file_sha256": accepted["raw_pass_file_sha256"],
                    "raw_pass_result_sha256": accepted["raw_pass_result_sha256"],
                    "final_manifest_sha256": accepted["final_manifest_sha256"],
                    "condition_file_hashes_verified": 856,
                }
            )
    return {
        "states": states,
        "manifest_sha256": sha256(manifest_bytes),
        "primary_raw_sha256": sha256(primary_bytes),
        "protocol_sha256": sha256(protocol_bytes),
    }, sources


def classify(
    states: dict[str, dict[tuple[str, int, int], dict[str, list[int | None]]]],
) -> dict[str, Any]:
    """Classify complete three-process arm states with an explicit unstable bucket."""
    classes: dict[str, Any] = {}
    for class_name, expected_gt in CLASSES.items():
        poses = states[class_name]
        if len(poses) != expected_gt:
            raise ValueError(f"{class_name} has {len(poses)} GT poses, expected {expected_gt}")
        buckets: dict[str, list[dict[str, Any]]] = {name: [] for name in CATEGORIES}
        unstable_states: list[dict[str, Any]] = []
        for (drive_id, frame_index, gt_track_id), values in sorted(poses.items()):
            a2, e2 = values["a2"], values["e2"]
            if len(a2) != 3 or len(e2) != 3 or any(bit not in (0, 1) for bit in a2 + e2):
                raise ValueError("missing or invalid primary process state")
            key = {"drive_id": drive_id, "frame_index": frame_index, "gt_track_id": gt_track_id}
            if len(set(a2)) > 1 or len(set(e2)) > 1:
                category = "unstable"
                unstable_states.append(
                    {"pose": key, "a2_process_states": a2, "e2_process_states": e2}
                )
            elif a2[0] and e2[0]:
                category = "shared"
            elif e2[0]:
                category = "e2_only"
            elif a2[0]:
                category = "a2_only"
            else:
                category = "neither"
            buckets[category].append(key)
        if sum(map(len, buckets.values())) != expected_gt:
            raise ValueError("partitions are not collectively exhaustive")
        result: dict[str, Any] = {}
        for category in CATEGORIES:
            members = buckets[category]
            result[category] = {
                "count": len(members),
                "canonical_json_sha256": sha256(canonical_json_bytes(members)),
                "identities": members,
            }
        result["unstable_process_states"] = unstable_states
        result["eligible_gt_count"] = expected_gt
        classes[class_name] = result
    return classes


def build_draft(archive_paths: list[Path]) -> dict[str, Any]:
    evidence, sources = read_accepted_passes(archive_paths)
    classes = classify(evidence["states"])
    car = classes["car"]
    a2 = car["shared"]["count"] + car["a2_only"]["count"]
    e2 = car["shared"]["count"] + car["e2_only"]["count"]
    gaps = [
        sum(values["e2"][index] for values in evidence["states"]["car"].values())
        - sum(values["a2"][index] for values in evidence["states"]["car"].values())
        for index in range(3)
    ]
    denominator_eligible = all(gap > 0 for gap in gaps) and len(set(gaps)) == 1 and gaps[0] >= 20
    partition_eligible = (
        car["unstable"]["count"] == 0
        and a2 == 19
        and e2 == 43
        and car["e2_only"]["count"] - car["a2_only"]["count"] == 24
    )
    return {
        "schema_version": "laserperception.m8.s2.partitions-draft.v1",
        "status": "PROSPECTIVE DRAFT FOR OWNER REVIEW; NOT FROZEN OR AUTHORIZED",
        "source": {
            "primary_execution_commit": PRIMARY_COMMIT,
            "primary_measurement_manifest_sha256": evidence["manifest_sha256"],
            "primary_compact_raw_sha256": evidence["primary_raw_sha256"],
            "frozen_s1_protocol_sha256": evidence["protocol_sha256"],
            "accepted_primary_passes": sources,
            "excluded_sources": ["incomplete attempts", "zero-intensity processes", "Stage R"],
            "pose_state_source": (
                "Per-condition target_observations, cross-checked with matched_gt_identity_set "
                "at score >=0.25 / oriented BEV IoU >=0.50"
            ),
        },
        "canonical_pose_key": ["drive_id", "frame_index", "gt_track_id"],
        "ordering": "lexicographic by canonical pose key",
        "canonical_list_hash_encoding": (
            "M7 canonical JSON: sorted keys, compact separators, ASCII, final LF, SHA256"
        ),
        "classes": classes,
        "car_denominator_proposal": {
            "per_process_e2_minus_a2_tp": gaps,
            "all_positive": all(gap > 0 for gap in gaps),
            "exactly_stable": len(set(gaps)) == 1,
            "proposed_minimum_gap": 20,
            "one_tp_normalized_increment_at_common_gap": 1 / gaps[0]
            if len(set(gaps)) == 1 and gaps[0]
            else None,
            "denominator_eligible": denominator_eligible,
            "partition_eligible": partition_eligible,
            "normalized_recovery_eligible_if_owner_freezes_proposal": denominator_eligible
            and partition_eligible,
            "stable_partitions_reproduce_a2_tp_19": a2 == 19,
            "stable_partitions_reproduce_e2_tp_43": e2 == 43,
            "stable_partitions_reproduce_gap_24": car["e2_only"]["count"] - car["a2_only"]["count"]
            == 24,
        },
        "pedestrian_role": (
            "descriptive paired partitions only; no normalized Pedestrian recovery proposed"
        ),
        "authorization": {
            "s2_protocol_frozen": False,
            "s2_implementation_frozen": False,
            "s2_input_ledger_frozen": False,
            "s2_ready_to_execute": False,
            "s2_authorized": False,
            "b2_c2_d2_f2_calls": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        action="append",
        type=Path,
        required=True,
        help="accepted primary archive, pass order 1/2/3",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draft = build_draft(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(json.dumps(draft, indent=2, ensure_ascii=True).encode() + b"\n")
    print(
        json.dumps(
            {
                "classes": {
                    name: {
                        key: value["count"] for key, value in result.items() if key in CATEGORIES
                    }
                    for name, result in draft["classes"].items()
                },
                "car_denominator_proposal": draft["car_denominator_proposal"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
