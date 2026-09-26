"""Stream the frozen M8 S2 CPU inputs from verified M7 interventions.

No detector, ground truth, optional accelerator runtime, or model is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.m7.interventions import construct_b, construct_c, construct_d, construct_f
from benchmarks.m7.prepare_inputs import CanonicalM7SourceAdapter
from benchmarks.m7.protocol import canonical_frame_ids
from benchmarks.m7.provenance import model_ready_sha256, selected_rows_sha256, verify_external_asset
from benchmarks.m7.structural_validation import validate_c_against_a_e, validate_f_against_a
from laserperception.detection.m8_capacity import candidate_dynamic_pillar_coordinates
from laserperception.detection.m8_s1_frozen_input import FrozenInputSource
from laserperception.detection.m8_s2_input import (
    ARM_ORDER,
    UINT64_LE,
    array_sha256,
    canonical_xyzt,
    lift_b2,
    lift_c2,
    lift_d2,
    lift_f2,
)

PROTOCOL_SHA256 = "218ef2dcc03fa4ff75562e02f368f7624f16a4c768b1758147c1d46ac1d9c53d"
M8_LEDGER_SHA256 = "474e87e34c64d669750d4b6f7a64ac46fc9c5c462693fad79ff7c9547a7f1f7c"
M7_MANIFEST_SHA256 = "8d4f74d783950d24956239f3a67a7a58fe10013e0e83a88d0f8b23e3139ffe90"
M6_LEDGER_SHA256 = "e25b3d62113cc7e8c1fcf736caa68b1ab698f965f007c758ff91d3e498ca6caa"
M6_RESULT_SHA256 = "87870b2aa0cc2a91d39331afc8154fdad0c8c796f1cabfb4f8530a3eb106de27"
M7_IMPLEMENTATION_COMMIT = "c989f7df5ca8c5ac8148c0ed3a2e91de48b754b2"
SENTINELS = (
    "2011_09_26_drive_0001/0000000010",
    "2011_09_26_drive_0001/0000000011",
    "2011_09_26_drive_0001/0000000015",
    "2011_09_26_drive_0001/0000000083",
    "2011_09_26_drive_0091/0000000010",
    "2011_09_26_drive_0091/0000000011",
    "2011_09_26_drive_0091/0000000012",
)
M7_ARMS = {
    "B2": "H10_LAG_COMPRESSED",
    "C2": "H10_POINT_COUNT_MATCHED",
    "D2": "H10_LAG_COMPRESSED_POINT_COUNT_MATCHED",
    "F2": "H10_ALTERNATE_FULL_SPAN",
}
SCHEMA = "laserperception.m8.s2.input-ledger.v1"


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL)


def _verify_code_and_assets(args: argparse.Namespace) -> str:
    if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
        raise RuntimeError("forbidden accelerator runtime was imported")
    protocol = Path("docs/m8/M8_S2_PROTOCOL.md")
    if (
        hashlib.sha256(_git("show", "HEAD:docs/m8/M8_S2_PROTOCOL.md")).hexdigest()
        != PROTOCOL_SHA256
    ):
        raise ValueError("frozen S2 protocol Git identity differs")
    if _git("diff", "--", str(protocol)):
        raise ValueError("frozen S2 protocol working copy differs")
    frozen_files = (
        "benchmarks/m7/interventions.py",
        "benchmarks/m7/provenance.py",
        "benchmarks/m7/prepare_inputs.py",
        "benchmarks/m7/structural_validation.py",
    )
    if _git("diff", M7_IMPLEMENTATION_COMMIT, "HEAD", "--", *frozen_files):
        raise ValueError("M7 intervention/provenance code drifted")
    if _git("diff", "--", *frozen_files):
        raise ValueError("M7 intervention/provenance working copy drifted")
    assets = (
        (args.m6_ledger, 5_837_452, M6_LEDGER_SHA256),
        (args.m6_result, 41_987_113, M6_RESULT_SHA256),
        (args.accepted_m8_ledger, args.accepted_m8_ledger.stat().st_size, M8_LEDGER_SHA256),
        (args.m7_manifest, args.m7_manifest.stat().st_size, M7_MANIFEST_SHA256),
    )
    for path, byte_count, digest in assets:
        verify_external_asset(path, expected_bytes=byte_count, expected_sha256=digest)
    commit = _git("rev-parse", "HEAD").decode().strip()
    if args.implementation_commit != commit:
        raise ValueError("generation must run at the exact committed implementation identity")
    return commit


def _load_manifest(path: Path) -> dict[str, Mapping[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("conditions")
    if payload.get("condition_count") != 1712 or not isinstance(rows, list):
        raise ValueError("frozen M7 input manifest does not have 1,712 conditions")
    result = {str(row["condition_id"]): row for row in rows}
    if len(result) != 1712:
        raise ValueError("frozen M7 input manifest has duplicate conditions")
    return result


def _coordinates(points: np.ndarray) -> tuple[int, str, np.ndarray]:
    coordinates = candidate_dynamic_pillar_coordinates(points)
    return len(coordinates), array_sha256(coordinates, dtype=np.dtype("<i4")), coordinates


def _sweep_details(result: Any) -> tuple[list[int], dict[str, int], dict[str, str]]:
    provenance = result.provenance
    ranks = sorted(int(rank) for rank in np.unique(provenance.history_rank))
    counts = {str(rank): int(np.count_nonzero(provenance.history_rank == rank)) for rank in ranks}
    source_ids = {str(item.history_rank): item.source_sweep_id for item in provenance.rank_sources}
    return ranks, counts, source_ids


def _seed_records(records: tuple[dict[str, object], ...]) -> list[dict[str, object]]:
    output = []
    for row in records:
        ordinals = np.asarray(row["selected_ordinals"], dtype=UINT64_LE)
        output.append(
            {
                "history_rank": row["history_rank"],
                "seed_text_utf8": row["seed_text_utf8"],
                "sha256": row["sha256"],
                "seed_uint64_hex": row["seed_uint64_hex"],
                "selected_ordinal_count": len(ordinals),
                "selected_ordinals_sha256": array_sha256(ordinals, dtype=UINT64_LE),
            }
        )
    return output


def generate_frame(
    frame_id: str,
    *,
    m7_adapter: CanonicalM7SourceAdapter,
    m8_adapter: FrozenInputSource,
    manifest: Mapping[str, Mapping[str, object]],
    implementation_commit: str,
) -> tuple[dict[str, object], ...]:
    """Verify both frozen source pairs, lift M7 B/C/D/F, and record one frame."""

    m7 = m7_adapter.frame_sources(frame_id)
    (a2, a_record), (e2, e_record) = m8_adapter.pair(frame_id)
    if array_sha256(a2) != a_record["input_sha256"] or array_sha256(e2) != e_record["input_sha256"]:
        raise ValueError(f"M8 A2/E2 source identity mismatch: {frame_id}")
    if array_sha256(canonical_xyzt(a2)) != m7.expected_a_sha256:
        raise ValueError(f"M7 A projection mismatch: {frame_id}")
    if array_sha256(canonical_xyzt(e2)) != m7.expected_e_sha256:
        raise ValueError(f"M7 E projection mismatch: {frame_id}")
    if (
        model_ready_sha256(m7.a_points) != m7.expected_a_sha256
        or model_ready_sha256(m7.e_points) != m7.expected_e_sha256
    ):
        raise ValueError(f"M7 A/E source commitment mismatch: {frame_id}")
    drive_id, frame_text = frame_id.split("/", 1)
    b, scale = construct_b(m7.a_points, m7.e_points, m7.a_provenance, m7.e_provenance)
    c = construct_c(
        m7.a_points,
        m7.e_points,
        m7.a_provenance,
        m7.e_provenance,
        drive_id=drive_id,
        frame_index=int(frame_text),
    )
    d = construct_d(c, scale)
    f = construct_f(m7.a_points, m7.a_provenance)
    validate_c_against_a_e(m7.a_points, m7.e_points, c)
    validate_f_against_a(m7.a_points, m7.a_provenance, f)
    c2 = lift_c2(a2, c.intervention, e2_count=len(e2))
    lifted = (lift_b2(a2, b), c2, lift_d2(c2, d), lift_f2(a2, f))
    if tuple(item.arm for item in lifted) != ARM_ORDER:
        raise ValueError("S2 arm order differs")
    m7_results = (b, c.intervention, d, f)
    a_count, a_coordinate_sha, a_coordinates = _coordinates(a2)
    result: list[dict[str, object]] = []
    for item, m7_result in zip(lifted, m7_results, strict=True):
        expected = manifest.get(f"{frame_id}|{M7_ARMS[item.arm]}")
        if (
            expected is None
            or expected.get("source_a_sha256") != m7.expected_a_sha256
            or expected.get("source_e_sha256") != m7.expected_e_sha256
        ):
            raise ValueError(f"M7 manifest source mismatch: {frame_id}/{item.arm}")
        xyzt_sha = array_sha256(canonical_xyzt(item.points))
        if xyzt_sha != expected.get(
            "model_ready_sha256"
        ) or item.selected_row_sha256 != expected.get("selected_row_sha256"):
            raise ValueError(f"M7 XYZT or selected-row mismatch: {frame_id}/{item.arm}")
        if selected_rows_sha256(item.selected_global_rows) != item.selected_row_sha256:
            raise ValueError(f"selected-row identity mismatch: {frame_id}/{item.arm}")
        if item.arm == "B2":
            if item.points[:, :4].tobytes(order="C") != a2[:, :4].tobytes(order="C"):
                raise ValueError(f"B2/A2 XYZ or intensity differs: {frame_id}")
        if item.arm == "D2":
            if item.points[:, :4].tobytes(order="C") != lifted[1].points[:, :4].tobytes(order="C"):
                raise ValueError(f"D2/C2 XYZ or intensity differs: {frame_id}")
            if item.selected_row_sha256 != lifted[1].selected_row_sha256:
                raise ValueError(f"D2/C2 selected rows differ: {frame_id}")
        if item.arm in ("C2", "F2"):
            rows = item.selected_global_rows.astype(np.int64)
            if item.points.tobytes(order="C") != a2[rows].tobytes(order="C"):
                raise ValueError(f"{item.arm} is not an exact A2 subset: {frame_id}")
        count, coordinate_sha, coordinates = _coordinates(item.points)
        if item.arm == "B2" and (
            count != a_count
            or coordinate_sha != a_coordinate_sha
            or not np.array_equal(coordinates, a_coordinates)
        ):
            raise ValueError(f"B2/A2 CPU analytic coordinates differ: {frame_id}")
        if item.arm == "C2":
            c_coordinates = coordinates
            c_coordinate_sha = coordinate_sha
            c_count = count
        if item.arm == "D2" and (
            count != c_count
            or coordinate_sha != c_coordinate_sha
            or not np.array_equal(coordinates, c_coordinates)
        ):
            raise ValueError(f"D2/C2 CPU analytic coordinates differ: {frame_id}")
        ranks, sweep_counts, sweep_ids = _sweep_details(m7_result)
        if item.arm == "F2" and ranks != [0, 2, 4, 6, 8, 10]:
            raise ValueError(f"F2 ranks differ: {frame_id}")
        if item.arm == "C2" and (
            len(item.points) != len(e2)
            or sweep_counts.get("0") != int(np.count_nonzero(m7.a_provenance.history_rank == 0))
        ):
            raise ValueError(f"C2 current or E2 count differs: {frame_id}")
        lag_support = np.unique(item.points[:, 4].view(np.dtype("<u4")))
        if int(item.points[0, 4].view(np.dtype("<u4"))) != 0:
            raise ValueError(f"current S2 lag is not positive zero: {frame_id}/{item.arm}")
        detailed = {
            "schema_version": SCHEMA,
            "implementation_commit": implementation_commit,
            "condition_id": f"{frame_id}/{item.arm}",
            "frame_id": frame_id,
            "arm": item.arm,
            "source_A2_full_XYZIT_sha256": a_record["input_sha256"],
            "source_E2_full_XYZIT_sha256": e_record["input_sha256"],
            "source_M7_A_XYZT_sha256": m7.expected_a_sha256,
            "source_M7_E_XYZT_sha256": m7.expected_e_sha256,
            "point_count": len(item.points),
            "selected_global_row_sha256": item.selected_row_sha256,
            "XYZ_sha256": array_sha256(item.points[:, :3]),
            "intensity_sha256": array_sha256(item.points[:, 3]),
            "XYZT_projection_sha256": xyzt_sha,
            "full_XYZIT_sha256": array_sha256(item.points),
            "M7_expected_XYZT_sha256": expected["model_ready_sha256"],
            "M7_XYZT_exact": True,
            "cpu_analytic_candidate_pillar_count": count,
            "cpu_analytic_candidate_coordinate_sha256": coordinate_sha,
            "sweep_ranks": ranks,
            "sweep_ids_by_rank": sweep_ids,
            "per_sweep_counts": sweep_counts,
            "lag_support_count": len(lag_support),
            "lag_span_seconds": float(np.max(item.points[:, 4]) - np.min(item.points[:, 4])),
            "lag_scale": scale.to_dict() if item.arm in ("B2", "D2") else None,
            "quota": c.quota.to_dict() if item.arm in ("C2", "D2") else None,
            "seed_identities": _seed_records(c.seed_identities)
            if item.arm in ("C2", "D2")
            else None,
            "f_selected_ranks": [2, 4, 6, 8, 10] if item.arm == "F2" else None,
            "source_A2_cpu_analytic_candidate_pillar_count": a_count,
            "source_A2_cpu_analytic_candidate_coordinate_sha256": a_coordinate_sha,
            "source_A2_point_count": len(a2),
            "source_E2_point_count": len(e2),
            "source_E2_lag_span_seconds": float(np.max(e2[:, 4]) - np.min(e2[:, 4])),
            "no_detector_call": True,
        }
        result.append(detailed)
    return tuple(result)


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "maximum": max(values),
    }


def _characterization(records: list[dict[str, object]]) -> dict[str, object]:
    by_arm: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_arm[str(record["arm"])].append(record)
    a_counts = {
        str(row["frame_id"]): int(row["source_A2_point_count"])
        for row in records
        if row["arm"] == "B2"
    }
    e_counts = {
        str(row["frame_id"]): int(row["source_E2_point_count"])
        for row in records
        if row["arm"] == "B2"
    }
    arms = {}
    for arm in ARM_ORDER:
        rows = by_arm[arm]
        arms[arm] = {
            "condition_count": len(rows),
            "point_count": _summary([float(row["point_count"]) for row in rows]),
            "point_count_ratio_to_A2": _summary(
                [float(row["point_count"]) / a_counts[str(row["frame_id"])] for row in rows]
            ),
            "point_count_ratio_to_E2": _summary(
                [float(row["point_count"]) / e_counts[str(row["frame_id"])] for row in rows]
            ),
            "lag_support_count": _summary([float(row["lag_support_count"]) for row in rows]),
            "lag_span_seconds": _summary([float(row["lag_span_seconds"]) for row in rows]),
            "cpu_analytic_candidate_pillar_count": _summary(
                [float(row["cpu_analytic_candidate_pillar_count"]) for row in rows]
            ),
            "per_sweep_counts": {
                rank: _summary([float(row["per_sweep_counts"].get(rank, 0)) for row in rows])
                for rank in map(str, range(11))
            },
        }
    arms["B2"]["lag_scale"] = _summary(
        [float.fromhex(str(row["lag_scale"]["scale_binary64_hex"])) for row in by_arm["B2"]]
    )
    arms["C2"]["retained_fraction_of_A2"] = arms["C2"]["point_count_ratio_to_A2"]
    arms["C2"]["zero_quota_frame_count"] = sum(
        bool(row["quota"]["zero_quota_ranks"]) for row in by_arm["C2"]
    )
    arms["F2"]["point_count_exceeds_E2_frames"] = sum(
        int(row["point_count"]) > e_counts[str(row["frame_id"])] for row in by_arm["F2"]
    )
    arms["F2"]["lag_span_exceeds_E2_frames"] = sum(
        float(row["lag_span_seconds"]) > float(row["source_E2_lag_span_seconds"])
        for row in by_arm["F2"]
    )
    published = json.loads(
        Path("benchmarks/m8/diagnostics/m8_h10_capacity_census.json").read_text(encoding="utf-8")
    )
    first = by_arm["B2"][0]
    census = next(
        row for row in published["records"] if row["condition_id"] == first["frame_id"] + "/H10"
    )
    if census["candidate_feature_sha256"] != first["source_A2_full_XYZIT_sha256"]:
        raise ValueError("published H10 census source identity differs")
    return {
        "schema_version": "laserperception.m8.s2.input-characterization.v1",
        "status": "CPU INPUT STRUCTURE ONLY; NO DETECTOR OUTPUT",
        "arithmetic_namespace": "CPU_ANALYTIC_STRUCTURAL_IDENTITY",
        "known_cross_runtime_example": {
            "condition_id": census["condition_id"],
            "cpu_analytic_candidate_pillar_count": first[
                "source_A2_cpu_analytic_candidate_pillar_count"
            ],
            "historical_cuda_candidate_pillar_count": census["candidate_dynamic_pillars"],
            "source_input_identity_matched": True,
            "count_equality_required": False,
            "exact_one_pillar_cause_localized": False,
        },
        "arms": arms,
        "detector_calls": 0,
    }


def _adapters(args: argparse.Namespace) -> tuple[CanonicalM7SourceAdapter, FrozenInputSource]:
    return (
        CanonicalM7SourceAdapter(args.dataset_root, args.m6_ledger),
        FrozenInputSource.load(
            date_root=args.dataset_root,
            full_ledger=args.m6_ledger,
            accepted_ledger=args.accepted_m8_ledger,
        ),
    )


def _frames(
    args: argparse.Namespace, frames: tuple[str, ...]
) -> Iterator[tuple[dict[str, object], ...]]:
    manifest = _load_manifest(args.m7_manifest)
    m7_adapter, m8_adapter = _adapters(args)
    for frame in frames:
        yield generate_frame(
            frame,
            m7_adapter=m7_adapter,
            m8_adapter=m8_adapter,
            manifest=manifest,
            implementation_commit=args.implementation_commit,
        )


def generate(args: argparse.Namespace) -> None:
    """Generate the complete private ledger and compact public records after preflight."""

    _verify_code_and_assets(args)
    frames = canonical_frame_ids()
    if len(frames) != 428:
        raise ValueError("frozen corpus must have 428 frames")
    args.private_ledger.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.private_ledger.with_suffix(".partial.jsonl")
    if args.private_ledger.exists() or temporary.exists():
        raise FileExistsError(
            "existing S2 ledger/partial ledger must be preserved before regeneration"
        )
    compact: list[dict[str, object]] = []
    with temporary.open("wb") as output:
        for index, group in enumerate(_frames(args, frames), start=1):
            if tuple(row["arm"] for row in group) != ARM_ORDER:
                raise ValueError("canonical S2 arm order changed")
            for row in group:
                output.write(_json_bytes(row))
                compact.append(row)
            if index % 20 == 0 or index == 428:
                print(f"frames={index}/428 conditions={len(compact)}", flush=True)
    if len(compact) != 1712 or len({str(row["condition_id"]) for row in compact}) != 1712:
        raise ValueError("missing or duplicate S2 condition")
    temporary.replace(args.private_ledger)
    ledger_bytes = args.private_ledger.stat().st_size
    ledger_sha = _sha256_file(args.private_ledger)
    characterization = _characterization(compact)
    _write_json(args.characterization, characterization)
    essential = (
        "condition_id",
        "frame_id",
        "arm",
        "point_count",
        "selected_global_row_sha256",
        "source_A2_full_XYZIT_sha256",
        "source_E2_full_XYZIT_sha256",
        "XYZ_sha256",
        "intensity_sha256",
        "XYZT_projection_sha256",
        "full_XYZIT_sha256",
        "M7_expected_XYZT_sha256",
        "M7_XYZT_exact",
        "cpu_analytic_candidate_pillar_count",
        "cpu_analytic_candidate_coordinate_sha256",
        "lag_support_count",
        "lag_span_seconds",
        "per_sweep_counts",
    )
    manifest = {
        "schema_version": "laserperception.m8.s2.input-manifest.v1",
        "status": "CPU INPUTS ONLY; NO DETECTOR OUTPUT; INFERENCE NOT AUTHORIZED",
        "implementation_commit": args.implementation_commit,
        "protocol_sha256": PROTOCOL_SHA256,
        "m7_input_manifest_sha256": M7_MANIFEST_SHA256,
        "condition_count": 1712,
        "canonical_order": "428 frames, each B2/C2/D2/F2",
        "conditions": [{key: row[key] for key in essential} for row in compact],
        "private_full_ledger": {
            "logical_filename": args.private_ledger.name,
            "bytes": ledger_bytes,
            "sha256": ledger_sha,
            "record_count": 1712,
        },
        "detector_calls": 0,
    }
    _write_json(args.compact_manifest, manifest)
    freeze = {
        "schema_version": "laserperception.m8.s2.input-freeze.v1",
        "implementation_commit": args.implementation_commit,
        "protocol_sha256": PROTOCOL_SHA256,
        "m7_manifest_sha256": M7_MANIFEST_SHA256,
        "full_ledger": manifest["private_full_ledger"],
        "compact_manifest": {
            "logical_filename": args.compact_manifest.name,
            "bytes": args.compact_manifest.stat().st_size,
            "sha256": _sha256_file(args.compact_manifest),
        },
        "characterization": {
            "logical_filename": args.characterization.name,
            "bytes": args.characterization.stat().st_size,
            "sha256": _sha256_file(args.characterization),
        },
        "M7_XYZT_proof": {arm: 428 for arm in ARM_ORDER} | {"total": 1712},
        "cpu_analytic_pairwise_coordinates": {"B2_A2": 428, "D2_C2": 428},
        "arm_relation_gates": {
            "B2_A2_row_XYZ_intensity_exact": 428,
            "C2_E2_point_count_and_A2_subset_exact": 428,
            "D2_C2_row_XYZ_intensity_exact": 428,
            "F2_frozen_ranks_and_A2_subset_exact": 428,
        },
        "replay_28_exact": False,
        "implementation_frozen": False,
        "input_ledger_frozen": False,
        "runtime_bound": False,
        "inference_authorized": False,
        "S2_calls": 0,
    }
    _write_json(args.input_freeze, freeze)
    print(f"ledger bytes={ledger_bytes} sha256={ledger_sha}", flush=True)


def replay(args: argparse.Namespace) -> None:
    """Regenerate seven sentinel frames in this fresh CPU process and compare all fields."""

    _verify_code_and_assets(args)
    expected = {}
    with args.private_ledger.open("rb") as source:
        for line in source:
            row = json.loads(line)
            if row["frame_id"] in SENTINELS:
                expected[row["condition_id"]] = row
    if len(expected) != 28:
        raise ValueError("private ledger lacks the 28 sentinel conditions")
    verified = 0
    for group in _frames(args, SENTINELS):
        for row in group:
            if row != expected.get(row["condition_id"]):
                raise ValueError(f"fresh-adapter replay differs: {row['condition_id']}")
            verified += 1
    if verified != 28:
        raise ValueError("fresh-adapter replay count differs")
    freeze = json.loads(args.input_freeze.read_text(encoding="utf-8"))
    if freeze["full_ledger"]["sha256"] != _sha256_file(args.private_ledger):
        raise ValueError("private ledger changed before replay")
    freeze["replay_28_exact"] = True
    freeze["implementation_frozen"] = True
    freeze["input_ledger_frozen"] = True
    _write_json(args.input_freeze, freeze)
    print("fresh-adapter input replay=28/28 exact", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("generate", "replay"))
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--m6-ledger", required=True, type=Path)
    parser.add_argument("--m6-result", required=True, type=Path)
    parser.add_argument("--accepted-m8-ledger", required=True, type=Path)
    parser.add_argument("--m7-manifest", required=True, type=Path)
    parser.add_argument("--private-ledger", required=True, type=Path)
    parser.add_argument("--compact-manifest", required=True, type=Path)
    parser.add_argument("--characterization", required=True, type=Path)
    parser.add_argument("--input-freeze", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "generate":
        generate(args)
    else:
        replay(args)


if __name__ == "__main__":
    main()
