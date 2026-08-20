from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from benchmarks.m6a import generate_offline_reconstruction as oracle
from laserperception.datasets.kitti_raw import KittiRawSequence


def _identity_calibration(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rigid = "R: 1 0 0 0 1 0 0 0 1\nT: 0 0 0\n"
    (root / "calib_imu_to_velo.txt").write_text(rigid, encoding="utf-8")
    (root / "calib_velo_to_cam.txt").write_text(rigid, encoding="utf-8")
    (root / "calib_cam_to_cam.txt").write_text("R_rect_00: 1 0 0 0 1 0 0 0 1\n", encoding="utf-8")


def _synthetic_drive(tmp_path: Path, frame_count: int = 3) -> tuple[Path, Path]:
    date_root = tmp_path / "2011_09_26"
    drive_root = date_root / "2011_09_26_drive_0001_sync"
    point_root = drive_root / "velodyne_points/data"
    oxts_root = drive_root / "oxts/data"
    point_root.mkdir(parents=True)
    oxts_root.mkdir(parents=True)
    _identity_calibration(date_root)
    timestamps: list[str] = []
    for index in range(frame_count):
        np.array(
            [[1.0 + index, 2.0, 0.0, 0.5], [3.0 + index, 4.0, 1.0, 0.75]],
            dtype="<f4",
        ).tofile(point_root / f"{index:010d}.bin")
        (oxts_root / f"{index:010d}.txt").write_text(
            f"49.0 {8.0 + index * 1e-6} 100.0 0.0 0.0 {index * 0.01}\n",
            encoding="utf-8",
        )
        timestamps.append(f"2011-09-26 00:00:00.{index * 100_000_000:09d}")
    (drive_root / "velodyne_points/timestamps.txt").write_text(
        "\n".join(timestamps) + "\n", encoding="utf-8"
    )
    return date_root, drive_root


def test_exact_raw_pose_gate_is_bit_exact_on_identical_arithmetic_path(tmp_path: Path) -> None:
    date_root, drive_root = _synthetic_drive(tmp_path)
    result = oracle.exact_raw_pose_gate(
        date_root,
        drive_root,
        frame_count=3,
        role="synthetic",
    )
    assert result["status"] == "pass"
    assert result["exact_equality_count"] == 3
    assert result["matrix_max_abs"] == 0.0
    assert result["rotation_matrix_max_abs"] == 0.0
    assert result["translation_norm_m"] == 0.0
    assert result["frame_zero"]["production_reference_exact"] is True


def test_exact_raw_pose_gate_stops_on_any_nonzero_difference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    date_root, drive_root = _synthetic_drive(tmp_path)
    original = oracle.official_oxts_poses

    def changed(records: object) -> tuple[np.ndarray, ...]:
        result = list(original(records))  # type: ignore[arg-type]
        result[1] = result[1].copy()
        result[1][0, 3] = np.nextafter(result[1][0, 3], np.inf)
        return tuple(result)

    monkeypatch.setattr(oracle, "official_oxts_poses", changed)
    with pytest.raises(RuntimeError, match="nonzero difference is the finding"):
        oracle.exact_raw_pose_gate(
            date_root,
            drive_root,
            frame_count=3,
            role="synthetic",
        )


def test_manual_builder_oracle_matches_unchanged_builder_bytes(tmp_path: Path) -> None:
    date_root, drive_root = _synthetic_drive(tmp_path)
    sequence = KittiRawSequence(date_root, drive_root)
    parts, selected, counts = oracle._manual_builder_parts(sequence, 2)
    expected = np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)
    production = sequence.reconstruct(2)
    assert selected == production.selected_indices
    assert counts == production.source_counts
    assert np.array_equal(expected, production.point_cloud.points_xyzt)


def test_candidate_pillar_count_deduplicates_xy_cells() -> None:
    points = np.array(
        [
            [-49.99, -49.99, 0.0, 0.0],
            [-49.80, -49.80, 0.0, 0.1],
            [-49.70, -49.70, 0.0, 0.2],
        ],
        dtype=np.float32,
    )
    assert oracle._candidate_pillars(points) == 2


def test_tracklet_contract_records_selected_drive_availability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "kitti"
    drive_root = data_root / "extracted/2011_09_26/2011_09_26_drive_0001_sync"
    drive_root.mkdir(parents=True)
    xml = """<?xml version="1.0"?>
<boost_serialization>
  <tracklets>
    <count>1</count>
    <item>
      <objectType>Car</objectType><h>1.5</h><w>1.6</w><l>4.0</l><first_frame>0</first_frame>
      <poses><count>2</count>
        <item><tx>1</tx><ty>2</ty><tz>0</tz><rx>0</rx><ry>0</ry><rz>0.1</rz><state>2</state><occlusion>0</occlusion><truncation>0</truncation></item>
        <item><tx>2</tx><ty>2</ty><tz>0</tz><rx>0</rx><ry>0</ry><rz>0.2</rz><state>1</state><occlusion>1</occlusion><truncation>1</truncation></item>
      </poses>
    </item>
  </tracklets>
</boost_serialization>
"""
    xml_path = drive_root / "tracklet_labels.xml"
    xml_path.write_text(xml, encoding="utf-8")
    archive = data_root / "archives/2011_09_26_drive_0001_tracklets.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"synthetic archive")
    monkeypatch.setattr(oracle, "EXPECTED_TRACKLET_XML_SHA256", oracle.sha256_file(xml_path))
    result = oracle.tracklet_contract(data_root, drive_root)
    assert result["status"] == "pass"
    assert result["selected_drive_available"] is True
    assert result["tracklet_count"] == 1
    assert result["pose_count"] == 2
    assert result["class_counts"] == {"Car": 1}
    assert result["covered_frame_min"] == 0
    assert result["covered_frame_max"] == 1
    assert result["archive_sha256"] == hashlib.sha256(b"synthetic archive").hexdigest()


def test_model_frame_evidence_uses_tracked_calibration_and_proper_alignment() -> None:
    result = oracle.model_frame_evidence()
    assert result["status"] == "pass"
    assert result["model_basis"] == {
        "+X": "vehicle right",
        "+Y": "vehicle forward",
        "+Z": "up",
    }
    assert result["determinant"] == pytest.approx(1.0)
    assert np.array_equal(
        np.asarray(result["inverse"]),
        np.asarray(result["model_from_kitti_rotation"]).T,
    )
