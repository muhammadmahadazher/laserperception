from __future__ import annotations

import numpy as np
import pytest

from laserperception.detection.multisweep import LidarPose, RawSweep
from laserperception.evaluation.m6c_projected_reference import (
    build_projected_reference_from_sources,
    projected_relative_transform,
    projected_world_pose,
)


def _pose(rotation: np.ndarray, translation: tuple[float, float, float]) -> LidarPose:
    return LidarPose(
        lidar_to_ego_rotation=np.eye(3, dtype=np.float64),
        lidar_to_ego_translation=np.zeros(3, dtype=np.float64),
        ego_to_global_rotation=np.asarray(rotation, dtype=np.float64),
        ego_to_global_translation=np.asarray(translation, dtype=np.float64),
    )


def test_projected_world_pose_is_a_proper_rotation() -> None:
    nearly_rotation = np.array(
        [[1.0, -1.0e-8, 0.0], [1.0e-8, 1.0 + 2.0e-8, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    projected = projected_world_pose(_pose(nearly_rotation, (1.0, 2.0, 3.0)))
    np.testing.assert_allclose(projected.rotation.T @ projected.rotation, np.eye(3), atol=1e-15)
    assert np.linalg.det(projected.rotation) == pytest.approx(1.0, abs=1e-15)
    assert np.linalg.norm(projected.quaternion_xyzw) == pytest.approx(1.0, abs=1e-15)
    np.testing.assert_array_equal(projected.translation, np.array([1.0, 2.0, 3.0]))


def test_projected_relative_transform_uses_builder_storage_mapping() -> None:
    yaw = np.pi / 2.0
    rotation = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    historical_pose = _pose(np.eye(3), (2.0, 0.0, 0.0))
    current_pose = _pose(rotation, (0.0, 0.0, 0.0))
    transform = projected_relative_transform(
        source_id="historical",
        target_id="current",
        historical_pose=historical_pose,
        current_pose=current_pose,
    )
    historical = projected_world_pose(historical_pose)
    current = projected_world_pose(current_pose)
    expected_relative_rotation = current.rotation.T @ historical.rotation
    expected_relative_translation = current.rotation.T @ (
        historical.translation - current.translation
    )
    expected = np.eye(4, dtype=np.float64)
    expected[:3, :3] = expected_relative_rotation.T
    expected[:3, 3] = -expected_relative_rotation.T @ expected_relative_translation
    np.testing.assert_array_equal(transform.lidar2sensor, expected.astype(np.float32))


@pytest.mark.parametrize("history_depth", [True, 0, -1, 1.5])
def test_projected_reference_history_depth_is_fail_closed(history_depth: object) -> None:
    from laserperception.evaluation.m6c_projected_reference import build_projected_reference

    with pytest.raises((TypeError, ValueError)):
        build_projected_reference(None, current_index=10, history_depth=history_depth)  # type: ignore[arg-type]


def test_projected_reference_accepts_preloaded_sources_without_changing_order() -> None:
    sweeps = {
        index: RawSweep(
            np.array([[float(index), 1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            timestamp_microseconds=1_000_000 + index * 100_000,
            source_id=f"frame-{index}",
        )
        for index in range(3)
    }
    poses = {index: _pose(np.eye(3), (float(index), 0.0, 0.0)) for index in range(3)}
    result = build_projected_reference_from_sources(
        current_index=2,
        history_depth=10,
        sweeps=sweeps,
        poses=poses,
    )
    assert result.historical_indices == (1, 0)
    assert [transform.source_id for transform in result.transforms] == ["frame-1", "frame-0"]
    assert result.point_cloud.points_xyzt.shape == (3, 4)
    np.testing.assert_array_equal(
        result.point_cloud.points_xyzt[:, 3],
        np.array([0.0, 0.1, 0.2], dtype=np.float32),
    )


def test_projected_reference_preloaded_sources_fail_closed_when_incomplete() -> None:
    sweep = RawSweep(
        np.array([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        timestamp_microseconds=1_000_000,
        source_id="frame-0",
    )
    with pytest.raises(KeyError, match="sources are incomplete"):
        build_projected_reference_from_sources(
            current_index=1,
            history_depth=1,
            sweeps={0: sweep},
            poses={0: _pose(np.eye(3), (0.0, 0.0, 0.0))},
        )
