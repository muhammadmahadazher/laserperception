import numpy as np

from laserperception.evaluation.m6b_pillars import (
    analyze_pillars,
    pillar_box_overlap_mask,
    pillar_centres,
    spatial_regions,
)


def test_candidate_order_and_capacity_follow_first_point_touch() -> None:
    points = np.array(
        [
            [0.10, 0.10, 0.0, 0.0],
            [1.10, 0.10, 0.0, 0.1],
            [0.20, 0.20, 0.0, 0.2],
            [2.10, 0.10, 0.0, 0.2],
        ],
        dtype=np.float32,
    )

    audit = analyze_pillars(points, max_voxels=2)

    assert audit.candidate_count == 3
    assert audit.retained_count == 2
    assert audit.discarded_count == 1
    assert audit.overflow
    assert audit.candidate_xy_indices.tolist() == [[200, 200], [204, 200], [208, 200]]
    assert audit.candidate_first_touch_sweep.tolist() == [0, 1, 2]
    assert audit.summary()["first_touch_sweep_histogram"] == {
        "candidate": {"0": 1, "1": 1, "2": 1},
        "retained": {"0": 1, "1": 1},
        "discarded": {"2": 1},
    }


def test_out_of_grid_points_do_not_create_candidates() -> None:
    points = np.array([[-50.1, 0.0, 0.0, 0.0], [49.9, 49.9, 0.0, 0.0]], dtype=np.float32)

    audit = analyze_pillars(points)

    assert audit.in_range_points == 1
    assert audit.candidate_count == 1


def test_pillar_centres_and_box_overlap_use_cell_area_not_only_centres() -> None:
    cells = np.array([[200, 200], [201, 200], [210, 210]], dtype=np.int32)

    centres = pillar_centres(cells)
    overlap = pillar_box_overlap_mask(
        cells,
        center_xy=(0.25, 0.125),
        size_lw=(0.25, 0.25),
        yaw_rad=0.0,
    )

    assert centres.tolist() == [[0.125, 0.125], [0.375, 0.125], [2.625, 2.625]]
    assert overlap.tolist() == [True, True, False]


def test_spatial_regions_use_frozen_sector_quadrant_and_range_edges() -> None:
    cells = np.array([[240, 200], [200, 240], [60, 200], [200, 60]], dtype=np.int32)

    regions = spatial_regions(cells)

    assert regions["azimuth_sector"].tolist() == [0, 2, 5, 9]
    assert regions["cartesian_quadrant"].tolist() == [0, 0, 2, 1]
    assert regions["radial_bin"].tolist() == [0, 0, 1, 1]
