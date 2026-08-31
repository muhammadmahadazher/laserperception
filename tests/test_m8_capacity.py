from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from laserperception.detection.m8_backend import (
    DsvtBackend,
    load_m8_candidate_manifest,
)
from laserperception.detection.m8_capacity import (
    DSVT_THEORETICAL_XY_CELLS,
    candidate_dynamic_pillar_coordinates,
    load_dsvt_capacity_contract,
    require_capacity,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/m8/dsvt_nuscenes_pillar.json"


def test_candidate_coordinates_match_selected_floor_unique_semantics() -> None:
    points = np.array(
        [
            [-50.0, -50.0, 0.0, 0.1, 0.0],
            [-49.99, -49.99, 1.0, 0.2, 0.1],
            [0.0, 0.0, 0.0, 0.3, 0.2],
            [49.999, 49.999, 0.0, 0.4, 0.3],
        ],
        dtype=np.float32,
    )

    coordinates = candidate_dynamic_pillar_coordinates(points)

    assert coordinates.dtype == np.int32
    assert coordinates.flags.c_contiguous
    assert coordinates.tolist() == [[13, 13], [180, 180], [346, 346]]


def test_manifest_capacity_contract_is_exact_and_uncapped() -> None:
    manifest = load_m8_candidate_manifest(MANIFEST)

    contract = load_dsvt_capacity_contract(manifest)

    assert contract.grid_size == (360, 360, 1)
    assert contract.voxel_size == (0.3, 0.3, 8.0)
    assert contract.theoretical_xy_cells == DSVT_THEORETICAL_XY_CELLS
    assert contract.dynamic_pillar_cap is None
    assert contract.set_size == 90
    assert contract.block_count == 4
    capacity = manifest["structural_capacity_contract"]
    assert capacity["max_number_of_voxels_present"] is False
    assert capacity["coordinate_dtype_before_unique"] == "int32"
    assert capacity["input_layer_coordinate_dtype"] == "int64"
    assert capacity["set_index_dtype"] == "int64"


def test_capacity_violation_fails_closed() -> None:
    contract = load_dsvt_capacity_contract(load_m8_candidate_manifest(MANIFEST))

    require_capacity(10_000, contract=contract)
    with pytest.raises(RuntimeError, match="spatial grid"):
        require_capacity(DSVT_THEORETICAL_XY_CELLS + 1, contract=contract)
    with pytest.raises(RuntimeError, match="TensorRT profile"):
        require_capacity(3_688, contract=contract, deployment_profile_max=3_687)


def test_no_zero_intensity_inference_entrypoint_exists() -> None:
    assert not hasattr(DsvtBackend, "infer_zero_intensity")
