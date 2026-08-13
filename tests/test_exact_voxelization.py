from pathlib import Path

from laserperception.detection.exact_voxelization import ExactDeterministicVoxelizer


def test_exact_voxelizer_module_is_cpu_importable() -> None:
    assert ExactDeterministicVoxelizer.__name__ == "ExactDeterministicVoxelizer"


def test_m3b_v2_protocol_freezes_exact_gates_and_scope() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = (root / "configs/detection/m3b_deterministic_voxelization_v2.yaml").read_text(
        encoding="utf-8"
    )

    assert "status: protocol_frozen_before_measurement" in protocol
    assert "required_exact_samples: 81" in protocol
    assert "stop_on_first_mismatch: true" in protocol
    assert "runs_per_sample: 30" in protocol
    assert "reference_deterministic: true" in protocol
    assert "coordinate_operation: pinned_mmcv_dynamic_voxel_coordinate_cuda" in protocol
    assert "stable_sort_required: false" in protocol
    assert "custom_cuda: false" in protocol
    assert "deterministic_false_fallback_allowed: false" in protocol
    assert "production_candidate_adoption_allowed: false" in protocol
    assert "postprocess_optimization_allowed: false" in protocol
    assert "ros_dds_optimization_allowed: false" in protocol
    assert (
        "cuda_launcher:\n"
        "    logical_name: mmcv/ops/csrc/pytorch/cuda/voxelization_cuda.cu\n"
        "    sha256: 9a089b79490c1a53648601d992bdacea4f1272af7c127dc0c2ac854ef3f79d2d" in protocol
    )


def test_m3_ros_config_explicitly_selects_exact_fast_live() -> None:
    root = Path(__file__).parents[1]
    config = (root / "ros2/laserperception_ros/config/m3_ros2.yaml").read_text(encoding="utf-8")

    assert "voxelization_mode: exact_fast" in config
    assert "provenance_mode: live" in config
    assert "start_index: 42" in config
