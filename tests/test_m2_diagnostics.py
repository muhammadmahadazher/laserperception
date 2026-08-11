from __future__ import annotations

from dataclasses import dataclass

import pytest

from laserperception.detection.m2_diagnostics import (
    assert_cuda0_model,
    assert_cuda0_tensor,
    assert_raw_outputs_cuda0,
    benchmark_review_flags,
)


@dataclass(frozen=True)
class FakeTensor:
    shape: tuple[int, ...]
    device: str = "cuda:0"
    dtype: str = "torch.float32"


class FakeModel:
    def __init__(self, parameter: FakeTensor) -> None:
        self.parameter = parameter

    def parameters(self) -> object:
        yield self.parameter


def _raw() -> dict[str, list[FakeTensor]]:
    return {
        "cls_score": [FakeTensor((1, 140, 200, 200))],
        "bbox_pred": [FakeTensor((1, 126, 200, 200))],
        "dir_cls_pred": [FakeTensor((1, 28, 200, 200))],
    }


def test_device_assertions_record_device_dtype_and_shape() -> None:
    tensor = FakeTensor((3, 4))

    assert assert_cuda0_tensor(
        tensor,
        name="voxels",
        expected_dtype="torch.float32",
        expected_shape=(3, 4),
    ) == {
        "name": "voxels",
        "device": "cuda:0",
        "dtype": "torch.float32",
        "shape": [3, 4],
    }
    assert assert_cuda0_model(FakeModel(tensor), name="native_model")["device"] == "cuda:0"


def test_device_assertions_fail_loudly_for_cpu() -> None:
    with pytest.raises(RuntimeError, match="must be on cuda:0"):
        assert_cuda0_tensor(FakeTensor((1,), device="cpu"), name="coors")


def test_raw_output_assertions_require_one_cuda_level_and_exact_shapes() -> None:
    shapes = {name: value[0].shape for name, value in _raw().items()}

    records = assert_raw_outputs_cuda0(
        _raw(), runtime_name="native", expected_dtype="torch.float32", expected_shapes=shapes
    )

    assert records["bbox_pred"]["shape"] == [1, 126, 200, 200]
    malformed = _raw()
    malformed["cls_score"] = []
    with pytest.raises(RuntimeError, match="exactly one tensor"):
        assert_raw_outputs_cuda0(malformed, runtime_name="native", expected_dtype="torch.float32")


def test_benchmark_review_flags_cover_requested_sanity_conditions() -> None:
    flags = benchmark_review_flags(
        native_e2e_median_ms=120.0,
        native_network_median_ms=130.0,
        tensorrt_e2e_median_ms=8.0,
        tensorrt_network_median_ms=5.0,
        native_e2e_p95_ms=250.0,
        tensorrt_e2e_p95_ms=20.0,
    )

    assert flags == [
        "native_pytorch_e2e_over_2x_historical_m1",
        "end_to_end_speedup_over_10x",
        "network_speedup_over_10x",
        "native_pytorch_e2e_p95_over_2x_median",
        "tensorrt_e2e_p95_over_2x_median",
        "native_pytorch_e2e_below_network",
    ]


def test_benchmark_review_flags_are_empty_for_plausible_measurements() -> None:
    assert (
        benchmark_review_flags(
            native_e2e_median_ms=62.0,
            native_network_median_ms=20.0,
            tensorrt_e2e_median_ms=55.0,
            tensorrt_network_median_ms=10.0,
            native_e2e_p95_ms=75.0,
            tensorrt_e2e_p95_ms=70.0,
        )
        == []
    )
