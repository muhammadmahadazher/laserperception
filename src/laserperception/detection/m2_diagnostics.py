"""CPU-testable device metadata and M2 benchmark sanity helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

RAW_OUTPUT_NAMES = ("cls_score", "bbox_pred", "dir_cls_pred")


def tensor_metadata(tensor: object, *, name: str) -> dict[str, object]:
    """Return a small device/dtype/shape record without importing PyTorch."""

    shape = getattr(tensor, "shape", None)
    device = getattr(tensor, "device", None)
    dtype = getattr(tensor, "dtype", None)
    if shape is None or device is None or dtype is None:
        raise TypeError(f"{name} must expose shape, device, and dtype")
    try:
        dimensions = [int(value) for value in shape]
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} has an invalid tensor shape") from error
    return {
        "name": name,
        "device": str(device),
        "dtype": str(dtype),
        "shape": dimensions,
    }


def assert_cuda0_tensor(
    tensor: object,
    *,
    name: str,
    expected_dtype: str | None = None,
    expected_shape: Sequence[int] | None = None,
) -> dict[str, object]:
    """Fail unless one tensor is on CUDA device zero with expected metadata."""

    metadata = tensor_metadata(tensor, name=name)
    if metadata["device"] != "cuda:0":
        raise RuntimeError(f"{name} must be on cuda:0, found {metadata['device']}")
    if expected_dtype is not None and metadata["dtype"] != expected_dtype:
        raise RuntimeError(f"{name} must have dtype {expected_dtype}, found {metadata['dtype']}")
    if expected_shape is not None:
        shape = [int(value) for value in expected_shape]
        if metadata["shape"] != shape:
            raise RuntimeError(f"{name} must have shape {shape}, found {metadata['shape']}")
    return metadata


def assert_cuda0_model(model: object, *, name: str) -> dict[str, object]:
    """Fail unless the model's first parameter is FP32 on CUDA device zero."""

    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        raise TypeError(f"{name} must expose parameters()")
    try:
        parameter = next(iter(parameters()))
    except StopIteration as error:
        raise RuntimeError(f"{name} exposes no parameters") from error
    return assert_cuda0_tensor(
        parameter, name=f"{name}.first_parameter", expected_dtype="torch.float32"
    )


def assert_raw_outputs_cuda0(
    raw: Mapping[str, Sequence[object]],
    *,
    runtime_name: str,
    expected_dtype: str,
    expected_shapes: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, dict[str, object]]:
    """Validate the three one-level PointPillars output tensors."""

    records: dict[str, dict[str, object]] = {}
    if set(raw) != set(RAW_OUTPUT_NAMES):
        raise RuntimeError(
            f"{runtime_name} outputs must be exactly {RAW_OUTPUT_NAMES}, found {tuple(raw)}"
        )
    for name in RAW_OUTPUT_NAMES:
        levels = raw[name]
        if not isinstance(levels, Sequence) or isinstance(levels, (str, bytes)) or len(levels) != 1:
            raise RuntimeError(f"{runtime_name}.{name} must contain exactly one tensor")
        expected_shape = None if expected_shapes is None else expected_shapes[name]
        records[name] = assert_cuda0_tensor(
            levels[0],
            name=f"{runtime_name}.{name}",
            expected_dtype=expected_dtype,
            expected_shape=expected_shape,
        )
    return records


def benchmark_review_flags(
    *,
    native_e2e_median_ms: float,
    native_network_median_ms: float,
    tensorrt_e2e_median_ms: float,
    tensorrt_network_median_ms: float,
    native_e2e_p95_ms: float,
    tensorrt_e2e_p95_ms: float,
    historical_m1_e2e_median_ms: float = 55.097,
) -> list[str]:
    """Return non-scientific reviewer flags for suspicious benchmark evidence."""

    values = {
        "native_e2e_median_ms": native_e2e_median_ms,
        "native_network_median_ms": native_network_median_ms,
        "tensorrt_e2e_median_ms": tensorrt_e2e_median_ms,
        "tensorrt_network_median_ms": tensorrt_network_median_ms,
        "native_e2e_p95_ms": native_e2e_p95_ms,
        "tensorrt_e2e_p95_ms": tensorrt_e2e_p95_ms,
        "historical_m1_e2e_median_ms": historical_m1_e2e_median_ms,
    }
    if any(value <= 0.0 for value in values.values()):
        raise ValueError("benchmark sanity inputs must be positive")

    flags: list[str] = []
    if native_e2e_median_ms > 2.0 * historical_m1_e2e_median_ms:
        flags.append("native_pytorch_e2e_over_2x_historical_m1")
    if native_e2e_median_ms / tensorrt_e2e_median_ms > 10.0:
        flags.append("end_to_end_speedup_over_10x")
    if native_network_median_ms / tensorrt_network_median_ms > 10.0:
        flags.append("network_speedup_over_10x")
    if native_e2e_p95_ms > 2.0 * native_e2e_median_ms:
        flags.append("native_pytorch_e2e_p95_over_2x_median")
    if tensorrt_e2e_p95_ms > 2.0 * tensorrt_e2e_median_ms:
        flags.append("tensorrt_e2e_p95_over_2x_median")
    if native_e2e_median_ms < native_network_median_ms:
        flags.append("native_pytorch_e2e_below_network")
    if tensorrt_e2e_median_ms < tensorrt_network_median_ms:
        flags.append("tensorrt_e2e_below_network")
    return flags
