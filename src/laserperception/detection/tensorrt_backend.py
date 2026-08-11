"""Lazy TensorRT loading and binding metadata inspection for M2."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType


class TensorRTEnvironmentError(RuntimeError):
    """Raised when the optional pinned TensorRT runtime is unavailable."""


def load_tensorrt(
    *, import_module: Callable[[str], ModuleType] = importlib.import_module
) -> ModuleType:
    """Load TensorRT lazily with a focused optional-dependency error."""

    try:
        tensorrt = import_module("tensorrt")
    except (ImportError, OSError) as error:
        raise TensorRTEnvironmentError(
            "TensorRT is optional and unavailable; run scripts/setup_detection_m2.sh "
            "inside the isolated M2 environment"
        ) from error
    version = str(getattr(tensorrt, "__version__", ""))
    if version != "8.6.1":
        raise TensorRTEnvironmentError(f"M2 requires TensorRT 8.6.1, found {version or 'unknown'}")
    return tensorrt


def inspect_engine(
    path: str | Path,
    *,
    expected_bindings: Sequence[str],
    expected_profile: Mapping[str, Mapping[str, Sequence[int]]],
) -> dict[str, object]:
    """Deserialize an engine and validate its public bindings and optimization profile."""

    trt = load_tensorrt()
    engine_path = Path(path)
    if not engine_path.is_file():
        raise FileNotFoundError("TensorRT engine was not found in the external M2 cache")
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, namespace="")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise TensorRTEnvironmentError("TensorRT could not deserialize the external M2 engine")
    names = [str(engine.get_binding_name(index)) for index in range(engine.num_bindings)]
    if names != list(expected_bindings):
        raise TensorRTEnvironmentError(
            f"TensorRT bindings differ from the frozen contract: expected "
            f"{list(expected_bindings)}, found {names}"
        )
    bindings: list[dict[str, object]] = []
    for index, name in enumerate(names):
        record: dict[str, object] = {
            "index": index,
            "name": name,
            "is_input": bool(engine.binding_is_input(index)),
            "dtype": str(engine.get_binding_dtype(index)),
            "shape": [int(value) for value in engine.get_binding_shape(index)],
        }
        if record["is_input"]:
            minimum, optimum, maximum = engine.get_profile_shape(0, index)
            actual = {
                "min_shape": [int(value) for value in minimum],
                "opt_shape": [int(value) for value in optimum],
                "max_shape": [int(value) for value in maximum],
            }
            frozen = {
                key: [int(value) for value in expected_profile[name][key]]
                for key in ("min_shape", "opt_shape", "max_shape")
            }
            if actual != frozen:
                raise TensorRTEnvironmentError(
                    f"TensorRT profile for {name} differs from the frozen contract"
                )
            record["profile"] = actual
        bindings.append(record)
    context = engine.create_execution_context()
    if context is None:
        raise TensorRTEnvironmentError("TensorRT could not create an execution context")
    return {
        "tensorrt_version": str(trt.__version__),
        "bindings": bindings,
        "optimization_profiles": int(engine.num_optimization_profiles),
        "engine_device_memory_size_bytes": int(engine.device_memory_size),
        "execution_context_created": True,
        "has_implicit_batch_dimension": bool(engine.has_implicit_batch_dimension),
    }
