from pathlib import Path

import pytest

from laserperception.detection.artifacts import ExternalArtifactMetadata
from laserperception.detection.tensorrt_backend import (
    TensorRTEnvironmentError,
    load_tensorrt,
)


def test_external_artifact_metadata_is_sanitized(tmp_path: Path) -> None:
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"external model")

    metadata = ExternalArtifactMetadata.from_file(artifact, logical_name="m2/pointpillars.onnx")

    serialized = metadata.to_dict()
    assert serialized["logical_name"] == "m2/pointpillars.onnx"
    assert serialized["size_bytes"] == len(b"external model")
    assert serialized["committed"] is False
    assert str(tmp_path) not in str(serialized)


@pytest.mark.parametrize(
    "logical_name",
    ["/tmp/model.onnx", "C:/tmp/model.onnx", "..\\model.onnx", "../model.onnx", ""],
)
def test_external_artifact_metadata_rejects_unsafe_names(logical_name: str) -> None:
    with pytest.raises(ValueError):
        ExternalArtifactMetadata(logical_name, "a" * 64, 10)


def test_tensorrt_dependency_error_is_lazy_and_focused() -> None:
    def unavailable(_: str):
        raise ImportError("synthetic missing dependency")

    with pytest.raises(TensorRTEnvironmentError, match="optional and unavailable"):
        load_tensorrt(import_module=unavailable)


def test_tensorrt_version_is_pinned() -> None:
    class WrongTensorRT:
        __version__ = "10.0.0"

    with pytest.raises(TensorRTEnvironmentError, match="requires TensorRT 8.6.1"):
        load_tensorrt(import_module=lambda _: WrongTensorRT())
