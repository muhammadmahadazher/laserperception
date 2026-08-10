"""Minimal lazy adapter for the pinned MMDetection3D M1 backend."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from laserperception.detection.types import Detection3D, DetectionFrame

EXPECTED_M1_VERSIONS = {
    "torch": "2.1.0+cu118",
    "mmcv": "2.1.0",
    "mmengine": "0.10.7",
    "mmdet": "3.2.0",
    "mmdet3d": "1.4.0",
}


class DetectionEnvironmentError(RuntimeError):
    """Raised when the optional M1 GPU environment is missing or incompatible."""


@dataclass(frozen=True, slots=True)
class _Mmdet3dRuntime:
    torch: ModuleType
    mmcv: ModuleType
    mmengine: ModuleType
    mmdet: ModuleType
    mmdet3d: ModuleType
    config_type: Any
    pseudo_collate: Any
    init_default_scope: Any
    init_model: Any
    datasets: Any

    @property
    def versions(self) -> dict[str, str]:
        return {
            "torch": str(self.torch.__version__),
            "mmcv": str(self.mmcv.__version__),
            "mmengine": str(self.mmengine.__version__),
            "mmdet": str(self.mmdet.__version__),
            "mmdet3d": str(self.mmdet3d.__version__),
        }


@dataclass(frozen=True, slots=True)
class PreparedMmdet3dSample:
    """One official nuScenes pipeline result ready for model execution."""

    batch: object
    points_xyz: np.ndarray
    sample_id: str
    sample_index: int
    split: str

    def __post_init__(self) -> None:
        points = np.asarray(self.points_xyz, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_xyz must have shape (N, 3)")
        if not np.isfinite(points).all():
            raise ValueError("points_xyz must contain only finite values")
        object.__setattr__(self, "points_xyz", points.copy())


def _load_mmdet3d_runtime() -> _Mmdet3dRuntime:
    try:
        torch = importlib.import_module("torch")
        mmcv = importlib.import_module("mmcv")
        mmengine = importlib.import_module("mmengine")
        mmdet = importlib.import_module("mmdet")
        mmdet3d = importlib.import_module("mmdet3d")
        config_type = importlib.import_module("mmengine.config").Config
        pseudo_collate = importlib.import_module("mmengine.dataset").pseudo_collate
        init_default_scope = importlib.import_module("mmengine.registry").init_default_scope
        init_model = importlib.import_module("mmdet3d.apis").init_model
        datasets = importlib.import_module("mmdet3d.registry").DATASETS
    except (AttributeError, ImportError, OSError) as error:
        raise DetectionEnvironmentError(
            "The optional M1 detection environment is unavailable. Run "
            "scripts/setup_detection_m1.sh inside Ubuntu 22.04 WSL2, activate "
            "~/.venvs/laserperception-m1, and retry."
        ) from error

    return _Mmdet3dRuntime(
        torch=torch,
        mmcv=mmcv,
        mmengine=mmengine,
        mmdet=mmdet,
        mmdet3d=mmdet3d,
        config_type=config_type,
        pseudo_collate=pseudo_collate,
        init_default_scope=init_default_scope,
        init_model=init_model,
        datasets=datasets,
    )


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a file SHA256 without loading the asset into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_numpy(value: object, *, name: str) -> np.ndarray:
    current = value
    for method_name in ("detach", "cpu"):
        method = getattr(current, method_name, None)
        if callable(method):
            current = method()
    numpy_method = getattr(current, "numpy", None)
    if callable(numpy_method):
        current = numpy_method()
    try:
        return np.asarray(current)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} cannot be converted to a NumPy array") from error


def convert_mmdet3d_prediction(
    prediction: object,
    *,
    class_names: Sequence[str],
    sample_id: str,
    metadata: Mapping[str, object] | None = None,
) -> DetectionFrame:
    """Convert one MMDetection3D prediction without retaining upstream objects.

    MMDetection3D LiDAR boxes expose geometric centers through
    ``gravity_center`` and dimensions in ``(length, width, height)`` order.
    Velocity is taken from tensor columns 7 and 8 when the head supplies it.
    No score threshold is applied here.
    """

    instances = getattr(prediction, "pred_instances_3d", None)
    if instances is None:
        raise TypeError("prediction must expose pred_instances_3d")

    boxes = getattr(instances, "bboxes_3d", None)
    if boxes is None:
        raise TypeError("pred_instances_3d must expose bboxes_3d")
    centers = _to_numpy(getattr(boxes, "gravity_center", None), name="gravity_center")
    dimensions = _to_numpy(getattr(boxes, "dims", None), name="dims")
    yaws = _to_numpy(getattr(boxes, "yaw", None), name="yaw").reshape(-1)
    box_tensor = _to_numpy(getattr(boxes, "tensor", None), name="box tensor")
    scores = _to_numpy(getattr(instances, "scores_3d", None), name="scores_3d").reshape(-1)
    labels = _to_numpy(getattr(instances, "labels_3d", None), name="labels_3d").reshape(-1)

    count = len(scores)
    expected_shapes = {
        "gravity_center": (count, 3),
        "dims": (count, 3),
    }
    for name, expected in expected_shapes.items():
        value = centers if name == "gravity_center" else dimensions
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
    if yaws.shape != (count,) or labels.shape != (count,):
        raise ValueError("box yaw, score, and label counts must match")
    if box_tensor.ndim != 2 or box_tensor.shape[0] != count or box_tensor.shape[1] < 7:
        raise ValueError("box tensor must have shape (N, 7+) matching scores")

    detections: list[Detection3D] = []
    for index in range(count):
        class_id = int(labels[index])
        if class_id < 0 or class_id >= len(class_names):
            raise ValueError(f"prediction label {class_id} is outside the upstream class taxonomy")
        velocity = (
            (float(box_tensor[index, 7]), float(box_tensor[index, 8]))
            if box_tensor.shape[1] >= 9
            else None
        )
        detections.append(
            Detection3D(
                center_xyz=(
                    float(centers[index, 0]),
                    float(centers[index, 1]),
                    float(centers[index, 2]),
                ),
                size_lwh=(
                    float(dimensions[index, 0]),
                    float(dimensions[index, 1]),
                    float(dimensions[index, 2]),
                ),
                yaw_rad=float(yaws[index]),
                score=float(scores[index]),
                class_id=class_id,
                class_name=str(class_names[class_id]),
                velocity_xy=velocity,
            )
        )

    frame_metadata = {
        "raw_detection_count": count,
        "box_center": "geometric_center",
        "box_dimension_order": "length_width_height",
        "axes": "x_forward_y_left_z_up",
        "yaw": "counter_clockwise_from_positive_x_radians",
        **({} if metadata is None else dict(metadata)),
    }
    return DetectionFrame(
        detections=tuple(detections),
        sample_id=sample_id,
        coordinate_frame="nuscenes_lidar_top",
        metadata=frame_metadata,
    )


class Mmdet3dBackend:
    """Pinned M1 backend using official MMDetection3D model and data pipelines."""

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        *,
        checkpoint_sha256: str,
        device: str = "cuda:0",
        enforce_versions: bool = True,
    ) -> None:
        self.config_path = Path(config_path).expanduser()
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        self.expected_checkpoint_sha256 = checkpoint_sha256.lower()
        self.device = device
        self.enforce_versions = enforce_versions
        self._runtime: _Mmdet3dRuntime | None = None
        self._model: Any = None
        self._datasets: dict[tuple[Path, str], Any] = {}

    @property
    def initialized(self) -> bool:
        return self._model is not None

    @property
    def versions(self) -> Mapping[str, str]:
        if self._runtime is None:
            raise RuntimeError("backend is not initialized")
        return self._runtime.versions

    def initialize(self) -> None:
        """Validate the environment and load the official model in FP32 eval mode."""

        if self.initialized:
            return
        if not self.config_path.is_file():
            raise FileNotFoundError("official MMDetection3D config file was not found")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError("official PointPillars checkpoint was not found")
        actual_checksum = sha256_file(self.checkpoint_path)
        if actual_checksum != self.expected_checkpoint_sha256:
            raise DetectionEnvironmentError(
                "PointPillars checkpoint SHA256 mismatch; delete the cached asset and rerun setup"
            )

        runtime = _load_mmdet3d_runtime()
        if self.enforce_versions and runtime.versions != EXPECTED_M1_VERSIONS:
            raise DetectionEnvironmentError(
                f"M1 requires {EXPECTED_M1_VERSIONS}, but found {runtime.versions}. "
                "Rerun scripts/setup_detection_m1.sh in the isolated environment."
            )
        if self.device != "cuda:0":
            raise ValueError("M1 supports only device 'cuda:0'")
        if not bool(runtime.torch.cuda.is_available()):
            raise DetectionEnvironmentError("PyTorch cannot access CUDA in the M1 environment")

        try:
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                model = runtime.init_model(
                    str(self.config_path), str(self.checkpoint_path), device=self.device
                )
        except Exception as error:
            message = str(error)
            for path in (self.config_path, self.checkpoint_path):
                message = message.replace(str(path), f"<{path.name}>")
            raise DetectionEnvironmentError(
                f"MMDetection3D failed to initialize the pinned PointPillars model: {message}"
            ) from error
        model.eval()
        self._runtime = runtime
        self._model = model

    def _dataset(self, data_root: str | Path, split: str) -> Any:
        self.initialize()
        if split not in {"mini_train", "mini_val"}:
            raise ValueError("split must be 'mini_train' or 'mini_val'")
        root = Path(data_root).expanduser().resolve()
        info_name = (
            "nuscenes_infos_train.pkl" if split == "mini_train" else "nuscenes_infos_val.pkl"
        )
        if not (root / "v1.0-mini").is_dir():
            raise FileNotFoundError("nuScenes root is missing the v1.0-mini metadata directory")
        if not (root / info_name).is_file():
            raise FileNotFoundError(
                f"prepared metadata {info_name} is missing; run prepare_nuscenes_mini.py"
            )
        key = (root, split)
        if key not in self._datasets:
            assert self._runtime is not None
            dataset_config = deepcopy(self._model.cfg.test_dataloader.dataset)
            dataset_config.data_root = f"{root}{os.sep}"
            dataset_config.ann_file = info_name
            dataset_config.test_mode = True
            dataset_config.lazy_init = False
            self._runtime.init_default_scope("mmdet3d")
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self._datasets[key] = self._runtime.datasets.build(dataset_config)
        return self._datasets[key]

    def dataset_size(self, data_root: str | Path, split: str = "mini_val") -> int:
        """Return the actual prepared split size."""

        return int(len(self._dataset(data_root, split)))

    def prepare_sample(
        self, data_root: str | Path, *, split: str = "mini_val", index: int = 0
    ) -> PreparedMmdet3dSample:
        """Run the official multi-sweep test pipeline for one dataset sample."""

        dataset = self._dataset(data_root, split)
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer")
        if not 0 <= index < len(dataset):
            raise IndexError(f"sample index {index} is outside split size {len(dataset)}")
        item = dataset[index]
        if not isinstance(item, dict) or "inputs" not in item or "points" not in item["inputs"]:
            raise RuntimeError("official nuScenes test pipeline did not produce point inputs")
        info = dataset.get_data_info(index)
        sample_id = str(info.get("token") or info.get("sample_idx") or index)
        points = _to_numpy(item["inputs"]["points"], name="prepared points")
        if points.ndim != 2 or points.shape[1] < 3:
            raise RuntimeError("official nuScenes pipeline returned malformed point inputs")
        assert self._runtime is not None
        batch = self._runtime.pseudo_collate([item])
        return PreparedMmdet3dSample(
            batch=batch,
            points_xyz=points[:, :3],
            sample_id=sample_id,
            sample_index=index,
            split=split,
        )

    def run_prepared(self, sample: PreparedMmdet3dSample) -> DetectionFrame:
        """Run one prepared sample in explicit FP32 and convert every raw result."""

        self.initialize()
        assert self._runtime is not None
        with (
            self._runtime.torch.inference_mode(),
            self._runtime.torch.autocast(device_type="cuda", enabled=False),
        ):
            predictions = self._model.test_step(sample.batch)
        if len(predictions) != 1:
            raise RuntimeError("M1 backend expected exactly one prediction")
        classes = tuple(str(name) for name in self._model.dataset_meta["classes"])
        return convert_mmdet3d_prediction(
            predictions[0],
            class_names=classes,
            sample_id=sample.sample_id,
            metadata={
                "sample_index": sample.sample_index,
                "split": sample.split,
                "precision": "fp32",
                "backend": "mmdetection3d",
                "backend_version": self._runtime.versions["mmdet3d"],
                "checkpoint_sha256": self.expected_checkpoint_sha256,
            },
        )
