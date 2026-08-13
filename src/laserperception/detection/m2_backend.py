"""Shared official voxelization and postprocessing for M2 backend parity."""

from __future__ import annotations

import hashlib
import importlib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from laserperception.detection.exact_voxelization import ExactDeterministicVoxelizer
from laserperception.detection.m2_diagnostics import (
    RAW_OUTPUT_NAMES,
    assert_cuda0_model,
    assert_cuda0_tensor,
    assert_raw_outputs_cuda0,
)
from laserperception.detection.mmdet3d_backend import (
    DetectionEnvironmentError,
    Mmdet3dBackend,
    PreparedMmdet3dSample,
)
from laserperception.detection.types import DetectionFrame

EXPECTED_MMDEPLOY_VERSION = "1.3.1"
EXPECTED_RAW_OUTPUT_SHAPES = {
    "cls_score": (1, 140, 200, 200),
    "bbox_pred": (1, 126, 200, 200),
    "dir_cls_pred": (1, 28, 200, 200),
}
ProvenanceMode = Literal["full", "live"]
VoxelizationMode = Literal["official", "exact_fast"]


def validate_provenance_mode(value: str) -> ProvenanceMode:
    """Validate the explicit full-fidelity or lightweight live metadata policy."""

    if value not in {"full", "live"}:
        raise ValueError("provenance_mode must be full or live")
    return cast(ProvenanceMode, value)


def validate_voxelization_mode(value: str) -> VoxelizationMode:
    """Validate the official-reference or exact deployment voxelization policy."""

    if value not in {"official", "exact_fast"}:
        raise ValueError("voxelization_mode must be official or exact_fast")
    return cast(VoxelizationMode, value)


@dataclass(frozen=True, slots=True)
class VoxelizedM2Sample:
    """Official MMDetection3D voxel tensors shared by both M2 runtimes."""

    prepared: PreparedMmdet3dSample
    voxels: Any
    num_points: Any
    coors: Any
    data_samples: tuple[Any, ...]

    @property
    def voxel_count(self) -> int:
        return int(self.voxels.shape[0])

    @property
    def shapes(self) -> dict[str, list[int]]:
        return {
            "voxels": [int(value) for value in self.voxels.shape],
            "num_points": [int(value) for value in self.num_points.shape],
            "coors": [int(value) for value in self.coors.shape],
        }

    def hashes(self) -> dict[str, str]:
        """Return SHA256 values for exact intermediate-array identity checks."""

        return {
            "voxels": _tensor_sha256(self.voxels),
            "num_points": _tensor_sha256(self.num_points),
            "coors": _tensor_sha256(self.coors),
        }

    def tensor_statistics(self) -> dict[str, dict[str, object]]:
        """Return sanitized shapes, dtypes, and finite numeric ranges."""

        return {
            name: _tensor_statistics(tensor)
            for name, tensor in (
                ("voxels", self.voxels),
                ("num_points", self.num_points),
                ("coors", self.coors),
            )
        }


class M2Backend(Mmdet3dBackend):
    """M2 adapter retaining one official voxelization and postprocess path."""

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        deploy_config_path: str | Path,
        *,
        checkpoint_sha256: str,
        device: str = "cuda:0",
        voxelization_mode: str = "official",
    ) -> None:
        super().__init__(
            config_path,
            checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            device=device,
        )
        self.deploy_config_path = Path(deploy_config_path).expanduser()
        self.voxelization_mode: VoxelizationMode = validate_voxelization_mode(voxelization_mode)
        self._deploy_config: Any = None
        self._task_processor: Any = None
        self._backend_models: dict[Path, Any] = {}
        self._exact_voxelizer: ExactDeterministicVoxelizer | None = None

    def initialize(self) -> None:
        """Load the pinned M1 model plus the pinned MMDeploy rewrite registry."""

        if self._task_processor is not None:
            return
        super().initialize()
        if not self.deploy_config_path.is_file():
            raise FileNotFoundError("official MMDeploy deployment config was not found")
        try:
            mmdeploy = importlib.import_module("mmdeploy")
            build_task_processor = importlib.import_module("mmdeploy.apis").build_task_processor
            config_type = importlib.import_module("mmengine.config").Config
        except (AttributeError, ImportError, OSError) as error:
            raise DetectionEnvironmentError(
                "The pinned MMDeploy 1.3.1 environment is unavailable. Run "
                "scripts/setup_detection_m2.sh and retry."
            ) from error
        if str(mmdeploy.__version__) != EXPECTED_MMDEPLOY_VERSION:
            raise DetectionEnvironmentError(
                f"M2 requires MMDeploy {EXPECTED_MMDEPLOY_VERSION}, found {mmdeploy.__version__}"
            )
        self._deploy_config = config_type.fromfile(str(self.deploy_config_path))
        self._task_processor = build_task_processor(
            self._model.cfg, self._deploy_config, self.device
        )
        if self.voxelization_mode == "exact_fast":
            try:
                official_layer = self._model.data_preprocessor.voxel_layer
                if bool(official_layer.deterministic) is not True:
                    raise RuntimeError("the pinned official voxel layer is not deterministic")
                self._exact_voxelizer = ExactDeterministicVoxelizer(official_layer).eval()
            except Exception as error:
                raise DetectionEnvironmentError(
                    "exact_fast voxelization initialization failed; startup is refused "
                    "and no fallback voxelizer will be selected"
                ) from error

    @property
    def deploy_config(self) -> Any:
        self.initialize()
        return self._deploy_config

    @property
    def official_nms_pre(self) -> int:
        """Return the pinned model's official pre-NMS top-anchor count."""

        self.initialize()
        test_config = getattr(self._model, "test_cfg", None)
        if test_config is None:
            model_config = self._model.cfg.model
            test_config = model_config.get("test_cfg")
        if test_config is None:
            raise RuntimeError("pinned PointPillars model does not expose test_cfg")
        points_config = test_config.get("pts") if hasattr(test_config, "get") else None
        test_config = points_config if points_config is not None else test_config
        value = (
            test_config.get("nms_pre")
            if hasattr(test_config, "get")
            else getattr(test_config, "nms_pre", None)
        )
        if isinstance(value, bool) or not isinstance(value, int) or value == 0:
            raise RuntimeError("pinned PointPillars test_cfg has an invalid nms_pre value")
        return value

    def voxelize(self, sample: PreparedMmdet3dSample) -> VoxelizedM2Sample:
        """Apply the configured fail-closed voxelization policy exactly once."""

        if self.voxelization_mode == "official":
            return self.voxelize_official(sample)
        if self.voxelization_mode == "exact_fast":
            return self._voxelize_exact_fast(sample)
        raise AssertionError(f"unsupported voxelization mode: {self.voxelization_mode}")

    def voxelize_official(self, sample: PreparedMmdet3dSample) -> VoxelizedM2Sample:
        """Apply official MMDetection3D preprocessing for reference and M2 evidence."""

        self.initialize()
        assert self._runtime is not None
        with (
            self._runtime.torch.inference_mode(),
            self._runtime.torch.autocast(device_type="cuda", enabled=False),
        ):
            processed = self._model.data_preprocessor(sample.batch, False)
        try:
            voxel_data = processed["inputs"]["voxels"]
            voxels = voxel_data["voxels"]
            num_points = voxel_data["num_points"]
            coors = voxel_data["coors"]
            data_samples = tuple(processed["data_samples"])
        except (KeyError, TypeError) as error:
            raise RuntimeError(
                "official MMDetection3D preprocessing did not return voxel inputs"
            ) from error
        if tuple(voxels.shape[1:]) != (64, 4):
            raise RuntimeError(f"M2 requires voxel shape (N, 64, 4), found {tuple(voxels.shape)}")
        if tuple(num_points.shape) != (int(voxels.shape[0]),):
            raise RuntimeError("official num_points shape does not match the voxel count")
        if tuple(coors.shape) != (int(voxels.shape[0]), 4):
            raise RuntimeError("official coors shape does not match the voxel count")
        if len(data_samples) != 1:
            raise RuntimeError("M2 requires exactly one data sample")
        return VoxelizedM2Sample(
            prepared=sample,
            voxels=voxels.contiguous(),
            num_points=num_points.contiguous(),
            coors=coors.contiguous(),
            data_samples=data_samples,
        )

    def _voxelize_exact_fast(self, sample: PreparedMmdet3dSample) -> VoxelizedM2Sample:
        """Run the validated LaserPerception exact deterministic deployment path."""

        self.initialize()
        if self._exact_voxelizer is None:
            raise RuntimeError(
                "exact_fast voxelization was requested but did not initialize; no fallback allowed"
            )
        assert self._runtime is not None
        preprocessor = self._model.data_preprocessor
        with (
            self._runtime.torch.inference_mode(),
            self._runtime.torch.autocast(device_type="cuda", enabled=False),
        ):
            collated = preprocessor.collate_data(sample.batch)
            try:
                points = list(collated["inputs"]["points"])
                data_samples = tuple(collated["data_samples"])
            except (KeyError, TypeError) as error:
                raise RuntimeError(
                    "MMDetection3D collate_data returned malformed point data"
                ) from error
            if len(points) != 1 or len(data_samples) != 1:
                raise RuntimeError("exact_fast requires batch size one")
            voxels, coors, num_points = self._exact_voxelizer(points[0])
            centers = (coors[:, [2, 1, 0]] + 0.5) * voxels.new_tensor(
                self._exact_voxelizer.voxel_size
            ) + voxels.new_tensor(self._exact_voxelizer.point_cloud_range[0:3])
            padded_coors = self._runtime.torch.nn.functional.pad(
                coors, (1, 0), mode="constant", value=0
            )
            result = VoxelizedM2Sample(
                prepared=sample,
                voxels=self._runtime.torch.cat([voxels], dim=0).contiguous(),
                num_points=self._runtime.torch.cat([num_points], dim=0).contiguous(),
                coors=self._runtime.torch.cat([padded_coors], dim=0).contiguous(),
                data_samples=data_samples,
            )
            self._runtime.torch.cat([centers], dim=0).contiguous()
        if tuple(result.voxels.shape[1:]) != (64, 4):
            raise RuntimeError(
                f"exact_fast requires voxel shape (N, 64, 4), found {tuple(result.voxels.shape)}"
            )
        if tuple(result.num_points.shape) != (result.voxel_count,):
            raise RuntimeError("exact_fast num_points shape does not match voxel count")
        if tuple(result.coors.shape) != (result.voxel_count, 4):
            raise RuntimeError("exact_fast coors shape does not match voxel count")
        self.assert_shared_cuda_inputs(result)
        return result

    def run_rewritten_pytorch_raw(self, sample: VoxelizedM2Sample) -> dict[str, list[Any]]:
        """Run the MMDeploy-rewritten PointPillars network in PyTorch FP32."""

        self.initialize()
        assert self._runtime is not None
        self.assert_shared_cuda_inputs(sample)
        assert_cuda0_model(self._model, name="rewritten_pytorch_model")
        rewriter_context = importlib.import_module("mmdeploy.core").RewriterContext
        with (
            self._runtime.torch.inference_mode(),
            self._runtime.torch.autocast(device_type="cuda", enabled=False),
            rewriter_context(cfg=self._deploy_config, backend="tensorrt"),
        ):
            raw = self._model(
                sample.voxels,
                sample.num_points,
                sample.coors,
                data_samples=list(sample.data_samples),
            )
        if not isinstance(raw, tuple) or len(raw) != 3:
            raise RuntimeError("rewritten PointPillars must return exactly three output tensors")
        result = {
            "cls_score": [raw[0]],
            "bbox_pred": [raw[1]],
            "dir_cls_pred": [raw[2]],
        }
        assert_raw_outputs_cuda0(
            result,
            runtime_name="rewritten_pytorch",
            expected_dtype="torch.float32",
            expected_shapes=EXPECTED_RAW_OUTPUT_SHAPES,
        )
        return result

    def run_native_pytorch_raw(self, sample: VoxelizedM2Sample) -> dict[str, list[Any]]:
        """Run native MMDetection3D PointPillars modules on shared voxels in FP32."""

        self.initialize()
        assert self._runtime is not None
        self.assert_shared_cuda_inputs(sample)
        assert_cuda0_model(self._model, name="native_pytorch_model")
        batch_input_metas = [data_sample.metainfo for data_sample in sample.data_samples]
        with (
            self._runtime.torch.inference_mode(),
            self._runtime.torch.autocast(device_type="cuda", enabled=False),
        ):
            _, point_features = self._model.extract_feat(self._inputs(sample), batch_input_metas)
            raw = self._model.pts_bbox_head(point_features)
        if not isinstance(raw, tuple) or len(raw) != 3:
            raise RuntimeError("native PointPillars bbox head must return exactly three outputs")
        result: dict[str, list[Any]] = {}
        for name, levels in zip(RAW_OUTPUT_NAMES, raw, strict=True):
            if not isinstance(levels, (list, tuple)) or len(levels) != 1:
                raise RuntimeError(f"native PointPillars {name} must contain one feature level")
            result[name] = [levels[0]]
        assert_raw_outputs_cuda0(
            result,
            runtime_name="native_pytorch",
            expected_dtype="torch.float32",
            expected_shapes=EXPECTED_RAW_OUTPUT_SHAPES,
        )
        return result

    def run_tensorrt_raw(
        self, sample: VoxelizedM2Sample, engine_path: str | Path
    ) -> dict[str, list[Any]]:
        """Run one external TensorRT engine through the official MMDeploy wrapper."""

        self.assert_shared_cuda_inputs(sample)
        backend_model = self._backend_model(engine_path)
        backend_device = str(getattr(backend_model, "device", ""))
        if backend_device != "cuda:0":
            found = backend_device or "unknown"
            raise RuntimeError(f"TensorRT backend must execute on cuda:0, found {found}")
        raw = backend_model.forward(self._inputs(sample), data_samples=None)
        if not isinstance(raw, dict):
            raise RuntimeError("official MMDeploy TensorRT wrapper returned malformed outputs")
        assert_raw_outputs_cuda0(
            raw,
            runtime_name="tensorrt",
            expected_dtype="torch.float32",
            expected_shapes=EXPECTED_RAW_OUTPUT_SHAPES,
        )
        return raw

    def assert_shared_cuda_inputs(self, sample: VoxelizedM2Sample) -> dict[str, object]:
        """Fail closed and record the device, dtype, and shape of shared inputs."""

        return {
            "model_parameter": assert_cuda0_model(self._model, name="m2_model"),
            "voxels": assert_cuda0_tensor(
                sample.voxels,
                name="voxels",
                expected_dtype="torch.float32",
            ),
            "num_points": assert_cuda0_tensor(sample.num_points, name="num_points"),
            "coors": assert_cuda0_tensor(sample.coors, name="coors"),
        }

    def run_official_postprocess_raw(
        self, raw: dict[str, list[Any]], sample: VoxelizedM2Sample
    ) -> Any:
        """Run the existing static MMDeploy postprocess, including head construction."""

        self.initialize()
        voxel_model = importlib.import_module(
            "mmdeploy.codebase.mmdet3d.deploy.voxel_detection_model"
        ).VoxelDetectionModel
        predictions = voxel_model.postprocess(
            model_cfg=self._model.cfg,
            deploy_cfg=self._deploy_config,
            outs=raw,
            metas=deepcopy(list(sample.data_samples)),
        )
        if len(predictions) != 1:
            raise RuntimeError("official MMDeploy postprocess must return exactly one prediction")
        return predictions[0]

    def convert_postprocessed_prediction(
        self,
        prediction: Any,
        sample: VoxelizedM2Sample,
        *,
        backend_name: str,
        precision: str,
        provenance_mode: ProvenanceMode = "full",
    ) -> DetectionFrame:
        """Convert one official prediction to the framework-independent output contract."""

        frame = self.convert_prediction(prediction, sample.prepared)
        return self.attach_runtime_metadata(
            frame,
            sample,
            backend_name=backend_name,
            precision=precision,
            provenance_mode=provenance_mode,
        )

    @staticmethod
    def attach_runtime_metadata(
        frame: DetectionFrame,
        sample: VoxelizedM2Sample,
        *,
        backend_name: str,
        precision: str,
        provenance_mode: ProvenanceMode = "full",
    ) -> DetectionFrame:
        """Attach runtime metadata under an explicit provenance-cost policy."""

        mode = validate_provenance_mode(provenance_mode)
        provenance: dict[str, object]
        if mode == "full":
            provenance = {"shared_voxel_hashes": sample.hashes()}
        else:
            provenance = {
                "voxel_provenance_mode": "live",
                "voxel_provenance_scope": "lightweight_semantic_metadata_only",
                "shared_voxel_hashes_omitted": True,
                "shared_voxel_shapes": sample.shapes,
            }
        return DetectionFrame(
            detections=frame.detections,
            sample_id=frame.sample_id,
            coordinate_frame=frame.coordinate_frame,
            metadata={
                **frame.metadata,
                "backend": backend_name,
                "precision": precision,
                "voxel_count": sample.voxel_count,
                **provenance,
            },
        )

    def postprocess_raw(
        self,
        raw: dict[str, list[Any]],
        sample: VoxelizedM2Sample,
        *,
        backend_name: str,
        precision: str,
        provenance_mode: ProvenanceMode = "full",
    ) -> DetectionFrame:
        """Apply the same official MMDeploy postprocess and LaserPerception conversion."""

        prediction = self.run_official_postprocess_raw(raw, sample)
        return self.convert_postprocessed_prediction(
            prediction,
            sample,
            backend_name=backend_name,
            precision=precision,
            provenance_mode=provenance_mode,
        )

    def run_native_pytorch(self, sample: VoxelizedM2Sample) -> DetectionFrame:
        """Run native PyTorch FP32 and the common official postprocess."""

        return self.postprocess_raw(
            self.run_native_pytorch_raw(sample),
            sample,
            backend_name="native_mmdetection3d_pytorch",
            precision="fp32",
        )

    def run_rewritten_pytorch(self, sample: VoxelizedM2Sample) -> DetectionFrame:
        """Run rewritten PyTorch FP32 and the common official postprocess."""

        return self.postprocess_raw(
            self.run_rewritten_pytorch_raw(sample),
            sample,
            backend_name="mmdeploy_rewritten_pytorch",
            precision="fp32",
        )

    def run_tensorrt(
        self,
        sample: VoxelizedM2Sample,
        engine_path: str | Path,
        *,
        provenance_mode: ProvenanceMode = "full",
    ) -> DetectionFrame:
        """Run TensorRT FP16 and the common official postprocess."""

        return self.postprocess_raw(
            self.run_tensorrt_raw(sample, engine_path),
            sample,
            backend_name="tensorrt",
            precision="fp16",
            provenance_mode=provenance_mode,
        )

    @staticmethod
    def _inputs(sample: VoxelizedM2Sample) -> dict[str, dict[str, Any]]:
        return {
            "voxels": {
                "voxels": sample.voxels,
                "num_points": sample.num_points,
                "coors": sample.coors,
            }
        }

    def _backend_model(self, engine_path: str | Path) -> Any:
        self.initialize()
        path = Path(engine_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError("TensorRT engine was not found in the external M2 cache")
        if path not in self._backend_models:
            self._backend_models[path] = self._task_processor.build_backend_model([str(path)])
        return self._backend_models[path]


def _tensor_sha256(tensor: Any) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _tensor_statistics(tensor: Any) -> dict[str, object]:
    detached = tensor.detach()
    result: dict[str, object] = {
        "shape": [int(value) for value in detached.shape],
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "sha256": _tensor_sha256(detached),
    }
    if int(detached.numel()) > 0:
        numeric = detached.float()
        result.update(
            {
                "minimum": float(numeric.min().item()),
                "maximum": float(numeric.max().item()),
                "mean": float(numeric.mean().item()),
            }
        )
    return result
