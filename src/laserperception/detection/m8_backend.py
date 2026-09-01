"""Lazy, identity-bound OpenPCDet/DSVT backend for M8 Phase 1.

Importing this module never imports Torch, CUDA, spconv, or OpenPCDet.  The
heavy runtime is loaded only when :class:`DsvtBackend` is initialized in the
separate pinned M8 environment.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from laserperception.detection.m8_capacity import (
    candidate_dynamic_pillar_count_cuda,
    load_dsvt_capacity_contract,
)
from laserperception.detection.m8_input import M8_FEATURE_NAMES, M8PointCloud
from laserperception.detection.types import Detection3D, DetectionFrame

M8_CLASS_NAMES = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)
M8_PRIMARY_CLASS_MAPPING = {"car": "car", "pedestrian": "pedestrian"}
M8_SCIENTIFIC_SCORE_THRESHOLD = 0.25


def load_m8_candidate_manifest(path: str | Path) -> Mapping[str, object]:
    """Load and minimally validate the tracked candidate identity manifest."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("M8 candidate manifest must be a JSON object")
    if payload.get("schema_version") != "1.0":
        raise ValueError("unsupported M8 candidate manifest schema")
    if payload.get("architecture") != "DSVT-Pillar with TransFusion head":
        raise ValueError("M8 candidate architecture identity mismatch")
    return cast(Mapping[str, object], payload)


def map_m8_class_to_primary(class_name: str) -> str | None:
    """Map only the prospectively frozen M6 primary classes."""

    return M8_PRIMARY_CLASS_MAPPING.get(class_name)


def dsvt_predictions_to_detection_frame(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    sample_id: str,
    metadata: Mapping[str, object] | None = None,
) -> DetectionFrame:
    """Convert native DSVT/OpenPCDet lidar boxes into ``DetectionFrame``.

    Native boxes are geometric-centre ``[x,y,z,dx,dy,dz,yaw,vx,vy]`` in a
    right-handed lidar frame (X forward, Y left, Z up). ``dx,dy,dz`` already
    equal LaserPerception length, width, height and heading is CCW from +X.
    Labels are OpenPCDet's one-based class identifiers.
    """

    native_boxes = np.asarray(boxes)
    native_scores = np.asarray(scores)
    native_labels = np.asarray(labels)
    if native_boxes.ndim != 2 or native_boxes.shape[1] not in (7, 9):
        raise ValueError("DSVT boxes must have shape (N, 7) or (N, 9)")
    count = native_boxes.shape[0]
    if native_scores.shape != (count,) or native_labels.shape != (count,):
        raise ValueError("DSVT scores and labels must have shape (N,)")
    if not np.isfinite(native_boxes).all() or not np.isfinite(native_scores).all():
        raise ValueError("DSVT predictions must contain only finite values")
    if np.any((native_scores < 0.0) | (native_scores > 1.0)):
        raise ValueError("DSVT scores must be in [0, 1]")
    if np.any(native_boxes[:, 3:6] <= 0.0):
        raise ValueError("DSVT box dimensions must be positive")
    integer_labels = native_labels.astype(np.int64)
    if not np.array_equal(native_labels, integer_labels):
        raise ValueError("DSVT labels must be integral")
    if np.any((integer_labels < 1) | (integer_labels > len(M8_CLASS_NAMES))):
        raise ValueError("DSVT label is outside the frozen class table")

    detections = []
    for box, score, one_based_label in zip(
        native_boxes, native_scores, integer_labels, strict=True
    ):
        class_id = int(one_based_label) - 1
        detections.append(
            Detection3D(
                center_xyz=(float(box[0]), float(box[1]), float(box[2])),
                size_lwh=(float(box[3]), float(box[4]), float(box[5])),
                yaw_rad=float(box[6]),
                score=float(score),
                class_id=class_id,
                class_name=M8_CLASS_NAMES[class_id],
                velocity_xy=(float(box[7]), float(box[8])) if native_boxes.shape[1] == 9 else None,
            )
        )
    return DetectionFrame(
        detections=tuple(detections),
        sample_id=sample_id,
        coordinate_frame="lidar",
        metadata={} if metadata is None else metadata,
    )


class DsvtBackend:
    """Pinned DSVT-Pillar inference with no caller-supplied model factory."""

    def __init__(
        self,
        *,
        manifest_path: str | Path,
        upstream_root: str | Path,
        checkpoint_path: str | Path,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.upstream_root = Path(upstream_root).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.manifest = load_m8_candidate_manifest(self.manifest_path)
        self.capacity_contract = load_dsvt_capacity_contract(self.manifest)
        _validate_timestamp_contract(self.manifest)
        self._model: Any = None
        self._torch: Any = None
        self._cfg: Any = None
        self._identity: dict[str, object] = {}
        self._initialize()

    @classmethod
    def from_environment(cls, *, manifest_path: str | Path) -> DsvtBackend:
        """Initialize from explicit external-root variables, failing closed."""

        manifest = load_m8_candidate_manifest(manifest_path)
        environment = _required_mapping(manifest, "environment")
        root_variable = _required_string(environment, "upstream_root_variable")
        checkpoint_variable = _required_string(environment, "checkpoint_variable")
        root = os.environ.get(root_variable)
        checkpoint = os.environ.get(checkpoint_variable)
        if not root or not checkpoint:
            raise RuntimeError(
                f"set {root_variable} and {checkpoint_variable} to the pinned external assets"
            )
        return cls(
            manifest_path=manifest_path,
            upstream_root=root,
            checkpoint_path=checkpoint,
        )

    @property
    def identity(self) -> Mapping[str, object]:
        """Return runtime identities established during fail-closed loading."""

        return dict(self._identity)

    def infer(self, points_xyzit: M8PointCloud | np.ndarray, *, sample_id: str) -> DetectionFrame:
        """Run batch-one FP32 inference and return framework-independent detections."""

        points = (
            points_xyzit.points
            if isinstance(points_xyzit, M8PointCloud)
            else M8PointCloud(points_xyzit).points
        )
        boxes, scores, labels, dropped = self._predict_arrays(points)
        return dsvt_predictions_to_detection_frame(
            boxes,
            scores,
            labels,
            sample_id=sample_id,
            metadata={
                "backend": "dsvt_pillar_nuscenes",
                "feature_columns": list(M8_FEATURE_NAMES),
                "candidate_range_dropped_points": dropped,
                "identity": self.identity,
            },
        )

    def _predict_arrays(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        batch, dropped = self._prepare_batch(points)
        with self._torch.inference_mode():
            predictions, _ = self._model(batch)
            self._torch.cuda.synchronize(0)
        prediction = predictions[0]
        for name in ("pred_boxes", "pred_scores", "pred_labels"):
            value = prediction[name]
            if value.device != self._torch.device("cuda:0"):
                raise RuntimeError(f"DSVT {name} is not on cuda:0")
        return (
            prediction["pred_boxes"].detach().cpu().contiguous().numpy(),
            prediction["pred_scores"].detach().cpu().contiguous().numpy(),
            prediction["pred_labels"].detach().cpu().contiguous().numpy(),
            dropped,
        )

    def run_structural_smoke(self, points: np.ndarray) -> tuple[int, int]:
        """Run the model while retaining only output count and capacity status.

        This engineering-only entry point deliberately does not transfer or
        expose prediction values. It exists for owner-authorized structural
        capacity checks, not scientific detector evaluation.
        """

        batch, dropped = self._prepare_batch(points)
        with self._torch.inference_mode():
            predictions, _ = self._model(batch)
            self._torch.cuda.synchronize(0)
        prediction = predictions[0]
        expected = {"pred_boxes", "pred_scores", "pred_labels"}
        if not expected.issubset(prediction):
            raise RuntimeError("DSVT structural output contract is incomplete")
        count = int(prediction["pred_boxes"].shape[0])
        if prediction["pred_scores"].shape != (count,) or prediction["pred_labels"].shape != (
            count,
        ):
            raise RuntimeError("DSVT structural output shapes are inconsistent")
        del prediction, predictions
        return count, dropped

    def run_gt_blind_timing_call(self, points: np.ndarray) -> None:
        """Complete one engineering timing call and immediately discard all outputs.

        The method intentionally returns no prediction count, boxes, scores,
        labels, or DetectionFrame.  It is the only backend boundary used by
        the separately authorized M8 S1 GT-blind sizing preflight.
        """

        batch, _ = self._prepare_batch(points)
        with self._torch.inference_mode():
            predictions, auxiliary = self._model(batch)
            self._torch.cuda.synchronize(0)
        del auxiliary, predictions, batch

    def run_gt_blind_capacity_call(self, points: np.ndarray) -> int:
        """Run the unchanged GT-blind path while observing only retained pillar count."""

        retained_pillars: list[int] = []

        def observe_vfe_output(_module: object, _inputs: object, output: object) -> None:
            if not isinstance(output, dict) or "voxel_coords" not in output:
                raise RuntimeError("DSVT VFE output contract is unavailable for capacity review")
            coordinates = output["voxel_coords"]
            retained_pillars.append(int(coordinates.shape[0]))

        handle = self._model.vfe.register_forward_hook(observe_vfe_output)
        try:
            self.run_gt_blind_timing_call(points)
        finally:
            handle.remove()
        if len(retained_pillars) != 1:
            raise RuntimeError("DSVT capacity review did not observe exactly one VFE execution")
        return retained_pillars[0]

    def synchronize(self) -> None:
        """Synchronize CUDA device 0 for an external wall-clock boundary."""

        self._torch.cuda.synchronize(0)

    def reset_cuda_peak_memory_stats(self) -> None:
        """Reset CUDA peak counters immediately before an engineering capacity call."""

        self.synchronize()
        self._torch.cuda.reset_peak_memory_stats(0)

    def cuda_memory_state(self) -> Mapping[str, int]:
        """Capture allocated, reserved, allocator-peak, and driver-free CUDA memory."""

        self.synchronize()
        torch = self._torch
        free_bytes, mem_get_info_total_bytes = torch.cuda.mem_get_info(0)
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated(0)),
            "reserved_bytes": int(torch.cuda.memory_reserved(0)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            "mem_get_info_free_bytes": int(free_bytes),
            "mem_get_info_total_bytes": int(mem_get_info_total_bytes),
            "device_total_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        }

    def candidate_pillar_count(self, points: np.ndarray) -> int:
        """Return the selected candidate's exact CUDA input-only pillar count."""

        return candidate_dynamic_pillar_count_cuda(
            points,
            torch_module=self._torch,
            device="cuda:0",
        )

    def runtime_state(self) -> Mapping[str, object]:
        """Capture the accepted P1-E policy and current runtime state without changing it."""

        import random

        torch = self._torch

        def state_sha256(value: object) -> str:
            return hashlib.sha256(repr(value).encode()).hexdigest()

        try:
            gpu_uuid = subprocess.run(
                [
                    "nvidia-smi",
                    "--id=0",
                    "--query-gpu=uuid,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            gpu_uuid = "unavailable"
        relevant_environment: dict[str, str | None] = {
            name: value
            for name, value in sorted(os.environ.items())
            if name.startswith(("CUDA", "CUBLAS", "CUDNN", "NVIDIA", "PYTORCH", "TORCH"))
        }
        relevant_environment["PYTORCH_CUDA_ALLOC_CONF"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        return {
            "python_exact_version": sys.version,
            "pytorch_exact_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "nvidia_driver_and_gpu_uuid_query": gpu_uuid,
            "gpu_name": torch.cuda.get_device_name(0),
            "spconv": self._identity["spconv"],
            "torch_scatter": self._identity["torch_scatter"],
            "numpy": np.__version__,
            "candidate_eval_train_state": "eval" if not self._model.training else "train",
            "inference_mode_enabled_at_capture": torch.is_inference_mode_enabled(),
            "grad_enabled_at_capture": torch.is_grad_enabled(),
            "python_random_policy": "not reseeded by LaserPerception S1 runtime",
            "python_random_state_sha256": state_sha256(random.getstate()),
            "numpy_random_policy": "not reseeded by LaserPerception S1 runtime",
            "numpy_random_state_sha256": state_sha256(np.random.get_state()),
            "torch_random_policy": "not reseeded by LaserPerception S1 runtime",
            "torch_cpu_initial_seed": int(torch.initial_seed()),
            "torch_cpu_rng_state_sha256": hashlib.sha256(
                torch.get_rng_state().cpu().numpy().tobytes()
            ).hexdigest(),
            "torch_cuda_initial_seed": int(torch.cuda.initial_seed()),
            "torch_cuda_rng_state_sha256": hashlib.sha256(
                torch.cuda.get_rng_state(0).cpu().numpy().tobytes()
            ).hexdigest(),
            "tf32": {
                "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            },
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "relevant_environment": relevant_environment,
            "point_order_policy": "preserve frozen source-row order; no random inference shuffle",
            "model_config_checkpoint_identities": dict(self._identity),
            "cuda_memory": dict(self.cuda_memory_state()),
        }

    def _prepare_batch(self, points: np.ndarray) -> tuple[dict[str, object], int]:
        point_range = np.asarray(self.capacity_contract.point_cloud_range, dtype=np.float32)
        inside = np.all(points[:, :3] >= point_range[:3], axis=1) & np.all(
            points[:, :3] < point_range[3:], axis=1
        )
        selected = np.ascontiguousarray(points[inside])
        dropped = int(points.shape[0] - selected.shape[0])
        if selected.shape[0] == 0:
            raise ValueError("candidate range removed every input point")
        batch_column = np.zeros((selected.shape[0], 1), dtype=np.float32)
        candidate_points = np.concatenate((batch_column, selected), axis=1)
        tensor = self._torch.from_numpy(candidate_points).to(device="cuda:0")
        if tensor.dtype != self._torch.float32 or tensor.device != self._torch.device("cuda:0"):
            raise RuntimeError("DSVT input did not materialize as CUDA FP32 on device 0")
        return {"batch_size": 1, "points": tensor, "frame_id": ["m8"]}, dropped

    def _initialize(self) -> None:
        if not (self.upstream_root / ".git").is_dir():
            raise RuntimeError("LASERPERCEPTION_M8_DSVT_ROOT is not a Git checkout")
        upstream = _required_mapping(self.manifest, "upstream")
        checkpoint = _required_mapping(self.manifest, "checkpoint")
        expected_commit = _required_string(upstream, "commit")
        actual_commit = subprocess.run(
            ["git", "-C", str(self.upstream_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if actual_commit != expected_commit:
            raise RuntimeError(f"DSVT checkout is {actual_commit}, expected {expected_commit}")
        config_path = self.upstream_root / _required_string(upstream, "config_relative_path")
        _require_file_identity(config_path, _required_string(upstream, "config_sha256"))
        _require_file_identity(
            self.checkpoint_path,
            _required_string(checkpoint, "sha256"),
            expected_bytes=_required_integer(checkpoint, "bytes"),
        )

        if str(self.upstream_root) not in sys.path:
            sys.path.insert(0, str(self.upstream_root))
        try:
            torch = importlib.import_module("torch")
            spconv = importlib.import_module("spconv")
            torch_scatter = importlib.import_module("torch_scatter")
            config_module = importlib.import_module("pcdet.config")
            models_module = importlib.import_module("pcdet.models")
        except ImportError as error:
            raise RuntimeError(
                "pinned M8 Torch/CUDA/spconv/DSVT environment is unavailable"
            ) from error
        runtime = _required_mapping(self.manifest, "runtime")
        if torch.__version__ != _required_string(runtime, "torch"):
            raise RuntimeError("Torch version does not match the M8 manifest")
        if torch.version.cuda != _required_string(runtime, "cuda"):
            raise RuntimeError("Torch CUDA version does not match the M8 manifest")
        if spconv.__version__ != _required_string(runtime, "spconv"):
            raise RuntimeError("spconv version does not match the M8 manifest")
        if torch_scatter.__version__ != _required_string(runtime, "torch_scatter"):
            raise RuntimeError("torch-scatter version does not match the M8 manifest")
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("M8 DSVT requires CUDA device 0")

        cfg = config_module.cfg
        with _working_directory(config_path.parent.parent.parent):
            config_module.cfg_from_yaml_file(str(config_path), cfg)
        dataset = SimpleNamespace(
            class_names=list(M8_CLASS_NAMES),
            point_feature_encoder=SimpleNamespace(num_point_features=5),
            grid_size=np.asarray(self.capacity_contract.grid_size, dtype=np.int64),
            point_cloud_range=np.asarray(
                self.capacity_contract.point_cloud_range, dtype=np.float32
            ),
            voxel_size=np.asarray(self.capacity_contract.voxel_size, dtype=np.float32),
            depth_downsample_factor=None,
        )
        model = models_module.build_network(
            model_cfg=cfg.MODEL,
            num_class=len(M8_CLASS_NAMES),
            dataset=dataset,
        )
        logger = logging.getLogger("laserperception.m8.dsvt")
        model.load_params_from_file(filename=str(self.checkpoint_path), logger=logger, to_cpu=True)
        model.cuda(0).eval()
        if next(model.parameters()).device != torch.device("cuda:0"):
            raise RuntimeError("DSVT parameters are not on cuda:0")
        self._model = model
        self._torch = torch
        self._cfg = cfg
        self._identity = {
            "architecture": self.manifest["architecture"],
            "upstream_commit": actual_commit,
            "config_sha256": _required_string(upstream, "config_sha256"),
            "checkpoint_sha256": _required_string(checkpoint, "sha256"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "spconv": spconv.__version__,
            "torch_scatter": torch_scatter.__version__,
            "device": str(next(model.parameters()).device),
            "class_mapping": dict(M8_PRIMARY_CLASS_MAPPING),
            "coordinate_adapter": "openpcdet_lidar_box_direct",
            "feature_contract": list(M8_FEATURE_NAMES),
        }


def _require_file_identity(path: Path, sha256: str, *, expected_bytes: int | None = None) -> None:
    if not path.is_file():
        raise RuntimeError(f"required M8 artifact is missing: {path}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise RuntimeError(f"M8 artifact size mismatch: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise RuntimeError(f"M8 artifact SHA256 mismatch: {path}")


def _required_mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"M8 manifest {key} must be a mapping")
    return value


def _required_string(parent: Mapping[str, object], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"M8 manifest {key} must be a non-empty string")
    return value


def _required_integer(parent: Mapping[str, object], key: str) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"M8 manifest {key} must be an integer")
    return value


def _validate_timestamp_contract(manifest: Mapping[str, object]) -> None:
    timestamp = _required_mapping(manifest, "timestamp_semantics")
    expected: dict[str, object] = {
        "unit": "seconds",
        "definition": "current_timestamp_seconds - historical_timestamp_seconds",
        "current_value": 0.0,
        "current_zero_sign": "positive",
        "older_history_sign": "positive",
        "reference_frame": "current/reference lidar acquisition",
        "dtype": "float32",
        "m6_m7_relation": (
            "semantically aligned current-minus-historical elapsed-seconds convention"
        ),
    }
    for key, value in expected.items():
        if timestamp.get(key) != value:
            raise ValueError(f"M8 manifest timestamp_semantics.{key} mismatch")


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
