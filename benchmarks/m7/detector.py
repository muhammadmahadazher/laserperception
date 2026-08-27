"""Canonical M7 adapter for the unchanged, artifact-bound M6b detector runtime."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from benchmarks.m7.execution import DetectorObservation, ExecutionIdentity, RuntimeArtifacts
from benchmarks.m7.protocol import (
    CHECKPOINT_SHA256,
    ENGINE_SHA256,
    EVALUATOR_IDENTITY,
    ONNX_SHA256,
    PROTOCOL_FREEZE_COMMIT,
    ProtocolViolation,
)
from laserperception.detection.m1_assets import resolve_m1_asset_paths
from laserperception.detection.m2_assets import resolve_m2_asset_paths
from laserperception.detection.m2_backend import EXPECTED_MMDEPLOY_VERSION, M2Backend
from laserperception.detection.mmdet3d_backend import sha256_file

M6B_CONFIG_RELATIVE = "configs/m6/kitti_m6b.yaml"
M1_MANIFEST_RELATIVE = "configs/detection/m1_pointpillars_nuscenes.yaml"
M6B_DEPLOYMENT_MANIFEST_RELATIVE = "configs/detection/m6_pointpillars_tensorrt_40k.yaml"
M6B_MODEL_CONFIG_RELATIVE = "configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py"
M6B_DEPLOY_CONFIG_RELATIVE = (
    "configs/mmdet3d/voxel-detection/voxel-detection_tensorrt_dynamic-nus-64x4.py"
)
M6B_MODEL_CONFIG_SHA256 = "1ffe085179a48b1bf47e15c12674fe0d58d518cb117ce7a2b7fa10dbbdbd4db1"
M6B_DEPLOY_CONFIG_SHA256 = "bdf6e1def90ddc3b3e89b8a958e2e4f8f2e33e01668cdfeba6b050d7afc99751"
M6B_MODEL_REPOSITORY_COMMIT = "fe25f7a51d36e3702f961e198894580d83c4387b"
M6B_DEPLOY_REPOSITORY_COMMIT = "bc75c9d6c8940aa03d0e1e5b5962bd930478ba77"
M6B_BACKEND_CLASS = "laserperception.detection.m2_backend.M2Backend"
M6B_DEVICE = "cuda:0"
M6B_VOXELIZATION_MODE = "exact_fast"
M6B_PROVENANCE_MODE = "full"
M6B_COORDINATE_FRAME = "kitti_model_aligned_lidar"
M6B_PRECISION = "fp16"
M6B_EVALUATION_SCORE_THRESHOLD = 0.25
RAW_OUTPUT_NAMES = ("cls_score", "bbox_pred", "dir_cls_pred")


class _IdentityBearingDetector(Protocol):
    @property
    def identity(self) -> CanonicalM7DetectorIdentity:
        """Return the immutable identity of the detector actually constructed."""


@dataclass(frozen=True, slots=True)
class CanonicalM7DetectorIdentity:
    """Read-only paths and public identities for the constructed M6b runtime."""

    engine_path: Path
    checkpoint_path: Path
    onnx_path: Path
    engine_sha256: str
    checkpoint_sha256: str
    onnx_sha256: str
    onnx_role: str
    voxelization_mode: str
    m6b_config_path: str
    m6b_config_sha256: str
    model_manifest_path: str
    model_manifest_sha256: str
    model_config_path: str
    model_config_sha256: str
    model_repository_commit: str
    deployment_manifest_path: str
    deployment_manifest_sha256: str
    deploy_config_path: str
    deploy_config_sha256: str
    deploy_repository_commit: str
    backend_class: str
    device: str
    expected_mmdeploy_version: str
    provenance_mode: str
    precision: str
    coordinate_frame: str
    evaluation_score_threshold: float

    def to_public_dict(self) -> dict[str, object]:
        """Return sanitized audit metadata without external/private filesystem paths."""

        return {
            "schema_version": "laserperception.m7.canonical-detector-identity.v1",
            "engine_sha256": self.engine_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "onnx_sha256": self.onnx_sha256,
            "onnx_role": self.onnx_role,
            "voxelization_mode": self.voxelization_mode,
            "m6b_config_path": self.m6b_config_path,
            "m6b_config_sha256": self.m6b_config_sha256,
            "model_manifest_path": self.model_manifest_path,
            "model_manifest_sha256": self.model_manifest_sha256,
            "model_config_path": self.model_config_path,
            "model_config_sha256": self.model_config_sha256,
            "model_repository_commit": self.model_repository_commit,
            "deployment_manifest_path": self.deployment_manifest_path,
            "deployment_manifest_sha256": self.deployment_manifest_sha256,
            "deploy_config_path": self.deploy_config_path,
            "deploy_config_sha256": self.deploy_config_sha256,
            "deploy_repository_commit": self.deploy_repository_commit,
            "backend_class": self.backend_class,
            "device": self.device,
            "expected_mmdeploy_version": self.expected_mmdeploy_version,
            "provenance_mode": self.provenance_mode,
            "precision": self.precision,
            "coordinate_frame": self.coordinate_frame,
            "postprocess_score_filter": None,
            "evaluation_score_threshold": self.evaluation_score_threshold,
        }


class CanonicalM7Detector:
    """Artifact-bound wrapper around the accepted M6b ``M2Backend`` path."""

    __slots__ = ("_backend", "_engine_path", "_identity")

    def __init__(
        self,
        backend: M2Backend,
        engine_path: Path,
        identity: CanonicalM7DetectorIdentity,
    ) -> None:
        self._backend = backend
        self._engine_path = engine_path
        self._identity = identity
        self._require_runtime_contract()

    @property
    def identity(self) -> CanonicalM7DetectorIdentity:
        """Return the immutable identity for the runtime actually held by this adapter."""

        return self._identity

    @classmethod
    def from_verified_artifacts(
        cls,
        artifacts: RuntimeArtifacts,
        expected: ExecutionIdentity,
        *,
        repository_root: str | Path | None = None,
    ) -> CanonicalM7Detector:
        """Construct M6b's detector from the exact files already verified by M7.

        The checkpoint and engine paths passed to ``M2Backend``/MMDeploy are the same
        resolved paths held by ``RuntimeArtifacts``. ONNX is verified and recorded as
        provenance-only because the accepted TensorRT runtime does not consume it.
        """

        _require_frozen_execution_identity(expected)
        artifacts.verify_runtime(expected)
        root = (
            Path(repository_root).expanduser().resolve()
            if repository_root is not None
            else Path(__file__).resolve().parents[2]
        )
        m6b_config_path = root / M6B_CONFIG_RELATIVE
        m1_manifest_path = root / M1_MANIFEST_RELATIVE
        deployment_manifest_path = root / M6B_DEPLOYMENT_MANIFEST_RELATIVE
        m6b_config = _load_yaml_mapping(m6b_config_path)
        m1_manifest = _load_yaml_mapping(m1_manifest_path)
        deployment_manifest = _load_yaml_mapping(deployment_manifest_path)
        _require_frozen_manifests(m6b_config, m1_manifest, deployment_manifest)

        m1_assets = resolve_m1_asset_paths(m1_manifest)
        m2_assets = resolve_m2_asset_paths(deployment_manifest)
        model_config_path = (m1_assets.mmdet3d_root / M6B_MODEL_CONFIG_RELATIVE).resolve()
        deploy_config_path = (m2_assets.mmdeploy_root / M6B_DEPLOY_CONFIG_RELATIVE).resolve()
        for name, path in (
            ("frozen MMDetection3D model config", model_config_path),
            ("frozen MMDeploy deployment config", deploy_config_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"M7 {name} is missing: {path}")
        model_config_sha256 = sha256_file(model_config_path)
        if model_config_sha256 != M6B_MODEL_CONFIG_SHA256:
            raise ProtocolViolation("M7 frozen MMDetection3D model config SHA256 differs")
        deploy_config_sha256 = sha256_file(deploy_config_path)
        if deploy_config_sha256 != M6B_DEPLOY_CONFIG_SHA256:
            raise ProtocolViolation("M7 frozen MMDeploy deployment config SHA256 differs")

        engine_path = artifacts.engine.expanduser().resolve()
        checkpoint_path = artifacts.checkpoint.expanduser().resolve()
        onnx_path = artifacts.onnx.expanduser().resolve()
        identity = CanonicalM7DetectorIdentity(
            engine_path=engine_path,
            checkpoint_path=checkpoint_path,
            onnx_path=onnx_path,
            engine_sha256=expected.engine_sha256,
            checkpoint_sha256=expected.checkpoint_sha256,
            onnx_sha256=expected.onnx_sha256,
            onnx_role="verified_provenance_only_not_runtime_consumed",
            voxelization_mode=M6B_VOXELIZATION_MODE,
            m6b_config_path=M6B_CONFIG_RELATIVE,
            m6b_config_sha256=sha256_file(m6b_config_path),
            model_manifest_path=M1_MANIFEST_RELATIVE,
            model_manifest_sha256=sha256_file(m1_manifest_path),
            model_config_path=M6B_MODEL_CONFIG_RELATIVE,
            model_config_sha256=model_config_sha256,
            model_repository_commit=_required_nested_string(m1_manifest, "backend", "commit"),
            deployment_manifest_path=M6B_DEPLOYMENT_MANIFEST_RELATIVE,
            deployment_manifest_sha256=sha256_file(deployment_manifest_path),
            deploy_config_path=M6B_DEPLOY_CONFIG_RELATIVE,
            deploy_config_sha256=deploy_config_sha256,
            deploy_repository_commit=_required_nested_string(
                deployment_manifest, "deployment", "exporter_commit"
            ),
            backend_class=M6B_BACKEND_CLASS,
            device=M6B_DEVICE,
            expected_mmdeploy_version=EXPECTED_MMDEPLOY_VERSION,
            provenance_mode=M6B_PROVENANCE_MODE,
            precision=M6B_PRECISION,
            coordinate_frame=M6B_COORDINATE_FRAME,
            evaluation_score_threshold=M6B_EVALUATION_SCORE_THRESHOLD,
        )
        backend = M2Backend(
            model_config_path,
            checkpoint_path,
            deploy_config_path,
            checkpoint_sha256=expected.checkpoint_sha256,
            device=M6B_DEVICE,
            voxelization_mode=M6B_VOXELIZATION_MODE,
        )
        backend.initialize()
        backend_model = backend._backend_model(engine_path)
        if str(getattr(backend_model, "device", "")) != M6B_DEVICE:
            raise ProtocolViolation("M7 TensorRT backend model did not initialize on cuda:0")
        detector = cls(backend, engine_path, identity)
        require_detector_runtime_identity(detector, artifacts, expected)
        return detector

    def _require_runtime_contract(self) -> None:
        if self._backend.__class__ is not M2Backend:
            raise ProtocolViolation("M7 canonical detector backend class differs from M2Backend")
        if self._backend.device != M6B_DEVICE:
            raise ProtocolViolation("M7 canonical detector device must be cuda:0")
        if self._backend.voxelization_mode != M6B_VOXELIZATION_MODE:
            raise ProtocolViolation("M7 canonical detector voxelization must be exact_fast")
        if self._backend.checkpoint_path.expanduser().resolve() != self._identity.checkpoint_path:
            raise ProtocolViolation(
                "M7 canonical detector checkpoint path is not the verified path"
            )
        if (
            self._backend.config_path.expanduser()
            .resolve()
            .as_posix()
            .endswith(M6B_MODEL_CONFIG_RELATIVE)
            is False
        ):
            raise ProtocolViolation("M7 canonical detector uses the wrong model config")
        if (
            self._backend.deploy_config_path.expanduser()
            .resolve()
            .as_posix()
            .endswith(M6B_DEPLOY_CONFIG_RELATIVE)
            is False
        ):
            raise ProtocolViolation("M7 canonical detector uses the wrong deploy config")
        if self._engine_path != self._identity.engine_path:
            raise ProtocolViolation("M7 canonical detector engine path is not the verified path")

    def infer(self, points: np.ndarray, *, condition_id: str) -> DetectorObservation:
        """Execute M6b's unchanged prepare/voxelize/TensorRT/postprocess chain.

        ``M7CorpusRunner`` hashes and passes the same read-only NumPy array here. The
        accepted ``M2Backend.prepare_model_ready_points`` boundary necessarily copies
        those values into its validated model-ready wrapper and then a PyTorch tensor.
        """

        if points.flags.writeable:
            raise ProtocolViolation("M7 canonical detector input must remain read-only")
        self._require_runtime_contract()
        prepared = self._backend.prepare_model_ready_points(
            points,
            sample_id=condition_id,
            coordinate_frame=M6B_COORDINATE_FRAME,
        )
        voxelized = self._backend.voxelize(prepared)
        shared_cuda_inputs = self._backend.assert_shared_cuda_inputs(voxelized)
        raw = self._backend.run_tensorrt_raw(voxelized, self._engine_path)
        frame = self._backend.postprocess_raw(
            raw,
            voxelized,
            backend_name="tensorrt",
            precision=M6B_PRECISION,
            provenance_mode=M6B_PROVENANCE_MODE,
        )
        return DetectorObservation(
            raw_outputs=_raw_arrays(raw),
            detection_frame=frame.to_dict(),
            payload={
                "schema_version": "laserperception.m7.detector-observation.v1",
                "condition_id": condition_id,
                "point_count": int(points.shape[0]),
                "voxel_count": voxelized.voxel_count,
                "voxel_hashes": voxelized.hashes(),
                "shared_cuda_inputs": shared_cuda_inputs,
                "detector_identity": self._identity.to_public_dict(),
                "detection_frame": frame.to_dict(),
            },
        )


def require_detector_runtime_identity(
    detector: _IdentityBearingDetector,
    artifacts: RuntimeArtifacts,
    expected: ExecutionIdentity,
) -> None:
    """Reject a private test substitution whose claimed runtime differs before inference."""

    identity = detector.identity
    comparisons = {
        "engine path": (
            identity.engine_path,
            artifacts.engine.expanduser().resolve(),
        ),
        "checkpoint path": (
            identity.checkpoint_path,
            artifacts.checkpoint.expanduser().resolve(),
        ),
        "ONNX path": (identity.onnx_path, artifacts.onnx.expanduser().resolve()),
        "engine SHA256": (identity.engine_sha256, expected.engine_sha256),
        "checkpoint SHA256": (identity.checkpoint_sha256, expected.checkpoint_sha256),
        "ONNX SHA256": (identity.onnx_sha256, expected.onnx_sha256),
        "ONNX role": (identity.onnx_role, "verified_provenance_only_not_runtime_consumed"),
        "voxelization mode": (identity.voxelization_mode, M6B_VOXELIZATION_MODE),
        "M6b config": (identity.m6b_config_path, M6B_CONFIG_RELATIVE),
        "model manifest": (identity.model_manifest_path, M1_MANIFEST_RELATIVE),
        "model config": (identity.model_config_path, M6B_MODEL_CONFIG_RELATIVE),
        "model config SHA256": (identity.model_config_sha256, M6B_MODEL_CONFIG_SHA256),
        "model repository commit": (
            identity.model_repository_commit,
            M6B_MODEL_REPOSITORY_COMMIT,
        ),
        "deployment manifest": (
            identity.deployment_manifest_path,
            M6B_DEPLOYMENT_MANIFEST_RELATIVE,
        ),
        "deploy config": (identity.deploy_config_path, M6B_DEPLOY_CONFIG_RELATIVE),
        "deploy config SHA256": (identity.deploy_config_sha256, M6B_DEPLOY_CONFIG_SHA256),
        "deploy repository commit": (
            identity.deploy_repository_commit,
            M6B_DEPLOY_REPOSITORY_COMMIT,
        ),
        "backend class": (identity.backend_class, M6B_BACKEND_CLASS),
        "device": (identity.device, M6B_DEVICE),
        "MMDeploy version": (identity.expected_mmdeploy_version, EXPECTED_MMDEPLOY_VERSION),
        "provenance mode": (identity.provenance_mode, M6B_PROVENANCE_MODE),
        "precision": (identity.precision, M6B_PRECISION),
        "coordinate frame": (identity.coordinate_frame, M6B_COORDINATE_FRAME),
        "evaluation score threshold": (
            identity.evaluation_score_threshold,
            M6B_EVALUATION_SCORE_THRESHOLD,
        ),
    }
    for name, (actual, frozen) in comparisons.items():
        if actual != frozen:
            raise ProtocolViolation(f"M7 detector runtime identity mismatch: {name}")
    for name, value in (
        ("M6b config SHA256", identity.m6b_config_sha256),
        ("model manifest SHA256", identity.model_manifest_sha256),
        ("model config SHA256", identity.model_config_sha256),
        ("deployment manifest SHA256", identity.deployment_manifest_sha256),
        ("deploy config SHA256", identity.deploy_config_sha256),
    ):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ProtocolViolation(f"M7 detector runtime identity is malformed: {name}")


def build_canonical_m7_detector(
    artifacts: RuntimeArtifacts,
    expected: ExecutionIdentity,
) -> CanonicalM7Detector:
    """Internal production builder; no public caller-selected runtime is accepted."""

    return CanonicalM7Detector.from_verified_artifacts(artifacts, expected)


def _raw_arrays(raw: Mapping[str, list[Any]]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    if set(raw) != set(RAW_OUTPUT_NAMES):
        raise ProtocolViolation("M7 TensorRT outputs do not match the frozen tensor names")
    for name in RAW_OUTPUT_NAMES:
        values = raw[name]
        if len(values) != 1:
            raise ProtocolViolation(f"M7 raw output {name} must contain one feature level")
        result[name] = values[0].detach().cpu().contiguous().numpy()
    return result


def _require_frozen_execution_identity(expected: ExecutionIdentity) -> None:
    frozen = {
        "engine SHA256": (expected.engine_sha256, ENGINE_SHA256),
        "checkpoint SHA256": (expected.checkpoint_sha256, CHECKPOINT_SHA256),
        "ONNX SHA256": (expected.onnx_sha256, ONNX_SHA256),
        "evaluator identity": (expected.evaluator_identity, EVALUATOR_IDENTITY),
        "protocol commit": (expected.protocol_commit, PROTOCOL_FREEZE_COMMIT),
    }
    for name, (actual, required) in frozen.items():
        if actual != required:
            raise ProtocolViolation(f"M7 canonical detector requires the frozen {name}")


def _load_yaml_mapping(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"M7 frozen configuration is missing: {path}")
    try:
        yaml = importlib.import_module("yaml")
    except (AttributeError, ImportError) as error:
        raise RuntimeError("PyYAML is required by the pinned M6b detector environment") from error
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ProtocolViolation(f"M7 frozen configuration is not a mapping: {path.name}")
    return value


def _required_nested_mapping(parent: Mapping[str, object], *keys: str) -> Mapping[str, object]:
    current = parent
    for key in keys:
        value = current.get(key)
        if not isinstance(value, Mapping):
            raise ProtocolViolation(f"M7 frozen configuration field is malformed: {'.'.join(keys)}")
        current = value
    return current


def _required_nested_string(parent: Mapping[str, object], *keys: str) -> str:
    current = _required_nested_mapping(parent, *keys[:-1]) if len(keys) > 1 else parent
    value = current.get(keys[-1])
    if not isinstance(value, str) or not value:
        raise ProtocolViolation(f"M7 frozen configuration field is malformed: {'.'.join(keys)}")
    return value


def _require_frozen_manifests(
    m6b: Mapping[str, object],
    m1: Mapping[str, object],
    deployment: Mapping[str, object],
) -> None:
    frozen_detector = _required_nested_mapping(m6b, "frozen_detector")
    expected_fields: tuple[tuple[str, object], ...] = (
        ("source_manifest", M6B_DEPLOYMENT_MANIFEST_RELATIVE),
        ("checkpoint_sha256", CHECKPOINT_SHA256),
        ("onnx_sha256", ONNX_SHA256),
        ("tensorrt_engine_sha256", ENGINE_SHA256),
        ("precision", M6B_PRECISION),
        ("device", M6B_DEVICE),
        ("voxelization_mode", M6B_VOXELIZATION_MODE),
        ("provenance_mode", M6B_PROVENANCE_MODE),
        ("score_threshold", M6B_EVALUATION_SCORE_THRESHOLD),
    )
    for name, expected in expected_fields:
        if frozen_detector.get(name) != expected:
            raise ProtocolViolation(f"M7 frozen M6b detector field differs: {name}")
    if _required_nested_string(m1, "model", "upstream_config") != M6B_MODEL_CONFIG_RELATIVE:
        raise ProtocolViolation("M7 M1 manifest selects a different PointPillars config")
    if _required_nested_string(m1, "backend", "commit") != M6B_MODEL_REPOSITORY_COMMIT:
        raise ProtocolViolation("M7 M1 manifest selects a different MMDetection3D commit")
    if _required_nested_string(m1, "model", "checkpoint", "sha256") != CHECKPOINT_SHA256:
        raise ProtocolViolation("M7 M1 manifest selects a different checkpoint")
    if (
        _required_nested_string(deployment, "deployment", "official_deployment_config")
        != M6B_DEPLOY_CONFIG_RELATIVE
    ):
        raise ProtocolViolation("M7 deployment manifest selects a different MMDeploy config")
    if (
        _required_nested_string(deployment, "deployment", "exporter_commit")
        != M6B_DEPLOY_REPOSITORY_COMMIT
    ):
        raise ProtocolViolation("M7 deployment manifest selects a different MMDeploy commit")
    if _required_nested_string(deployment, "deployment", "device") != M6B_DEVICE:
        raise ProtocolViolation("M7 deployment manifest selects a different device")
    if _required_nested_string(deployment, "deployment", "precision") != M6B_PRECISION:
        raise ProtocolViolation("M7 deployment manifest selects a different precision")
    if (
        _required_nested_string(deployment, "source_model", "checkpoint_sha256")
        != CHECKPOINT_SHA256
    ):
        raise ProtocolViolation("M7 deployment manifest selects a different checkpoint")
    if _required_nested_string(deployment, "artifacts", "onnx", "sha256") != ONNX_SHA256:
        raise ProtocolViolation("M7 deployment manifest selects a different ONNX artifact")
    if _required_nested_string(deployment, "artifacts", "engine", "sha256") != ENGINE_SHA256:
        raise ProtocolViolation("M7 deployment manifest selects a different TensorRT engine")
