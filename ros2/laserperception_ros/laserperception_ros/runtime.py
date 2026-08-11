"""One-time construction of the frozen M2 runtime for ROS 2 M3A."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory

from laserperception.detection.m1_assets import resolve_m1_asset_paths
from laserperception.detection.m2_assets import resolve_m2_asset_paths
from laserperception.detection.m2_backend import M2Backend
from laserperception.detection.mmdet3d_backend import sha256_file
from laserperception.detection.ros2_contract import ModelReadyPointCloud
from laserperception.detection.types import DetectionFrame

EXPECTED_CHECKPOINT_SHA256 = "f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0"
EXPECTED_ONNX_SHA256 = "61ce22a8ca31498675c32576bfb94f0093d31dc95d2762f7254bf915a59ecc16"
EXPECTED_ENGINE_SHA256 = "a005f75852097cd9b193750560b214cc3d5237ae9b6c106c7fca3d4fc348714b"


@dataclass(frozen=True, slots=True)
class M3Assets:
    """Frozen manifests and external M2 paths used by M3."""

    m1_manifest: dict[str, object]
    m2_manifest: dict[str, object]
    config_path: Path
    checkpoint_path: Path
    deploy_config_path: Path
    onnx_path: Path
    engine_path: Path


def resolve_m3_assets(*, engine_override: str = "") -> M3Assets:
    """Resolve package-installed manifests and external cache artifacts."""

    share = Path(get_package_share_directory("laserperception_ros"))
    m1_manifest = dict(
        yaml.safe_load((share / "config/detection/m1_pointpillars_nuscenes.yaml").read_text())
    )
    m2_manifest = dict(
        yaml.safe_load((share / "config/detection/m2_pointpillars_tensorrt.yaml").read_text())
    )
    m1_assets = resolve_m1_asset_paths(m1_manifest)
    m2_assets = resolve_m2_asset_paths(m2_manifest)
    engine_path = (
        Path(engine_override).expanduser().resolve()
        if engine_override.strip()
        else m2_assets.engine_directory / "pointpillars_fp16.engine"
    )
    return M3Assets(
        m1_manifest=m1_manifest,
        m2_manifest=m2_manifest,
        config_path=m1_assets.mmdet3d_root / str(m1_manifest["model"]["upstream_config"]),
        checkpoint_path=m1_assets.checkpoint_path,
        deploy_config_path=m2_assets.mmdeploy_root
        / str(m2_manifest["deployment"]["official_deployment_config"]),
        onnx_path=m2_assets.artifact_directory / "pointpillars.onnx",
        engine_path=engine_path,
    )


def create_backend(assets: M3Assets) -> M2Backend:
    """Create the shared official backend without modifying model semantics."""

    checkpoint_sha = str(assets.m1_manifest["model"]["checkpoint"]["sha256"])
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("package M1 manifest does not identify the frozen M3 checkpoint")
    if str(assets.m2_manifest["artifacts"]["onnx"]["sha256"]) != EXPECTED_ONNX_SHA256:
        raise RuntimeError("package M2 manifest does not identify the frozen M3 ONNX artifact")
    if str(assets.m2_manifest["artifacts"]["engine"]["sha256"]) != EXPECTED_ENGINE_SHA256:
        raise RuntimeError("package M2 manifest does not identify the frozen M3 engine")
    return M2Backend(
        assets.config_path,
        assets.checkpoint_path,
        assets.deploy_config_path,
        checkpoint_sha256=checkpoint_sha,
    )


class M3DetectorRuntime:
    """Initialized-once in-memory PointCloud2-to-DetectionFrame runtime."""

    def __init__(self, *, engine_override: str = "") -> None:
        self.assets = resolve_m3_assets(engine_override=engine_override)
        if not self.assets.engine_path.is_file():
            raise FileNotFoundError("frozen TensorRT engine is missing from the external M2 cache")
        actual_engine_sha = sha256_file(self.assets.engine_path)
        if actual_engine_sha != EXPECTED_ENGINE_SHA256:
            raise RuntimeError(
                f"TensorRT engine SHA256 mismatch: expected {EXPECTED_ENGINE_SHA256}, "
                f"found {actual_engine_sha}"
            )
        self.backend = create_backend(self.assets)
        self.backend.initialize()
        # Build and retain the official wrapper/engine/context before the first callback.
        self.backend._backend_model(self.assets.engine_path)
        self.engine_sha256 = actual_engine_sha

    def infer(
        self,
        points: ModelReadyPointCloud,
        *,
        sample_id: str,
        coordinate_frame: str,
    ) -> DetectionFrame:
        """Run official voxelization, frozen TensorRT FP16, and shared postprocessing."""

        prepared = self.backend.prepare_model_ready_points(
            points,
            sample_id=sample_id,
            coordinate_frame=coordinate_frame,
        )
        voxelized = self.backend.voxelize(prepared)
        return self.backend.run_tensorrt(voxelized, self.assets.engine_path)
