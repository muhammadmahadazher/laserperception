from pathlib import Path

from laserperception.detection.m1_assets import resolve_m1_asset_paths


def _manifest() -> dict[str, object]:
    return {
        "cache": {
            "root_environment_variable": "LASERPERCEPTION_M1_CACHE",
            "default_root": "~/.cache/laserperception",
            "mmdet3d_checkout_relative": "mmdetection3d-v1.4.0",
            "checkpoint_directory_relative": "checkpoints",
        },
        "model": {
            "checkpoint": {
                "filename": "pointpillars.pth",
            }
        },
    }


def test_m1_assets_use_default_home_cache(monkeypatch) -> None:
    monkeypatch.delenv("LASERPERCEPTION_M1_CACHE", raising=False)

    paths = resolve_m1_asset_paths(_manifest())

    expected_root = (Path.home() / ".cache" / "laserperception").resolve()
    assert paths.cache_root == expected_root
    assert paths.mmdet3d_root == expected_root / "mmdetection3d-v1.4.0"
    assert paths.checkpoint_directory == expected_root / "checkpoints"
    assert paths.checkpoint_path == expected_root / "checkpoints" / "pointpillars.pth"


def test_m1_assets_use_environment_override(monkeypatch, tmp_path: Path) -> None:
    configured_root = tmp_path / "shared-m1-cache"
    monkeypatch.setenv("LASERPERCEPTION_M1_CACHE", str(configured_root))

    paths = resolve_m1_asset_paths(_manifest())

    expected_root = configured_root.resolve()
    assert paths.cache_root == expected_root
    assert paths.mmdet3d_root == expected_root / "mmdetection3d-v1.4.0"
    assert paths.checkpoint_directory == expected_root / "checkpoints"
    assert paths.checkpoint_path == expected_root / "checkpoints" / "pointpillars.pth"
