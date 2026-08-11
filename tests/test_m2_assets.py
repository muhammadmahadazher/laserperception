from pathlib import Path

from laserperception.detection.m2_assets import resolve_m2_asset_paths


def _manifest() -> dict[str, object]:
    return {
        "cache": {
            "root_environment_variable": "LASERPERCEPTION_M2_CACHE",
            "default_root": "~/.cache/laserperception",
            "mmdeploy_checkout_relative": "mmdeploy-v1.3.1",
            "artifact_directory_relative": "m2",
            "engine_directory_relative": "engines",
        }
    }


def test_m2_assets_use_default_home_cache(monkeypatch) -> None:
    monkeypatch.delenv("LASERPERCEPTION_M2_CACHE", raising=False)

    paths = resolve_m2_asset_paths(_manifest())

    expected_root = (Path.home() / ".cache" / "laserperception").resolve()
    assert paths.cache_root == expected_root
    assert paths.mmdeploy_root == expected_root / "mmdeploy-v1.3.1"
    assert paths.artifact_directory == expected_root / "m2"
    assert paths.engine_directory == expected_root / "m2" / "engines"


def test_m2_assets_use_environment_override(monkeypatch, tmp_path: Path) -> None:
    configured_root = tmp_path / "shared-m2-cache"
    monkeypatch.setenv("LASERPERCEPTION_M2_CACHE", str(configured_root))

    paths = resolve_m2_asset_paths(_manifest())

    expected_root = configured_root.resolve()
    assert paths.cache_root == expected_root
    assert paths.mmdeploy_root == expected_root / "mmdeploy-v1.3.1"
    assert paths.artifact_directory == expected_root / "m2"
    assert paths.engine_directory == expected_root / "m2" / "engines"
