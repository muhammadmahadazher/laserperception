from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_v0_3_release_metadata_is_consistent() -> None:
    pyproject = _text("pyproject.toml")
    assert 'version = "0.3.0"' in pyproject
    description = (
        'description = "Reproducible 3D LiDAR detection, raw ROS 2 ingestion, '
        'and TensorRT deployment."'
    )
    assert description in pyproject

    citation = _text("CITATION.cff")
    assert "version: 0.3.0" in citation
    assert "date-released:" not in citation
    assert "time-aware raw PointCloud2 multi-sweep ingestion" in citation
    assert "semantic segmentation" not in citation.lower()
    assert "No software release" + " exists yet" not in citation

    assert '__version__ = "0.3.0"' in _text("src/laserperception/__init__.py")
    assert 'version="0.3.0"' in _text("ros2/laserperception_ros/setup.py")
    assert "<version>0.3.0</version>" in _text("ros2/laserperception_ros/package.xml")

    changelog = _text("CHANGELOG.md")
    assert "## [Unreleased]" in changelog
    unreleased, current_release = changelog.split("## [0.3.0] - Release date pending", maxsplit=1)
    assert "M6" not in unreleased
    assert "856 frozen H10/H5 conditions" in current_release
    assert "## [0.2.0] - 2026-08-20" in current_release

    release_notes = _text("docs/releases/v0.3.0.md")
    quickstart = _text("docs/QUICKSTART_V0_2.md")
    assert "LaserPerception v0.3.0 release notes" in release_notes
    assert "release-preparation candidate, not yet released" in release_notes
    assert "860/860 exact" in release_notes
    assert "LaserPerception v0.2.0 quickstart" in quickstart
    assert "LaserPerception v0.2.0 release notes" in _text("docs/releases/v0.2.0.md")
    assert "# LaserPerception v0.1.0 release notes" in _text("docs/releases/v0.1.0.md")
    assert "# LaserPerception v0.1.0 quickstart" in _text("docs/QUICKSTART_V0_1.md")


def test_release_story_keeps_performance_and_parity_roles_distinct() -> None:
    agents = _text("AGENTS.md")
    assert "M2 parity reference: MMDeploy-rewritten PyTorch FP32" in agents
    assert "M2 performance baseline: native MMDetection3D PyTorch FP32" in agents
    assert "One coding implementer works on a feature or release branch at a time" in agents

    readme = _text("README.md")
    release_notes = _text("docs/releases/v0.1.0.md")
    for document in (readme, release_notes):
        assert "10 Hz was the highest tested clean sustained" in document
        assert "15 Hz" in document and "20 Hz" in document
        assert "not portable hardware capability guarantees" in document
        assert "hard voxel layer" in document
        assert "not ROS callback or loopback latency" in document

    assert "PR #4 " + "remains" not in readme
    assert "M3 complete" + " — final review" not in readme
    assert "0.1.0" + ".dev0" not in readme


def test_release_demo_wrapper_is_validation_only() -> None:
    wrapper = _text("scripts/run_v0_1_demo.sh")
    validator = _text("scripts/detection/check_v0_1_assets.py")

    assert "check_v0_1_assets.py" in wrapper
    assert "m3_demo.launch.py" in wrapper
    assert "torch.ones" in wrapper
    assert "curl " not in wrapper
    assert "wget " not in wrapper
    assert "build_m2_tensorrt.py" not in wrapper

    assert "exact_fast" in validator and "live" in validator
    assert 'start_index": 42' in validator
    assert "f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0" in validator
    assert "61ce22a8ca31498675c32576bfb94f0093d31dc95d2762f7254bf915a59ecc16" in validator
    assert "a005f75852097cd9b193750560b214cc3d5237ae9b6c106c7fca3d4fc348714b" in validator
