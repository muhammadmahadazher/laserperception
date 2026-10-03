"""CPU fixtures for the pinned generated file and future-worker pre-build gate."""

from __future__ import annotations

import hashlib
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from laserperception.detection import m8_s2_runtime as runtime
from laserperception.detection.m8_s2_bootstrap import (
    REQUIRED_HEADERS,
    REQUIRED_LIBRARIES,
    REQUIRED_PACKAGES,
    M8S2BootstrapError,
    verify_development_environment,
)

ROOT = Path(__file__).resolve().parents[1]
VERSION_BYTES = b'__version__ = "0.6.0+8cfc2a6"\n'


@pytest.fixture
def upstream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init")
    git("config", "user.name", "CPU Fixture")
    git("config", "user.email", "fixture@example.invalid")
    source = tmp_path / "pcdet/__init__.py"
    source.parent.mkdir()
    source.write_bytes(b"# frozen fixture\n")
    git("add", ".")
    git("commit", "-m", "fixture")
    monkeypatch.setattr(runtime, "git_head", lambda _: runtime.DSVT_GENERATED_VERSION_COMMIT)
    (tmp_path / "pcdet/version.py").write_bytes(VERSION_BYTES)
    return tmp_path


def verify_upstream(root: Path) -> None:
    runtime.verify_clean_tracked_tree(
        root, allow_native_extensions=True, allow_pinned_dsvt_generated_files=True
    )


def test_exact_generated_version_and_native_hash_policy(upstream: Path) -> None:
    assert (
        hashlib.sha256(VERSION_BYTES).hexdigest()
        == runtime.DSVT_GENERATED_FILES["pcdet/version.py"]
    )
    extension = upstream / "pcdet/ops/fixture.so"
    extension.parent.mkdir()
    extension.write_bytes(b"synthetic native build")
    verify_upstream(upstream)
    assert runtime.native_extension_hashes(upstream) == {
        "pcdet/ops/fixture.so": hashlib.sha256(extension.read_bytes()).hexdigest()
    }
    with pytest.raises(runtime.M8S2ProtocolViolation, match="untracked importable"):
        runtime.verify_clean_tracked_tree(upstream, allow_native_extensions=True)


@pytest.mark.parametrize(
    "contents",
    [
        b'__version__ = "0.6.1+8cfc2a6"\n',
        VERSION_BYTES + b"\n",
        b'__version__ = "0.6.0+abcdef0"\n',
        VERSION_BYTES + b"exec('malicious')\n",
    ],
)
def test_wrong_generated_bytes_fail(upstream: Path, contents: bytes) -> None:
    (upstream / "pcdet/version.py").write_bytes(contents)
    with pytest.raises(runtime.M8S2ProtocolViolation, match="generated-version bytes"):
        verify_upstream(upstream)


@pytest.mark.parametrize("name", ["pcdet/version_extra.py", "pcdet/shadow.py", "pcdet/version.pyc"])
def test_additional_python_fails(upstream: Path, name: str) -> None:
    (upstream / name).write_bytes(b"unexpected")
    with pytest.raises(runtime.M8S2ProtocolViolation, match="untracked importable"):
        verify_upstream(upstream)


def test_wrong_upstream_commit_fails(upstream: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "git_head", lambda _: "a" * 40)
    with pytest.raises(runtime.M8S2ProtocolViolation, match="upstream commit"):
        verify_upstream(upstream)


def test_tracked_source_change_still_fails(upstream: Path) -> None:
    (upstream / "pcdet/__init__.py").write_bytes(b"changed\n")
    with pytest.raises(runtime.M8S2ProtocolViolation, match="tracked execution tree"):
        verify_upstream(upstream)


def test_generated_version_symlink_fails(upstream: Path) -> None:
    version = upstream / "pcdet/version.py"
    target = upstream / "version.txt"
    target.write_bytes(VERSION_BYTES)
    version.unlink()
    try:
        version.symlink_to(target)
    except OSError:
        pytest.skip("fixture symlinks unavailable on this host")
    with pytest.raises(runtime.M8S2ProtocolViolation, match="directory or symlink"):
        verify_upstream(upstream)


def test_untracked_directory_inventory_fails(
    upstream: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unexpected = upstream / "nested"
    unexpected.mkdir()
    original = subprocess.run

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["git", "ls-files", "--others"]:
            return subprocess.CompletedProcess(command, 0, "nested/\0", "")
        return original(command, **kwargs)  # type: ignore[call-overload,no-any-return]

    monkeypatch.setattr(runtime.subprocess, "run", run)
    with pytest.raises(runtime.M8S2ProtocolViolation, match="directory or symlink"):
        verify_upstream(upstream)


@pytest.fixture
def development(tmp_path: Path) -> dict[str, object]:
    for relative in ("bin/nvcc", *(f"include/{name}" for name in REQUIRED_HEADERS)):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture\n")
    (tmp_path / "include/cuda.h").write_bytes(b"#define CUDA_VERSION 11080\n")
    for name in REQUIRED_LIBRARIES:
        path = tmp_path / f"lib64/lib{name}.so"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"fixture library\n")
    return {
        "cuda_home": tmp_path,
        "python_version": (3, 10),
        "packages": dict(REQUIRED_PACKAGES),
        "nvcc_output": "Cuda compilation tools, release 11.8, V11.8.89",
        "gcc_version": "11.5.0",
        "gxx_version": "11.5.0",
    }


def test_complete_development_environment_passes(development: dict[str, object]) -> None:
    before = set(sys.modules)
    receipt = verify_development_environment(**development)  # type: ignore[arg-type]
    assert receipt["complete"] is True
    assert receipt["extension_build_verified"] is False
    assert not any(
        name == "torch" or name.startswith("torch.") for name in set(sys.modules) - before
    )


@pytest.mark.parametrize("relative", ["bin/nvcc", "include/cusparse.h", "lib64/libcusparse.so"])
def test_missing_development_file_fails(development: dict[str, object], relative: str) -> None:
    root = development["cuda_home"]
    assert isinstance(root, Path)
    (root / relative).unlink()
    with pytest.raises(M8S2BootstrapError, match=relative.split("/")[-1]):
        verify_development_environment(**development)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("nvcc_output", "release 12.8, V12.8.1", "release 11.8"),
        ("gcc_version", "13.3.0", "GCC 11"),
        ("gxx_version", "11.4.0", "versions must match"),
        ("python_version", (3, 12), "Python 3.10"),
        ("packages", {"torch": "2.8.0+cu128"}, "torch==2.1.0"),
    ],
)
def test_wrong_environment_facts_fail(
    development: dict[str, object], key: str, value: object, message: str
) -> None:
    development[key] = value
    with pytest.raises(M8S2BootstrapError, match=message):
        verify_development_environment(**development)  # type: ignore[arg-type]


def test_wrong_header_version_fails(development: dict[str, object]) -> None:
    root = development["cuda_home"]
    assert isinstance(root, Path)
    (root / "include/cuda.h").write_bytes(b"#define CUDA_VERSION 12080\n")
    with pytest.raises(M8S2BootstrapError, match="CUDA_VERSION 11080"):
        verify_development_environment(**development)  # type: ignore[arg-type]


def test_wrapper_refuses_local_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapper = runpy.run_path(str(ROOT / "scripts/detection/check_m8_s2_development.py"))
    monkeypatch.setattr(sys, "argv", ["check_m8_s2_development.py"])

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("local probe attempted")

    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(SystemExit) as error:
        wrapper["main"]()
    assert error.value.code == 2


def test_wrapper_complete_fixture_without_real_commands(
    development: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import importlib.metadata
    import json

    wrapper = runpy.run_path(str(ROOT / "scripts/detection/check_m8_s2_development.py"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["checker", "--external-worker", "--cuda-home", str(development["cuda_home"])],
    )
    monkeypatch.setattr(sys, "flags", SimpleNamespace(isolated=True))
    monkeypatch.setattr(sys, "version_info", (3, 10))
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: REQUIRED_PACKAGES[name])
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        value = development["nvcc_output"] if command[0].endswith("nvcc") else "11.5.0"
        return subprocess.CompletedProcess(command, 0, str(value), "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    try:
        assert wrapper["main"]() == 0
        assert json.loads(capsys.readouterr().out)["complete"] is True
        assert len(commands) == 3
        assert {command[-1] for command in commands} == {"--version", "-dumpversion"}
    finally:
        # The future wrapper disables caches; do not change the rest of the test session.
        monkeypatch.undo()


def test_ignored_python_still_rejected(upstream: Path) -> None:
    (upstream / ".git/info/exclude").write_text("pcdet/*.py\n", encoding="utf-8")
    verify_upstream(upstream)
    (upstream / "pcdet/injected.py").write_bytes(b"malicious = True\n")
    with pytest.raises(runtime.M8S2ProtocolViolation, match="untracked importable"):
        verify_upstream(upstream)


def test_native_shadow_of_generated_version_rejected(upstream: Path) -> None:
    (upstream / "pcdet/version.so").write_bytes(b"shadow")
    with pytest.raises(runtime.M8S2ProtocolViolation, match="untracked native"):
        verify_upstream(upstream)
