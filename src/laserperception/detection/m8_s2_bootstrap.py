"""CPU-safe validation of supplied CUDA development-toolchain facts.

This module never imports Torch, discovers a GPU, builds extensions, or loads GT.
The external-only wrapper supplies compiler output and installed package metadata.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

REQUIRED_HEADERS = (
    "cuda.h",
    "cuda_runtime.h",
    "cuda_runtime_api.h",
    "cusparse.h",
    "cublas_v2.h",
    "cusolverDn.h",
    "curand.h",
)
REQUIRED_LIBRARIES = ("cudart", "cusparse", "cublas", "cusolver", "curand")
REQUIRED_PACKAGES = {
    "torch": "2.1.0+cu118",
    "torchvision": "0.16.0+cu118",
    "numpy": "1.23.5",
    "spconv-cu118": "2.3.8",
    "torch-scatter": "2.1.2+pt21cu118",
    # Eager imports reached by the exact pinned pcdet.models registry.
    "easydict": "1.13",
    "PyYAML": "6.0.3",
    "SharedArray": "3.2.4",
    "scipy": "1.10.1",
    "numba": "0.57.1",
    "scikit-image": "0.21.0",
    "tqdm": "4.66.5",
    "packaging": "26.3",
    "onnx": "1.14.1",
    "tensorrt": "8.6.1",
}


class M8S2BootstrapError(ValueError):
    """The supplied external development environment is incomplete or incompatible."""


def _readable(path: Path, toolkit_root: Path) -> bool:
    try:
        if not path.is_file() or not path.resolve().is_relative_to(toolkit_root.resolve()):
            return False
        with path.open("rb") as stream:
            return bool(stream.read(1))
    except OSError:
        return False


def verify_development_environment(
    *,
    cuda_home: Path,
    python_version: tuple[int, int],
    packages: Mapping[str, str],
    nvcc_output: str,
    gcc_version: str,
    gxx_version: str,
) -> dict[str, object]:
    """Validate fixtureable facts and readable Linux toolkit files; execute no commands.

    Require the prospective GCC/G++ 11 toolchain and the CUDA 11.8 header version,
    with unversioned development-library names inside the same CUDA_HOME. Readable
    symlinks within the toolkit are legitimate; foreign, dangling or empty files fail. This cheap
    gate is not a substitute for a complete extension build and import validation.
    """

    errors: list[str] = []
    if not cuda_home.is_dir():
        errors.append(f"invalid CUDA_HOME: {cuda_home}")
    if not _readable(cuda_home / "bin/nvcc", cuda_home):
        errors.append("missing/readability failure: CUDA_HOME/bin/nvcc")
    if re.search(r"\brelease 11\.8(?:,|\s|$)", nvcc_output) is None:
        errors.append("nvcc must report release 11.8")
    if python_version != (3, 10):
        errors.append("Python 3.10 required")
    for name, expected in REQUIRED_PACKAGES.items():
        if packages.get(name) != expected:
            errors.append(f"{name}=={expected} required (found {packages.get(name)!r})")
    for name, version in (("GCC", gcc_version), ("G++", gxx_version)):
        if re.fullmatch(r"11\.\d+(?:\.\d+)?", version.strip()) is None:
            errors.append(f"{name} 11 required (found {version!r})")
    if gcc_version.strip() != gxx_version.strip():
        errors.append("GCC/G++ versions must match")
    include = cuda_home / "include"
    for name in REQUIRED_HEADERS:
        if not _readable(include / name, cuda_home):
            errors.append(f"missing/readability failure: CUDA_HOME/include/{name}")
    if _readable(include / "cuda.h", cuda_home):
        try:
            header = (include / "cuda.h").read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            errors.append("cannot read CUDA_HOME/include/cuda.h version")
        else:
            if re.search(r"^\s*#\s*define\s+CUDA_VERSION\s+11080\b", header, re.M) is None:
                errors.append("cuda.h must define CUDA_VERSION 11080")
    library_paths: dict[str, str] = {}
    for name in REQUIRED_LIBRARIES:
        candidates = [cuda_home / directory / f"lib{name}.so" for directory in ("lib64", "lib")]
        library = next((path for path in candidates if _readable(path, cuda_home)), None)
        if library is None:
            errors.append(f"missing/readability failure: lib{name}.so in CUDA_HOME/lib64 or lib")
        else:
            library_paths[name] = str(library.resolve())
    if errors:
        raise M8S2BootstrapError("S2 pre-build development gate failed:\n- " + "\n- ".join(errors))
    return {
        "schema_version": "laserperception.m8.s2.development-gate.v1",
        "complete": True,
        "cuda_home": str(cuda_home.resolve()),
        "nvcc_output": nvcc_output,
        "gcc_version": gcc_version,
        "gxx_version": gxx_version,
        "python_version": list(python_version),
        "packages": dict(packages),
        "headers": list(REQUIRED_HEADERS),
        "development_libraries": library_paths,
        "extension_build_verified": False,
        "inference_authorized": False,
    }
