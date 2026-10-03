#!/usr/bin/env python3
"""External-only pre-build gate; no Torch import, GPU discovery, or compilation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-worker", action="store_true")
    parser.add_argument("--cuda-home", type=Path)
    parser.add_argument("--cc", default="gcc-11")
    parser.add_argument("--cxx", default="g++-11")
    args = parser.parse_args()
    if not args.external_worker:
        parser.error("--external-worker is required; do not probe a local workstation")
    if not sys.flags.isolated:
        parser.error("invoke with isolated Python: python -I check_m8_s2_development.py ...")
    cuda_home = args.cuda_home or (
        Path(os.environ["CUDA_HOME"]) if os.getenv("CUDA_HOME") else None
    )
    if cuda_home is None or not cuda_home.is_dir():
        parser.error("set a valid CUDA_HOME or --cuda-home")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from laserperception.detection.m8_s2_bootstrap import (
        REQUIRED_PACKAGES,
        M8S2BootstrapError,
        verify_development_environment,
    )

    failures: list[str] = []

    def output(command: list[str]) -> str:
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError) as error:
            failures.append(f"cannot validate {command[0]}: {error}")
            return ""
        return result.stdout.strip()

    packages: dict[str, str] = {}
    for name in REQUIRED_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "absent"
    nvcc = output([str(cuda_home / "bin/nvcc"), "--version"])
    gcc = output([args.cc, "-dumpfullversion", "-dumpversion"])
    gxx = output([args.cxx, "-dumpfullversion", "-dumpversion"])
    try:
        receipt = verify_development_environment(
            cuda_home=cuda_home,
            python_version=sys.version_info[:2],
            packages=packages,
            nvcc_output=nvcc,
            gcc_version=gcc,
            gxx_version=gxx,
        )
    except M8S2BootstrapError as error:
        print(str(error), file=sys.stderr)
        failures.append("development requirements failed")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
