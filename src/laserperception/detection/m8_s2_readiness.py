"""CPU-safe receipt validation for external, zero-forward candidate initialization."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path

from laserperception.detection.m8_s1_runtime import CANDIDATE_MANIFEST_PATH, sha256_file
from laserperception.detection.m8_s2_bootstrap import REQUIRED_PACKAGES
from laserperception.detection.m8_s2_runtime import (
    DSVT_GENERATED_FILES,
    INPUT_FREEZE_SHA256,
    PARTITIONS_SHA256,
    PROTOCOL_SHA256,
    M8S2ProtocolViolation,
    native_extension_hashes,
    verify_runtime_policy_document,
)

READINESS_SCHEMA = "laserperception.m8.s2.candidate-readiness.v1"


def verify_preimport_runtime(policy: Mapping[str, object], upstream: Path) -> None:
    """Check installed metadata and exact native bytes without importing Torch."""
    for package, expected in REQUIRED_PACKAGES.items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError as error:
            raise M8S2ProtocolViolation(f"readiness package absent: {package}") from error
        if actual != expected:
            raise M8S2ProtocolViolation(f"readiness package differs: {package}")
    if policy["python_exact_version"] != sys.version:
        raise M8S2ProtocolViolation("readiness Python identity differs")
    if native_extension_hashes(upstream) != policy["upstream_native_extensions"]:
        raise M8S2ProtocolViolation("readiness native inventory differs")


def readiness_document(
    *,
    execution_commit: str,
    input_gate_receipt_sha256: str,
    runtime_policy_sha256: str,
    qualification_receipt_sha256: str,
    policy: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    """Construct the exact receipt identity from already verified static bindings."""
    verify_runtime_policy_document(policy, execution_commit=execution_commit, candidate=candidate)
    identity = policy["candidate_identity"]
    if not isinstance(identity, Mapping):
        raise M8S2ProtocolViolation("readiness candidate identity is malformed")
    extensions = policy["upstream_native_extensions"]
    if not extensions:
        raise M8S2ProtocolViolation("readiness native inventory is empty")
    return {
        "schema_version": READINESS_SCHEMA,
        "status": "ACCEPTED",
        "execution_commit": execution_commit,
        "protocol_sha256": PROTOCOL_SHA256,
        "partitions_sha256": PARTITIONS_SHA256,
        "input_freeze_sha256": INPUT_FREEZE_SHA256,
        "input_gate_receipt_sha256": input_gate_receipt_sha256,
        "runtime_policy_sha256": runtime_policy_sha256,
        "qualification_receipt_sha256": qualification_receipt_sha256,
        "candidate_import_packages": policy["candidate_import_packages"],
        **identity,
        **{
            key: policy[key]
            for key in (
                "worker_hostname",
                "gpu_uuid",
                "python_exact_version",
                "pytorch_exact_version",
                "torchvision",
                "cuda_runtime",
                "numpy",
                "spconv",
                "torch_scatter",
            )
        },
        "generated_version_sha256": DSVT_GENERATED_FILES["pcdet/version.py"],
        "native_extension_inventory_sha256": hashlib.sha256(
            json.dumps(extensions, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "model_device": "cuda:0",
        "model_eval": True,
        "initialization_complete": True,
        "inference_calls": 0,
        "engineering_calls": 0,
        "scientific_calls": 0,
        "GT_loaded": False,
        "evaluator_loaded": False,
    }


def initialize_candidate(root: Path, expected: Mapping[str, object]) -> dict[str, object]:
    """External-only implementation; caller must finish every authorization first.

    The production backend factory loads the frozen checkpoint. No points, loader,
    predictions, evaluator or inference method are used here.
    """
    backend_module = importlib.import_module("laserperception.detection.m8_backend")
    backend = backend_module.DsvtBackend.from_environment(
        manifest_path=root / CANDIDATE_MANIFEST_PATH
    )
    state = backend.initialization_state()
    for key in ("architecture", "upstream_commit", "config_sha256", "checkpoint_sha256"):
        if backend.identity.get(key) != expected[key]:
            raise M8S2ProtocolViolation(f"readiness initialized {key} differs")
    for key, receipt_key in (
        ("torch", "pytorch_exact_version"),
        ("cuda", "cuda_runtime"),
        ("spconv", "spconv"),
        ("torch_scatter", "torch_scatter"),
        ("device", "model_device"),
    ):
        if backend.identity.get(key) != expected[receipt_key]:
            raise M8S2ProtocolViolation(f"readiness initialized {key} differs")
    if state != {
        key: expected[key] for key in ("model_device", "model_eval", "initialization_complete")
    }:
        raise M8S2ProtocolViolation("readiness model initialization state differs")
    return dict(expected)


def verify_readiness_receipt(path: Path, **bindings: object) -> str:
    """Reject missing, extra or mismatching evidence before any accelerator import."""
    expected = readiness_document(**bindings)  # type: ignore[arg-type]
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise M8S2ProtocolViolation("candidate readiness receipt is absent or malformed") from error
    # Compare canonical JSON as well: Python equality otherwise accepts True == 1.
    if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise M8S2ProtocolViolation("candidate readiness receipt binding differs")
    return sha256_file(path)
