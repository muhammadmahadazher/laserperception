from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "scripts" / "cloud" / "verify_drive_object.py"
SPEC = importlib.util.spec_from_file_location("verify_drive_object", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_task_manifest_schema_is_valid_json_and_requires_safety_fields() -> None:
    schema = json.loads((ROOT / "scripts/cloud/TASK_MANIFEST.schema.json").read_text())
    assert schema["properties"]["worker_unique_state_remaining"]["const"] is False
    assert (
        schema["properties"]["detector_call_accounting"]["properties"]["cloud_detector_calls"][
            "const"
        ]
        == 0
    )


def process_for(payload: bytes, returncode: int = 0) -> Mock:
    process = Mock(stdout=io.BytesIO(payload))
    process.wait.return_value = returncode
    return process


def test_verify_drive_object_accepts_matching_hash() -> None:
    payload = b"tiny synthetic object\n"
    digest = MODULE.hashlib.sha256(payload).hexdigest()
    with patch.object(MODULE.subprocess, "Popen", return_value=process_for(payload)) as popen:
        MODULE.verify("artifacts/object.bin", digest)
    popen.assert_called_once_with(
        ["rclone", "cat", "lpdrive:artifacts/object.bin"], stdout=MODULE.subprocess.PIPE
    )


def test_verify_drive_object_rejects_mismatch() -> None:
    with (
        patch.object(MODULE.subprocess, "Popen", return_value=process_for(b"wrong")),
        pytest.raises(RuntimeError, match="SHA256 mismatch"),
    ):
        MODULE.verify("artifacts/object.bin", "a" * 64)


def test_verify_drive_object_fails_on_transfer_error() -> None:
    with (
        patch.object(MODULE.subprocess, "Popen", return_value=process_for(b"", 1)),
        pytest.raises(RuntimeError, match="rclone failed"),
    ):
        MODULE.verify("artifacts/object.bin", "a" * 64)


@pytest.mark.parametrize("path", ["", "../secret", "safe/../secret"])
def test_verify_drive_object_rejects_unsafe_path(path: str) -> None:
    with pytest.raises(ValueError, match="Drive path"):
        MODULE.verify(path, "a" * 64)
