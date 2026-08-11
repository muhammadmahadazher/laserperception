from pathlib import Path

import pytest

from laserperception.detection.runtime_metadata import nvidia_smi_value, repository_git_sha


def test_repository_git_sha_is_sanitized() -> None:
    repository_root = Path(__file__).resolve().parents[1]

    sha = repository_git_sha(repository_root)

    assert len(sha) == 40
    assert set(sha) <= set("0123456789abcdef")


@pytest.mark.parametrize("field", ["", "name,uuid", "Name", "driver-version"])
def test_nvidia_smi_field_is_validated_before_optional_command(field: str) -> None:
    with pytest.raises(ValueError, match="lowercase letters and underscores"):
        nvidia_smi_value(field)
