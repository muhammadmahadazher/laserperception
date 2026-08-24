"""Fail-closed identities and resumable input ledger for M6c."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from laserperception.detection.mmdet3d_backend import sha256_file

M6C_ENGINE_SHA256 = "2e790b1cdbdc1b88c2aafdc81b5921ebee152edd8408158f88437ae4dd1f3e7f"
M6C_PROFILE_COUNTS = (4352, 18207, 40000)


def require_file_sha256(
    path: str | Path,
    expected_sha256: str,
    *,
    artifact_name: str,
) -> str:
    """Require one external artifact to match its frozen SHA256."""

    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"{artifact_name} is missing: {artifact}")
    actual = sha256_file(artifact)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{artifact_name} SHA256 mismatch: expected {expected_sha256}, found {actual}"
        )
    return actual


@dataclass(frozen=True, slots=True)
class M6cProgressIdentity:
    """Frozen identities that make an input-gate record resumable."""

    protocol_commit: str
    implementation_commit: str
    m6a_evidence_sha256: str
    m6b_input_ledger_sha256: str

    def to_dict(self) -> dict[str, str]:
        """Return the stable JSON representation."""

        return {
            "protocol_commit": self.protocol_commit,
            "implementation_commit": self.implementation_commit,
            "m6a_evidence_sha256": self.m6a_evidence_sha256,
            "m6b_input_ledger_sha256": self.m6b_input_ledger_sha256,
        }


class M6cInputProgress:
    """Atomic fail-closed local ledger for the 856-condition ROS gate."""

    def __init__(
        self,
        path: str | Path,
        identity: M6cProgressIdentity,
        condition_keys: Sequence[str],
    ) -> None:
        self.path = Path(path)
        self.identity = identity
        keys = tuple(condition_keys)
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("M6c progress condition keys must be non-empty and unique")
        self._keys = keys
        self.record = self._load_or_initialize()

    def passed(self, key: str) -> bool:
        """Return whether one condition passed under the frozen identities."""

        return self._condition(key).get("status") == "PASS"

    def mark(
        self,
        key: str,
        *,
        status: str,
        expected_sha256: str,
        observed_sha256: str,
        point_count: int,
        history_depth: int,
        timestamp_nanoseconds: int,
        elapsed_seconds: float,
    ) -> None:
        """Atomically record one exact comparison before continuing."""

        if status not in {"PASS", "FAIL"}:
            raise ValueError("M6c progress status must be PASS or FAIL")
        self._condition(key)
        conditions = self.record["conditions"]
        assert isinstance(conditions, dict)
        conditions[key] = {
            "status": status,
            "expected_input_sha256": expected_sha256,
            "observed_input_sha256": observed_sha256,
            "point_count": point_count,
            "history_depth": history_depth,
            "timestamp_nanoseconds": timestamp_nanoseconds,
            "elapsed_seconds": elapsed_seconds,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._write()

    def totals(self) -> dict[str, int]:
        """Return current PASS/FAIL/PENDING counts."""

        conditions = self.record["conditions"]
        assert isinstance(conditions, Mapping)
        statuses = [str(value["status"]) for value in conditions.values()]
        return {name.lower(): statuses.count(name) for name in ("PASS", "FAIL", "PENDING")}

    def _condition(self, key: str) -> dict[str, object]:
        conditions = self.record["conditions"]
        assert isinstance(conditions, dict)
        if key not in conditions:
            raise KeyError(f"condition is outside the frozen M6c corpus: {key}")
        value = conditions[key]
        if not isinstance(value, dict):
            raise RuntimeError("M6c progress condition record is malformed")
        return value

    def _load_or_initialize(self) -> dict[str, object]:
        expected_identity = self.identity.to_dict()
        if self.path.exists():
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if value.get("identity") != expected_identity:
                raise RuntimeError("M6c progress identity differs from the frozen run")
            conditions = value.get("conditions")
            if not isinstance(conditions, Mapping) or set(conditions) != set(self._keys):
                raise RuntimeError("M6c progress condition set differs from the frozen corpus")
            return dict(value)
        record: dict[str, object] = {
            "schema_version": 1,
            "identity": expected_identity,
            "conditions": {key: {"status": "PENDING"} for key in self._keys},
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.record = record
        self._write()
        return record

    def _write(self) -> None:
        self.record["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        self.record["totals"] = self.totals()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)


@dataclass(frozen=True, slots=True)
class M6cR3ProgressIdentity:
    """Frozen identities binding every resumable final-R3 live comparison."""

    protocol_commit: str
    implementation_commit: str
    projected_manifest_sha256: str

    def to_dict(self) -> dict[str, str]:
        """Return the stable JSON representation."""

        return {
            "protocol_commit": self.protocol_commit,
            "implementation_commit": self.implementation_commit,
            "projected_manifest_sha256": self.projected_manifest_sha256,
        }


class M6cR3InputProgress:
    """Atomic fail-closed ledger for the 860 unique projected-reference conditions."""

    def __init__(
        self,
        path: str | Path,
        identity: M6cR3ProgressIdentity,
        condition_keys: Sequence[str],
    ) -> None:
        self.path = Path(path)
        self.identity = identity
        keys = tuple(condition_keys)
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("M6c R3 progress condition keys must be non-empty and unique")
        self._keys = keys
        self.record = self._load_or_initialize()

    def passed(
        self,
        key: str,
        *,
        expected_sha256: str,
        expected_point_count: int,
        expected_history_depth: int,
        expected_timestamp_nanoseconds: int,
    ) -> bool:
        """Return whether one condition passed under every identical frozen identity."""

        condition = self._condition(key)
        return condition.get("status") == "PASS" and all(
            (
                condition.get("expected_sha256") == expected_sha256,
                condition.get("observed_sha256") == expected_sha256,
                condition.get("expected_point_count") == expected_point_count,
                condition.get("observed_point_count") == expected_point_count,
                condition.get("expected_history_depth") == expected_history_depth,
                condition.get("observed_history_depth") == expected_history_depth,
                condition.get("expected_timestamp_nanoseconds") == expected_timestamp_nanoseconds,
                condition.get("observed_timestamp_nanoseconds") == expected_timestamp_nanoseconds,
            )
        )

    def mark(
        self,
        key: str,
        *,
        status: str,
        expected_sha256: str,
        observed_sha256: str,
        expected_point_count: int,
        observed_point_count: int,
        expected_history_depth: int,
        observed_history_depth: int,
        expected_timestamp_nanoseconds: int,
        observed_timestamp_nanoseconds: int,
        elapsed_seconds: float,
    ) -> None:
        """Atomically record one comparison before continuing."""

        if status not in {"PASS", "FAIL"}:
            raise ValueError("M6c R3 progress status must be PASS or FAIL")
        self._condition(key)
        conditions = self.record["conditions"]
        assert isinstance(conditions, dict)
        conditions[key] = {
            "status": status,
            "expected_sha256": expected_sha256,
            "observed_sha256": observed_sha256,
            "expected_point_count": expected_point_count,
            "observed_point_count": observed_point_count,
            "expected_history_depth": expected_history_depth,
            "observed_history_depth": observed_history_depth,
            "expected_timestamp_nanoseconds": expected_timestamp_nanoseconds,
            "observed_timestamp_nanoseconds": observed_timestamp_nanoseconds,
            "elapsed_seconds": elapsed_seconds,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._write()

    def totals(self) -> dict[str, int]:
        """Return current PASS/FAIL/PENDING counts."""

        conditions = self.record["conditions"]
        assert isinstance(conditions, Mapping)
        statuses = [str(value["status"]) for value in conditions.values()]
        return {name.lower(): statuses.count(name) for name in ("PASS", "FAIL", "PENDING")}

    def conditions(self) -> Mapping[str, object]:
        """Return the current condition records for compact evidence generation."""

        conditions = self.record["conditions"]
        assert isinstance(conditions, Mapping)
        return conditions

    def _condition(self, key: str) -> dict[str, object]:
        conditions = self.record["conditions"]
        assert isinstance(conditions, dict)
        if key not in conditions:
            raise KeyError(f"condition is outside the frozen M6c R3 corpus: {key}")
        value = conditions[key]
        if not isinstance(value, dict):
            raise RuntimeError("M6c R3 progress condition record is malformed")
        return value

    def _load_or_initialize(self) -> dict[str, object]:
        expected_identity = self.identity.to_dict()
        if self.path.exists():
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if value.get("identity") != expected_identity:
                raise RuntimeError("M6c R3 progress identity differs from the frozen run")
            conditions = value.get("conditions")
            if not isinstance(conditions, Mapping) or set(conditions) != set(self._keys):
                raise RuntimeError("M6c R3 progress condition set differs from the frozen corpus")
            return dict(value)
        record: dict[str, object] = {
            "schema_version": 1,
            "identity": expected_identity,
            "conditions": {key: {"status": "PENDING"} for key in self._keys},
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.record = record
        self._write()
        return record

    def _write(self) -> None:
        self.record["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        self.record["totals"] = self.totals()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)
