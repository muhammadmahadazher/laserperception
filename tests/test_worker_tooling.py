"""All transfers, hardware collection and repository commands are CPU mocks."""

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from laserperception.cli import main
from laserperception.worker import (
    ArtifactTransfer,
    AuthorizationReference,
    RemoteObject,
    RuntimeDescriptor,
    TaskManifest,
    TaskType,
    VerifiedArtifact,
    plan_worker,
    verify_artifact,
)
from laserperception.worker.bootstrap import GPURecord, QualificationRecord, bootstrap
from laserperception.worker.paths import resolve_artifact_path, validate_relative_path
from laserperception.worker.transfer import run_command


def identity(data=b"synthetic CPU fixture", relative="inputs/tiny.bin"):
    return VerifiedArtifact(
        "tiny",
        relative,
        len(data),
        hashlib.sha256(data).hexdigest(),
        "fixture",
        "synthetic CPU test",
    )


def task_fixture():
    return TaskManifest(
        "1.0",
        "test-task",
        "2026-09-13T00:00:00Z",
        "muhammadmahadazher/laserperception",
        "a" * 40,
        None,
        TaskType.ENGINEERING,
        RuntimeDescriptor("unselected-runtime", None, False, "planning only"),
        (),
        (),
        (),
        ("verify-inputs",),
        (),
        0,
        0,
        0,
        "pending",
        "planned",
        False,
    )


class FakeRemote:
    def __init__(self):
        self.objects = {}
        self.calls = []
        self.return_code = 0
        self.corrupt_download = False

    def __call__(self, command):
        self.calls.append(command)
        if self.return_code:
            return subprocess.CompletedProcess(command, self.return_code, "", "secret stderr")
        if command[1] == "lsjson":
            return subprocess.CompletedProcess(
                command, 0 if command[2] in self.objects else 4, "{}", ""
            )
        source, target = command[2:4]
        if source.startswith("lpdrive:"):
            data = self.objects[source]
            Path(target).write_bytes(data[:-1] if self.corrupt_download else data)
        else:
            self.objects.setdefault(target, Path(source).read_bytes())
        return subprocess.CompletedProcess(command, 0, "", "")


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/a",
        "C:/a",
        "C:a",
        "//host/share",
        "a\\b",
        "a/../b",
        "./a",
        "a//b",
        "a/",
        "a/./b",
        "a\x00b",
        "a\nb",
        "NUL",
        "a/name.",
        "a/name ",
        "file:stream",
        "a/*",
    ],
)
def test_invalid_paths(path):
    with pytest.raises(ValueError):
        validate_relative_path(path)


def test_symlink_escape(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    try:
        (root / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("OS does not permit symlink creation; exercised on Linux CI")
    with pytest.raises(ValueError, match="symlink"):
        resolve_artifact_path(root, "link/file")


@pytest.mark.parametrize("data", [b"short", b"x" * len(b"synthetic CPU fixture")])
def test_wrong_size_or_hash(tmp_path, data):
    path = tmp_path / "tiny"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="mismatch"):
        verify_artifact(path, identity())


def test_pull_and_push_roundtrip(tmp_path):
    remote = RemoteObject("lpdrive", "_CLOUD_WORK/test/tiny")
    fake = FakeRemote()
    fake.objects[remote.address] = b"synthetic CPU fixture"
    transfer = ArtifactTransfer(tmp_path / "attempts", runner=fake)
    path = transfer.pull(identity(), remote, tmp_path / "download")
    verify_artifact(path, identity())
    with pytest.raises(FileExistsError):
        transfer.pull(identity(), remote, tmp_path / "download")
    uploaded = RemoteObject("lpdrive", "_CLOUD_WORK/test/uploaded")
    attempt = transfer.push(identity(), tmp_path / "download", uploaded)
    assert (attempt / "roundtrip").read_bytes() == path.read_bytes()
    with pytest.raises(FileExistsError):
        transfer.push(identity(), tmp_path / "download", uploaded)
    assert all("sync" not in call and "delete" not in call for call in fake.calls)


@pytest.mark.parametrize("failure", ["command", "truncated", "hash"])
def test_pull_failures_preserve_attempt(tmp_path, failure):
    fake = FakeRemote()
    remote = RemoteObject("lpdrive", "test/tiny")
    fake.objects[remote.address] = b"synthetic CPU fixture"
    if failure == "command":
        fake.return_code = 5
    elif failure == "truncated":
        fake.corrupt_download = True
    else:
        fake.objects[remote.address] = b"x" * len(b"synthetic CPU fixture")
    with pytest.raises((RuntimeError, ValueError)):
        ArtifactTransfer(tmp_path / "attempts", runner=fake).pull(
            identity(), remote, tmp_path / "out"
        )
    events = next((tmp_path / "attempts").glob("*/events.jsonl")).read_text()
    assert '"failed"' in events and "secret stderr" not in events
    assert not (tmp_path / "out/inputs/tiny.bin").exists()


def test_push_verification_failure_preserves_remote_and_roundtrip(tmp_path):
    source = tmp_path / "source/inputs/tiny.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"synthetic CPU fixture")
    fake = FakeRemote()
    fake.corrupt_download = True
    remote = RemoteObject("lpdrive", "test/tiny")
    with pytest.raises(ValueError, match="byte size"):
        ArtifactTransfer(tmp_path / "attempts", runner=fake).push(
            identity(), tmp_path / "source", remote
        )
    assert fake.objects[remote.address] == source.read_bytes()
    assert next((tmp_path / "attempts").glob("*/roundtrip")).exists()


def test_rclone_missing(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(subprocess, "run", missing)
    with pytest.raises(RuntimeError, match="rclone is required"):
        run_command(["rclone", "lsjson", "lpdrive:tiny"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("task_type", "invalid"),
        ("schema_version", "2"),
        ("attempted_calls", True),
        ("accepted_calls", 1),
        ("completion_state", "complete"),
        ("task_type", "primary"),
        ("execution_commit", "short"),
    ],
)
def test_invalid_task(field, value):
    raw = task_fixture().to_dict()
    raw[field] = value
    with pytest.raises(ValueError):
        TaskManifest.from_dict(raw)


def test_plan_is_cpu_metadata():
    task = task_fixture()
    assert TaskManifest.from_json(task.to_json()) == task
    plan = plan_worker(task, "dsvt-pillar-transfusion-m8")
    assert not plan.executes
    assert plan.model_artifacts[0].sha256
    assert plan.task.runtime.provider is None
    assert "not verified" in plan.authorization_state


def qualification_fixture(tmp_path):
    task = task_fixture()
    payload = {
        "schema_version": "laserperception.worker.qualification-authorization.v1",
        "owner_approved": True,
        "task_id": task.task_id,
        "runtime_id": "fixture-external",
        "execution_commit": task.execution_commit,
        "mode": "qualification",
    }
    data = json.dumps(payload).encode()
    (tmp_path / "auth.json").write_bytes(data)
    auth = AuthorizationReference(
        TaskType.QUALIFICATION,
        "auth.json",
        hashlib.sha256(data).hexdigest(),
        "fixture-external",
        task.execution_commit,
    )
    return replace(
        task,
        task_type=TaskType.QUALIFICATION,
        runtime=RuntimeDescriptor("fixture-external", None, True, "CPU mocked external context"),
        authorizations=(auth,),
    )


def test_local_bootstrap_refuses_before_git_or_probe(tmp_path):
    calls = []

    def forbidden(*args):
        calls.append(args)
        raise AssertionError("must never reach collection")

    with pytest.raises(ValueError, match="external-worker"):
        bootstrap(
            task_fixture(),
            root=tmp_path,
            output=tmp_path / "result.json",
            external_worker=False,
            expected_commit="a" * 40,
            runtime_id="none",
            qualification=True,
            git_reader=forbidden,
            collector=forbidden,
        )
    assert not calls


@pytest.mark.parametrize("failure", [None, "authorization", "commit", "dirty", "collector"])
def test_bootstrap_mocked_qualification(tmp_path, failure):
    task = qualification_fixture(tmp_path)
    calls = []
    if failure == "authorization":
        (tmp_path / "auth.json").write_text("{}")

    def git_reader(command):
        if "rev-parse" in command:
            return "b" * 40 if failure == "commit" else "a" * 40
        if "status" in command:
            return " M file" if failure == "dirty" else ""
        return "https://github.com/muhammadmahadazher/laserperception.git"

    def collect(task):
        calls.append("mock collection")
        if failure == "collector":
            raise RuntimeError("mock collection failure")
        return QualificationRecord(
            "1.0",
            task.task_id,
            task.runtime.runtime_id,
            task.execution_commit,
            (GPURecord("CPU-mock-not-a-measurement", 8, 4, "mock"),),
            "mock",
            "mock",
            (),
            (),
            "recorded_for_owner_review_not_scientific_authorization",
        )

    kwargs = dict(
        root=tmp_path,
        output=tmp_path / "qualification.json",
        external_worker=True,
        expected_commit=task.execution_commit,
        runtime_id=task.runtime.runtime_id,
        qualification=True,
        git_reader=git_reader,
        collector=collect,
    )
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            bootstrap(task, **kwargs)
        assert bool(calls) == (failure == "collector")
    else:
        record = bootstrap(task, **kwargs)
        assert QualificationRecord.from_json(record.to_json()) == record
        assert len(calls) == 1


def test_worker_cli(tmp_path, capsys):
    path = tmp_path / "task.json"
    path.write_text(task_fixture().to_json())
    assert main(["worker", "manifest", "validate", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["task_id"] == "test-task"
    assert main(["worker", "plan", str(path), "--model", "dsvt-pillar-transfusion-m8"]) == 0
    assert not json.loads(capsys.readouterr().out)["executes"]


def test_differing_remote_is_never_overwritten(tmp_path):
    source = tmp_path / "inputs/tiny.bin"
    source.parent.mkdir()
    source.write_bytes(b"synthetic CPU fixture")
    fake = FakeRemote()
    remote = RemoteObject("lpdrive", "test/existing")
    fake.objects[remote.address] = b"valuable existing bytes"
    with pytest.raises(FileExistsError):
        ArtifactTransfer(tmp_path / "attempts", runner=fake).push(identity(), tmp_path, remote)
    assert fake.objects[remote.address] == b"valuable existing bytes"
    assert len(fake.calls) == 1


def test_invalid_authorization_role_and_runtime(tmp_path):
    task = qualification_fixture(tmp_path)
    raw = task.to_dict()
    raw["authorizations"][0]["role"] = "unrestricted"
    with pytest.raises(ValueError):
        TaskManifest.from_dict(raw)
    with pytest.raises(ValueError, match="runtime/commit"):
        replace(task, runtime=replace(task.runtime, runtime_id="different"))


def test_training_plan_is_rejected():
    task = task_fixture()
    auth = AuthorizationReference(
        TaskType.TRAINING, "auth.json", "a" * 64, task.runtime.runtime_id, task.execution_commit
    )
    training = replace(
        task,
        task_type=TaskType.TRAINING,
        authorizations=(auth,),
        runtime=replace(task.runtime, external=True),
    )
    with pytest.raises(ValueError, match="training"):
        plan_worker(training, "dsvt-pillar-transfusion-m8")


def test_engineering_plan_has_no_bootstrap_command():
    plan = plan_worker(task_fixture(), "pointpillars-nuscenes-v0.3")
    assert all("bootstrap" not in command for command in plan.remote_commands)
