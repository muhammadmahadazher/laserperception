"""Synthetic CPU tracking regressions; no detector, benchmark, or downloads."""

import importlib.abc
import json
import sys
from dataclasses import FrozenInstanceError, replace

import pytest

from laserperception.cli import main
from laserperception.detection.serialization import (
    TimedDetectionFrame,
    detection_frame_from_json,
)
from laserperception.detection.types import Detection3D, DetectionFrame
from laserperception.tracking import (
    ConstantVelocityTracker,
    TrackerConfig,
    TrackFrame,
    TrackState,
    track_sequence,
)


def box(x=0, *, y=0, name="car", class_id=0, velocity=None, score=0.9):
    return Detection3D((x, y, 0), (4, 2, 1.5), 0, score, class_id, name, velocity)


def frame(*boxes, sample="sample", coordinate="world"):
    return DetectionFrame(tuple(boxes), sample, coordinate, {"precomputed": True})


def test_lifecycle_miss_reacquisition_retirement_and_ids():
    tracker = ConstantVelocityTracker("sequence", TrackerConfig(min_hits=2, max_missed=1))
    first = tracker.update(frame(box()), timestamp_ns=0)
    assert first.tracks[0].lifecycle is TrackState.TENTATIVE
    second = tracker.update(frame(box(1)), timestamp_ns=1_000_000_000)
    track = second.tracks[0]
    assert (track.track_id, track.hits, track.age, track.missed) == (1, 2, 2, 0)
    assert track.lifecycle is TrackState.CONFIRMED
    assert track.velocity_xy == (1, 0)
    missed = tracker.update(frame(sample="empty"), timestamp_ns=2_000_000_000).tracks[0]
    assert missed.lifecycle is TrackState.LOST
    assert missed.predicted_center_xyz == (2, 0, 0)
    assert missed.detection is track.detection
    assert missed.observation_timestamp_ns == 1_000_000_000
    assert missed.source_sample_id == "sample"
    reacquired = tracker.update(frame(box(3)), timestamp_ns=3_000_000_000).tracks[0]
    assert reacquired.track_id == 1 and reacquired.lifecycle is TrackState.CONFIRMED
    tracker.update(frame(), timestamp_ns=4_000_000_000)
    assert tracker.update(frame(), timestamp_ns=5_000_000_000).tracks == ()
    assert tracker.update(frame(box()), timestamp_ns=6_000_000_000).tracks[0].track_id == 2


def test_empty_frames_and_immediate_confirmation_or_retirement():
    tracker = ConstantVelocityTracker("s", TrackerConfig(min_hits=1, max_missed=0))
    assert tracker.update(frame(), timestamp_ns=0).tracks == ()
    assert tracker.update(frame(box()), timestamp_ns=1).tracks[0].lifecycle is TrackState.CONFIRMED
    assert tracker.update(frame(), timestamp_ns=2).tracks == ()


def test_class_aware_default_and_explicit_class_agnostic_policy():
    for aware, expected in ((True, 2), (False, 1)):
        tracker = ConstantVelocityTracker("s", TrackerConfig(class_aware=aware))
        tracker.update(frame(box()), timestamp_ns=0)
        result = tracker.update(frame(box(name="truck", class_id=1)), timestamp_ns=1)
        assert len(result.tracks) == expected
        if aware:
            assert result.tracks[0].lifecycle is TrackState.LOST


def test_radius_boundary_and_class_name_are_both_gated():
    tracker = ConstantVelocityTracker("s", TrackerConfig(association_radius_m=1))
    tracker.update(frame(box()), timestamp_ns=0)
    assert tracker.update(frame(box(1)), timestamp_ns=1).tracks[0].hits == 2
    assert len(tracker.update(frame(box(3.1)), timestamp_ns=2).tracks) == 2
    tracker.reset(sequence_id="new")
    tracker.update(frame(box()), timestamp_ns=0)
    assert len(tracker.update(frame(box(name="truck")), timestamp_ns=1).tracks) == 2


def test_deterministic_ties_and_input_permutation():
    outputs = []
    for reverse in (False, True):
        tracker = ConstantVelocityTracker("s")
        boxes = (box(-1), box(1))
        tracker.update(frame(*((tuple(reversed(boxes))) if reverse else boxes)), timestamp_ns=0)
        observations = (box(0, y=-1), box(0, y=1))
        result = tracker.update(
            frame(*((tuple(reversed(observations))) if reverse else observations)), timestamp_ns=1
        )
        assert result.tracks[0].detection.center_xyz == (0, -1, 0)
        outputs.append(result.to_json())
    assert outputs[0] == outputs[1]


def test_crossing_with_explicit_motion_preserves_continuity():
    tracker = ConstantVelocityTracker("crossing", TrackerConfig(association_radius_m=0.6))
    tracker.update(frame(box(-2, velocity=(2, 0)), box(2, velocity=(-2, 0))), timestamp_ns=0)
    tracker.update(
        frame(box(-0.2, velocity=(2, 0)), box(0.2, velocity=(-2, 0))), timestamp_ns=900_000_000
    )
    result = tracker.update(
        frame(box(2, velocity=(2, 0)), box(-2, velocity=(-2, 0))), timestamp_ns=2_000_000_000
    )
    assert [t.detection.center_xyz[0] for t in result.tracks] == [2, -2]
    assert all(t.track_id in (1, 2) and t.hits == 3 for t in result.tracks)


def test_explicit_velocity_precedes_estimate_and_estimates_smooth():
    tracker = ConstantVelocityTracker(
        "s", TrackerConfig(association_radius_m=10, velocity_smoothing=0.25)
    )
    tracker.update(frame(box(velocity=(2, 1))), timestamp_ns=0)
    result = tracker.update(frame(box(4)), timestamp_ns=1_000_000_000)
    assert result.tracks[0].velocity_xy == (2.5, 0.75)
    assert tracker.update(frame(box(5, velocity=(9, 8))), timestamp_ns=2_000_000_000).tracks[
        0
    ].velocity_xy == (9, 8)


def test_zero_dt_never_divides_and_backward_failure_is_atomic():
    tracker = ConstantVelocityTracker("s")
    tracker.update(frame(box()), timestamp_ns=100)
    result = tracker.update(frame(box(1)), timestamp_ns=100)
    assert result.tracks[0].velocity_xy is None
    assert result.tracks[0].hits == 2
    with pytest.raises(ValueError, match="backwards"):
        tracker.update(frame(box()), timestamp_ns=99)
    assert tracker.state() is result
    with pytest.raises(ValueError, match="coordinate"):
        tracker.update(frame(box(), coordinate="sensor"), timestamp_ns=101)
    assert tracker.state() is result
    tracker.reset(sequence_id="reset-sequence")
    assert tracker.state() is None
    assert tracker.update(frame(box(), coordinate="sensor"), timestamp_ns=0).tracks[0].track_id == 1


@pytest.mark.parametrize("stamp", [-1, 1.2, True, None, 10**400])
def test_invalid_timestamps(stamp):
    with pytest.raises(ValueError, match="timestamp"):
        ConstantVelocityTracker("s").update(frame(), timestamp_ns=stamp)


@pytest.mark.parametrize(
    "change",
    [
        {"min_hits": 0},
        {"min_hits": True},
        {"max_missed": -1},
        {"association_radius_m": 0},
        {"association_radius_m": float("nan")},
        {"association_radius_m": float("inf")},
        {"velocity_smoothing": -0.1},
        {"velocity_smoothing": 1.1},
        {"class_aware": 1},
    ],
)
def test_invalid_policy(change):
    with pytest.raises((ValueError, TypeError)):
        TrackerConfig(**change)


def test_immutable_snapshots_and_frame_content_roundtrip():
    original = frame(box(1, velocity=(2, 3)))
    tracker = ConstantVelocityTracker("s")
    tracked = tracker.update(original, timestamp_ns=10)
    assert tracked.tracks[0].detection is original.detections[0]
    assert original.metadata == {"precomputed": True}
    assert TrackFrame.from_json(tracked.to_json()) == tracked
    assert tracked.to_json() == TrackFrame.from_json(tracked.to_json()).to_json()
    assert detection_frame_from_json(json.dumps(original.to_dict())) == original
    with pytest.raises(FrozenInstanceError):
        tracked.tracks[0].hits = 100
    timed = TimedDetectionFrame(original, 10)
    assert TimedDetectionFrame.from_json(json.dumps(timed.to_dict())) == timed


@pytest.mark.parametrize("mutation", ["timestamp_bool", "unknown", "nan", "duplicate", "schema"])
def test_strict_serialization_rejects_invalid_envelopes(mutation):
    result = ConstantVelocityTracker("s").update(frame(box()), timestamp_ns=1)
    payload = result.to_dict()
    if mutation == "timestamp_bool":
        payload["timestamp_ns"] = True
    elif mutation == "unknown":
        payload["unexpected"] = 1
    elif mutation == "nan":
        payload["tracks"][0]["detection"]["score"] = float("nan")
    elif mutation == "schema":
        payload["schema_version"] = "future"
    text = json.dumps(payload)
    if mutation == "duplicate":
        text = '{"sample_id":"duplicate",' + text[1:]
    with pytest.raises((ValueError, TypeError)):
        TrackFrame.from_json(text)


def test_streaming_consumes_only_requested_frames():
    events = []

    def source():
        for i in range(3):
            events.append(i)
            yield TimedDetectionFrame(frame(box(i)), i * 1_000_000_000)

    output = track_sequence(source(), sequence_id="s")
    assert events == []
    assert next(output).tracks[0].track_id == 1
    assert events == [0]
    assert len(list(output)) == 2


def test_cli_tracks_precomputed_jsonl_and_diagnoses_timestamp(tmp_path, capsys):
    path = tmp_path / "frames.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(TimedDetectionFrame(frame(box(i)), i * 1_000_000_000).to_dict())
            for i in range(2)
        ),
        encoding="utf-8",
    )
    assert main(["track", str(path), "--sequence-id", "s", "--min-hits", "2", "--json"]) == 0
    results = [TrackFrame.from_json(line) for line in capsys.readouterr().out.splitlines()]
    assert results[-1].tracks[0].lifecycle is TrackState.CONFIRMED
    path.write_text(json.dumps(frame(box()).to_dict()), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["track", str(path), "--sequence-id", "s"])
    assert "line 1" in capsys.readouterr().err


def test_tracking_and_cli_do_not_import_optional_frameworks(monkeypatch, tmp_path, capsys):
    attempted = []

    class Guard(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in {"torch", "scipy", "tensorrt", "pcdet", "spconv", "rclpy"}:
                attempted.append(fullname)
                raise AssertionError(fullname)

    guard = Guard()
    monkeypatch.setattr(sys, "meta_path", [guard, *sys.meta_path])
    tracker = ConstantVelocityTracker("s")
    tracker.update(frame(box()), timestamp_ns=0)
    with pytest.raises(SystemExit) as exc:
        main(["track", "--help"])
    assert exc.value.code == 0
    assert not attempted


def test_snapshot_rejects_missing_prediction_and_json_exponent_overflow():
    result = ConstantVelocityTracker("s").update(frame(box()), timestamp_ns=0)
    with pytest.raises(ValueError, match="predicted_center"):
        replace(result.tracks[0], predicted_center_xyz=None)
    with pytest.raises(ValueError, match="finite"):
        detection_frame_from_json('{"metadata":{"overflow":1e999}}')


def test_reset_requires_new_sequence_scope_and_config_is_readonly():
    tracker = ConstantVelocityTracker("s")
    tracker.update(frame(box()), timestamp_ns=0)
    with pytest.raises(ValueError, match="new sequence_id"):
        tracker.reset(sequence_id="s")
    with pytest.raises(AttributeError):
        tracker.config = TrackerConfig(min_hits=1)
