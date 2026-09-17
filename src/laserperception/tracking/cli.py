"""Streaming CPU tracking of explicitly timed precomputed detections."""

from __future__ import annotations

import argparse
from pathlib import Path

from laserperception.detection.serialization import TimedDetectionFrame

from .config import TrackerConfig
from .tracker import ConstantVelocityTracker


def configure_track(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path, help="timed DetectionFrame JSONL")
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--max-missed", type=int, default=2)
    parser.add_argument("--association-radius-m", type=float, default=3.0)
    parser.add_argument("--velocity-smoothing", type=float, default=0.5)
    parser.add_argument("--class-agnostic", action="store_true")
    parser.add_argument("--json", action="store_true", help="JSONL is the default output")


def run_track(args: argparse.Namespace) -> int:
    tracker = ConstantVelocityTracker(
        args.sequence_id,
        TrackerConfig(
            args.min_hits,
            args.max_missed,
            args.association_radius_m,
            not args.class_agnostic,
            args.velocity_smoothing,
        ),
    )
    with args.input.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = TimedDetectionFrame.from_json(line)
                result = tracker.update(value.frame, timestamp_ns=value.timestamp_ns)
            except (ValueError, TypeError) as error:
                raise ValueError(f"line {line_number}: {error}") from error
            print(result.to_json())
    return 0
