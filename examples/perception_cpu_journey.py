"""Generate a reusable synthetic CPU fixture; no detector or benchmark claim."""

import argparse
from dataclasses import replace
from pathlib import Path

import laspy
import numpy as np

from laserperception.data import inspect_input, load_input, registry
from laserperception.detection.serialization import TimedDetectionFrame
from laserperception.detection.types import Detection3D, DetectionFrame
from laserperception.perception.contracts import CoordinateContract
from laserperception.semantic import (
    EXPERIMENT_001_TAXONOMY,
    evaluate_semantics,
    save_semantic_frame,
)
from laserperception.semantic.datasets import ground_truth_from_point_cloud
from laserperception.tracking import track_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=False)
    path = args.output_directory / "synthetic-dales.las"
    las = laspy.LasData(laspy.LasHeader(point_format=3, version="1.2"))
    las.x = [1, 2, 3]
    las.y = [0, 0, 0]
    las.z = [0, 0, 0]
    las.classification = [1, 8, 3]
    las.write(path)
    print("Adapters:", ", ".join(m.adapter_id for m in registry.list_adapters()))
    print(inspect_input(path, model_id="dsvt-pillar-transfusion-m8").to_json())
    loaded = load_input(path, adapter_id="dales")
    coordinates = CoordinateContract(
        "synthetic_map",
        "right",
        ("east", "north", "up"),
        "metre",
        "not_applicable",
        ("not_applicable",) * 3,
        "not_applicable",
        "synthetic fixture",
    )
    gt = ground_truth_from_point_cloud(
        loaded.cloud,
        sample_id=loaded.sample_id,
        frame_id="map",
        coordinates=coordinates,
        taxonomy=EXPERIMENT_001_TAXONOMY,
        mapping="dales",
    )
    prediction = replace(
        gt,
        class_ids=np.array([0, 1, -2], dtype=np.int64),
        provenance=replace(gt.provenance, operation="synthetic fixture prediction"),
    )
    save_semantic_frame(gt, args.output_directory / "ground-truth.json", inline=True)
    save_semantic_frame(prediction, args.output_directory / "prediction.json", inline=True)
    print("Synthetic semantic metrics:", evaluate_semantics(prediction, gt).to_json())
    inputs = []
    for i in range(3):
        box = Detection3D((float(i), 0, 0), (4, 2, 1.5), 0, 0.9, 0, "car", (1, 0))
        frame = DetectionFrame((box,), f"synthetic:{i}", "fixed-world-meters")
        inputs.append(TimedDetectionFrame(frame, i * 1_000_000_000))
    with (args.output_directory / "detections.jsonl").open("x", encoding="utf-8") as stream:
        import json

        for item in inputs:
            stream.write(json.dumps(item.to_dict(), sort_keys=True, allow_nan=False) + "\n")
    for tracked in track_sequence(iter(inputs), sequence_id="synthetic-journey"):
        print("Synthetic track IDs:", [track.track_id for track in tracked.tracks])


if __name__ == "__main__":
    main()
