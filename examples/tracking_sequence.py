"""Synthetic CPU engineering example: no detector and no benchmark claim."""

from laserperception.detection.types import Detection3D, DetectionFrame
from laserperception.tracking import ConstantVelocityTracker


def main() -> None:
    tracker = ConstantVelocityTracker("synthetic-moving-car")
    for index in range(5):
        detection = Detection3D((float(index), 0.0, 0.0), (4.0, 2.0, 1.5), 0.0, 0.9, 0, "car")
        frame = DetectionFrame((detection,), f"synthetic-{index}", "fixed-world-meters")
        print(tracker.update(frame, timestamp_ns=index * 1_000_000_000).to_json())


if __name__ == "__main__":
    main()
