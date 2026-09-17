# CPU 3D tracking

The tracking package consumes precomputed `DetectionFrame` values. It runs no detector, changes no
scores, and makes no benchmark accuracy claim. Install the minimal CPU package; no SciPy or PyTorch
is required.

```python
from laserperception.tracking import ConstantVelocityTracker, TrackerConfig

tracker = ConstantVelocityTracker("my-sequence", TrackerConfig(min_hits=3, max_missed=2))
# frame is an existing DetectionFrame in a common fixed frame expressed in meters.
tracked = tracker.update(frame, timestamp_ns=explicit_acquisition_timestamp_ns)
```

Timestamps are explicit non-negative signed-64-bit integer nanoseconds in one clock domain. They may be equal;
at zero elapsed time the tracker keeps previous estimated velocity, or accepts explicitly supplied
detection velocity. Backwards time and changed coordinate-frame names fail without changing state.
`reset(sequence_id="new-sequence")` begins a new ID scope; reusing the active sequence ID is rejected. Same frame names alone do not prove a fixed
physical frame: callers must explicitly compensate ego motion before tracking. No transform is inferred.

The baseline predicts XY motion with constant velocity and holds Z constant. A detection's finite
`velocity_xy` takes priority. Otherwise matched observations estimate XY velocity using elapsed time
since the last accepted observation, including missed frames; estimates blend with the previous
velocity using `velocity_smoothing` (default 0.5). Before velocity is available, predictions retain the
last observed center and velocity remains `None`.

Association considers Euclidean 3D center distance in meters. Class ID and name must both agree by
default. Eligible pairs within `association_radius_m` (default 3) are globally sorted by distance,
track ID, then the existing deterministic DetectionFrame row order. Greedy selection accepts each
track/detection at most once. This is deterministic, but is not globally optimal. Class-agnostic
matching requires an explicit policy change.

A new observation creates a tentative track; `min_hits` accepted observations (default 3, cumulative)
confirm it. A miss marks a retained track lost. Reacquisition preserves its ID and restores tentative
or confirmed state according to total hits. `max_missed` misses (default 2) are retained; the next miss
retires the track. `min_hits=1` confirms on creation; `max_missed=0` retires on the first miss. IDs grow
monotonically and are never recycled within a sequence. Age counts updates since creation; hits count
accepted observations; missed counts consecutive misses.

Snapshots are immutable. Lost tracks retain the original last observed Detection3D, its source sample
and observation timestamp; `predicted_center_xyz` separately reports the estimated current center.
Every snapshot has the current explicit timestamp, tracker identity, config, and sequence/frame ID.
No detector output is rewritten to look like a new observation.

`track_sequence()` lazily consumes `TimedDetectionFrame` values. `TrackFrame.to_json()` and
`TrackFrame.from_json()` use a strict versioned envelope, stable ID ordering, and finite JSON numbers.
Deterministic serialization does not establish detector numeric determinism.

```console
laserperception track frames.jsonl --sequence-id my-sequence --min-hits 3 --json
python examples/tracking_sequence.py
```

Each input JSONL row contains `schema_version: "laserperception.timed-detection.v1"`, an explicit
`timestamp_ns`, and `frame` holding the existing DetectionFrame schema 1.0 representation. Construct
rows with `TimedDetectionFrame(frame, timestamp_ns).to_dict()`. CLI output is one TrackFrame JSON object
per input row; blank rows are skipped. Input is streamed and malformed lines identify their row.

Limitations include greedy suboptimal assignments, identity switches at ambiguous crossings, no box
IoU/appearance association, no covariance or learned motion, no ego-motion compensation, and no
cross-class matching by default. Identical coincident detections remain indistinguishable. Association
considers all active tracks and detections, so per-frame pair cost is quadratic; active history is
bounded by the miss policy, without retaining a full sequence. Crossing fixtures are synthetic
engineering tests, not tracking benchmark results.
