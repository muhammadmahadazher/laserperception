# M6b characterization preregistration draft — complete but not active

M6b has not started. This complete draft exists so the later detector study cannot choose questions
after viewing predictions. It requires explicit owner/reviewer authorization after M6a; nothing in
this file activates detector, ROS, matching, or visualization work.

Frozen inputs proposed for review:

- official KITTI Raw drives `2011_09_26_drive_0001`, `0017`, and `0019`;
- unchanged nuScenes-trained PointPillars checkpoint, ONNX, TensorRT engine, `exact_fast`, voxel
  geometry, class mapping, and 0.25 display threshold;
- the M6a timestamp, pose, model-frame alignment, and offline-oracle bytes;
- history 10 as primary input and at most one separately preregistered history-5 ablation;
- full score distributions and PR curves rather than a single attractive screenshot;
- BEV IoU levels, matching rules, ignored classes, and denominators frozen before predictions;
- range bands, KITTI occlusion/truncation states, heading-error distributions, input-density,
  candidate occupied-pillar counts, and `max_voxels` engagement slices;
- ROS PointCloud2/tf2 reconstruction must reproduce the M6a offline oracle before detector use.

The primary proposed detector setting remains the frozen v0.2 history-10 path and display/base
threshold 0.25. One history-5 reconstruction ablation may be admitted only if its frame set and
comparison rule are frozen before any detector result. No threshold search or presentation-only
frame selection is permitted.

Owner review remains required for:

- the exact KITTI-tracklet to nuScenes taxonomy mapping (the canonical drive has 12 `Car`, two
  `Cyclist`, and one `Tram` tracklets);
- treatment of `Van`, `Truck`, `Tram`, seated persons, `Misc`, and especially `Cyclist` versus
  `bicycle`/`motorcycle`;
- IoU levels, matching policy, ignored labels, and coverage denominators;
- the exact characterization-frame set and any visualization selection.

The canonical drive's tracklet XML covers frames 0-107, so a future approved study can use the same
data as M6a. Availability does not resolve semantic comparability: especially `Cyclist` must not be
silently mapped to either nuScenes `bicycle` or `motorcycle`.

No KITTI detector, ground-truth matching, PR/AP computation, predicted boxes, scores, ROS replay,
or visualization was run for this draft.
