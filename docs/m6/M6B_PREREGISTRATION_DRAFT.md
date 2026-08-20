# M6b characterization preregistration draft — not active

M6b has not started. This draft exists so the later detector study cannot choose questions after
viewing predictions. It requires owner/reviewer approval after M6a.

Frozen inputs proposed for review:

- official KITTI Raw drives `2011_09_26_drive_0001`, `0017`, and `0019`;
- unchanged nuScenes-trained PointPillars checkpoint, ONNX, TensorRT engine, `exact_fast`, voxel
  geometry, class mapping, and 0.25 display threshold;
- the M6a timestamp, pose, model-frame alignment, and offline-oracle bytes;
- history 10 as primary input and at most one separately preregistered history-5 ablation;
- full score distributions and PR curves rather than a single attractive screenshot;
- BEV IoU at levels fixed before predictions;
- range, occlusion, heading-error, input-density, occupied-pillar, and `max_voxels` engagement
  slices;
- ROS PointCloud2/tf2 reconstruction must reproduce the M6a offline oracle before detector use.

Owner review remains required for:

- the exact KITTI-tracklet to nuScenes taxonomy mapping;
- treatment of `Van`, `Truck`, `Tram`, seated persons, `Misc`, and especially `Cyclist` versus
  `bicycle`/`motorcycle`;
- IoU levels, matching policy, ignored labels, and coverage denominators;
- the exact characterization-frame set and any visualization selection.

No KITTI detector, ground-truth matching, PR/AP computation, predicted boxes, scores, ROS replay,
or visualization was run for this draft.
