# M6b Protocol R2 — draft structural-engine revision

> **DRAFT — REQUIRES OWNER APPROVAL BEFORE M6b RESUMES**

## Prospective revision and preserved chronology

The original frozen M6b protocol remains failed at its engine-shape preflight. It reproduced
856/856 H10/H5 inputs, then the historical 30,000-voxel TensorRT profile rejected a valid
40,000-row `exact_fast` tensor before network execution. It produced zero KITTI network outputs and
zero detector predictions. Neither that failure nor its evidence is replaced by this draft.

Protocol R2 prospectively changes exactly one detector artifact:

- historical engine: TensorRT FP16 with a 30,000-voxel maximum profile;
- candidate engine: the same byte-identical ONNX network converted to a second TensorRT FP16
  artifact with a structural 40,000-voxel maximum profile.

This is not the same engine binary and exact output equality is not assumed. The candidate SHA256
is `2e790b1cdbdc1b88c2aafdc81b5921ebee152edd8408158f88437ae4dd1f3e7f`.

## Frozen candidate profile

| Binding | Minimum | Optimum | Maximum |
| --- | --- | --- | --- |
| `voxels` | `[4352, 64, 4]` | `[18207, 64, 4]` | `[40000, 64, 4]` |
| `num_points` | `[4352]` | `[18207]` | `[40000]` |
| `coors` | `[4352, 4]` | `[18207, 4]` | `[40000, 4]` |

Only the insufficient maximum changes. The optimum remains the historical nuScenes-derived 18,207
instead of a KITTI-derived statistic, so the candidate is not intentionally tactic-tuned to the
evaluation domain.

## M6b-R1 validation evidence

The candidate was built once from ONNX SHA256
`61ce22a8ca31498675c32576bfb94f0093d31dc95d2762f7254bf915a59ecc16` at build commit
`7feb5be8a2c529b55928a4bda180ae4bcb050cc7`. The serialized engine has the six expected bindings,
the exact profile above, and TensorRT device-memory size 1,602,800,640 bytes versus 1,212,340,736
bytes for the historical engine.

At clean validation commit `d1f534d79b85f6d67c54ebc70b99d7b92cd31413`:

- the exact frozen 20-sample nuScenes parity-v2 suite passed Stage 1 with unchanged thresholds;
- a same-session historical-engine versus candidate-engine characterization was recorded;
- an input-only census froze 12 H10 frames from non-evaluation drive
  `2011_09_30_drive_0016`, spanning 29,423–40,000 retained voxels;
- rewritten PyTorch FP32 versus candidate TensorRT FP16 passed unchanged parity-v2 Stage 1 on all
  12 frozen frames; and
- raw candidate outputs were SHA256-exact across five repeats on both the 40,000-voxel frame and
  the closest available full-history frame to OPT (29,423 voxels).

The evidence is tracked in [`benchmarks/m6b/engine`](../../benchmarks/m6b/engine). No network output
was produced for either evaluation drive during M6b-R1.

## Scientific definitions retained from frozen M6b

If the owner approves this draft, every other definition in [`M6B_PROTOCOL.md`](M6B_PROTOCOL.md)
remains unchanged:

- evaluation drives `2011_09_26_drive_0001` and `2011_09_26_drive_0091`;
- all 428 paired evaluation frames and their frozen H10/H5 inputs and hashes;
- ground-truth mapping, annotation field of view, class mapping, and neighbour-ignore rules;
- score threshold 0.25, metric definitions, preregistered hypotheses, and visualization selection;
- deterministic `exact_fast`, `max_voxels=40000`, unchanged voxel geometry, model, checkpoint,
  ONNX, postprocess, precision, and no target-domain tuning.

R2 does not authorize a second candidate, performance work, M6c, training, or release activity.
Until explicit owner approval is recorded, the 428-frame M6b characterization must not resume.
