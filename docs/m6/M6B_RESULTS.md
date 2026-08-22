# M6b frozen-engine input-profile blocker

Status: **M6b NOT READY**. This is a failed preflight record, not a detector result and not an
official KITTI benchmark result.

## What passed

The prospective portability correction was frozen at protocol commit
`16e2f7734061a5d0c2c2dec7b44f8b31e21591ae`. From clean measurement commit
`438e755d46f5768e429c1359ee99c353b325bad7`, WSL2 reproduced the frozen model-ready inputs exactly
for all 428 frames under both H10 and H5: 856/856 hashes, history lists, point counts, and time-lag
sets passed before backend initialization.

The source KITTI files were byte-identical across the Windows oracle and WSL2 measurement staging.
The unchanged `MultiSweepBuilder` consumed the canonical float32 sweep transforms recorded before
the first detector prediction.

## Failed frozen-engine gate

The first repeatability sentinel, `2011_09_26_drive_0001/0000000010` H10, produced 41,437 candidate
pillars and retained the frozen `max_voxels=40000`. The exact-fast input therefore had shape
`(40000, 64, 4)`. The unchanged TensorRT engine's only optimization profile accepts at most
`(30000, 64, 4)` voxels, with matching 30,000-row maxima for `num_points` and `coors`. The TensorRT
wrapper rejected the shape before network execution.

This is not an isolated sentinel: 218/428 H10 frames and 174/428 H5 frames retain more than 30,000
voxels. Sixty-eight H10 frames reach 40,000. The frozen engine is therefore incompatible with a
material part of the preregistered KITTI input corpus.

## Scientific boundary

No raw TensorRT output, postprocessed score, predicted box, DetectionFrame, metric, PR curve, AP,
or detector visualization was generated. The preregistration was not contaminated by predictions.
The failure is recorded in
[`failed_engine_shape_preflight.json`](../../benchmarks/m6b/diagnostics/failed_engine_shape_preflight.json).

The authorized protocol forbids truncating the input, changing `max_voxels`, changing voxel
geometry, rebuilding/replacing the engine, or skipping incompatible frames. Any path forward would
require a separate owner-reviewed prospective protocol and detector-artifact decision. M6b remains
incomplete; M6c and M5 remain inactive.
