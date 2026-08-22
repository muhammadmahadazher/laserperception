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

The frozen protocol forbade truncating the input, changing `max_voxels`, changing voxel geometry,
rebuilding/replacing the engine, or skipping incompatible frames. The owner subsequently authorized
the separate prospective M6b-R1 remediation below. The original failure and zero-prediction status
remain unchanged.

## Prospective M6b-R1 structural remediation

M6b-R1 built one distinct candidate engine from the byte-identical M2 ONNX. It retained the
historical 4,352 minimum and 18,207 optimum while expanding only the profile maximum from 30,000 to
the voxelizer's structural 40,000 ceiling. The original engine remains the canonical artifact for
M2 through v0.2.0.

The candidate engine passed serialized profile and binding inspection. Its SHA256 is
`2e790b1cdbdc1b88c2aafdc81b5921ebee152edd8408158f88437ae4dd1f3e7f`; it requires 1,602,800,640
bytes from TensorRT's `getDeviceMemorySize`, 390,459,904 bytes more than the historical engine's
1,212,340,736 bytes.

The frozen 20-sample nuScenes parity-v2 suite passed at validation commit
`d1f534d79b85f6d67c54ebc70b99d7b92cd31413`. All preregistered per-metric Stage 1 gates passed;
7/752 high-confidence matches exceeded at least one continuous tolerance and remain included in the
metric denominators. Same-session old-versus-new characterization found different raw values and
three continuous final-detection outliers among 754 high-confidence matches, while both exported
885 detections and the characterization satisfied the unchanged Stage 1 checks. This comparison is
diagnostic, not an exact-equality requirement.

An input-only census of all 269 eligible H10 frames from non-evaluation drive
`2011_09_30_drive_0016` found 29,423–40,000 retained voxels; 251 exceeded 30,000 and 108 reached
40,000. The deterministic frozen 12-frame set spans 29,423–40,000. Rewritten PyTorch FP32 versus
candidate TensorRT FP16 then passed the unchanged parity-v2 Stage 1 gate: 42/42 exported detections,
39/39 high-confidence matches, and no continuous-tolerance failures. Neither evaluation drive was
used.

Candidate raw outputs were bit-exact across five repeats on the 40,000-voxel frame and across five
repeats on frame 114 at 29,423 voxels. That second frame is the closest available full-history 0016
frame to the frozen 18,207 optimum; the census minimum is 29,423.

Sanitized tracked evidence is under [`benchmarks/m6b/engine`](../../benchmarks/m6b/engine).

## Owner-approved Protocol R2 boundary

The owner approved the structural candidate subject to one final non-evaluation H5 parity gate in
the uncovered 22,547–29,422-voxel interval. An input-only census of 274 eligible drive-0016 H5
frames froze frames 131, 109, 101, and 193 at 23,488, 24,982, 26,981, and 29,011 voxels. The
unchanged parity-v2 Stage 1 gate passed with 16/16 exported detections, 15/15 high-confidence
matches, and no continuous-tolerance failure.

[`M6B_PROTOCOL_R2.md`](M6B_PROTOCOL_R2.md) is therefore the final prospective evaluation contract.
The original protocol and failure remain preserved. M6c and M5 remain inactive.
