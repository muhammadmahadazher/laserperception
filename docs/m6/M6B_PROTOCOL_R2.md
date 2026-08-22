# M6b Protocol R2 — owner-approved structural-engine revision

Status: **OWNER APPROVED AND FROZEN BEFORE EVALUATION PREDICTION**.

## Preserved chronology

The original M6b protocol at `16e2f7734061a5d0c2c2dec7b44f8b31e21591ae` remains failed at its
engine-shape preflight. Measurement `438e755d46f5768e429c1359ee99c353b325bad7` reproduced 856/856
H10/H5 inputs, then the historical 30,000-voxel TensorRT profile rejected valid 40,000-row
`exact_fast` input before network execution. It produced zero KITTI evaluation network outputs and
zero predictions. The blocker evidence SHA256 remains
`dfd595dcab5ce41e8846e128de85092c2c8f9d3f98b9aba99f488b03332ed2fb`.

M6b-R1 was separately preregistered at
`c3c4fd9faf41396ad5a7553757d222fc20981169`. It built one candidate at
`7feb5be8a2c529b55928a4bda180ae4bcb050cc7` and validated it at
`d1f534d79b85f6d67c54ebc70b99d7b92cd31413`; sanitized evidence was committed at
`b28a273694b2d943bb2fe8797001d9ff366cd640`. None of those prospective records rewrites the
original failure.

## Approved candidate

The sole prospective detector-artifact change is:

- historical engine: TensorRT FP16, maximum profile 30,000, SHA256
  `a005f75852097cd9b193750560b214cc3d5237ae9b6c106c7fca3d4fc348714b`;
- accepted candidate: TensorRT FP16, maximum profile 40,000, SHA256
  `2e790b1cdbdc1b88c2aafdc81b5921ebee152edd8408158f88437ae4dd1f3e7f`.

The candidate uses the byte-identical ONNX SHA256
`61ce22a8ca31498675c32576bfb94f0093d31dc95d2762f7254bf915a59ecc16` and checkpoint SHA256
`f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0`. Its input profile remains
MIN 4,352, OPT 18,207, MAX 40,000. OPT remains nuScenes-derived and was not retuned to KITTI.

TensorRT `getDeviceMemorySize` is 1,602,800,640 bytes for the candidate versus 1,212,340,736 bytes
for the historical engine, an increase of 390,459,904 bytes. This is compatibility metadata, not a
performance result.

## Accepted parity and repeatability gates

R1 established:

- Stage 1 PASS on the exact frozen 20-sample nuScenes parity-v2 suite;
- same-session historical-versus-candidate characterization;
- Stage 1 PASS on 12 frozen non-evaluation drive-0016 H10 frames spanning 29,423–40,000 voxels;
- exact five-run raw-output repeatability at 40,000 voxels and the closest available 0016 H10
  frame to OPT.

Before R2 evaluation, an input-only census of all 274 eligible drive-0016 H5 frames froze four
blind-band frames nearest 23k, 25k, 27k, and 29k. Their retained voxel counts were 23,488, 24,982,
26,981, and 29,011. Rewritten PyTorch FP32 versus candidate TensorRT FP16 passed unchanged
parity-v2 Stage 1: 16/16 exported detections, 15/15 high-confidence matches, and 100% pass fraction
for every continuous metric. Neither evaluation drive was used. Evidence is
[`structural_40k_h5_profile_gap_parity.json`](../../benchmarks/m6b/engine/structural_40k_h5_profile_gap_parity.json).

This closes the disclosed gap between the initial low-shape evidence (through approximately 22.5k)
and initial H10 evidence (29.4k–40k). No tolerance, model, profile, or input contract changed.

## Frozen evaluation

R2 authorizes exactly the existing corpus and scientific contract from
[`M6B_PROTOCOL.md`](M6B_PROTOCOL.md):

- drives `2011_09_26_drive_0001` frames 10–107 and
  `2011_09_26_drive_0091` frames 10–339;
- 428 current frames, each under H10 and H5, for 856 detector conditions;
- the already-validated model-ready hashes and sweep transforms in the frozen input ledger;
- deterministic `exact_fast`, `max_voxels=40000`, FP16 candidate engine, unchanged MMDeploy
  postprocess, score threshold 0.25, taxonomy, camera-FOV rule, neighbour ignores, GT geometry,
  oriented-BEV matching, IoU thresholds, hypotheses, range bins, and visualization rules.

H10 versus H5 remains a compound temporal-and-density history ablation. It changes temporal span,
time-lag values, point count, density, occupied pillars, and cap pressure simultaneously; no isolated
time-lag causal claim is permitted.

`time_lag` is an explicit detector input feature. KITTI H10 exposes the frozen model to lag values
extending to approximately 1.035 s, substantially beyond the temporal values represented by the
nuScenes training input, while H5 reduces the span to approximately 0.518 s. Temporal-span/
`time_lag` mismatch is therefore a motivated hypothesis for a future controlled study, not an
established cause. Point density and other accumulated-history effects remain confounded. A future
preregistered isolation experiment would need to hold temporal/history geometry fixed while
controlling density, or hold the point/pillar population fixed while manipulating only `time_lag`;
neither experiment is authorized or run here.

The five frozen H10 sentinels must pass ten exact repetitions for all raw outputs and final
`DetectionFrame` before the full run. Repeat #1 becomes the canonical H10 result, preventing an
unnecessary eleventh inference. The remaining conditions run in frame order, H10 then H5, with
atomic local checkpoints and fail-closed resume identity.

No evaluation definition may change after the first prediction. A fundamental defect requires a
stop with the partial run preserved. R2 does not authorize tuning, training, another engine,
performance optimization, M6c, ROS KITTI replay, or release activity.

## Post-measurement evidence packaging

This section is a non-normative repository-packaging record added after the completed measurement;
it does not revise the frozen R2 protocol, inputs, outputs, metrics, or conclusions.

Git tracks the compact canonical result at
[`kitti_raw_cross_domain_characterization.json`](../../benchmarks/m6b/results/kitti_raw_cross_domain_characterization.json)
and the compact 856-condition audit ledger at
[`pre_inference_input_ledger.json`](../../benchmarks/m6b/diagnostics/pre_inference_input_ledger.json).
The complete per-frame measurement artifact remains immutable external generated evidence under
logical name `kitti_raw_cross_domain_characterization_full.json`, size 41,987,113 bytes, SHA256
`87870b2aa0cc2a91d39331afc8154fdad0c8c796f1cabfb4f8530a3eb106de27`. The complete input ledger
remains immutable external evidence under logical name `pre_inference_input_ledger_full.json`, size
5,837,452 bytes, SHA256
`e25b3d62113cc7e8c1fcf736caa68b1ab698f965f007c758ff91d3e498ca6caa`. Neither full artifact is
stored in the final tracked tree.

The measurement commit remains `9159682fadfc069eeb70e07acb76dd0a929db98f`; the evidence-
packaging commit is `969ee69d06685025ca09794ef7e1ef33f2b892b7`; the later PR #11 squash-merge
commit is pending. At an appropriate future release, the immutable full artifacts may be published
as hash-pinned GitHub Release assets. No release or asset publication occurs during M6b. PR #11
must be squash-merged so the original large blobs in feature-branch history do not enter `main`;
a normal merge is prohibited.
