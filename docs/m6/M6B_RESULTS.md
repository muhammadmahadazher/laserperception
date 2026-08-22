# M6b frozen PointPillars characterization on KITTI Raw

Status: **M6b READY FOR REVIEW**. The owner-approved Protocol R2 characterization completed on the
entire frozen corpus. This is an offline cross-domain engineering study, not an official KITTI
benchmark and not a claim about ROS or live-sensor operation.

## Result in one paragraph

The unchanged nuScenes-trained PointPillars detector produced valid outputs for all 428 paired
KITTI Raw frames under H10 and H5. The strongest paired result was recall: Car increased from
0.242 under H10 to 0.727 under H5, and Pedestrian increased from 0.553 to 0.677. At the frozen
score threshold 0.25 and primary oriented BEV IoU 0.50, annotation-conditioned precision against
the available incomplete KITTI Raw tracklets was 0.100/0.155 for Car and 0.054/0.065 for
Pedestrian under H10/H5. H10 versus H5 is a compound temporal-and-density history ablation. The
improvement under H5 establishes strong sensitivity to the accumulated-history configuration,
but this experiment does not isolate temporal span, `time_lag`, point density, or another single
factor.

## Preserved failure and remediation chronology

The original protocol was frozen at `16e2f7734061a5d0c2c2dec7b44f8b31e21591ae`. Its clean
measurement at `438e755d46f5768e429c1359ee99c353b325bad7` reproduced all 856 H10/H5 inputs
exactly, then stopped before network execution when the historical 30,000-profile TensorRT engine
rejected a valid 40,000-voxel `exact_fast` input. The blocker record SHA256 is
`dfd595dcab5ce41e8846e128de85092c2c8f9d3f98b9aba99f488b03332ed2fb`. That run remains failed
with zero KITTI evaluation network outputs and zero predictions.

M6b-R1 was separately preregistered at `c3c4fd9faf41396ad5a7553757d222fc20981169`. It built one
prospective candidate from the byte-identical M2 ONNX, with the historical 4,352 minimum and
18,207 optimum and a structural 40,000 maximum. R1 then passed the frozen 20-sample nuScenes
parity-v2 suite, same-session old-versus-new characterization, non-evaluation KITTI H10 parity,
40,000-shape repeatability, and exact 856/856 WSL2 H10/H5 input reproduction. None of those accepted
campaigns was rerun for R2.

The remaining 22,547–29,422 profile band was closed before evaluation using H5 from non-evaluation
drive `2011_09_30_drive_0016`. Its 274-frame input-only census ranged from 23,488 to 40,000 retained
voxels: 125 frames were at or below 30,000, 149 exceeded 30,000, and 71 reached 40,000. The frozen
four frames were 131/23,488, 109/24,982, 101/26,981, and 193/29,011 voxels. Rewritten PyTorch FP32
versus candidate TensorRT FP16 passed the unchanged parity-v2 Stage 1 gate: 16/16 exported
detections, 15/15 high-confidence matches, every per-metric pass fraction 1.0, and no continuous
tolerance outlier.

Only then was [`M6B_PROTOCOL_R2.md`](M6B_PROTOCOL_R2.md) committed at
`9159682fadfc069eeb70e07acb76dd0a929db98f`. The first evaluation prediction occurred afterward
from the same clean commit. This commit is both the final preregistration boundary and the
measurement implementation identity.

## Execution and repeatability

The corpus contains frames 10–107 from `2011_09_26_drive_0001` (98) and 10–339 from
`2011_09_26_drive_0091` (330): 428 current frames, each evaluated as H10 and H5, for 856 conditions.
All 856 completed. Each condition verified its identity, frozen model-ready SHA, and candidate
engine SHA before inference. Atomic local checkpoints were used; the five accepted sentinel H10
records were loaded as their canonical corpus outputs, and no completed condition was unnecessarily
rerun.

Each H10 sentinel ran ten times before the corpus sweep. All ten hashes were identical for each raw
output and final `DetectionFrame`:

| Frame | `cls_score` SHA256 | `bbox_pred` SHA256 | `dir_cls_pred` SHA256 | `DetectionFrame` SHA256 |
|---|---|---|---|---|
| 0001/10 | `cd065f7381305c14ffb1353d9c170a8be4a65d0227e0db8ce867731b006a7242` | `f19ba02133b698bdac6a56408eb0a3378b199517d965e43ac4899e7c925ba678` | `dd7a096abcc5591adeb59444304aa1430a66b23507c72b3c329fe2fa03dcd5ac` | `565cdd71123b78f2fd5b23456702c7ef0e6410d6f1195a36a0a02c1b4f47b132` |
| 0001/83 | `dae2ef1c872016920d096a5c6256477936a5515946eadb3e65278389c4777c63` | `ebd67cb18f2cbcde0443c20ca20b68df7008bd368ad0cd524b3ed7e851209524` | `098097306c2c69a783427894c6070ba0293d7f771aeffe2c70c556bc802bbe2c` | `cd18f732c3bf936d4f348eed8f89a99d76bbc2685f16539af8203b9c2eba1605` |
| 0001/11 | `27ccb58b20c84744b31a0594dc4e72282be7da09687e4d0d1ec76a287779cdb2` | `dc276e48a923626119b095111964678b62bec2ad52ceb479a30853862ff65fda` | `be11dcb1b7b2cd86e1bdfdaf6290ed7e232473c65b746f346cb4edc2ce2a7714` | `085aeb067ec7b8b4297df77f44067c757e41fe33d81da9ac418948fd9f4dc367` |
| 0001/15 | `bfeac9f14ae8dc5abd4e4786f8e23342769fe778051923252c851e71b8f3e5f5` | `8a576541db1b680e5f84c611bf67301f11b104cb2b8b0b75d6cf7938b4e453c2` | `a2390bd2194e3e2bc3c6e8c62d26113cbb7c7ba7df4dba0ffce996a72c895024` | `a737f4fb8e54ee63a9e6d40936b40936bc486b3c7b0bca1ffccb6d7ef88a174c` |
| 0091/10 | `4daa22d58a500baad6f4d7c56a5525b6d692166675e266216830f6a2683cf3c2` | `6207f4a6d7510655cc08096a9f6469f918195fdbe35f516173f391e275740f0e` | `ba436a9635c6e23d5b42e6008f2a4dbcebc1b5c9340f3a6b6dbb7daa7874ac02` | `536cd45814100ca8cd3389e734fabdf44c98e8bed81ebd60ca42ec9e81cf661f` |

## Annotation and scoring contract

The frozen reference-camera-0 FOV contains 66 eligible Car and 396 eligible Pedestrian poses. Car
maps only to detector `car`; Pedestrian maps only to `pedestrian`. Van and Person (sitting) are
neighbour-ignore classes. Truck, Cyclist, Tram, and Misc remain unmapped. No Raw `DontCare` boxes
are synthesized, and precision is never claimed for the full 360-degree LiDAR field.

Ground-truth conversion uses the frozen bottom/contact-centre to geometric-centre adjustment,
`(h,w,l)` to `(l,w,h)`, the fixed KITTI-to-model basis, and `yaw_model = yaw_kitti + pi/2` with
normalization. The analytic fixture passed exactly before aggregation.

The following operating results use score at least 0.25 and oriented BEV IoU at least 0.50.
Precision is annotation-conditioned against the available incomplete KITTI Raw tracklets; it is a
conservative observed quantity under this frozen annotation protocol, not a true physical
false-positive rate. AP is the preregistered all-points score-ranked PR-envelope area over the
postprocessed population and carries the same annotation caveat. It is a LaserPerception M6b
Raw-tracklet characterization, not KITTI benchmark AP.

| Condition | Class | TP | FP | FN | Ignored | Annotation-conditioned precision | Recall | F1 | Raw-tracklet AP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H10 | Car | 16 | 144 | 50 | 6 Van | 0.100 | 0.242 | 0.142 | 0.0601 |
| H5 | Car | 48 | 261 | 18 | 8 Van | 0.155 | 0.727 | 0.256 | 0.2424 |
| H10 | Pedestrian | 219 | 3,831 | 177 | 0 | 0.054 | 0.553 | 0.099 | 0.0865 |
| H5 | Pedestrian | 268 | 3,868 | 128 | 0 | 0.065 | 0.677 | 0.118 | 0.1269 |

At IoU 0.30/0.50/0.70, TP counts were respectively 16/16/16 for H10 Car, 48/48/44 for H5 Car,
280/219/54 for H10 Pedestrian, and 316/268/76 for H5 Pedestrian. Person (sitting) ignored 4 H10 and
3 H5 Pedestrian predictions at IoU 0.30, but none at the primary 0.50 gate. At score 0.25, Car had
661 H10 and 944 H5 predictions outside the annotation FOV; Pedestrian had 8,540 and 7,610. Across
all detector classes, the corresponding outside-FOV counts were 82,591 and 81,233. Those outputs
are excluded from scored FP rather than mislabeled as hallucinations.

## Range and track characterization

| Condition | Class | 0–20 m TP/GT (recall) | 20–35 m TP/GT (recall) | 35–50 m TP/GT (recall) |
|---|---|---|---|---|
| H10 | Car | 12/27 (0.444) | 4/30 (0.133) | 0/6 (0.000) |
| H5 | Car | 27/27 (1.000) | 17/30 (0.567) | 4/6 (0.667) |
| H10 | Pedestrian | 168/288 (0.583) | 51/106 (0.481) | 0/2 (0.000) |
| H5 | Pedestrian | 191/288 (0.663) | 75/106 (0.708) | 2/2 (1.000) |

Three eligible Car poses lie outside the frozen 0–50 m range bins; the reported class denominator
remains 66. Across 11 Car tracks, aggregate detected-frame continuity was 16/66 (0.242) for H10 and
48/66 (0.727) for H5; median per-track continuity was 0.250 and 0.778. Across 41 Pedestrian tracks,
the aggregate figures were 219/396 (0.553) and 268/396 (0.677), with median per-track continuity
0.500 and 0.750. The longest hit/miss runs were 2/3 and 2/2 for Car, and 4/4 for Pedestrian under
both conditions. These are labelled-track summaries, not tracking output.

## H10 versus H5 compound ablation

At the primary gate, 16 Car poses were detected by both conditions, 32 were gained in H5, and 18
were missed by both; none was lost in H5. For Pedestrian, 204 were detected by both, 64 were gained
in H5, 15 were lost in H5, and 113 were missed by both. Relative to H5, H10 added a median 606,146
points and 5,891 candidate pillars per frame.

These paired changes do not isolate time lag. H10 is current plus ten history sweeps; H5 is current
plus five. The full generated artifact's field named `history_sweep_count` records the ten frozen
source transform records made available to reconstruction for both conditions. It is a
provenance-ledger count, not the consumed H5 depth. H5 consumption is fixed by the builder
configuration and verified by exact H5 model-ready hashes, six distinct time-lag values, a maximum
0.518104 s span, and its separate point/pillar summaries.

`time_lag` is an explicit detector input feature. KITTI H10 exposes the frozen model to lag values
extending to approximately 1.035 s, substantially beyond the temporal values represented by the
nuScenes training input. H5 reduces the span to approximately 0.518 s. This makes temporal-span/
`time_lag` mismatch a motivated hypothesis for future controlled study, not an established cause;
point density and related history effects remain confounded. A future preregistered isolation
experiment would need either to hold H10 temporal/history geometry fixed while controlling point
density, or preserve the H10 point/pillar population while manipulating only the `time_lag`
representation. Neither experiment was run, and these results do not select a production history
configuration.

## Input shift and capacity

| Statistic | H10 | H5 |
|---|---:|---:|
| Mean points | 1,330,654 | 726,561 |
| Median points | 1,334,420 | 728,279 |
| Time span, median | 1.035418 s | 0.517701 s |
| Distinct time-lag values | 11 | 6 |
| Candidate pillars, mean | 31,192 | 25,320 |
| Candidate pillars, median | 30,699 | 24,792 |
| Candidate pillars, range | 19,023–43,810 | 14,957–35,157 |
| Retained pillars, maximum | 40,000 | 35,157 |
| Overflow frames | 68/428 (15.89%) | 0/428 (0%) |
| Discarded pillars, total | 124,696 | 0 |
| Discarded pillars, maximum frame | 3,810 | 0 |

The candidate engine SHA256 is
`2e790b1cdbdc1b88c2aafdc81b5921ebee152edd8408158f88437ae4dd1f3e7f`, with MIN/OPT/MAX
4,352/18,207/40,000. Its TensorRT device-memory requirement is 1,602,800,640 bytes, an increase of
390,459,904 bytes (about 32.2%) over the immutable historical engine's 1,212,340,736 bytes. The
candidate was not rebuilt and no performance campaign was run.

## Truncation, sweep provenance, and target overlap

For H10 overflow frames, the preregistered Pearson 2x12 sector homogeneity statistic was
19,354.05 versus the 24.72497 critical value at alpha 0.01, so sector homogeneity was rejected.
The largest/smallest nonzero sector drop-rate ratio was 5.52. The additional practical gate did
not pass because the aggregate highest-drop sector was each frame's highest in only 19.12% of
overflow frames, below the frozen greater-than-50% rule. Therefore the preregistered combined
classification is **not spatially non-uniform truncation**; the sector differences remain
descriptive. H5 had no discarded pillars, so the test was inapplicable.

First-touch provenance measured, rather than assumed, the deterministic retention behavior. H10
discarded pillars were touched first only by the three oldest sweeps: sweep 8 contributed 5,165
(4.14%), sweep 9 contributed 48,096 (38.57%), and sweep 10 contributed 71,435 (57.29%). Every
pillar first touched by the current sweep or history sweeps 1–7 was retained. H5 discarded none.
This shows that first-occurrence ordering protected recent/current first touches in this corpus;
it does not establish a detector-quality cause.

Of 66 Car targets, 65 overlapped at least one retained pillar under both conditions; four H10 Car
targets overlapped discarded pillars. All 396 Pedestrian targets overlapped retained pillars; four
also overlapped H10 discarded pillars. No H5 target overlapped discarded pillars. H10 Car recall
was 0.231 on overflow frames versus 0.245 on non-overflow frames; H10 Pedestrian recall was 0.638
versus 0.535. These mixed associations, plus the compound H10/H5 intervention, do not support a
causal statement that dropped pillars caused misses or that removing overflow caused H5's gains.
The 40k cap was not supported as the primary explanation for the corpus-wide H10 deficit. This
does not establish that overflow has no effect in general or on another dataset.

## Frozen visualizations

All images are real outputs from the completed offline KITTI Raw evaluation. Red boxes are eligible
ground truth, purple boxes are detector outputs, and orange points in truncation views mark
discarded-pillar support. They are not ROS, live-LiDAR, or real-time-sensor evidence.

![Paired H10 and H5 output](../assets/m6b/paired_H10_H5.png)

The five preregistered sentinels are
[`0001/10`](../assets/m6b/sentinel_canonical_first_full_history.png),
[`0001/83`](../assets/m6b/sentinel_canonical_lower_median_candidate_pillars.png),
[`0001/11`](../assets/m6b/sentinel_highest_overflow_full_history_m6a_sentinel.png),
[`0001/15`](../assets/m6b/sentinel_non_overflow_closest_to_40000.png), and
[`0091/10`](../assets/m6b/sentinel_second_drive_first_eligible_pedestrian.png).

Frozen metric selection produced
[`Car best`](../assets/m6b/car_best.png),
[`Car median`](../assets/m6b/car_median.png),
[`Car worst`](../assets/m6b/car_worst.png),
[`highest overflow`](../assets/m6b/highest_overflow.png),
[`closest non-overflow comparison`](../assets/m6b/non_overflow_comparison.png), and the paired H10/H5
frame above. No example was manually substituted for visual quality.

## Evidence identity and packaging

The tracked canonical result is now a compact summary:
[`kitti_raw_cross_domain_characterization.json`](../../benchmarks/m6b/results/kitti_raw_cross_domain_characterization.json),
111,529 bytes, SHA256
`b9d47120aabb38733d79f987f16277ebab626d2b7c13159c40c30a68b76c1d26`. It preserves the frozen
population, denominators, metrics, engine and artifact identities, structural validation,
repeatability hashes, mechanism summaries, visualization hashes, and scope guards without
embedding the complete per-frame detection payload.

The tracked compact input audit ledger is
[`pre_inference_input_ledger.json`](../../benchmarks/m6b/diagnostics/pre_inference_input_ledger.json),
594,264 bytes, SHA256
`2413224808b0140856a0e00f884c53bc9b49ec6b545172f5e7fa57d00802dc15`. It retains all 856 frozen
condition identities, model-ready hashes, source/history identities, point and voxel counts,
compact lag identities, and completion status.

The immutable full per-frame measurement artifact is external generated evidence, intentionally
excluded from Git history. Its logical name is
`kitti_raw_cross_domain_characterization_full.json`, its size is 41,987,113 bytes, and its SHA256
is `87870b2aa0cc2a91d39331afc8154fdad0c8c796f1cabfb4f8530a3eb106de27`. The immutable full
pre-inference ledger is likewise external; its logical name is
`pre_inference_input_ledger_full.json`, its size is 5,837,452 bytes, and its SHA256 is
`e25b3d62113cc7e8c1fcf736caa68b1ab698f965f007c758ff91d3e498ca6caa`. At an appropriate future
release, these hash-pinned artifacts may be attached as GitHub Release assets for independent
download and verification. They are not published during M6b.

The inference measurement and final preregistration identity remains
`9159682fadfc069eeb70e07acb76dd0a929db98f`. The evidence-packaging commit is
`969ee69d06685025ca09794ef7e1ef33f2b892b7`; it did not run inference or change any measurement.
The eventual squash-merge commit is pending owner-approved squash merge of PR #11. A normal merge
is prohibited because the earlier feature-branch history contains the original large generated
blobs.

## Limitations

The characterization is limited to two official KITTI Raw drives, their incomplete Raw tracklet
annotations, the reference-camera FOV, the frozen nuScenes-trained PointPillars model, and one
pinned software/GPU environment. It does not measure official KITTI AP, generalization beyond the
frozen corpus, safety, latency, or deployment performance. No threshold, model, ONNX, checkpoint,
precision, taxonomy, matching rule, `exact_fast` behavior, voxel geometry, or `max_voxels` was
changed. No training, tuning, ROS KITTI replay, M6c work, M5 work, or release activity occurred.

M6a is complete. M6b is complete and ready for review. M6 remains in progress because M6c is not
started and requires separate owner authorization. M5 remains conditional and inactive.
