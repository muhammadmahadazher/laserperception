# M3B-V1 hard-voxelization diagnostic

Status: **diagnostic complete; experimental fast voxelizer not adopted.**

The frozen result is
[`diagnostics/voxelization_v1_ad0d38b.json`](diagnostics/voxelization_v1_ad0d38b.json), measured at
exact implementation commit `ad0d38b6e926f3a03b471c192d3e815cd07d34d1`. Its SHA256 is
`98315416fa148a52ed14f734f923661cb70a70c8c032549fe54bd0dbf4354423`. This is diagnostic
evidence, not canonical performance and not a production configuration.

The protocol was frozen before measurement in
[`configs/detection/m3b_voxelization_fidelity_v1.yaml`](../../configs/detection/m3b_voxelization_fidelity_v1.yaml)
(SHA256 `03422de2d59e7bffae75eac73dd7c9dd47e40bd1e9daba6588762cdfdbf80019`). It used 20 warmups
and 100 measurements per timing boundary, plus 30 repeatability runs each for W1 and W2.

## Scope and immutable assets

The experiment changed only an in-memory clone of the upstream MMCV hard voxel layer from
`deterministic=True` to `deterministic=False`. It did not edit the official config or live runtime.
The checkpoint, ONNX, and engine SHA256 values remained respectively
`f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0`,
`61ce22a8ca31498675c32576bfb94f0093d31dc95d2762f7254bf915a59ecc16`, and
`a005f75852097cd9b193750560b214cc3d5237ae9b6c106c7fca3d4fc348714b`.

Runtime settings were:

| Setting | Official reference | Experimental candidate |
|---|---:|---:|
| Type | hard | hard |
| Voxel size | `[0.25, 0.25, 8.0]` m | unchanged |
| Point-cloud range | `[-50, -50, -5, 50, 50, 3]` m | unchanged |
| Max points per voxel | 64 | unchanged |
| Max voxels, train/test | 30,000 / 40,000 | unchanged |
| Deterministic | true | false |
| Production status | current official path | experimental only |

The pinned deterministic MMCV implementation performs a per-point search over prior point
coordinates and serial voxel-number assignment. The measurements below establish that the hard
voxel-layer call itself, rather than center creation or coordinate padding, dominates full-history
preprocessing.

## Deterministic preprocessing decomposition

All values are synchronized wall-clock milliseconds. `Sum B–F` adds cast/transfer, hard layer,
centers, padding, and final cat/contiguous/bookkeeping for each measured iteration. The separately
measured official complete-preprocessing median is within 6.21% for W0, 0.55% for W1, and 0.84% for
W2, so the decomposition reconciles with the official boundary.

### W0 — index 0, 33,587 points, scene start / zero history

| Stage | Mean | Median | P95 | Min | Max | Population std. |
|---|---:|---:|---:|---:|---:|---:|
| A. CPU model-ready batch | 0.891 | 0.850 | 1.216 | 0.708 | 1.777 | 0.154 |
| B. `cast_data` / CPU→CUDA | 0.627 | 0.577 | 0.943 | 0.403 | 1.311 | 0.156 |
| C. Hard voxel layer | 7.564 | 7.493 | 8.349 | 6.754 | 8.385 | 0.437 |
| D. Voxel centers | 1.222 | 1.138 | 1.788 | 0.630 | 2.437 | 0.385 |
| E. Coordinate batch padding | 0.261 | 0.210 | 0.557 | 0.107 | 1.008 | 0.148 |
| F. Cat/contiguous/bookkeeping | 0.364 | 0.302 | 0.627 | 0.178 | 1.260 | 0.167 |
| Sum B–F | 10.038 | 10.051 | 11.235 | 8.371 | 11.608 | 0.752 |
| Official complete preprocessing | 9.589 | 9.463 | 10.975 | 8.171 | 12.760 | 0.803 |

### W1 — index 42, 354,182 points, full history

| Stage | Mean | Median | P95 | Min | Max | Population std. |
|---|---:|---:|---:|---:|---:|---:|
| A. CPU model-ready batch | 9.404 | 9.133 | 11.676 | 7.377 | 16.464 | 1.227 |
| B. `cast_data` / CPU→CUDA | 2.408 | 2.345 | 3.091 | 1.814 | 3.521 | 0.334 |
| C. Hard voxel layer | 303.705 | 302.522 | 323.123 | 284.325 | 330.969 | 9.569 |
| D. Voxel centers | 1.547 | 1.464 | 2.356 | 0.819 | 2.544 | 0.398 |
| E. Coordinate batch padding | 0.314 | 0.285 | 0.546 | 0.134 | 1.146 | 0.148 |
| F. Cat/contiguous/bookkeeping | 0.561 | 0.498 | 1.158 | 0.296 | 1.734 | 0.237 |
| Sum B–F | 308.534 | 307.712 | 328.185 | 288.382 | 336.572 | 9.585 |
| Official complete preprocessing | 308.083 | 306.018 | 327.288 | 282.110 | 393.388 | 14.280 |

### W2 — index 49, 346,073 points, full history

| Stage | Mean | Median | P95 | Min | Max | Population std. |
|---|---:|---:|---:|---:|---:|---:|
| A. CPU model-ready batch | 8.587 | 8.374 | 10.340 | 7.367 | 11.857 | 0.856 |
| B. `cast_data` / CPU→CUDA | 2.542 | 2.421 | 3.104 | 1.862 | 4.844 | 0.470 |
| C. Hard voxel layer | 288.803 | 290.071 | 307.037 | 245.147 | 311.819 | 13.091 |
| D. Voxel centers | 1.729 | 1.610 | 2.823 | 0.810 | 4.350 | 0.612 |
| E. Coordinate batch padding | 0.381 | 0.306 | 0.686 | 0.122 | 1.614 | 0.211 |
| F. Cat/contiguous/bookkeeping | 0.600 | 0.551 | 1.032 | 0.379 | 1.418 | 0.199 |
| Sum B–F | 294.055 | 295.570 | 311.561 | 250.205 | 318.355 | 12.979 |
| Official complete preprocessing | 299.644 | 298.066 | 315.744 | 276.913 | 351.779 | 10.066 |

## Deterministic versus experimental fast timing

All boundaries used isolated reference-then-candidate blocks. Direct E2E includes model-ready batch
construction, preprocessing, TensorRT, unchanged MMDeploy postprocessing, DetectionFrame conversion,
and current provenance hashes. The no-hash boundary is a projected experimental live path only.
Hashed and no-hash conversions from the exact same postprocessed prediction produced identical
detection values for W1 and W2.

| Workload / boundary | Mean | Median | P95 | Min | Max | Population std. |
|---|---:|---:|---:|---:|---:|---:|
| W0 official layer | 8.752 | 8.376 | 11.066 | 6.823 | 12.089 | 1.316 |
| W0 fast layer | 3.196 | 2.895 | 5.753 | 1.846 | 6.566 | 1.076 |
| W0 official complete preprocessing | 9.589 | 9.463 | 10.975 | 8.171 | 12.760 | 0.803 |
| W0 fast complete preprocessing | 3.524 | 3.315 | 4.686 | 2.802 | 5.812 | 0.606 |
| W0 current deterministic E2E | 94.231 | 92.822 | 109.307 | 78.027 | 118.211 | 8.220 |
| W0 fast E2E + hashes | 94.510 | 94.083 | 110.438 | 76.354 | 122.533 | 8.923 |
| W1 official layer | 275.983 | 270.937 | 298.328 | 258.209 | 419.466 | 24.188 |
| W1 fast layer | 4.901 | 4.782 | 6.583 | 2.733 | 7.670 | 0.929 |
| W1 official complete preprocessing | 308.083 | 306.018 | 327.288 | 282.110 | 393.388 | 14.280 |
| W1 fast complete preprocessing | 7.706 | 7.279 | 10.539 | 5.093 | 13.338 | 1.470 |
| W1 current deterministic E2E | 426.455 | 424.020 | 475.297 | 374.901 | 544.829 | 27.566 |
| W1 fast E2E + hashes | 125.516 | 124.960 | 136.448 | 106.172 | 145.633 | 7.798 |
| W1 fast E2E, no hashes | 99.592 | 98.910 | 111.428 | 86.226 | 121.702 | 6.794 |
| W2 official layer | 292.531 | 291.729 | 315.086 | 266.777 | 326.351 | 10.293 |
| W2 fast layer | 5.197 | 5.040 | 7.397 | 2.774 | 10.220 | 1.189 |
| W2 official complete preprocessing | 299.644 | 298.066 | 315.744 | 276.913 | 351.779 | 10.066 |
| W2 fast complete preprocessing | 7.785 | 7.596 | 10.342 | 4.912 | 11.530 | 1.346 |
| W2 current deterministic E2E | 420.442 | 419.733 | 451.448 | 356.212 | 567.530 | 26.038 |
| W2 fast E2E + hashes | 141.805 | 140.649 | 167.435 | 116.094 | 174.613 | 12.927 |
| W2 fast E2E, no hashes | 106.824 | 106.112 | 126.749 | 79.889 | 139.983 | 10.644 |

The isolated layer median speedups were 2.893× for W0, 56.662× for W1, and 57.882× for W2.
Despite that acceleration, the final direct path did not demonstrate 20 Hz.

### Cross-session instability

Two preliminary exact-commit sessions exposed result-shape issues after their timing blocks; they
were not promoted. Their timing code path was unchanged and is retained only to disclose material
session instability. The first/second/final summary SHA256 values were respectively
`c0a561b5c145ffaaeb112dc8c22d9dae74c8bd5df4f4916b004b158b0cf4720b`,
`407ae51bb9e8fb06b3c655171e80bdef8ca7f092a696ac9ea19accbc409b508c`, and the authoritative hash
above.

| Session | Pre-run SM / memory clock (MHz) | W1 fast / no-hash E2E median | W2 fast / no-hash E2E median |
|---|---:|---:|---:|
| `a1ff0a2` preliminary | 1890 / 8001 | 64.922 / 52.002 ms | 65.262 / 50.072 ms |
| `a00d7ac` preliminary | 930 / 810 | 129.943 / 105.332 ms | 132.762 / 102.724 ms |
| `ad0d38b` authoritative | 855 / 6001 | 124.960 / 98.910 ms | 140.649 / 106.112 ms |

Before/after telemetry is only a snapshot and does not establish the causal mechanism. The large
session disagreement prevents promotion of any fast-path latency as stable canonical performance.

## Saturation analysis

No sample reached the 40,000 test-voxel cap, so valid points minus retained points is attributable
to the 64-point per-voxel limit. Official and experimental paths had identical voxel counts,
`num_points`, retained totals, and discarded totals. They frequently selected different retained
subsets inside saturated voxels, analyzed in the next section.

| Index | Valid points | Voxels | Saturated voxels | Saturated fraction | Retained | Discarded at 64-point cap |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (W0) | 33,587 | 4,352 | 22 | 0.51% | 21,022 | 12,565 |
| 4 | 370,352 | 16,203 | 805 | 4.97% | 191,842 | 178,510 |
| 8 | 366,395 | 19,512 | 901 | 4.62% | 227,518 | 138,877 |
| 12 | 367,096 | 19,579 | 901 | 4.60% | 216,201 | 150,895 |
| 16 | 366,359 | 15,037 | 1,051 | 6.99% | 214,520 | 151,839 |
| 21 | 364,175 | 13,606 | 1,077 | 7.92% | 215,894 | 148,281 |
| 25 | 362,081 | 17,453 | 798 | 4.57% | 203,255 | 158,826 |
| 29 | 356,035 | 19,589 | 846 | 4.32% | 222,212 | 133,823 |
| 33 | 357,519 | 18,130 | 858 | 4.73% | 235,935 | 121,584 |
| 37 | 364,318 | 16,270 | 920 | 5.65% | 216,526 | 147,792 |
| 42 (W1) | 354,182 | 18,207 | 927 | 5.09% | 242,307 | 111,875 |
| 46 | 346,729 | 22,546 | 752 | 3.34% | 240,277 | 106,452 |
| 49 (W2, outside frozen 20) | 346,073 | 20,085 | 781 | 3.89% | 238,192 | 107,881 |
| 50 | 346,436 | 19,974 | 788 | 3.95% | 234,371 | 112,065 |
| 54 | 347,495 | 18,207 | 808 | 4.44% | 230,268 | 117,227 |
| 58 | 349,245 | 19,134 | 774 | 4.05% | 222,496 | 126,749 |
| 63 | 340,767 | 16,074 | 754 | 4.69% | 193,577 | 147,190 |
| 67 | 342,907 | 17,921 | 798 | 4.45% | 198,517 | 144,390 |
| 71 | 338,012 | 18,602 | 708 | 3.81% | 209,662 | 128,350 |
| 75 | 337,553 | 19,262 | 800 | 4.15% | 213,302 | 124,251 |
| 80 | 340,695 | 18,528 | 782 | 4.22% | 224,417 | 116,278 |

## Coordinate-canonical voxel fidelity

Across the frozen 20 samples, all 348,186 voxel coordinates were present in both paths; Jaccard,
bidirectional coordinate coverage, and per-coordinate `num_points` agreement were 1.0. All
non-saturated point multisets were identical. Output coordinate order differed for every sample,
and 179,667 same-multiset voxels also differed in point ordering.

The semantic risk is saturated selection: 12,136 of 16,070 saturated voxels (75.5196%) retained a
different point subset in the candidate. Per-sample saturated-subset difference fractions ranged
from 67.78% to 95.45%. These are real retained-input changes, not merely output-order changes.

## Frozen 20-sample detector fidelity

The single candidate draw passed the separately frozen M3B diagnostic yardstick. This does not
reinterpret M2 parity, which remains unchanged.

| Metric over 754 high-confidence matches | Median | P95 | P99 | Maximum | Pass fraction |
|---|---:|---:|---:|---:|---:|
| XY displacement (m) | 0.000153 | 0.002914 | 0.027601 | 0.183484 | 1.000000 |
| Absolute Z difference (m) | 0.000210 | 0.001827 | 0.011121 | 0.092251 | 1.000000 |
| Maximum L/W/H relative error | 0.000122 | 0.002091 | 0.010589 | 0.041825 | 1.000000 |
| Axis-yaw difference (degrees) | 0.000000 | 0.252488 | 2.335195 | 46.468895 | 0.996021 |
| Absolute score difference | 0.000313 | 0.002437 | 0.010697 | 0.021875 | 1.000000 |

Exported totals were 885/885, with one fewer candidate detection at index 54 and one more at index
58. Bidirectional high-confidence coverage was 1.0, class mismatches were zero, and full-heading
agreement was 1.0. Three of 754 matches exceeded at least one continuous tolerance; all three were
axis-yaw exceptions. There were two threshold-edge disagreements. Raw TensorRT output maximum
absolute differences reached 2.09375 (`cls_score`), 2.10547 (`bbox_pred`), and 0.83301
(`dir_cls_pred`); maxima are diagnostic evidence, not new acceptance gates.

## Thirty-run repeatability

W1 and W2 voxel counts and coordinate sets were invariant across all 30 fast runs. `num_points` and
all non-saturated point multisets were exact. Candidate coordinate ordering matched the first fast
run only for that first run. Saturated retained subsets varied substantially:

| Workload / comparison | Median different saturated subsets | P95 | Maximum |
|---|---:|---:|---:|
| W1 fast run vs first fast run | 77.02% | 78.54% | 79.07% |
| W1 fast run vs deterministic | 75.24% | 76.43% | 76.91% |
| W2 fast run vs first fast run | 77.14% | 77.99% | 78.87% |
| W2 fast run vs deterministic | 74.01% | 76.08% | 76.44% |

Raw outputs were observably nondeterministic. Across W1 runs, maximum absolute differences versus
the first fast run reached 0.85938/0.57227/0.37549 for classification/box/direction outputs; versus
deterministic they reached 1.37500/0.73828/0.54883. W2 maxima were
0.43750/0.30713/0.16211 versus the first fast run and 0.90625/0.63672/0.28418 versus deterministic.

W1 final detections passed the reused yardstick both against the first fast run and the deterministic
baseline. W2 passed against the first fast run, but **failed against the deterministic baseline**:
19 of 1,869 high-confidence matches exceeded the 5-degree axis-yaw tolerance, giving a pass fraction
of `0.9898341359` below the frozen 0.99 requirement; maximum axis-yaw difference was 10.1694 degrees.
The 30 W2 comparisons also contained three exported-count changes, three threshold-edge
disagreements, and 20 distinct continuous outliers. Coverage remained approximately 0.99946 in both
directions, class mismatches were zero, and no full-heading reversal occurred.

## Decision and next action

The authoritative direct-path classification is:

- W1 fast + current hashes: 124.960 ms — additional optimization required.
- W1 fast + no hashes: 98.910 ms — meaningful acceleration, roughly 10–20 Hz regime.
- W2 fast + current hashes: 140.649 ms — additional optimization required.
- W2 fast + no hashes: 106.112 ms — additional optimization required.

The candidate is substantially faster but has material saturated-subset, raw-output, and W2
detector repeatability variation. Therefore it is not recommended for production integration and
no fast path becomes the default. The next review should evaluate an upstream-supported efficient
**deterministic** hard-voxelization alternative under the same protocol and stabilize/record GPU
clock and power state before making performance claims. Custom CUDA, postprocess optimization, and
ROS/DDS profiling remain out of scope for this result.

No model, weights, checkpoint, ONNX, engine, point-cloud range, voxel size, voxel limits, sweep
count, threshold, postprocess, ROS/DDS configuration, or historical evidence changed. M4 was not
started.
