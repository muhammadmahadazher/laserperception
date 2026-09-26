# M8 P1-S1 zero-intensity raw measurement

## Measurement identity

Three complete, independent zero-intensity processes on one external NVIDIA A40 produced 2,568 accepted canonical calls: 856 per process, with 428 H10 and 428 H5 conditions each. The zero-intensity execution source was `95fb66ac1f57c41f06f05bd9ef5dac27b1e3ea54`; the historical raw-intensity primary remains bound to `6994d72c3e7691a86116d1417ac3ae08256d163f`. The A40 GPU UUID was `GPU-e43b9709-1c26-6b50-f52b-b32bb621048b`. A fresh GT-blind qualification and accepted 10-process/140-call Stage R preceded the zero-intensity authorization.

The intervention replaced candidate-consumed intensity with float32 positive zero (`0x00000000`) while preserving the other frozen input fields and source-row order. `A2_zeroI` is H10 under that intervention; `E2_zeroI` is H5. All 2,568 condition records carry the exact intervention label, primary-input hash, and transformed-input hash, and each pass matches the same 856-row transformation ledger in order.

## Accepted process accounting

| Pass | Attempt | Process UUID | Accepted calls | Archive SHA-256 |
| --- | --- | --- | ---: | --- |
| 1 | `zi01-a40-20260926T083114Z` | `8f6f2fcb-8ebb-425b-972c-75f6b4f78869` | 856 | `afa685227b604ccf275faf66a651ff0c8b2ee30124d8619db375343b2ae24796` |
| 2 | `zi02-a40-20260926T094058Z` | `c658f15e-c8eb-484f-a8f9-73ff0ba0289b` | 856 | `7ee351681786a78c8c2155a3a8f800bb9ff3497af0ed2e5c06353292a752d15b` |
| 3 | `zi03-a40-20260926T105947Z` | `ffb7d228-7d43-40de-95d8-a56ba6e118f9` | 856 | `7998980c7584751249959ced10c97e430eba20320fac5eea0c04d62405dcb041` |

Each final manifest has `COMPLETE` status, zero failed calls, and no next condition or failure reason. The archive and per-condition hashes were verified offline against the preserved Drive-backed evidence. The [machine-readable manifest](../../benchmarks/m8/results/m8_s1_zero_intensity_manifest.json) carries the remaining evidence and control hashes.

## A2_zeroI / H10 and E2_zeroI / H5 metrics

At score ≥ 0.25 and oriented BEV IoU 0.50, all three passes had the same TP/FP/FN and recall values below. The linked [raw result](../../benchmarks/m8/results/m8_s1_zero_intensity_raw.json) retains pass 1/2/3, minimum, median, and maximum for TP, FP, FN, recall, annotation-conditioned precision, F1, ignored predictions, and annotation-conditioned AP, at IoU 0.30, 0.50, and 0.70. AP is ranked over predictions and can be nonzero even when recall at the displayed score threshold is zero.

| Arm | Class | TP / FP / FN (each pass) | Recall (each pass) | AP at IoU 0.50, pass 1 / 2 / 3 |
| --- | --- | --- | ---: | --- |
| A2_zeroI / H10 | Car | 17 / 81 / 49 | 0.257576 | 0.113260 / 0.113261 / 0.113296 |
| A2_zeroI / H10 | Pedestrian | 0 / 2 / 396 | 0 | 0.000159902 / 0.000159915 / 0.000159887 |
| E2_zeroI / H5 | Car | 43 / 169 / 23 | 0.651515 | 0.199850 / 0.199859 / 0.199762 |
| E2_zeroI / H5 | Pedestrian | 0 / 8 / 396 | 0 | 0.001792067 / 0.001817663 / 0.001792220 |

## Within-zero-intensity H10-to-H5 contrast

The within-process contrast is `recall(E2_zeroI_i) - recall(A2_zeroI_i)` at IoU 0.50. Car was `0.393939` in each pass (minimum/median/maximum all `0.393939`); Pedestrian was `0` in each pass. This calculation compares H10 and H5 inside the same zero-intensity realization.

## Range, tracks, prediction population, FOV, and ignore behavior

The [secondary raw result](../../benchmarks/m8/results/m8_s1_zero_intensity_secondary.json) retains pass 1/2/3 and minimum/median/maximum for the preregistered 0–20 m, 20–35 m, and 35–50 m range bands; track continuity; total postprocessed predictions; inside-FOV and outside-annotation-FOV counts; neighbouring ignore GT; and ignored predictions at each IoU threshold. The range-band median recalls at IoU 0.50 were:

| Arm and class | 0–20 m | 20–35 m | 35–50 m |
| --- | ---: | ---: | ---: |
| A2_zeroI Car | 0.111111 | 0.433333 | 0.111111 |
| E2_zeroI Car | 0.814815 | 0.666667 | 0.111111 |
| A2_zeroI Pedestrian | 0 | 0 | 0 |
| E2_zeroI Pedestrian | 0 | 0 | 0 |

At score ≥ 0.25, inside-FOV prediction counts per pass were 98 Car and 2 Pedestrian for H10, and 212 Car and 8 Pedestrian for H5. Outside-annotation-FOV counts at that threshold were 194 Car and 1 Pedestrian for H10, and 505 Car and 6 Pedestrian for H5. The all-class, all-score postprocessed population ranged from 50,154 to 50,157 for H10 and 49,605 to 49,611 for H5. The neighbouring-ignore GT counts were 19 Car and 68 Pedestrian per arm; ignored prediction counts were zero at all three IoU thresholds. Track-level and per-pass values remain in the secondary JSON rather than being compressed into a single narrative value.

## Descriptive comparison with raw-intensity primary

The [comparison artifact](../../benchmarks/m8/results/m8_s1_zero_intensity_comparison.json) reports separate three-process distributions at IoU 0.50. Primary and zero-intensity process indices are **not paired**. The differences below are differences of observed medians only.

| History / class | Primary recalls, passes 1/2/3 | Zero-intensity recalls, passes 1/2/3 | ZeroI median − primary median |
| --- | --- | --- | ---: |
| H10 Car | 0.287879 / 0.287879 / 0.287879 | 0.257576 / 0.257576 / 0.257576 | −0.030303 |
| H5 Car | 0.651515 / 0.651515 / 0.651515 | 0.651515 / 0.651515 / 0.651515 | 0 |
| H10 Pedestrian | 0 / 0 / 0 | 0 / 0 / 0 | 0 |
| H5 Pedestrian | 0.002525 / 0.002525 / 0.002525 | 0 / 0 / 0 | −0.002525 |

## Evidence identities and claim boundaries

The frozen protocol SHA-256 is `c132f60257c6a39debb548461c79bd59c98325484d233db6095b441c638d8e88`. The zero-intensity input-gate receipt SHA-256 is `947cb0c764fb8e10e9d480b899821afd6a091f4dd1eb8c949442730063d551aa`; its transformation-ledger SHA-256 is `6ba3287a381a4b86222e22184dc5065d771757104c72848987f6b6802e592162`. The fresh Stage R acceptance review SHA-256 is `972060173a8a55fe51f0ceff6bbb2eb29aeaf77ab10226e6283fccfc25f93cae`. The zero-intensity authorization SHA-256 is `f1b033c2f73d43ba56b2dccea07429a8b3135658c811efcf87e898254fc42b61`.

The core aggregation and publication reduction each produced byte-identical outputs in two independent CPU processes. No detector call was added during offline publication. This document publishes raw measurement only. It does not claim intensity causality, architecture causality, statistical significance, a universal model winner, or production readiness. Zero-intensity scientific interpretation is pending. S2 and training remain blocked.
