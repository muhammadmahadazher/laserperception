# M8 P1-S1 three-pass primary raw measurement

Status: **RAW PREREGISTERED MEASUREMENT — SCIENTIFIC INTERPRETATION NOT YET FROZEN.**

This record publishes the accepted three-process A2/E2 measurement without selecting a winner,
assigning architecture or intensity causality, or making statistical, production-readiness, or
deployment claims. The three processes are repeated runtime/numerical realizations of the same
frozen 428-frame corpus, not independent dataset samples. Min/median/max therefore describe only
the observed three-process spread. Zero-intensity, S2, and training remain unexecuted.

## Evidence and accounting

Scientific execution used frozen commit
`6994d72c3e7691a86116d1417ac3ae08256d163f`. Each accepted fresh process executed 428 H10 and 428
H5 conditions in the frozen frame-major order. The accepted total is three complete processes and
2,568 canonical calls. The historical 779-condition incomplete attempt and the interrupted
26-condition attempt before the pass-1 restart each retain zero accepted canonical calls and are
excluded from every aggregate.

| Pass | Process UUID | Accepted calls | Archive SHA-256 |
| --- | --- | ---: | --- |
| 1 | `3256b511-921e-4456-92c6-1fd5d5a8c380` | 856 | `fab1db337c41788d38bf2560129494be65e2f98a3e153a130bfd03dc915e26d1` |
| 2 | `21f32e37-58c3-42e2-b9c1-2011b20e47b7` | 856 | `7df28ade47aacd3dde1a2ab836c6796522df75a242ec52ba0e5e2ab5a67443ee` |
| 3 | `b9c1ab4b-9ea2-428a-a41e-e70fb8528d87` | 856 | `5bf95e23058c4e01bbf79a35682df24454342f39ac9398e2da3b70c4e33e33a1` |

The exact archive, raw-pass, final-manifest, result, receipt, protocol, model, input, evaluator, and
runtime bindings are recorded in
[`m8_s1_measurement_manifest.json`](../../benchmarks/m8/results/m8_s1_measurement_manifest.json).
The large raw passes and condition dumps remain in private durable storage and are not committed.

## Primary raw numbers

All three passes had identical threshold counts and derived recall, annotation-conditioned
precision, and F1. The values below therefore have equal minimum, median, and maximum. AP is the
frozen annotation-conditioned all-points AP at IoU 0.50; its three values are reported separately.

| Arm | Class | IoU | TP | FP | FN | Recall | Precision | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A2 / H10 | Car | 0.30 | 19 | 89 | 47 | 0.287879 | 0.175926 | 0.218391 |
| A2 / H10 | Car | 0.50 | 19 | 89 | 47 | 0.287879 | 0.175926 | 0.218391 |
| A2 / H10 | Car | 0.70 | 8 | 100 | 58 | 0.121212 | 0.074074 | 0.091954 |
| A2 / H10 | Pedestrian | 0.30 | 0 | 1 | 396 | 0 | 0 | 0 |
| A2 / H10 | Pedestrian | 0.50 | 0 | 1 | 396 | 0 | 0 | 0 |
| A2 / H10 | Pedestrian | 0.70 | 0 | 1 | 396 | 0 | 0 | 0 |
| E2 / H5 | Car | 0.30 | 44 | 187 | 22 | 0.666667 | 0.190476 | 0.296296 |
| E2 / H5 | Car | 0.50 | 43 | 188 | 23 | 0.651515 | 0.186147 | 0.289562 |
| E2 / H5 | Car | 0.70 | 17 | 214 | 49 | 0.257576 | 0.073593 | 0.114478 |
| E2 / H5 | Pedestrian | 0.30 | 1 | 8 | 395 | 0.002525 | 0.111111 | 0.004938 |
| E2 / H5 | Pedestrian | 0.50 | 1 | 8 | 395 | 0.002525 | 0.111111 | 0.004938 |
| E2 / H5 | Pedestrian | 0.70 | 0 | 9 | 396 | 0 | 0 | 0 |

| Arm | Class | AP pass 1 | AP pass 2 | AP pass 3 | Min | Median | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A2 / H10 | Car | 0.119872 | 0.120013 | 0.119867 | 0.119867 | 0.119872 | 0.120013 |
| A2 / H10 | Pedestrian | 0.000296481 | 0.000296659 | 0.000296650 | 0.000296481 | 0.000296650 | 0.000296659 |
| E2 / H5 | Car | 0.203850 | 0.203841 | 0.203838 | 0.203838 | 0.203841 | 0.203850 |
| E2 / H5 | Pedestrian | 0.00450507 | 0.00450639 | 0.00450517 | 0.00450507 | 0.00450517 | 0.00450639 |

The paired H5-minus-H10 recall contrast is computed within each pass first. Its three values are
`0.363636` for Car and `0.00252525` for Pedestrian, so each class has the same minimum, median, and
maximum. Against the fixed historical PointPillars realization, the per-pass recall deltas are
`+0.0454545` (A2 Car), `-0.553030` (A2 Pedestrian), `-0.0757576` (E2 Car), and `-0.674242`
(E2 Pedestrian); each vector is constant across the three DSVT passes. These are descriptive
cross-stack comparisons under the frozen protocol.

## Secondary characterization from captured evidence

The secondary reducer performed no detector calls. Each range numerator below is TP/eligible GT
and is identical in all three passes.

| Arm | Class | IoU | 0–20 m | 20–35 m | 35–50 m |
| --- | --- | ---: | ---: | ---: | ---: |
| A2 | Car | 0.30 | 5/27 | 13/30 | 1/9 |
| A2 | Car | 0.50 | 5/27 | 13/30 | 1/9 |
| A2 | Car | 0.70 | 4/27 | 4/30 | 0/9 |
| A2 | Pedestrian | 0.30 | 0/288 | 0/106 | 0/2 |
| A2 | Pedestrian | 0.50 | 0/288 | 0/106 | 0/2 |
| A2 | Pedestrian | 0.70 | 0/288 | 0/106 | 0/2 |
| E2 | Car | 0.30 | 22/27 | 21/30 | 1/9 |
| E2 | Car | 0.50 | 22/27 | 20/30 | 1/9 |
| E2 | Car | 0.70 | 11/27 | 6/30 | 0/9 |
| E2 | Pedestrian | 0.30 | 0/288 | 1/106 | 0/2 |
| E2 | Pedestrian | 0.50 | 0/288 | 1/106 | 0/2 |
| E2 | Pedestrian | 0.70 | 0/288 | 0/106 | 0/2 |

The full per-track detected-frame continuity record is retained in
[`m8_s1_secondary_raw.json`](../../benchmarks/m8/results/m8_s1_secondary_raw.json). At IoU 0.50,
A2 records 19 detected Car frames across 11 tracks/66 eligible frames and zero detected Pedestrian
frames across 41 tracks/396 eligible frames. E2 records 43 detected Car frames across the same 11
tracks/66 frames and one detected Pedestrian frame across the same 41 tracks/396 frames.

At score 0.25, A2 has 108 inside-FOV and 220 outside-FOV Car predictions in every pass, plus one
inside and one outside Pedestrian prediction. E2 has 231 inside-FOV Car predictions in every pass,
538/539/539 outside-FOV Car predictions, nine inside-FOV Pedestrian predictions, and eight
outside-FOV Pedestrian predictions. Total postprocessed all-score/all-class populations are
50,460/50,461/50,465 for A2 and 49,820/49,825/49,819 for E2. The corpus contains 19 eligible Car
neighbour-ignore boxes and 68 eligible Pedestrian neighbour-ignore boxes per arm; zero predictions
were ignored at IoU 0.30, 0.50, or 0.70 in all accepted passes.

## Reproducibility and boundary

The frozen primary aggregator ran twice in independent CPU processes and produced byte-identical
full output with SHA-256
`3962fbe69a2d88b5437b2ea38821b3f80392cd5f30b26711f2f5068f4284ca83`. The tracked compact primary
artifact and secondary reducer also ran twice with exact byte equality. This offline publication
made zero RunPod actions, created no resources, executed no GPU probes, touched no local GPU, and
made zero scientific detector calls.
