# M7 raw preregistered measurement R2

**M7 MEASUREMENT COMPLETE — RAW PREREGISTERED RESULTS ONLY.**

**SCIENTIFIC INTERPRETATION NOT YET FROZEN.**

This record reports the frozen operating-point numbers without a scientific interpretation
narrative. The operating point is score `>= 0.25` and oriented BEV IoU `>= 0.50`. Precision and
AP are annotation-conditioned by the incomplete KITTI Raw tracklets.

Machine-readable records:

- [raw arm table](../../benchmarks/m7/results/m7_raw_arm_table.json)
- [factorial contrasts](../../benchmarks/m7/results/m7_raw_factorial_contrasts.json)
- [measurement manifest](../../benchmarks/m7/results/m7_measurement_manifest.json)

## Complete-corpus and repeatability gates

| Gate | Result |
| --- | ---: |
| Repeatability groups | 20/20 exact PASS |
| Repeatability detector calls | 200 |
| B | 428/428 |
| C | 428/428 |
| D | 428/428 |
| F | 428/428 |
| Canonical corpus conditions | 1,712/1,712 |
| Ordinary non-sentinel calls | 1,692 |
| Total detector calls | 1,892 |
| R1 checkpoints reused | 0 |
| Canonical sentinel repeat-1 outputs reused | 20 |

Repeat 1 is the canonical result for each sentinel/arm group because all ten frozen repetitions
were exact. No eleventh sentinel invocation was made.

## Car raw table

| Arm | TP | FP | FN | Recall | Ann.-conditioned precision | F1 | AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 16 | 144 | 50 | 0.2424242424 | 0.1000000000 | 0.1415929204 | 0.0600703809 |
| B | 41 | 254 | 25 | 0.6212121212 | 0.1389830508 | 0.2271468144 | 0.2121835464 |
| C | 17 | 152 | 49 | 0.2575757576 | 0.1005917160 | 0.1446808511 | 0.0877469896 |
| D | 49 | 260 | 17 | 0.7424242424 | 0.1585760518 | 0.2613333333 | 0.2415680459 |
| E | 48 | 261 | 18 | 0.7272727273 | 0.1553398058 | 0.2560000000 | 0.2423801048 |
| F | 21 | 148 | 45 | 0.3181818182 | 0.1242603550 | 0.1787234043 | 0.0998995397 |

### Car preregistered paired numbers and booleans

| Arm | G_car | E-only recovered | R_gain | Shared retained | R_shared | Neither recovered | R_novel | G gate | Gain gate | Shared gate | All three |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| B | 0.781250 | 23/32 | 0.718750 | 16/16 | 1.000000 | 2/18 | 0.111111 | PASS | PASS | PASS | PASS |
| C | 0.031250 | 2/32 | 0.062500 | 14/16 | 0.875000 | 1/18 | 0.055556 | FAIL | FAIL | FAIL | FAIL |
| D | 1.031250 | 29/32 | 0.906250 | 16/16 | 1.000000 | 4/18 | 0.222222 | PASS | PASS | PASS | PASS |
| F | 0.156250 | 7/32 | 0.218750 | 13/16 | 0.812500 | 1/18 | 0.055556 | FAIL | FAIL | FAIL | FAIL |

The exact detected and missed pose identities are retained in the raw arm-table JSON.

## Pedestrian raw table

| Arm | TP | FP | FN | Recall | Ann.-conditioned precision | F1 | AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 219 | 3,831 | 177 | 0.5530303030 | 0.0540740741 | 0.0985155196 | 0.0864770435 |
| B | 199 | 3,975 | 197 | 0.5025252525 | 0.0476760901 | 0.0870897155 | 0.0497325802 |
| C | 224 | 3,656 | 172 | 0.5656565657 | 0.0577319588 | 0.1047708138 | 0.1033048200 |
| D | 212 | 3,751 | 184 | 0.5353535354 | 0.0534948272 | 0.0972700161 | 0.0629102678 |
| E | 268 | 3,868 | 128 | 0.6767676768 | 0.0647969052 | 0.1182700794 | 0.1268901645 |
| F | 236 | 3,729 | 160 | 0.5959595960 | 0.0595208071 | 0.1082320569 | 0.1092505902 |

| Arm | G_ped | E-only recovery | A-only retention | Shared retention | Neither recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| B | -0.408163 | 28/64 (0.437500) | 2/15 (0.133333) | 159/204 (0.779412) | 10/113 (0.088496) |
| C | 0.102041 | 21/64 (0.328125) | 4/15 (0.266667) | 190/204 (0.931373) | 9/113 (0.079646) |
| D | -0.142857 | 33/64 (0.515625) | 4/15 (0.266667) | 166/204 (0.813725) | 9/113 (0.079646) |
| F | 0.346939 | 29/64 (0.453125) | 7/15 (0.466667) | 187/204 (0.916667) | 13/113 (0.115044) |

## Descriptive Car-recall factorial numbers

F is excluded from these preregistered equations.

| Quantity | Value |
| --- | ---: |
| L | 0.4318181818 |
| P | 0.0681818182 |
| I | 0.1060606061 |

These are numerical descriptive contrasts only.

## F raw comparison

F is the preregistered natural, unthinned, long-span comparator at matched history-sweep count.
It does not isolate temporal span as a unique cause.

| Class | Baseline | Delta TP | Delta recall | Delta AP |
| --- | --- | ---: | ---: | ---: |
| Car | A | +5 | +0.0757575758 | +0.0398291588 |
| Car | E | -27 | -0.4090909091 | -0.1424805651 |
| Pedestrian | A | +17 | +0.0429292929 | +0.0227735467 |
| Pedestrian | E | -32 | -0.0808080808 | -0.0176395742 |

## Frozen input structural context

- A/B: 68 overflow frames.
- C/D/E/F: 0 overflow frames.
- C and E: exact matched total point count.
- C: more candidate pillars than E.
- F: near-E point population with intermediate pillar count.

## Session and identities

The single R2 session ran from `2026-08-28T12:06:13.600987Z` through
`2026-08-28T17:43:48.955441Z` with a wall time of `13,674.09046414` seconds. It
recorded `2,020,376,576` bytes process peak RSS and `280,403,456` bytes peak allocated GPU
memory. It completed without failure or restart.

- Scientific implementation: `c989f7df5ca8c5ac8148c0ed3a2e91de48b754b2`
- Corrected reviewed measurement runtime: `5a8c02e8ba279ee44a8bb87eb2ec2984ca95e729`
- Prospective R2 authorization/execution: `8a7e0128092e30c1b249f2dca4c2541f2792574a`
- Frozen input ledger: `577a7ee3da5495611592ca3226a2adefd577fa54821bb859d25892d0cbcbb8ea`
- Frozen paired GT: `0f4ecf564bff30913a0cb35b2043a9a5cd0c8fdb26b220c4cb12072e186f8ba5`
- TensorRT engine: `2e790b1cdbdc1b88c2aafdc81b5921ebee152edd8408158f88437ae4dd1f3e7f`

The 46,859,349-byte external checkpoint/session package is
`m7_r2_checkpoint_evidence.tar.gz`, SHA256
`0e35af22f5f83751d09422861ade6aafd3f2f7cb8b483752377b1c38c119d466`. It remains outside
Git under `.local/m7-measurement-r2/` and contains no raw network tensors.

## Aggregation reproducibility

Two fresh processes aggregated the completed R2 checkpoints without detector rerun. Both produced
byte-identical compact JSON:

- arm table: `2539286bc4ddf05e0526e0301aeb93e295afa1d549140d2ef341edc6cb725f44`;
- factorial contrasts: `f6e6f7a25759948be894d3419055064775ae168081acfc2c9ae77422052bbb06`.

No scientific interpretation narrative is frozen in this record.
