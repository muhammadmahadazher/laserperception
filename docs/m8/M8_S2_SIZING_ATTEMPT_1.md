# M8 P1-S2 first sizing attempt: initialization failure, zero calls

The temporary NVIDIA A40 worker `a5z5ko7mnyepa4` used reviewed execution commit
`7cf628050a693d4e0725c654b75c0ccebcc2f455`. Fresh structural qualification passed:
1,712/1,712 input identities, 28/28 sentinel XYZIT identities, 7/7 B2/A2 CUDA coordinate
comparisons and 7/7 D2/C2 comparisons. This structural qualification remains valid historical
evidence; it did not prove full candidate model initialization.

| Historical binding | SHA256 |
|---|---|
| Runtime policy | `4634702a068759f5fcf0f997fd0e1707fa8728c9e0de64591c2bb3d167688efc` |
| Qualification receipt | `8ef05b49e016de54b834c8df999f26f9d42f7de52101fb0f23a94d66be64144d` |
| Sealed process-1 manifest | `078caec7ec8dfc1abf3ac4d5bb81f0228dab6e4c4d07ec6ccc84f0958ccc0297` |

Sizing process 1 (`s2-sizing-01-b1b8bdc67066`) started model initialization, then failed before
the first detector call with `ModuleNotFoundError: No module named 'torchvision'`. The pinned
`pcdet.models` eager registry imports `backbones_3d/focal_sparse_conv/SemanticSeg/sem_deeplabv3.py`,
which requires torchvision. No model initialization completed. The runner sealed **INCOMPLETE**
and released its sizing lock; process 2 never started. Engineering detector calls = 0; scientific
calls = 0; GT loaded = false; evaluator loaded = false. There is no accepted sizing aggregate,
latency measurement or scientific result.

The private orchestrator then unnecessarily called interrupted-process recovery and encountered
the already-released lock. This secondary error masked the primary error in its outcome summary;
both errors and the sealed evidence were retained. Prospective private orchestration inspects
canonical status first: already-sealed INCOMPLETE/COMPLETE requires no recovery; RUNNING with
a retained lock delegates same-worker/dead-process verification to the reviewed recovery runner.
Missing/corrupt status fails closed, with evidence retained and the primary exception preserved.

Private Drive-backed evidence root:
`_CLOUD_WORK/m8-s2-runtime-sizing-20261004-a5z5ko7mnyepa4/`, including `CAMPAIGN_REPORT.md`,
`FAILURE_ANALYSIS.md`, `completion.json`, `orchestration-outcome.json` and `snapshot-final/`.
Private authorization contents are not published. The Pod was deleted, verified by DELETE 204,
GET 404 and an empty provider Pod list. Its runtime policy and grants are historical and cannot
authorize another worker.

The [prospective candidate-readiness gate](M8_S2_CANDIDATE_READINESS.md) closes the full-model
initialization gap without changing the frozen protocol, partitions, inputs, candidate or sizing plan.
