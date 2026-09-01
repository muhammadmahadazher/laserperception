# M8 P1-S1 Stage R authorization

Status: **M8 P1-S1 STAGE R — OWNER AUTHORIZED BEFORE ANY GT-RELATIVE V2
MEASUREMENT.**

**NO STAGE R DETECTOR CALL HAS OCCURRED. NO GT-RELATIVE V2 RESULT EXISTS.**

This authorization act permits only the preregistered Stage R repeatability gate. It does not
contain detector output and does not expand the frozen P1-S1 scientific protocol.

## Authorized scope

- Mode: `stage-r` only.
- Logical passes: `stage-r-1` through `stage-r-10` only.
- Future accepted calls: 140 total.
- Process model: 10 fresh sequential processes, each executing the same 14 sentinel conditions.
- Condition order: the seven frozen frames below, `H10` then `H5` for each frame.

The seven sentinels are:

1. `2011_09_26_drive_0001/0000000010`
2. `2011_09_26_drive_0001/0000000011`
3. `2011_09_26_drive_0001/0000000015`
4. `2011_09_26_drive_0001/0000000083`
5. `2011_09_26_drive_0091/0000000010`
6. `2011_09_26_drive_0091/0000000011`
7. `2011_09_26_drive_0091/0000000012`

The following remain **unauthorized**:

- primary A2/E2 corpus passes;
- zero-intensity passes;
- B2/C2/D2/F2 interventions;
- training or fine-tuning.

## Frozen execution and authorization identities

- Stage R execution commit:
  `d8e5012312b6ee0b3c891e1c2d794424f8a35c36`
- Reviewed runtime feature head:
  `610de73312487367a47105b85f3f9f84aa3fca13`
- Runtime normal merge commit:
  `d8e5012312b6ee0b3c891e1c2d794424f8a35c36`
- Frozen protocol commit:
  `5061d5d2c6a6057fed1f3f537c5857d2d84f6b3f`
- Frozen protocol Markdown SHA256:
  `1ad58ebbdd04897558ef9802fee6288b806c5e633d393f1bed957ecc6d6f6b10`
- Frozen protocol JSON SHA256:
  `c132f60257c6a39debb548461c79bd59c98325484d233db6095b441c638d8e88`
- Owner memory-margin review:
  `benchmarks/m8/diagnostics/m8_s1_memory_margin_owner_review.json`, SHA256
  `16d74a9672b351bc5cec3cd0b51bb654bed6327d7fb5df5ab9748229115826d0`, decision
  `ACCEPTED_FOR_S1_RUNTIME`.

The GT-blind runtime-policy binding is
`benchmarks/m8/preregistration/m8_s1_stage_r_runtime_policy.json`: 2,028 bytes, SHA256
`703e453a8bca0e6e2e4b1c4b976deaa5bc4ed27b3a4847144204193baab77563`. It records
`repository_execution_commit = d8e5012312b6ee0b3c891e1c2d794424f8a35c36` and was captured
without GT, an evaluator, DSVT construction, or detector inference.

The owner authorization is
`benchmarks/m8/preregistration/m8_s1_stage_r_authorization.json`: 2,225 bytes, SHA256
`3c16a0c0ff9680b6418a53b20a5a51dd8f2a40d864a75c380397f2454ce06b9c`. It uses schema
`laserperception.m8.s1.authorization.v2`, binds the runtime-policy file SHA256, and contains only
the Stage R mode and ten Stage R logical pass IDs.

## Operational constraints

Scientific execution uses exactly one LaserPerception S1 detector process at a time. Stage R
processes remain sequential; primary and zero-intensity processes do not run in parallel; and no
other user-launched CUDA workload may occupy the GPU. `PYTORCH_CUDA_ALLOC_CONF` remains unset.
Allocator policy/tuning and deterministic-algorithm settings remain unchanged.

These constraints implement the accepted owner memory-margin decision. They are operational
constraints, not new scientific acceptance criteria.

## Future detached execution-worktree contract

After this authorization merges, Stage R must execute from a dedicated detached worktree such as
`.local/m8-s1-stage-r-execution`, pinned exactly to
`d8e5012312b6ee0b3c891e1c2d794424f8a35c36`. It must not execute from the newer authorization
merge commit. On the reviewed WSL checkout, create the worktree with `core.autocrlf=false` for the
checkout operation so the frozen protocol Markdown materializes at its bound 23,802-byte LF
identity; a different byte identity must fail rather than be rewritten after checkout.

The detached runner receives absolute paths to the merged authorization JSON and runtime-policy
JSON from the authoritative `main` workspace, or byte-identical external copies. Before the first
detector call it must verify:

1. repository `HEAD` equals the Stage R execution commit;
2. authorization fields and SHA-bound runtime policy;
3. live stable runtime policy equals the bound policy;
4. frozen protocol, candidate, config, checkpoint, and input identities.

No source file from later `main` may be copied into the detached execution worktree. Only the two
authorization/policy **data artifacts** may be supplied externally. A failed logical repeat is
preserved as incomplete and restarts from condition 1 in a new process.

## Repository branch convergence

Branch convergence occurred before selecting the execution commit. No valid work required a
consolidation merge, so `main` remained at `d8e5012312b6ee0b3c891e1c2d794424f8a35c36`.

| Branch | Tip | Classification | Unique commits | Action and reason |
| --- | --- | --- | ---: | --- |
| `feat/m8-detector-v2-integration` | `77369c02e3486650cd06624cb796cf1efbc6e3d4` | Already in `main` | 0 | Deleted locally; tip is an ancestor of `main`. |
| `fix/m7-runtime-binding-contract` | `5a8c02e8ba279ee44a8bb87eb2ec2984ca95e729` | Already in `main` | 0 | Deleted locally; tip is an ancestor of `main`. |
| `fix/m7-streaming-ledger-runtime` | `5d8cc81653d27ef513b1fd83f98e58793983506e` | Already in `main` | 0 | Deleted locally; tip is an ancestor of `main`. |
| `evidence/m7-measurement` | `dc4da349ef94a95ac87e84417fc1e10eb41588f1` | Superseded/stale | 1 | Deleted locally without merge. Its authorization binds the failed/superseded `5d8cc816…` runtime; authoritative `main` retains corrected M7 R2 runtime `5a8c02e…` and its accepted evidence. |

No corresponding remote non-main branch existed after `git fetch origin --prune`; the initial and
post-convergence remote inventory contained only `origin/main` plus symbolic `origin/HEAD`.
