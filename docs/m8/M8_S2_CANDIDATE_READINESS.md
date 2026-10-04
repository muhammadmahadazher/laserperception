# M8 P1-S2 prospective candidate readiness

This CPU-only repair follows the [zero-call sizing initialization failure](M8_S2_SIZING_ATTEMPT_1.md).
No worker, real grant, GPU probe, model execution, GT access, sizing or scientific call is performed
by this PR. Actual readiness is external-only and requires fresh owner authorization.

## Dependencies and static import audit

The metadata-only [development checker](../../scripts/detection/check_m8_s2_development.py)
now requires `torchvision==0.16.0+cu118`, paired with `torch==2.1.0+cu118` from the
[official cu118 wheel index](https://download.pytorch.org/whl/cu118). CPU/plain/CUDA-12 wheels
and later torchvision versions fail before build. Core-wheel dependencies remain unchanged.

Static inspection used pinned DSVT commit `8cfc2a6f23eed0b10aabcdc4768c60b184357061`,
without importing upstream code. The `pcdet.models` eager registry loads every registered
detector, backbone and head, beyond the selected TransFusion/DSVT components:

| Import path | Required direct distribution and prospective pin |
|---|---|
| `pcdet.config` | easydict 1.13; PyYAML 6.0.3 |
| focal sparse semantic segmentation | torchvision 0.16.0+cu118 |
| `common_utils`, box utilities and Hungarian assigner | SharedArray 3.2.4; scipy 1.10.1 |
| CenterNet utility decorators | numba 0.57.1 |
| MPPNet registry → dataset/augmentation/processor registry | scikit-image 0.21.0; tqdm 4.66.5 |
| DSVT → `TRTWrapper` → TensorRT utilities | tensorrt 8.6.1; onnx 1.14.1; packaging 26.3 |
| DSVT VFE, sparse utilities and native wrappers | Torch 2.1.0+cu118; NumPy 1.23.5; spconv-cu118 2.3.8; torch-scatter 2.1.2+pt21cu118 |

The existing external package inventory establishes the ancillary pins. TensorRT 8.6.1 is
already recorded by the frozen candidate; ONNX 1.14.1 is the established detection setup pin.
TensorRT/ONNX are eager import dependencies here; readiness constructs no TensorRT engine and
exports no ONNX graph. Keep the compatible TensorRT binding in the isolated Python environment.
Kornia, pandas/pandaset and dataset SDK/evaluation imports are guarded or deferred and are not
needed to initialize the selected LiDAR-only candidate. Do not install unrelated SDKs, invoke
dataset constructors, or load evaluators. Ordinary transitive dependencies are resolved by the
external environment installer; this audit is not an ABI or initialization success claim.

## Receipt and execution boundary

The [runner](../../scripts/detection/run_m8_s2.py) adds `candidate-readiness`. It verifies the
reviewed execution commit/source tree, frozen inputs and receipt, accepted structural qualification
and its qualification-only grant, exact policy/candidate/upstream/config/checkpoint, generated
version file and native inventory. Installed distribution metadata and exact Python identity are
checked before Torch imports. The grant must match the live worker hostname and sole GPU UUID.
Live policy capture must then exactly match the bound policy.

Only after those checks does it call the existing `DsvtBackend.from_environment` with the frozen
manifest, parse the frozen YAML, build the model, load the checkpoint, move to `cuda:0`, enter
eval state, verify all model parameters/buffers and submodules, and synchronize. There are no
points, forward calls, structural-smoke calls, timing/capacity calls, predictions, GT or evaluator.
Initialization exceptions emit no accepted receipt; output paths must be fresh.

```sh
python -I scripts/detection/run_m8_s2.py candidate-readiness --external-worker \
  --execution-commit <new-reviewed-merge> --full-ledger <frozen-ledger> \
  --input-gate-receipt <input-receipt> --runtime-policy-binding <fresh-policy> \
  --qualification-authorization <fresh-qualification-only-grant> \
  --qualification-receipt <accepted-qualification> --upstream-root <pinned-DSVT> \
  --checkpoint <frozen-checkpoint> --output <fresh-readiness-receipt>
```

`laserperception.m8.s2.candidate-readiness.v1` binds the execution/frozen identities, three
prior receipt/policy hashes, candidate/config/checkpoint/upstream, exact runtime/package versions,
hostname/GPU UUID, generated version SHA and canonical native inventory SHA. It records
initialization complete, model device `cuda:0`, eval true, zero inference/engineering/scientific
calls and GT/evaluator false. No initialization timing or detector output is included.

Prospective S2 authorization, policy, attempt/condition/claim, sizing-result,
repeatability-review/owner-attestation and scientific aggregate schemas are **v2**;
historical v1 files remain unchanged.
Qualification grants carry a null `candidate_readiness_receipt_sha256` and cannot imply later scope.
Each separate sizing/repeatability/full-pass grant requires that SHA; runner modes require
`--candidate-readiness-receipt` and validate its complete identity before accelerator imports.
Readiness evidence grants no inference permission. Scientific and sizing claims, attempts,
condition identities and final aggregates retain the readiness SHA and require it to agree across
processes. Scientific repeatability review and the separate owner attestation retain the same
binding. Recovery preserves it; missing, malformed or mixed readiness identities fail closed.

## Next paid-worker handoff

After a separate owner task authorizes billing: create one fresh worker → fresh worker-specific
qualification grant → exact reviewed checkout → hydrate/hash frozen assets → exact pinned
environment including Torch/torchvision cu118 and the import dependencies above → CUDA
development gate → build/import all seven extensions → input gate → fresh runtime policy →
structural qualification → full candidate-readiness initialization → persist/hash-verify receipt
off-worker → fresh sizing-only grant bound to readiness → sizing process 1 → persist/hash-verify →
sizing process 2 → persist/hash-verify → CPU aggregate → persist/hash-verify → delete worker →
return to owner. Failures also require persistence and worker deletion. No repeatability or full
pass is included in that future sizing sequence. Existing timing/selection and scientific gates
remain unchanged. This CPU-only task stops after the reviewed PR merges.
