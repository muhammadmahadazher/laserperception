# M2 benchmark protocol and diagnostic status

Status: **Diagnostic complete; reviewer approval required before a replacement benchmark.**

There is no accepted canonical M2 performance result. The run at
e2f9b6babb541d52beaa0bcd58e841a0a56cc851 is retained only in
benchmarks/m2/diagnostics/rejected_e2f9b6b.json with status rejected_measurement. Its 124.297×
network and 23.101× end-to-end ratios are not publication evidence.

Parity v2 remains valid and PASS. The frozen ONNX and TensorRT engine are unchanged.

## Runtime roles

- Parity reference: MMDeploy-rewritten PyTorch FP32.
- Performance baseline: native MMDetection3D PyTorch FP32.
- Deployment runtime: TensorRT FP16.
- Common path: identical official voxelization, existing MMDeploy postprocess, and DetectionFrame.

The diagnostic runner performs the 20-sample native-vs-rewritten fidelity check and a
diagnostic-only component profile. Raw diagnostic output stays in the external cache.

## Exact-commit diagnosis

The full non-canonical diagnostic ran at commit
4e12374dec8eecaf0e772b2b5776e0b266fbe09e. Its external JSON SHA256 is
2b537a4415cc981c6cc64f0b617726e82ca38a92c5fafd440c42c06baffb16c2 and its sanitized summary is
diagnostics/diagnosis_4e12374.json.

Native and rewritten FP32 outputs were exactly equal across all 20 frozen samples. Component
medians were 5.567 ms prepare, 8.356 ms voxelize, 20.800 ms native raw, 1910.464 ms rewritten raw,
6.917 ms TensorRT raw, 24.093 ms current postprocess, 0.999 ms bbox-head construction, and 5.160 ms
DetectionFrame conversion. No direct canonical end-to-end comparison was run.

## Future measurement boundary

A reviewer-approved canonical run will use batch size one, mini_val index 0, 10 warmups, and 100
measurements per runtime/boundary in isolated blocks. Network timing uses identical precomputed
voxels and CUDA events. End-to-end timing uses synchronized wall time through common postprocess
and DetectionFrame. The headline remains end-to-end median speedup.

This is a warm-cache repeated-single-sample microbenchmark, not cold-storage I/O, whole-dataset
sequential throughput, a sensor-throughput guarantee, or production evidence. No new canonical
benchmark will be run or promoted until the diagnostic report and methodology receive review.

## Retained parity disclosures

All preregistered per-metric Stage 1 gates passed. Separately, 8/753 (1.06%) high-confidence
matches exceeded at least one continuous tolerance. The 47.626393° index-50 pedestrian axis-yaw
case remains unexplained and fully retained. Raw cls_score p99 absolute difference was 0.056829
with a rare maximum tail of 0.721187.
