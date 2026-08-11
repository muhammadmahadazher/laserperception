# M2 benchmark protocol and diagnostic status

Status: **Previous benchmark rejected; diagnostic review required.**

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
