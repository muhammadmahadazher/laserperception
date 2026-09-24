# Preserved failures and negative results

LaserPerception retains failed, rejected, incomplete, and negative evidence. These categories have
different meanings and must not be collapsed into a single success/failure label.

| Item | Category | Preserved conclusion |
|---|---|---|
| [M2 rejected benchmark](../benchmarks/m2/diagnostics/rejected_e2f9b6b.json) | Rejected methodology | Rewritten eager PyTorch was not the valid performance denominator |
| [M3A 20 Hz record](../benchmarks/m3/results/rtx4060_ros2_humble_exact_tensorrt_fp16.json) | Scientific/performance failure | Representative 20 Hz operation was not sustained |
| [M3B-V1 diagnostic](../benchmarks/m3/diagnostics/voxelization_v1_ad0d38b.json) | Rejected methodology | Nondeterministic upstream voxelization changed observable semantics |
| [M4.5b W1 failure](../benchmarks/m45b/diagnostics/w1_raw_ros_hash_failure.json) | Engineering correctness failure | Initial raw reconstruction did not match the frozen reference |
| [M6a original failure](m6/M6A_POSE_ORACLE_DIAGNOSIS.md) | Scientific protocol failure | Odom equality assumptions were invalid; diagnosis is retained |
| [M6b 30k stop](../benchmarks/m6b/diagnostics/failed_engine_shape_preflight.json) | Engineering/runtime failure | Structural engine capacity was insufficient |
| [M6c R2 failure](m6/M6C_RESULTS.md) | Scientific protocol failure | Byte-exactness gate failed before accepted R3 repair |
| [M7 preflight failures](m7/M7_INFERENCE_PREFLIGHT_FAILURE.md) | Engineering/runtime failure | Frozen runtime binding/preflight blocked execution until corrected |
| [M8 incomplete primary](../benchmarks/m8/diagnostics/external_runtime_primary_incomplete_summary.json) | Incomplete engineering execution | 779 conditions attempted; zero accepted canonical calls |
| [M8 capacity status](../benchmarks/m8/diagnostics/external_runtime_capacity_status_20260922.json) | Infrastructure blocker | Requested workers were unavailable before allocation |
| [OmniLink/OmniSim](external/OMNILINK_OMNISIM_EVALUATION.md) | External scientific negative result | No valid intended traffic-cone match at threshold 0.25 |

Historical records remain unchanged. Follow-up repairs or later accepted runs do not erase the
original failure.
