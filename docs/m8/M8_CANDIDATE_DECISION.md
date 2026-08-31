# M8 Phase 1 candidate decision

> **M8 Phase 1 engineering tier only. No KITTI comparative result exists, and no scientific
> claim is authorized.**

## Decision

DSVT-Pillar with the nuScenes TransFusion head is the immutable candidate for a future,
owner-gated M8 Phase 1 scientific comparison. The candidate was selected on engineering
feasibility before any KITTI detector output was observed. CenterPoint was the prospective
fallback and was not exercised.

Status: **M8 PHASE 1 V2 CANDIDATE SELECTED ON ENGINEERING FEASIBILITY — SCIENTIFIC COMPARISON
NOT YET AUTHORIZED.**

## Timebox record

- Attempt start: 2026-08-31 10:56 PKT (creation of `feat/m8-detector-v2-integration`).
- Engineering decision completed: 2026-08-31 12:01 PKT (selected-config TensorRT boundary smoke
  completed).
- Focused engineering interval: 1 hour 5 minutes, within the preregistered 12 focused-hour and
  two-working-day limits.

| DSVT feasibility requirement | Result | Evidence |
|---|---:|---|
| Official config/checkpoint provenance | PASS | Official DSVT repository README, config, and linked checkpoint |
| Checkpoint loads | PASS | 449/449 parameters loaded |
| Official-domain inference | PASS | nuScenes mini validation index 0 |
| Batch one fits 8 GB | PASS | 1,454,273,536 B peak allocated; 1,904,214,016 B peak reserved |
| Point-feature schema known | PASS | float32 `[x,y,z,intensity,timestamp]`; timestamp is consumed |
| DetectionFrame conversion | PASS | CPU analytic fixtures and source output conversion |
| Coordinate convention identified | PASS | OpenPCDet lidar geometric-centre `dx,dy,dz,heading` contract |
| Deployment route concrete/smoked | PASS | selected-config ONNX export plus TensorRT FP16 build/deserialization at the official partial boundary |
| Training/model surgery required | PASS (not required) | official checkpoint used unchanged |

There was one environment blocker during the attempt: the official dynamic pillar VFE references
`torch_scatter`, which was not initially installed. Installing the official PyG wheel
`torch-scatter 2.1.2+pt21cu118` matched to the pinned PyTorch/CUDA stack resolved it. DSVT therefore
did not reach the rejection condition and no fallback selection occurred.

## Immutable identity

- Architecture: DSVT-Pillar with TransFusion head.
- DSVT repository: `https://github.com/Haiyang-W/DSVT`, commit
  `8cfc2a6f23eed0b10aabcdc4768c60b184357061`, package `0.6.0+8cfc2a6`.
- Reference OpenPCDet audit: `https://github.com/open-mmlab/OpenPCDet`, commit
  `233f849829b6ac19afb8af8837a0246890908755`, setup-derived version
  `0.6.0+233f849` (audit checkout, not installed).
- Config: `tools/cfgs/dsvt_models/dsvt_plain_1f_onestage_nusences.yaml` (upstream spelling),
  SHA256 `b0832e03ad11d4e0b61f0fb07d977e687763caae472a4f87ed750bdc2d13be0f`.
- Checkpoint: `DSVT_Nuscenes_val.pth`, 28,665,215 bytes, SHA256
  `a675149d095eef8ddc0c137ae46eeac075ccc504c7608162c71e7adf318793fb`.
- Official source URL:
  `https://drive.google.com/file/d/10d7c-uJxg5w4GN-JmRBQi4gQDwHiOHxP/view?usp=drive_link`.
- Official repository-reported nuScenes validation context: 66.4 mAP and 71.1 NDS. These values
  were not re-benchmarked by LaserPerception.

No unofficial checkpoint conversion, training, fine-tuning, model surgery, or KITTI model
selection was used. The machine-readable identity is
[`configs/m8/dsvt_nuscenes_pillar.json`](../../configs/m8/dsvt_nuscenes_pillar.json).
