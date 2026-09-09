# M8 P1-S1 cloud runtime migration

> **Status: DRAFT ENGINEERING MIGRATION PLAN — NO CLOUD SCIENTIFIC INFERENCE AUTHORIZED.**

This plan migrates a runtime; it does not redesign or amend the frozen S1 protocol. The prior RTX
4060 Laptop GPU runtime is retired. Its runtime policy and primary authorization are historical,
machine-specific evidence. The authorization was never exercised (zero A2/E2 calls) and is **not
portable**. **OLD MACHINE STAGE R != CLOUD STAGE R.**

## Frozen state and authorization boundary

M1–M7 and v0.3.0 remain historical/frozen as applicable. M8 Detector V2 is active, DSVT-Pillar with
TransFusion is selected, P1-E is complete, and S1 is paused for migration. The retired-machine
Stage R is complete and reviewed. Cloud Stage R and cloud primary are not authorized; cloud
scientific detector calls are zero. Zero-intensity, S2, and training remain unauthorized.

The only permitted work before a new authorization is GT-blind engineering qualification. Do not
load ground truth, construct the detector for a scientific call, or retain a semantic result.

## Required cloud sequence

1. Select a fixed provider and GPU SKU.
2. Launch a clean Linux GPU worker.
3. Hydrate the exact repository and reviewed artifacts from GitHub and Drive.
4. Recreate the frozen software stack where possible and record every unavoidable difference.
5. Run a GT-blind environment/runtime audit.
6. Run a GT-blind capacity/timing preflight.
7. Capture a new cloud-machine runtime-policy binding.
8. Obtain owner review.
9. Obtain a **cloud Stage-R-only** authorization.
10. Repeat Stage R on the cloud GPU.
11. Upload and verify every Stage R process on Drive immediately.
12. Review and freeze cloud Stage R raw evidence.
13. Obtain a separate **cloud primary** authorization.
14. Run primary pass 1.
15. Upload and verify pass 1 immediately.
16. Run primary pass 2.
17. Upload and verify pass 2 immediately.
18. Run primary pass 3.
19. Upload and verify pass 3 immediately.
20. Aggregate the raw primary evidence under the frozen protocol.
21. Leave zero-intensity for a later, explicit authorization.

Any failed process is preserved and uploaded; it is not silently replaced. Scientific processes run
one at a time, and a cloud authorization binds to the recorded provider, SKU, GPU identity,
software, artifacts, and policy.

## Existing-evidence GPU assessment

The maximum frozen H10 condition contains 1,339,216 points and 32,774 retained dynamic pillars.
On the retired 8,585,216,000-byte device, that condition completed without OOM or pillar
truncation. Peak active allocation was 2,922,354,688 bytes, while PyTorch's cached reservation
reached 8,443,133,952 bytes. Earlier sizing reserved 7,073,693,696 bytes. The evidence establishes
that 8 GB was workable only under exclusive, single-process constraints; it does not qualify a new
GPU or establish a portable minimum.

**Practical target: one NVIDIA GPU with at least 16 GB VRAM**, exclusive to one S1 detector process,
with sufficient host memory and disk for the hydrated corpus and per-process evidence. More VRAM
reduces allocator pressure and migration risk; it does not waive qualification.

| GPU class | Suitability for this workload |
|---|---|
| RTX 4090 (24 GB) | Strong practical target with ample VRAM; consumer scheduling/telemetry and provider configuration must be frozen. |
| L40S (48 GB) | Excellent margin and datacenter operation; more capacity than the existing evidence requires. |
| A10/A10G (24 GB) | Suitable practical baseline with adequate VRAM; likely slower than newer classes, to be measured only after authorization. |
| A100 (40/80 GB) | Technically excellent but substantially beyond the demonstrated memory need; use only if operational availability motivates it. |

No provider or performance claim is made, and current pricing was not considered. The historical
partial TensorRT artifacts are diagnostic only: the scientific candidate remains the frozen eager
DSVT/OpenPCDet runtime unless owner-reviewed policy says otherwise.

## Cloud migration input inventory

Paths were found using repository identities, retirement accounting, and bounded Drive metadata;
large objects were not downloaded. A SHA marked `tree` identifies the recorded directory tree.

| Artifact | Canonical Drive path | Bytes | SHA256 | Status |
|---|---|---:|---|---|
| DSVT checkpoint | `_MACHINE_RETIREMENT_2026-09-06/04_MODELS_AND_DEPLOYMENT/DSVT_Nuscenes_val.pth` | 28,665,215 | `a675149d095eef8ddc0c137ae46eeac075ccc504c7608162c71e7adf318793fb` | Present; external official weight |
| Frozen candidate manifest | `_MACHINE_RETIREMENT_2026-09-06/03_SCIENTIFIC_EVIDENCE/tracked_critical_state/configs/m8/dsvt_nuscenes_pillar.json` | 4,915 | `aa456e0386e46e9d089a957b1f1a8a4f74ceae70435c7ad8e6ca5e67bb90f4e7` | Present and Git-tracked |
| Official DSVT config | `_MACHINE_RETIREMENT_2026-09-06/04_MODELS_AND_DEPLOYMENT/dsvt_plain_1f_onestage_nusences.yaml` | 5,048 | `b0832e03ad11d4e0b61f0fb07d977e687763caae472a4f87ed750bdc2d13be0f` | Present |
| Frozen input ledger | `_MACHINE_RETIREMENT_2026-09-06/03_SCIENTIFIC_EVIDENCE/tracked_critical_state/benchmarks/m8/diagnostics/m8_input_projection_ledger.json` | 669,345 | `474e87e34c64d669750d4b6f7a64ac46fc9c5c462693fad79ff7c9547a7f1f7c` | Present and Git-tracked |
| Input revalidation | repository `benchmarks/m8/diagnostics/m8_input_projection_revalidation.json` | 966 | `71ac9418c29da5efd64f9eaeb03e859f85d6b1c56dc2fe47cef6563a9f960341` | Git-tracked; no separate Drive mirror located |
| Old runtime policy | `_MACHINE_RETIREMENT_2026-09-06/03_SCIENTIFIC_EVIDENCE/tracked_critical_state/benchmarks/m8/preregistration/m8_s1_stage_r_runtime_policy.json` | 2,028 | `703e453a8bca0e6e2e4b1c4b976deaa5bc4ed27b3a4847144204193baab77563` | Historical only |
| Old Stage R raw evidence | `_MACHINE_RETIREMENT_2026-09-06/03_SCIENTIFIC_EVIDENCE/m8_s1_stage_r_external/` | 30,023,835 | tree `85fbffcb48f83d28ff74b7ab2bab77f99166090dc994a7436ac8b4fc6d3a3c7d` | Present; 190 files |
| Old Stage R compact evidence | `_MACHINE_RETIREMENT_2026-09-06/03_SCIENTIFIC_EVIDENCE/tracked_critical_state/benchmarks/m8/results/m8_s1_stage_r.json` | 28,402 | `1549386c9bc9185c4240082398c5e0e5a64dc066ee5695c462bbfefde873952f` | Present and Git-tracked |
| Frozen S1 protocol | `_MACHINE_RETIREMENT_2026-09-06/03_SCIENTIFIC_EVIDENCE/tracked_critical_state/benchmarks/m8/preregistration/m8_s1_protocol.json` | 15,956 | `c132f60257c6a39debb548461c79bd59c98325484d233db6095b441c638d8e88` | Present and Git-tracked |
| Measurement runtime implementation | repository at `f23de4d3bbf538166f25ba6fa5e96dccd3a5c7e3` (sizing) / `2655c26ce3f4298438a80f265fb3884c7046e40c` (capacity) | — | Git object identity | Present in authoritative Git history |
| Primary authorization history | `_MACHINE_RETIREMENT_2026-09-06/03_SCIENTIFIC_EVIDENCE/tracked_critical_state/benchmarks/m8/preregistration/m8_s1_primary_authorization.json` | 2,044 | `0f67b939dd57fd782ca55a525f609c58d0349b216f2ed89a697e328995cb4ddd` | Historical, non-portable, zero calls |
| DSVT source | `Haiyang-W/DSVT` | — | Git commit `8cfc2a6f23eed0b10aabcdc4768c60b184357061` | Reconstructible exact checkout |
| OpenPCDet audit source | `open-mmlab/OpenPCDet` | — | Git commit `233f849829b6ac19afb8af8837a0246890908755` | Reconstructible exact checkout |

### Dataset inventory

| Dataset/input | Drive path | Required content | Status |
|---|---|---|---|
| KITTI Raw drive 0001 | `.local/assets/m6b/` and `.local/external/kitti_raw/` | `2011_09_26_drive_0001_sync`, Velodyne, OXTS, tracklets | Present; hydrate only required files |
| KITTI Raw drive 0091 | `.local/assets/m6b/` and `.local/external/kitti_raw/` | `2011_09_26_drive_0091_sync`, Velodyne, OXTS, tracklets | Present; hydrate only required files |
| Shared date calibration | same roots | `2011_09_26` camera/Velodyne/IMU calibration | Present; verify recorded identities |
| nuScenes source-domain data | no retirement-backup copy | Config-selected official subset | Re-download only if an authorized task needs it |

KITTI and nuScenes remain subject to upstream terms and are never redistributed through GitHub.
Drive presence is not execution permission.

## Bootstrap handoff

`scripts/cloud/bootstrap_gpu_worker.sh` checks Linux, NVIDIA visibility, repository identity, Drive
connectivity, reviewed artifact hashes, and exact DSVT/OpenPCDet commits. Dependencies are installed
only when the owner supplies a hash-pinned requirements file. The script then stops explicitly
before inference. The artifact manifest and rclone configuration are external inputs; neither
credentials nor private machine paths belong in Git.
