# Independent OmniLink / OmniSim evaluation

## Claim boundary

OmniLink independently produced this synthetic evaluation of LaserPerception v0.3.0. Current
maintainers did not rerun it. It is not a canonical LaserPerception benchmark, dataset-level
accuracy result, official nuScenes or KITTI evaluation, physical-LiDAR validation, TensorRT
throughput evidence, M8 evidence, or a production claim.

The evaluated LaserPerception commit was
`0f93c480acb6c98bc07781db8ed64b8433ec9238`. The external report used the historical official
PointPillars checkpoint with SHA256
`f19d00a38e6b775f38a45a9a3ca3ecaec20a5585a3caf44622423e2d5f75d5d0` and score threshold 0.25.
The pinned MMDetection3D source commit was
`fe25f7a51d36e3702f961e198894580d83c4387b`.
Reported detector runtime was native PyTorch FP32 with PyTorch 2.1.0+cu118, MMDetection3D 1.4.0,
MMDetection 3.2.0, MMCV 2.1.0, MMEngine 0.10.7, and autocast disabled. Reported detector timings
on an RTX 3060 Laptop GPU exclude preprocessing and must not be compared with canonical internal
TensorRT measurements.

OmniSim is publicly available at <https://github.com/omnilink-tech/omnisim>. No OmniSim source,
binary, emailed script, or raw capture is redistributed here.

## Synthetic accumulation diagnostic

The deterministic synthetic sequence contained 11 LiDAR poses at 0.1 s intervals, sensor motion
from x=0.0 to 0.8 m, and a moving obstacle from y=-0.5 to +0.5 m. The adapter accepted 4,070 of
4,092 supplied rows and rejected 22 deliberately non-finite rows.

| Input | Points | Unique 0.25 m voxels | Repeat-observation voxels | Repeated fraction | Maximum observations/voxel |
|---|---:|---:|---:|---:|---:|
| One sweep | 370 | 269 | 38 | 0.141264 | 9 |
| Eleven sweeps | 4,070 | 317 | 311 | 0.981073 | 51 |

Static geometry aligned repeatably. The synthetic obstacle centroid error was approximately
0.000004 m for one sweep; the accumulated moving obstacle produced a 0.500 m history trail. This
was a geometric diagnostic, not a detector result.

## Sparse PointPillars result

The intended object was a `traffic_cone` at center [7.2, 0.5, 0.65] m, dimensions
[0.6, 0.7, 0.8] m, yaw 0. Matching required the exact class and at most 1.0 m 3D center error.

| Input | Points | Raw proposals | Detections ≥0.25 | Intended match | Median | Min | Max |
|---|---:|---:|---:|---|---:|---:|---:|
| One sweep | 370 | 2 | 0 | null | 0.133299999 s | 0.103750011 s | 0.173318120 s |
| Eleven sweeps | 4,070 | 3 | 0 | null | 0.115038864 s | 0.099573329 s | 0.116985814 s |

Each timing has five repetitions and excludes preprocessing. Backend initialization was reported as
26.565913873 s. Three rather than two raw proposals is not an accuracy improvement; neither
condition detected the intended object.

## Native OmniSim scene

The separate native scene used OmniSim's LiDAR point-cloud API at 1024 horizontal samples, 32
layers, 2.4 rad horizontal FOV, 0.52 rad vertical FOV, and 0.2–25 m range. It included a ground
slab, barriers, far wall, vehicle proxies, poles, and one authored traffic cone. The current sweep
contained 21,483 points; eleven sweeps accumulated 233,950 points. Capture time was 0.8868392 s.
The world authored traffic-cone center was [8.0, 0.5, 0.45] m; evaluation used the current-sensor
center [7.2, 0.5, -0.75] m, dimensions [0.7, 0.7, 0.9] m, and yaw 0.

| Input | Raw proposals | Detections ≥0.25 | Intended match | Median | Min | Max |
|---|---:|---:|---|---:|---:|---:|
| One sweep | 23 | 2 unrelated truck/car boxes | null | 0.190670852 s | 0.145792361 s | 0.331061805 s |
| Eleven sweeps | 28 | 0 | null | 0.261350272 s | 0.185615007 s | 0.397753021 s |

The denser native input did not produce a valid intended traffic-cone match. This is preserved as a
negative domain-gap result and does not prove that multi-sweep accumulation helps detection.

## Transform and reconstruction verification

For this identity-rotation trajectory, the physical relation is
`p_current = p_historical + (x_source - x_current)`. LaserPerception stores
`x_current - x_source`, and the builder subtracts that stored translation. An independent static
point test across ten historical poses had maximum absolute coordinate error
`2.861022947442393e-07` m. All constructed matrices exactly matched
`SweepTransform.from_poses`.

The rebuilt full input had shape [233950, 4], dtype float32, and was exactly equal to the preserved
model input, with maximum absolute difference 0.0. Rebuilt value SHA256 was
`e00d05c6562449d52a2ca7d84692e50b6ccc9c512d5cc5f2f4a3551383094472`; preserved NPY SHA256 was
`3e8fdddf277e5a173a92f38a4b3d71557a84940153216a4bc5e46054e6b4d107`. Because the detector input
was unchanged, no detector rerun was needed.

## Capture provenance

- Native capture archive SHA256:
  `01575b979a7b7b698e838aa1083a175a132edfc07070f032eb944efadb7693fd`
- OmniSim version: 8.1.17
- Source checkout: `64a083d652c931d29274ae549957b464852ffff9`
- Capture log start: 2026-09-01 21:47:58 Europe/Berlin
- Capture executable binary SHA256: null; the capture-time binary had been replaced before hashing

Raw correspondence and attachments remain external because redistribution rights were not
established.
