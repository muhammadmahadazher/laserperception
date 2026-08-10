# M1 detection environment

This document records the isolated environment that was actually validated for M1. The base
LaserPerception package remains lightweight; none of these CUDA packages are mandatory core
dependencies.

## Verified matrix

| Component | Verified value |
|---|---|
| Host | Windows 11 with WSL 2.7.11.0 |
| Linux | Ubuntu 22.04.5 LTS, WSL kernel 6.18.33.2 |
| Python | 3.10.12 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB reported |
| NVIDIA driver | 610.88 |
| PyTorch | 2.1.0+cu118 |
| PyTorch CUDA runtime | 11.8 |
| torchvision | 0.16.0+cu118 |
| MMDetection3D | 1.4.0, commit `fe25f7a51d36e3702f961e198894580d83c4387b` |
| MMDetection | 3.2.0 |
| MMCV | 2.1.0 prebuilt `cu118/torch2.1` wheel |
| MMEngine | 0.10.7 |
| NumPy | 1.26.4 |
| OpenCV | 4.10.0.84 |
| GCC/G++ | 11.4.0 |

This selection satisfies MMDetection3D 1.4.0's official compatibility ranges while using an
official prebuilt MMCV wheel instead of compiling CUDA operators in the Google Drive workspace.
The environment was validated with a CUDA matrix multiplication, imports of `mmcv.ops.nms` and
`mmcv.ops.Voxelization`, and initialization of the pinned PointPillars checkpoint on `cuda:0`.

The driver may advertise a newer maximum CUDA capability in `nvidia-smi`; M1 reports the CUDA 11.8
runtime bundled with PyTorch, not that driver capability, as its runtime.

## Prerequisites

Install an Ubuntu 22.04 WSL distribution, enable NVIDIA GPU access in WSL, and verify that
`nvidia-smi` shows the intended GPU. Inside Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y build-essential curl git python3.10 python3.10-venv
```

If the repository is on a Google Drive virtual drive that WSL does not automount, mount that drive
for the current WSL session. For a Windows `J:` drive, for example:

```bash
sudo mkdir -p /mnt/j
sudo mount -t drvfs J: /mnt/j
cd /mnt/j/path/to/laserperception
```

The drive letter and repository path are local choices, not assumptions in source code.

## Reproduce the environment

From the repository root inside WSL:

```bash
bash scripts/setup_detection_m1.sh
source ~/.venvs/laserperception-m1/bin/activate
```

The script pins every version above, installs LaserPerception editable, runs `pip check`, verifies
CUDA with a small operation, prints the detected GPU, and imports compiled MMCV operators. It
accepts any NVIDIA GPU that CUDA and the pinned stack can use; the RTX 4060 Laptop GPU remains the
canonical hardware for the committed M1 measurement, not an installation requirement.

The cache root is `LASERPERCEPTION_M1_CACHE` when that variable is set and
`~/.cache/laserperception` otherwise. The setup and runtime scripts derive the checkout as
`<cache_root>/mmdetection3d-v1.4.0` and the checkpoint as
`<cache_root>/checkpoints/<checkpoint filename>`. For example:

```bash
export LASERPERCEPTION_M1_CACHE=/path/to/external/m1-cache
bash scripts/setup_detection_m1.sh
```

The setup script clones only the pinned official checkout and downloads the official checkpoint
into that external cache, verifies the checkpoint SHA256, and rejects a different upstream commit
or checkpoint checksum. It is idempotent.

The setup script requires network access to the official PyTorch, OpenMMLab, GitHub, and Python
package sources. It never creates a virtual environment, framework clone, checkpoint, or cache in
the repository.

## Verify an existing environment

```bash
source ~/.venvs/laserperception-m1/bin/activate
python - <<'PY'
import torch
import mmcv
import mmdet
import mmdet3d
import mmengine
from mmcv.ops import Voxelization, nms

print("torch", torch.__version__, "CUDA runtime", torch.version.cuda)
print("CUDA available", torch.cuda.is_available())
print("GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("mmcv", mmcv.__version__, "mmengine", mmengine.__version__)
print("mmdet", mmdet.__version__, "mmdet3d", mmdet3d.__version__)
print("compiled operators", nms.__name__, Voxelization.__name__)
PY
```

An importable environment is necessary but not sufficient evidence for an M1 benchmark. A result is
accepted only after actual nuScenes-mini inference completes and the benchmark records its hardware,
software, asset, sample, timing, memory, timestamp, and commit provenance.

## Primary upstream references

- [MMDetection3D v1.4.0 release](https://github.com/open-mmlab/mmdetection3d/releases/tag/v1.4.0)
- [MMDetection3D compatibility table](https://github.com/open-mmlab/mmdetection3d/blob/v1.4.0/docs/en/notes/faq.md)
- [PyTorch previous-version CUDA 11.8 packages](https://pytorch.org/get-started/previous-versions/)
- [Official OpenMMLab MMCV wheel index](https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html)
