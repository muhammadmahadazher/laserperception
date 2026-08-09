# CVGC Reference Investigation

## Scope and provenance

This note records a time-boxed, read-only investigation of the public
[CVGC-DG repository](https://github.com/KintomZi/CVGC-DG), the implementation
linked from the paper
[Cross-view Domain Generalization via Geometric Consistency for LiDAR Semantic Segmentation](https://arxiv.org/abs/2602.14525).

- Investigated on: 2026-08-09
- Default branch: `main`
- Inspected commit: `10c2aac0635f79508c3702481875aa3dc18a161c`
- Repository license: none declared and no license file found
- Dependency manifest: none found
- CVGC source copied into LaserPerception: no

The absent license means the repository is used only as a behavioral and
configuration reference. No implementation code, model weights, or dataset
artifacts were imported.

## Verified interface and preprocessing behavior

The committed data loader expects structured NumPy files with `x`, `y`, `z`,
and `label` fields. Dataset paths and file lists are supplied in YAML. The
Group 2 configuration names source-domain STPLS3D data and Toronto3D and DALES
target-domain data. Its path names contain `S50(0)-50(0)`, which suggests
50 m by 50 m prepared tiles, but the repository does not include the tool that
creates those tiles. This is therefore preprocessing provenance, not an
implemented patch-generation rule.

The inspected Group 2 configuration and loader use:

| Item | Committed reference setting |
| --- | --- |
| Coordinate initialization | Per-cloud minimum XYZ subtraction (`center_type: -1`) |
| Voxel size | 0.3 m |
| Input channels | XYZ only (`input_dim: 3`) |
| Sparse quantization | Floor-scaled coordinates followed by MinkowskiEngine sparse quantization |
| Training augmentation | `DGLSS`: Z rotation, XY flip, and isotropic scale |
| Shared output classes | Ground, Building, Tree, Car, Light pole, Fence |
| Ignore label | -1 |

The loader uses the transformed coordinates as input features when no separate
features are present. LaserPerception's experiment config intentionally keeps
the reproducible, backend-independent subset of these choices: 0.3 m voxels,
minimum-XYZ normalization, and XYZ input. Its ontology remains independently
defined and tested against the official source-dataset label definitions.

## Model and optimization snapshot

The public tree defines a MinkowskiEngine `MinkUNet34`-style sparse U-Net. The
Group 2 YAML contains reference training settings including 100 epochs, batch
sizes of 4/2/2 for train/validation/test, Adam with a 0.001 learning rate, and
a test interval of 10 epochs. These values describe the inspected upstream
configuration; they are not LaserPerception benchmark claims or recommended
defaults.

The basic entry point creates a cross-entropy loss with ignore index -1. In the
checked-in pipeline, each training iteration forwards the `origin` variant and
optimizes only its semantic loss. The collation layer recognizes optional
occupancy fields (`coords_Occ`, `label_Occ`, `index_Occ`, and `weight_Occ`), but
the inspected training step does not consume them. No clearly runnable
cross-view augmentation plus geometric-consistency objective was found in the
committed entry point.

This does not establish that the paper's full method was never implemented; it
only describes the cited repository at the pinned commit. Reproducing the full
method would require clarification or additional artifacts from the authors.

## Evaluation behavior

The evaluation pipeline constructs a confusion matrix after excluding ignored
labels and derives per-class intersection-over-union, mean IoU, overall
accuracy, mean accuracy, and mean F1. The validation and test loaders are both
constructed from the configured test folder and test file list in the inspected
pipeline, so independent validation splits must be verified before using its
model-selection results.

No numerical result from the repository has been copied into
`docs/BENCHMARKS.md`. LaserPerception continues to use the exact placeholder
`Pending measurement` until results are produced by a logged run.

## Local reproduction status

The repository was inspected but not installed or executed. The local reference
machine had:

- Windows 11, 64-bit
- Python 3.12.10
- NVIDIA GeForce RTX 4060 Laptop GPU (8 GB class)
- NVIDIA driver 610.88
- CUDA compiler 12.9
- No PyTorch or MinkowskiEngine in the LaserPerception development environment

Installing the upstream stack was outside the core-library bootstrap and would
have required inferring unpinned dependencies from an unlicensed repository.
No installation failure is claimed because installation was not attempted.

## Implications and next steps

1. Keep CVGC-related choices isolated behind optional experiment/backend code;
   do not add MinkowskiEngine to the CPU core dependency set.
2. Implement and validate explicit SemanticKITTI and DALES dataset-directory
   adapters before adding a sparse training backend.
3. Treat tile generation as a separately versioned preprocessing step with
   manifests, checksums, split provenance, and tests.
4. Ask the CVGC authors for the intended license, dependency lock information,
   preprocessing scripts, and the full geometric-consistency training path
   before attempting a faithful reproduction.
5. Record every future run's environment, seed, ontology mapping, split, and
   artifact hashes before publishing metrics.
