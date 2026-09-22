# Contributing

LaserPerception welcomes focused contributions to its CPU perception platform, reviewed data
adapters, deterministic tracking, semantic-result infrastructure, detector/deployment wrappers,
documentation, and reproducibility tooling.

Read [AGENTS.md](AGENTS.md), [PROJECT_STATUS.md](docs/PROJECT_STATUS.md), and the relevant protocol
before changing evidence or M8 code. Historical accepted, failed, rejected, and incomplete records
must remain intact.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src
python -m pytest
python -m build
git diff --check
```

Normal development is CPU-only. Tests must skip optional GPU/ROS integrations before hardware
discovery. Do not probe or assume a local GPU. PyTorch, CUDA, OpenMMLab, DSVT/OpenPCDet, TensorRT,
and ROS environments are optional and separately authorized.

## Contribution boundaries

- Keep the core wheel lightweight and importable without GPU or ROS dependencies.
- Use explicit coordinates, units, feature order, timestamps, taxonomies, and artifact identities.
- Keep tracking deterministic and based on explicitly timed `DetectionFrame` input.
- Keep semantic results row-aligned and identity-bound; do not imply a segmentation model exists.
- Reuse canonical readers and registries when adding an adapter. Document limitations and optional
  dependencies.
- Do not change frozen detector, evaluator, protocol, or evidence semantics without an explicitly
  scoped scientific review.
- Never commit datasets, point-cloud captures, checkpoints, weights, engines, archives, raw cloud
  logs, credentials, private paths, or unreviewed visualizations.

## Evidence and documentation

Measurements require exact commit, configuration, upstream versions, artifact hashes, data
identity, environment, hardware, timing boundaries, and memory method. Use `Pending measurement`
for unknown values. Preserve negative and failed evidence with its original status.

External evaluations must be labeled external, document provenance and claim boundaries, and omit
raw third-party material unless redistribution rights are established.

## Pull requests

Use a focused branch and Conventional Commit messages. Describe the final behavior, validation,
claim boundaries, and any intentionally preserved historical files. Before requesting review,
inspect the full diff, built wheel and sdist, secrets/private paths, and large files. GPU or ROS
results require their separately provisioned and authorized environments; CPU pull requests must
not manufacture them.
