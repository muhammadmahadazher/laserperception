# Roadmap

Progress is evidence-gated. Dates and capabilities are not promised before prerequisites pass.

## V0.1 — data foundation

- [x] `PointCloud`, KITTI/SemanticKITTI I/O, LAS/LAZ loading
- [x] Explicit `min_xyz` normalization and verified six-class ontology
- [x] Experiment 001 config, directory adapters, deterministic patches, and dataset-audit CLI
- [ ] Run bounded real-dataset audits without committing data or generated reports

## Next — after real-dataset audit

- Deliberate sparse backend selection and minimal sparse-voxel baseline
- SemanticKITTI source training and zero-shot DALES evaluation
- One ablation selected from observed failure evidence

Further domains, tasks, and deployment work are conditional. See [VISION.md](VISION.md).
